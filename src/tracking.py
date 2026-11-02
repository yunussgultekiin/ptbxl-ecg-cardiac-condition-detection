import copy
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, TensorSpec
from src.config import EXPERIMENT_NAME, PYTHON_VERSION, Runtime
from src.data import FileDigest

STAGE_TAG = "stage"
FOLD_TAG = "fold"
MODEL_FAMILY_TAG = "model_family"
AUGMENTATION_TAG = "augmentation"
CHECKPOINT_PATH_TAG = "checkpoint_path"
CHECKPOINT_SHA_TAG = "checkpoint_sha256"
USER_TAG = "mlflow.user"
RUN_USER = "root"
SOURCE_TAG = "mlflow.source.name"
RUN_SOURCE = "src/training.py"
FIGURES_DIR = "figures"
CONFIDENCE_INTERVALS_FILE = "confidence_intervals.json"
REGISTRY_MODEL_NAME = "model"
SERIALIZATION_FORMAT = "pt2"
SIGNATURE_INPUTS = ("signals", "metadata")
MODEL_PACKAGES = ("torch", "mlflow")
BUILD_PACKAGES = ("pip", "setuptools", "wheel")
CONDA_CHANNEL = "conda-forge"
CONDA_ENV_NAME = "mlflow-env"
REQUIREMENTS_FILE_NAME = "requirements.txt"
PYTHON_ENV_FILE_NAME = "python_env.yaml"
ARTIFACT_LOCATION_UPDATE = "update experiments set artifact_location = ? || '/' || name"

class TrackingStore:
    def __init__(self, paths, experiment_name=EXPERIMENT_NAME):
        self.paths = paths
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None

    @property
    def artifact_location(self):
        return self.paths.artifacts_dir / self.experiment_name

    def connect(self):
        for directory in (self.artifact_location, self.paths.run_checkpoints_dir, self.paths.registry_checkpoints_dir):
            directory.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.paths.tracking_uri)
        mlflow.set_registry_uri(self.paths.tracking_uri)
        self.client = MlflowClient()
        self.experiment_id = self.ensure_experiment()
        mlflow.set_experiment(experiment_id=self.experiment_id)
        return self

    def ensure_experiment(self):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(self.experiment_name)
            self.relative_artifact_locations()
            return experiment_id
        if experiment.lifecycle_stage == "deleted":
            self.client.restore_experiment(experiment.experiment_id)
        return experiment.experiment_id

    def relative_artifact_locations(self):
        with sqlite3.connect(self.paths.database_file) as database:
            database.execute(ARTIFACT_LOCATION_UPDATE, (self.paths.relative(self.paths.artifacts_dir),))

class RunLogger:
    def __init__(self, paths):
        self.paths = paths

    def parent_run(self, run_name, model_family, params, augmentation=None):
        tags = {USER_TAG: RUN_USER, SOURCE_TAG: RUN_SOURCE, STAGE_TAG: "cv", MODEL_FAMILY_TAG: model_family} | ({} if augmentation is None else {AUGMENTATION_TAG: augmentation})
        active_run = mlflow.start_run(run_name=run_name, tags=tags)
        lineage = {f"data_{name}": digest for name, digest in FileDigest.lineage(self.paths).items()}
        mlflow.log_params(params | lineage)
        return active_run

    @staticmethod
    def child_run(stage, fold=None):
        tags = {USER_TAG: RUN_USER, SOURCE_TAG: RUN_SOURCE, STAGE_TAG: stage} | ({} if fold is None else {FOLD_TAG: str(fold)})
        return mlflow.start_run(run_name=stage if fold is None else f"{stage}_{fold}", nested=True, tags=tags)

    @staticmethod
    def resume(run_id):
        return mlflow.start_run(run_id=run_id)

    @staticmethod
    def log_metrics(prefix, values, step=None):
        mlflow.log_metrics({f"{prefix}/{name}": float(value) for name, value in values.items()}, step=step)

class RunArtifacts:
    def __init__(self, store):
        self.client = store.client

    def log_figure(self, run_id, figure, file_name):
        self.client.log_figure(run_id, figure, f"{FIGURES_DIR}/{file_name}")
        plt.close(figure)

    def log_dict(self, run_id, payload, file_name):
        self.client.log_dict(run_id, payload, file_name)

class RunHistory:
    def __init__(self, store):
        self.client = store.client
        self.experiment_id = store.experiment_id

    def child_runs(self, parent_run_id, stage):
        filter_string = f"tags.mlflow.parentRunId = '{parent_run_id}' and tags.{STAGE_TAG} = '{stage}'"
        runs = self.client.search_runs([self.experiment_id], filter_string=filter_string)
        return sorted(runs, key=lambda run: int(run.data.tags.get(FOLD_TAG, -1)))

    def metric_curve(self, run_id, key):
        return [metric.value for metric in sorted(self.client.get_metric_history(run_id, key), key=lambda metric: metric.step)]

    def fold_loss_histories(self, parent_run_id):
        return [{split: self.metric_curve(run.info.run_id, f"{split}/loss") for split in ("train", "val")} for run in self.child_runs(parent_run_id, "fold")]

@dataclass(frozen=True)
class CheckpointContext:
    model_name: str
    model_config: dict
    head_dropout: float
    class_names: tuple
    stage: str
    epoch: int
    fold: int | None = None
    normalization: dict | None = None
    thresholds: dict | None = None
    augmentation: str | None = None

class CheckpointStore:
    def __init__(self, paths):
        self.paths = paths

    def run_path(self, architecture, parent_run_id, stage, fold=None):
        return self.paths.run_checkpoints_dir / architecture / parent_run_id / (f"{stage}.pt" if fold is None else f"{stage}_{fold}.pt")

    def save(self, path, model, context):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "run_id": mlflow.active_run().info.run_id,
            "data_lineage": FileDigest.lineage(self.paths),
            **asdict(context)
        }
        torch.save(payload, path)
        mlflow.set_tags({CHECKPOINT_PATH_TAG: self.paths.relative(path), CHECKPOINT_SHA_TAG: FileDigest.sha256(path)})
        return path

    @staticmethod
    def load(path):
        return torch.load(path, map_location=Runtime.device, weights_only=True)

class ProjectEnvironment:
    def __init__(self, paths):
        self.paths = paths

    def pinned_versions(self):
        lines = [line.strip() for line in self.paths.requirements_file.read_text().splitlines()]
        return dict(line.split("==", 1) for line in lines if "==" in line)

    def pinned(self, packages):
        versions = self.pinned_versions()
        missing = [package for package in packages if package not in versions]
        if missing:
            raise RuntimeError(f"{self.paths.relative(self.paths.requirements_file)} does not pin {', '.join(missing)}")
        return [f"{package}=={versions[package]}" for package in packages]

    @property
    def pip_requirements(self):
        return self.pinned(MODEL_PACKAGES)

    @property
    def build_dependencies(self):
        return self.pinned(BUILD_PACKAGES)

    @property
    def conda_env(self):
        pip_dependency = self.pinned(("pip",))[0].replace("==", "<=")
        return {
            "channels": [CONDA_CHANNEL],
            "dependencies": [f"python={PYTHON_VERSION}", pip_dependency, {"pip": self.pip_requirements}],
            "name": CONDA_ENV_NAME
        }

    @property
    def python_env(self):
        return {"python": PYTHON_VERSION, "build_dependencies": self.build_dependencies, "dependencies": [f"-r {REQUIREMENTS_FILE_NAME}"]}

    def write_python_env(self, directory):
        path = Path(directory) / PYTHON_ENV_FILE_NAME
        path.write_text(yaml.safe_dump(self.python_env, sort_keys=False, default_flow_style=False))
        return path

class ModelRegistry:
    def __init__(self, store):
        self.paths = store.paths
        self.client = store.client
        self.environment = ProjectEnvironment(store.paths)
        torch.fx.config.do_not_emit_stack_traces = True

    def export_path(self, registered_name, version):
        return self.paths.registry_checkpoints_dir / registered_name / f"v{version}.pt"

    @staticmethod
    def signature(model, input_example):
        with torch.no_grad():
            output = model(*[torch.from_numpy(array) for array in input_example])
        inputs = Schema([TensorSpec(array.dtype, (-1,) + array.shape[1:], name) for array, name in zip(input_example, SIGNATURE_INPUTS)])
        return ModelSignature(inputs=inputs, outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, output.shape[1]))]))

    def register(self, model, registered_name, checkpoint_path, alias, input_example):
        portable_model = copy.deepcopy(model).cpu().eval()
        model_info = mlflow.pytorch.log_model(
            portable_model,
            name=REGISTRY_MODEL_NAME,
            registered_model_name=registered_name,
            input_example=input_example,
            signature=self.signature(portable_model, input_example),
            serialization_format=SERIALIZATION_FORMAT,
            conda_env=self.environment.conda_env
        )
        self.client.set_logged_model_tags(model_info.model_id, {USER_TAG: RUN_USER, SOURCE_TAG: RUN_SOURCE})
        with tempfile.TemporaryDirectory() as directory:
            self.client.log_model_artifact(model_info.model_id, str(self.environment.write_python_env(directory)))
        version = model_info.registered_model_version
        export_path = self.export_path(registered_name, version)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, export_path)
        self.client.set_model_version_tag(registered_name, version, CHECKPOINT_PATH_TAG, self.paths.relative(export_path))
        self.client.set_model_version_tag(registered_name, version, CHECKPOINT_SHA_TAG, FileDigest.sha256(export_path))
        self.client.set_registered_model_alias(registered_name, alias, version)
        return version

    def version(self, registered_name, alias):
        return self.client.get_model_version_by_alias(registered_name, alias)

    def load_checkpoint(self, registered_name, alias):
        return CheckpointStore.load(self.export_path(registered_name, self.version(registered_name, alias).version))

    @staticmethod
    def load_model(registered_name, alias):
        return mlflow.pytorch.load_model(f"models:/{registered_name}@{alias}")