import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEAD_NAMES = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
LIKELIHOOD_COLUMNS = tuple(f"likelihood_{superclass}" for superclass in SUPERCLASSES)
METADATA_COLUMNS = ("age", "sex")
VALIDATED_COLUMN = "validated_by_human"
PATIENT_COLUMN = "patient_id"
CERTAIN_LIKELIHOOD = 100
SAMPLING_RATE = 100
SIGNAL_LENGTH = 1000
EXPERIMENT_NAME = "ptbxl-superclass"
REGISTERED_MODEL_PREFIX = "ptbxl-superclass"
CANDIDATE_ALIAS = "candidate"
CUBLAS_WORKSPACE = ":4096:8"
PYTHON_VERSION = "3.12.13"
BOOTSTRAP_REPLICATES = 1000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 42

@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT

    @property
    def requirements_file(self):
        return self.root / "requirements.txt"

    @property
    def processed_dir(self):
        return self.root / "data" / "processed"

    @property
    def signals_file(self):
        return self.processed_dir / "cohort_signals_100hz.npy"

    @property
    def records_file(self):
        return self.processed_dir / "cohort_records.parquet"

    @property
    def split_file(self):
        return self.processed_dir / "split_folds.json"

    @property
    def preprocessing_report_file(self):
        return self.processed_dir / "preprocessing_report.json"

    @property
    def normalization_file(self):
        return self.processed_dir / "normalization_stats.json"

    @property
    def lineage_files(self):
        return (self.split_file, self.preprocessing_report_file)

    @property
    def mlflow_dir(self):
        return self.root / "mlflow"

    @property
    def architecture_reports_dir(self):
        return self.mlflow_dir / "reports" / "architectures"

    @property
    def database_file(self):
        return self.mlflow_dir / "mlflow.db"

    @property
    def artifacts_dir(self):
        return self.mlflow_dir / "artifacts"

    @property
    def run_checkpoints_dir(self):
        return self.mlflow_dir / "models" / "runs"

    @property
    def registry_checkpoints_dir(self):
        return self.mlflow_dir / "models" / "registry"

    @property
    def tracking_uri(self):
        return f"sqlite:///{self.database_file}"

    def relative(self, path):
        return str(Path(path).relative_to(self.root))

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.3
    gradient_clip: float = 1.0
    head_dropout: float = 0.5
    use_amp: bool = False
    seed: int = 42

    def as_params(self):
        return {f"training_{name}": value for name, value in asdict(self).items()}

class Runtime:
    device = torch.device("cuda")

    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def enforce_determinism():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def setup(seed):
        Runtime.enforce_determinism()
        Runtime.seed_everything(seed)
        return Runtime.device

    @staticmethod
    def autocast(enabled):
        return torch.autocast(device_type=Runtime.device.type, dtype=torch.bfloat16, enabled=enabled)