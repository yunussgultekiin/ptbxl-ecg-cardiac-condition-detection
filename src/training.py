import argparse
import copy
from dataclasses import dataclass
import matplotlib
import numpy as np
import torch
from torch import nn
from src.architectures import ARCHITECTURES, ArchitectureFactory
from src.augmentations import AUGMENTATIONS, AugmentationFactory, NoAugmentation
from src.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CANDIDATE_ALIAS, CONFIDENCE_LEVEL, PATIENT_COLUMN, REGISTERED_MODEL_PREFIX, SUPERCLASSES, ProjectPaths, Runtime, TrainingConfig
from src.data import BatchStream, FoldNormalizer, GpuCohort, NormalizationStore
from src.metrics import MultilabelScores, PatientBootstrap, ThresholdSearch
from src.plots import RunFigures
from src.tracking import CONFIDENCE_INTERVALS_FILE, CheckpointContext, CheckpointStore, ModelRegistry, RunArtifacts, RunHistory, RunLogger, TrackingStore

TEST_SUBSETS = ("validated", "certain")
DECAY_MODULES = (nn.Conv1d, nn.Linear, nn.LSTM)
PARAMETRIZATION_PREFIX = "parametrizations."
AUGMENTED_RUN_SEPARATOR = "__"
SUMMARY_COLUMNS = (("records", "records", None), ("macro_auroc", "AUROC", "auroc"), ("macro_auprc", "AP", None), ("macro_precision", "precision", None), ("macro_recall", "recall", None), ("macro_f1", "F1", "f1"))
CLASS_COLUMNS = (("threshold", "threshold", None), ("auroc", "AUROC", "auroc"), ("precision", "precision", None), ("recall", "recall", None), ("f1", "F1", "f1"))
LABEL_WIDTH = 16
VALUE_WIDTH = 11
INTERVAL_WIDTH = 25

class Predictor:
    def __init__(self, model, use_amp, augmentation=None):
        self.model = model
        self.use_amp = use_amp
        self.augmentation = augmentation or NoAugmentation()
        self.loss = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict(self, batches):
        self.model.eval()
        logits, labels = [], []
        for signals, metadata, batch_labels in batches:
            with Runtime.autocast(self.use_amp):
                logits.append(self.augmentation.predict(self.model, signals, metadata).float())
            labels.append(batch_labels)
        logits, labels = torch.cat(logits), torch.cat(labels)
        return self.loss(logits, labels).item(), torch.sigmoid(logits).cpu(), labels.cpu().long()

class Trainer:
    def __init__(self, model, config, steps_per_epoch, augmentation=None):
        self.model = model.to(Runtime.device)
        self.config = config
        self.augmentation = augmentation or NoAugmentation()
        self.loss = nn.BCEWithLogitsLoss()
        self.predictor = Predictor(self.model, config.use_amp, self.augmentation)
        self.optimizer = torch.optim.AdamW(Trainer.parameter_groups(self.model, config.weight_decay), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.max_epochs * steps_per_epoch,
            pct_start=config.warmup_fraction
        )

    @staticmethod
    def parameter_groups(model, weight_decay):
        decayed = {
            id(parameter)
            for module in model.modules() if isinstance(module, DECAY_MODULES)
            for name, parameter in module.named_parameters() if name.removeprefix(PARAMETRIZATION_PREFIX).startswith("weight")
        }
        parameters = list(model.parameters())
        return [
            {"params": [parameter for parameter in parameters if id(parameter) in decayed], "weight_decay": weight_decay},
            {"params": [parameter for parameter in parameters if id(parameter) not in decayed], "weight_decay": 0.0}
        ]

    def train_epoch(self, batches):
        self.model.train()
        epoch_logits, epoch_labels = [], []
        for signals, metadata, labels in batches:
            signals, metadata, targets, clean = self.augmentation.apply(signals, metadata, labels)
            self.optimizer.zero_grad(set_to_none=True)
            with Runtime.autocast(self.config.use_amp):
                logits = self.model(signals, metadata).float()
            loss = self.loss(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            epoch_logits.append(logits.detach()[clean])
            epoch_labels.append(labels[clean])
        logits, labels = torch.cat(epoch_logits), torch.cat(epoch_labels)
        return self.loss(logits, labels).item(), torch.sigmoid(logits).cpu(), labels.cpu().long()

class ResultsTable:
    @staticmethod
    def widths(columns):
        return [VALUE_WIDTH if metric is None else INTERVAL_WIDTH for _, _, metric in columns]

    @staticmethod
    def row(label, values, widths):
        return f"{label:<{LABEL_WIDTH}}" + "".join(f"{value:>{width}}" for value, width in zip(values, widths))

    @staticmethod
    def value(value):
        return f"{value:,}" if isinstance(value, int) else f"{value:.4f}"

    @staticmethod
    def interval(value, estimate):
        if estimate is None or estimate["lower"] is None:
            return ResultsTable.value(value)
        return f"{ResultsTable.value(value)} [{estimate['lower']:.4f}, {estimate['upper']:.4f}]"

    @staticmethod
    def splits(architecture, split_scores, intervals):
        widths = ResultsTable.widths(SUMMARY_COLUMNS)
        lines = [f"{architecture} | results with OOF thresholds and {intervals['confidence_level']:.0%} CI", ResultsTable.row("split", [title for _, title, _ in SUMMARY_COLUMNS], widths)]
        for split, scores in split_scores.items():
            estimates = intervals["sections"].get(split, {}).get("macro", {})
            lines.append(ResultsTable.row(split, [ResultsTable.interval(scores[key], estimates.get(metric)) for key, _, metric in SUMMARY_COLUMNS], widths))
        return lines

    @staticmethod
    def classes(architecture, scores, thresholds, intervals):
        widths = ResultsTable.widths(CLASS_COLUMNS)
        lines = [f"{architecture} | test per class", ResultsTable.row("class", [title for _, title, _ in CLASS_COLUMNS], widths)]
        for name, threshold in zip(SUPERCLASSES, thresholds.tolist()):
            estimates = intervals["sections"]["test"]["classes"][name]
            values = [f"{threshold:.2f}"] + [ResultsTable.interval(scores[f"{key}_{name}"], estimates.get(metric)) for key, _, metric in CLASS_COLUMNS[1:]]
            lines.append(ResultsTable.row(name, values, widths))
        return lines

    @staticmethod
    def show(architecture, split_scores, thresholds, intervals):
        lines = ResultsTable.splits(architecture, split_scores, intervals) + ResultsTable.classes(architecture, split_scores["test"], thresholds, intervals)
        print("\n".join(lines), flush=True)

@dataclass
class FoldOutcome:
    fold: int
    best_epoch: int
    best_auroc: float
    labels: torch.Tensor
    probabilities: torch.Tensor

class ConfidenceIntervalReport:
    def __init__(self, records, plan):
        self.records = records
        self.plan = plan
        self.bootstrap = PatientBootstrap(BOOTSTRAP_REPLICATES, CONFIDENCE_LEVEL, BOOTSTRAP_SEED)

    def sections(self, oof_predictions, test_predictions, split_scores):
        oof_predictions = tuple(tensor.cpu() for tensor in oof_predictions)
        test_labels, test_probabilities = (tensor.cpu() for tensor in test_predictions)
        test_patients = self.records.patient_ids(self.plan.test_ids)
        sections = {"oof": (*oof_predictions, self.records.patient_ids(self.plan.ids(self.plan.folds))), "test": (test_labels, test_probabilities, test_patients)}
        masks = self.records.subset_masks(self.plan.test_ids)
        for subset in TEST_SUBSETS:
            mask = torch.tensor(masks[subset])
            sections[f"test_{subset}"] = (test_labels[mask], test_probabilities[mask], test_patients[masks[subset]])
        return {name: (*arrays, split_scores[name]) for name, arrays in sections.items()}

    def build(self, thresholds, oof_predictions, test_predictions, split_scores):
        return self.bootstrap.report(self.sections(oof_predictions, test_predictions, split_scores), thresholds, SUPERCLASSES, PATIENT_COLUMN)

class ArchitecturePipeline:
    def __init__(self, architecture, config, paths=ProjectPaths(), augmentation=None):
        AugmentationFactory.validate(augmentation, architecture)
        self.architecture = architecture
        self.augmentation = augmentation
        self.config = config
        self.paths = paths
        self.normalization = NormalizationStore(paths)
        if not self.normalization.is_current():
            raise RuntimeError("normalization statistics are missing or outdated, run ./run.sh prepare")
        Runtime.setup(config.seed)
        self.store = TrackingStore(paths).connect()
        self.logger = RunLogger(paths)
        self.artifacts = RunArtifacts(self.store)
        self.checkpoints = CheckpointStore(paths)
        self.registry = ModelRegistry(self.store)
        self.cohort = GpuCohort(paths)
        self.plan = self.cohort.plan
        self.confidence_intervals = ConfidenceIntervalReport(self.cohort.records, self.plan)

    @property
    def run_name(self):
        return self.architecture if self.augmentation is None else f"{self.architecture}{AUGMENTED_RUN_SEPARATOR}{self.augmentation}"

    @property
    def registered_name(self):
        return f"{REGISTERED_MODEL_PREFIX}-{self.run_name.replace(AUGMENTED_RUN_SEPARATOR, '-')}"

    def build_model(self):
        Runtime.seed_everything(self.config.seed)
        return ArchitectureFactory.build(self.architecture, self.config.head_dropout)

    def build_augmentation(self, normalizer, seed):
        return AugmentationFactory.build(self.augmentation, normalizer, seed)

    def stream(self, rows, normalizer, shuffle=False, seed=0):
        return BatchStream(self.cohort, rows, normalizer, self.config.batch_size, shuffle, seed)

    def context(self, model_config, normalizer, stage, epoch, fold=None, thresholds=None):
        return CheckpointContext(self.architecture, model_config, self.config.head_dropout, SUPERCLASSES, stage, epoch, fold, normalizer.statistics, thresholds, self.augmentation)

    @staticmethod
    def scored(split, loss, scores):
        return f"{split} loss {loss:.4f} AUROC {scores['macro_auroc']:.4f} F1 {scores['macro_f1']:.4f}"

    def report(self, stage, epoch, train_loss, train_scores, val_loss=None, val_scores=None):
        validation = "" if val_loss is None else f" | {self.scored('val', val_loss, val_scores)}"
        print(f"{self.run_name} | {stage} | epoch {epoch:02d}/{self.config.max_epochs} | {self.scored('train', train_loss, train_scores)}{validation}", flush=True)

    @staticmethod
    def log_epoch(split, loss, labels, probabilities, epoch):
        scores = MultilabelScores.monitoring(labels, probabilities, SUPERCLASSES)
        RunLogger.log_metrics(split, {"loss": loss} | scores, step=epoch)
        return scores

    def train_fold(self, fold, parent_run_id):
        partition = f"cv_{fold}"
        normalizer = FoldNormalizer(self.normalization.statistics(partition))
        model, model_config = self.build_model()
        train_stream = self.stream(self.cohort.fold_rows(self.plan.train_folds(partition)), normalizer, shuffle=True, seed=self.config.seed + fold)
        val_stream = self.stream(self.cohort.fold_rows([fold]), normalizer)
        trainer = Trainer(model, self.config, len(train_stream), self.build_augmentation(normalizer, self.config.seed + fold))
        best_auroc, best_epoch, best_state, stale_epochs = -np.inf, 0, None, 0
        with self.logger.child_run("fold", fold):
            for epoch in range(1, self.config.max_epochs + 1):
                train_loss, train_probabilities, train_labels = trainer.train_epoch(train_stream)
                val_loss, probabilities, labels = trainer.predictor.predict(val_stream)
                train_scores = self.log_epoch("train", train_loss, train_labels, train_probabilities, epoch)
                val_scores = self.log_epoch("val", val_loss, labels, probabilities, epoch)
                self.report(f"fold {fold}", epoch, train_loss, train_scores, val_loss, val_scores)
                if val_scores["macro_auroc"] > best_auroc:
                    best_auroc, best_epoch, best_state, stale_epochs = val_scores["macro_auroc"], epoch, copy.deepcopy(model.state_dict()), 0
                else:
                    stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
            model.load_state_dict(best_state)
            _, probabilities, labels = trainer.predictor.predict(val_stream)
            RunLogger.log_metrics("best", {"epoch": best_epoch, "val_macro_auroc": best_auroc})
            checkpoint_path = self.checkpoints.run_path(self.run_name, parent_run_id, "fold", fold)
            self.checkpoints.save(checkpoint_path, model, self.context(model_config, normalizer, "fold", best_epoch, fold))
        return FoldOutcome(fold, best_epoch, best_auroc, labels, probabilities)

    def cross_validate(self, parent_run_id):
        outcomes = [self.train_fold(fold, parent_run_id) for fold in self.plan.folds]
        labels = torch.cat([outcome.labels for outcome in outcomes])
        probabilities = torch.cat([outcome.probabilities for outcome in outcomes])
        thresholds = ThresholdSearch.fit(labels, probabilities)
        final_epoch = int(np.median([outcome.best_epoch for outcome in outcomes]))
        fold_aurocs = [outcome.best_auroc for outcome in outcomes]
        RunLogger.log_metrics("cv", {"val_macro_auroc_mean": np.mean(fold_aurocs), "val_macro_auroc_std": np.std(fold_aurocs), "final_epoch": final_epoch})
        oof_scores = MultilabelScores.summary(labels, probabilities, SUPERCLASSES, thresholds) | {"records": len(labels)}
        RunLogger.log_metrics("oof", oof_scores)
        RunLogger.log_metrics("oof", {f"threshold_{name}": threshold for name, threshold in zip(SUPERCLASSES, thresholds.tolist())})
        return thresholds, final_epoch, oof_scores, (labels, probabilities)

    def train_final(self, parent_run_id, thresholds, final_epoch):
        normalizer = FoldNormalizer(self.normalization.statistics("final"))
        model, model_config = self.build_model()
        rows = self.cohort.fold_rows(self.plan.folds)
        train_stream = self.stream(rows, normalizer, shuffle=True, seed=self.config.seed + len(self.plan.folds))
        trainer = Trainer(model, self.config, len(train_stream), self.build_augmentation(normalizer, self.config.seed + len(self.plan.folds)))
        with self.logger.child_run("final") as final_run:
            for epoch in range(1, final_epoch + 1):
                train_loss, train_probabilities, train_labels = trainer.train_epoch(train_stream)
                self.report("final", epoch, train_loss, self.log_epoch("train", train_loss, train_labels, train_probabilities, epoch))
            _, probabilities, labels = trainer.predictor.predict(self.stream(rows, normalizer))
            train_scores = MultilabelScores.summary(labels, probabilities, SUPERCLASSES, thresholds) | {"records": len(labels)}
            RunLogger.log_metrics("train_eval", train_scores)
            threshold_map = {name: round(threshold, 2) for name, threshold in zip(SUPERCLASSES, thresholds.tolist())}
            checkpoint_path = self.checkpoints.save(
                self.checkpoints.run_path(self.run_name, parent_run_id, "final"),
                model,
                self.context(model_config, normalizer, "final", final_epoch, thresholds=threshold_map)
            )
            self.registry.register(model, self.registered_name, checkpoint_path, CANDIDATE_ALIAS, self.cohort.example_inputs(normalizer))
        return final_run.info.run_id, model, normalizer, train_scores

    def evaluate(self, final_run_id, parent_run_id, model, normalizer, thresholds, final_epoch, oof_scores, oof_predictions):
        _, probabilities, labels = Predictor(model, self.config.use_amp, self.build_augmentation(normalizer, self.config.seed)).predict(self.stream(self.cohort.test_rows(), normalizer))
        masks = {subset: torch.tensor(mask) for subset, mask in self.cohort.records.subset_masks(self.plan.test_ids).items()}
        test_scores = {"test": MultilabelScores.summary(labels, probabilities, SUPERCLASSES, thresholds) | {"records": len(labels)}}
        for subset in TEST_SUBSETS:
            mask = masks[subset]
            test_scores[f"test_{subset}"] = MultilabelScores.summary(labels[mask], probabilities[mask], SUPERCLASSES, thresholds) | {"records": mask.sum().item()}
        intervals = self.confidence_intervals.build(thresholds, oof_predictions, (labels, probabilities), {"oof": oof_scores} | test_scores)
        with RunLogger.resume(final_run_id):
            for split, scores in test_scores.items():
                RunLogger.log_metrics(split, scores)
        self.artifacts.log_figure(parent_run_id, RunFigures.roc_curve(labels, probabilities, SUPERCLASSES), "roc_curve.png")
        self.artifacts.log_figure(parent_run_id, RunFigures.confusion_matrix(labels, probabilities, thresholds, SUPERCLASSES), "confusion_matrix.png")
        self.artifacts.log_figure(parent_run_id, RunFigures.loss_curve(RunHistory(self.store).fold_loss_histories(parent_run_id), final_epoch), "loss_curve.png")
        self.artifacts.log_dict(parent_run_id, intervals, CONFIDENCE_INTERVALS_FILE)
        return test_scores, intervals

    def run(self):
        model_config = {f"model_{name}": value for name, value in ArchitectureFactory.default_config(self.architecture).items()}
        params = {"architecture": self.architecture} | self.config.as_params() | model_config | AugmentationFactory.params(self.augmentation)
        with self.logger.parent_run(self.run_name, self.architecture, params, self.augmentation) as parent_run:
            thresholds, final_epoch, oof_scores, oof_predictions = self.cross_validate(parent_run.info.run_id)
            final_run_id, model, normalizer, train_scores = self.train_final(parent_run.info.run_id, thresholds, final_epoch)
        test_scores, intervals = self.evaluate(final_run_id, parent_run.info.run_id, model, normalizer, thresholds, final_epoch, oof_scores, oof_predictions)
        ResultsTable.show(self.run_name, {"train": train_scores, "oof": oof_scores} | test_scores, thresholds, intervals)

if __name__ == "__main__":
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=sorted(ARCHITECTURES))
    parser.add_argument("--augmentation", choices=sorted(AUGMENTATIONS))
    arguments = parser.parse_args()
    ArchitecturePipeline(arguments.architecture, TrainingConfig(), augmentation=arguments.augmentation).run()