import hashlib
import json
import numpy as np
import pandas as pd
import torch
from src.config import CERTAIN_LIKELIHOOD, LEAD_NAMES, LIKELIHOOD_COLUMNS, METADATA_COLUMNS, PATIENT_COLUMN, SUPERCLASSES, VALIDATED_COLUMN, ProjectPaths, Runtime

STATISTICS_CHUNK_SIZE = 1024
FINAL_PARTITION = "final"

class FileDigest:
    @staticmethod
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def lineage(paths):
        return {f"{path.stem}_sha256": FileDigest.sha256(path) for path in paths.lineage_files}

class SplitPlan:
    def __init__(self, split):
        self.test_ids = split["test"]
        self.fold_ids = {int(fold): ids for fold, ids in split["folds"].items()}

    @classmethod
    def load(cls, paths):
        return cls(json.loads(paths.split_file.read_text()))

    @property
    def folds(self):
        return sorted(self.fold_ids)

    @property
    def partitions(self):
        return [f"cv_{fold}" for fold in self.folds] + [FINAL_PARTITION]

    def train_folds(self, partition):
        if partition == FINAL_PARTITION:
            return self.folds
        held_out = int(partition.removeprefix("cv_"))
        return [fold for fold in self.folds if fold != held_out]

    def ids(self, folds):
        return [ecg_id for fold in folds for ecg_id in self.fold_ids[fold]]

class CohortRecords:
    def __init__(self, paths):
        self.frame = pd.read_parquet(paths.records_file).set_index("ecg_id")
        self.row_of = pd.Series(np.arange(len(self.frame)), index=self.frame.index)

    def rows(self, ecg_ids):
        return self.row_of.loc[ecg_ids].to_numpy()

    def labels(self):
        return self.frame[list(SUPERCLASSES)].to_numpy(np.float32)

    def metadata(self):
        return self.frame[list(METADATA_COLUMNS)].to_numpy(np.float32)

    def patient_ids(self, ecg_ids):
        return self.frame.loc[ecg_ids, PATIENT_COLUMN].to_numpy(np.int64)

    def subset_masks(self, ecg_ids):
        subset = self.frame.loc[ecg_ids]
        return {
            "validated": subset[VALIDATED_COLUMN].to_numpy(bool),
            "certain": ~subset[list(LIKELIHOOD_COLUMNS)].lt(CERTAIN_LIKELIHOOD).any(axis=1).to_numpy()
        }

class NormalizationStatistics:
    @staticmethod
    def compute(signals, rows, ages):
        lead_sum = np.zeros(signals.shape[1])
        lead_square_sum = np.zeros(signals.shape[1])
        for start in range(0, len(rows), STATISTICS_CHUNK_SIZE):
            chunk = signals[np.sort(rows[start:start + STATISTICS_CHUNK_SIZE])].astype(np.float64)
            lead_sum += chunk.sum(axis=(0, 2))
            lead_square_sum += np.square(chunk).sum(axis=(0, 2))
        count = len(rows) * signals.shape[2]
        lead_mean = lead_sum / count
        return {
            "lead_mean": lead_mean.tolist(),
            "lead_std": np.sqrt(lead_square_sum / count - np.square(lead_mean)).tolist(),
            "age_mean": float(ages[rows].mean()),
            "age_std": float(ages[rows].std())
        }

class NormalizationStore:
    def __init__(self, paths):
        self.paths = paths

    def is_current(self):
        if not self.paths.normalization_file.exists():
            return False
        return self.read()["source"]["lineage"] == FileDigest.lineage(self.paths)

    def read(self):
        return json.loads(self.paths.normalization_file.read_text())

    def statistics(self, partition):
        return self.read()["statistics"][partition]

    def prepare(self):
        if self.is_current():
            print(f"normalization statistics are up to date: {self.paths.relative(self.paths.normalization_file)}")
            return
        records = CohortRecords(self.paths)
        plan = SplitPlan.load(self.paths)
        signals = np.load(self.paths.signals_file, mmap_mode="r")
        ages = records.frame["age"].to_numpy(np.float64)
        statistics = {}
        for partition in plan.partitions:
            rows = records.rows(plan.ids(plan.train_folds(partition)))
            statistics[partition] = {"train_folds": plan.train_folds(partition)} | NormalizationStatistics.compute(signals, rows, ages)
            print(f"{partition}: {len(rows)} training records")
        payload = {
            "source": {"signals_file": self.paths.signals_file.name, "lineage": FileDigest.lineage(self.paths)},
            "lead_names": list(LEAD_NAMES),
            "statistics": statistics
        }
        self.paths.normalization_file.write_text(json.dumps(payload, indent=2))
        print(f"normalization statistics written: {self.paths.relative(self.paths.normalization_file)}")

class FoldNormalizer:
    def __init__(self, statistics):
        self.statistics = statistics
        self.lead_mean = torch.tensor(statistics["lead_mean"], dtype=torch.float32, device=Runtime.device).view(1, -1, 1)
        self.lead_std = torch.tensor(statistics["lead_std"], dtype=torch.float32, device=Runtime.device).view(1, -1, 1)

    def signals(self, signals):
        return (signals - self.lead_mean) / self.lead_std

    def metadata(self, metadata):
        age = (metadata[:, :1] - self.statistics["age_mean"]) / self.statistics["age_std"]
        return torch.cat([age, metadata[:, 1:]], dim=1)

class GpuCohort:
    def __init__(self, paths):
        self.records = CohortRecords(paths)
        self.plan = SplitPlan.load(paths)
        self.signals = torch.from_numpy(np.load(paths.signals_file)).to(Runtime.device)
        self.labels = torch.tensor(self.records.labels(), device=Runtime.device)
        self.metadata = torch.tensor(self.records.metadata(), device=Runtime.device)

    def rows(self, ecg_ids):
        return torch.tensor(self.records.rows(ecg_ids), device=Runtime.device)

    def fold_rows(self, folds):
        return self.rows(self.plan.ids(folds))

    def test_rows(self):
        return self.rows(self.plan.test_ids)

    def example_inputs(self, normalizer, count=2):
        rows = self.test_rows()[:count]
        return normalizer.signals(self.signals[rows]).cpu().numpy(), normalizer.metadata(self.metadata[rows]).cpu().numpy()

class BatchStream:
    def __init__(self, cohort, rows, normalizer, batch_size, shuffle=False, seed=0):
        self.cohort = cohort
        self.rows = rows
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def order(self):
        if not self.shuffle:
            return self.rows
        return self.rows[torch.randperm(len(self.rows), generator=self.generator).to(self.rows.device)]

    def __len__(self):
        return len(self.rows) // self.batch_size if self.shuffle else -(-len(self.rows) // self.batch_size)

    def __iter__(self):
        order = self.order()
        stop = len(self) * self.batch_size if self.shuffle else len(order)
        for start in range(0, stop, self.batch_size):
            rows = order[start:start + self.batch_size]
            yield self.normalizer.signals(self.cohort.signals[rows]), self.normalizer.metadata(self.cohort.metadata[rows]), self.cohort.labels[rows]

if __name__ == "__main__":
    NormalizationStore(ProjectPaths()).prepare()