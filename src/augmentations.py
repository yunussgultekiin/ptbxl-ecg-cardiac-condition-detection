import math
from dataclasses import asdict, dataclass
import numpy as np
import torch
from src.architectures import ArchitectureFactory
from src.config import LEAD_NAMES, Runtime

FIXED_LENGTH_FIELD = "signal_length"
VECTOR_AXES = 3
INDEPENDENT_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6", "I", "II")
INVERSE_DOWER = (
    (-0.172, -0.074, 0.122, 0.231, 0.239, 0.194, 0.156, -0.010),
    (0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227, 0.887),
    (-0.229, -0.310, -0.246, -0.063, 0.055, 0.108, 0.022, 0.102)
)
DOWER = (
    (-0.515, 0.157, -0.917),
    (0.044, 0.164, -1.387),
    (0.882, 0.098, -1.277),
    (1.213, 0.127, -0.601),
    (1.125, 0.127, -0.086),
    (0.831, 0.076, 0.230),
    (0.632, -0.235, 0.059),
    (0.235, 1.066, -0.132)
)
DERIVED_LEADS = {"III": (-1.0, 1.0), "AVR": (-0.5, -0.5), "AVL": (1.0, -0.5), "AVF": (-0.5, 1.0)}

class Augmentation:
    changes_length = False

    def __init__(self, config=None, normalizer=None, seed=0):
        self.config = config
        self.normalizer = normalizer
        self.generator = torch.Generator(device=Runtime.device).manual_seed(seed)

    def uniform(self, *size):
        return torch.rand(size, generator=self.generator, device=Runtime.device)

    def selected(self, count):
        return self.uniform(count) < self.config.probability

    def segments(self, count, length, selected):
        starts = torch.randint(length - self.config.window + 1, (count, 1), generator=self.generator, device=Runtime.device)
        positions = torch.arange(length, device=Runtime.device).unsqueeze(0)
        return (selected.unsqueeze(1) & (positions >= starts) & (positions < starts + self.config.window)).unsqueeze(1)

    def partners(self, count):
        return torch.randperm(count, generator=self.generator, device=Runtime.device)

    @staticmethod
    def unchanged_labels(labels):
        return torch.ones(len(labels), dtype=torch.bool, device=labels.device)

    def apply(self, signals, metadata, labels):
        return signals, metadata, labels, self.unchanged_labels(labels)

    def predict(self, model, signals, metadata):
        return model(signals, metadata)

class NoAugmentation(Augmentation):
    pass

@dataclass(frozen=True)
class RandomCropConfig:
    window: int = 250
    stride: int = 125

class RandomCrop(Augmentation):
    changes_length = True

    def apply(self, signals, metadata, labels):
        count, channels, length = signals.shape
        starts = torch.randint(length - self.config.window + 1, (count, 1), generator=self.generator, device=Runtime.device)
        positions = (starts + torch.arange(self.config.window, device=Runtime.device)).unsqueeze(1).expand(-1, channels, -1)
        return signals.gather(2, positions), metadata, labels, self.unchanged_labels(labels)

    def predict(self, model, signals, metadata):
        count, channels, _ = signals.shape
        windows = signals.unfold(2, self.config.window, self.config.stride).permute(0, 2, 1, 3)
        window_count = windows.size(1)
        logits = model(windows.reshape(-1, channels, self.config.window), metadata.repeat_interleave(window_count, dim=0))
        return logits.view(count, window_count, -1).amax(dim=1)

@dataclass(frozen=True)
class MixupConfig:
    alpha: float = 0.4
    probability: float = 0.5

class Mixup(Augmentation):
    def __init__(self, config, normalizer, seed):
        super().__init__(config, normalizer, seed)
        self.beta = np.random.default_rng(seed)

    def weights(self, selected):
        drawn = self.beta.beta(self.config.alpha, self.config.alpha, len(selected)).astype(np.float32)
        return torch.where(selected, torch.from_numpy(drawn).to(Runtime.device), 1.0)

    def apply(self, signals, metadata, labels):
        selected = self.selected(len(signals))
        partners = self.partners(len(signals))
        weights = self.weights(selected).unsqueeze(1)
        mixed_signals = weights.unsqueeze(2) * signals + (1 - weights.unsqueeze(2)) * signals[partners]
        mixed_metadata = weights * metadata + (1 - weights) * metadata[partners]
        return mixed_signals, mixed_metadata, weights * labels + (1 - weights) * labels[partners], ~selected

@dataclass(frozen=True)
class CutMixConfig:
    window: int = 200
    probability: float = 0.5

class CutMix(Augmentation):
    def apply(self, signals, metadata, labels):
        count, _, length = signals.shape
        selected = self.selected(count)
        partners = self.partners(count)
        share = selected.unsqueeze(1) * self.config.window / length
        mixed_signals = torch.where(self.segments(count, length, selected), signals[partners], signals)
        return mixed_signals, metadata, (1 - share) * labels + share * labels[partners], ~selected

@dataclass(frozen=True)
class VectorRotationConfig:
    max_degrees: float = 10.0
    probability: float = 0.5

class VectorRotation(Augmentation):
    def __init__(self, config, normalizer, seed):
        super().__init__(config, normalizer, seed)
        leads = {name: index for index, name in enumerate(LEAD_NAMES)}
        to_vector = torch.zeros(VECTOR_AXES, len(LEAD_NAMES))
        to_leads = torch.zeros(len(LEAD_NAMES), VECTOR_AXES)
        for position, name in enumerate(INDEPENDENT_LEADS):
            to_vector[:, leads[name]] = torch.tensor(INVERSE_DOWER)[:, position]
            to_leads[leads[name]] = torch.tensor(DOWER[position])
        for name, (lead_i, lead_ii) in DERIVED_LEADS.items():
            to_leads[leads[name]] = lead_i * to_leads[leads["I"]] + lead_ii * to_leads[leads["II"]]
        self.to_vector = to_vector.to(Runtime.device)
        self.to_leads = to_leads.to(Runtime.device)

    def rotations(self, selected):
        limit = math.radians(self.config.max_degrees)
        angles = (2 * self.uniform(len(selected), VECTOR_AXES) - 1) * limit * selected.unsqueeze(1)
        cosine, sine = angles.cos(), angles.sin()
        zero, one = torch.zeros_like(cosine[:, 0]), torch.ones_like(cosine[:, 0])
        around_x = torch.stack([one, zero, zero, zero, cosine[:, 0], -sine[:, 0], zero, sine[:, 0], cosine[:, 0]], dim=1)
        around_y = torch.stack([cosine[:, 1], zero, sine[:, 1], zero, one, zero, -sine[:, 1], zero, cosine[:, 1]], dim=1)
        around_z = torch.stack([cosine[:, 2], -sine[:, 2], zero, sine[:, 2], cosine[:, 2], zero, zero, zero, one], dim=1)
        return around_z.view(-1, VECTOR_AXES, VECTOR_AXES) @ around_y.view(-1, VECTOR_AXES, VECTOR_AXES) @ around_x.view(-1, VECTOR_AXES, VECTOR_AXES)

    def apply(self, signals, metadata, labels):
        selected = self.selected(len(signals))
        millivolts = signals * self.normalizer.lead_std + self.normalizer.lead_mean
        vectors = self.to_vector @ millivolts
        shifted = (self.rotations(selected) - torch.eye(VECTOR_AXES, device=Runtime.device)) @ vectors
        return signals + self.to_leads @ shifted / self.normalizer.lead_std, metadata, labels, self.unchanged_labels(labels)

@dataclass(frozen=True)
class TimeMaskConfig:
    window: int = 100
    probability: float = 0.5

class TimeMask(Augmentation):
    def apply(self, signals, metadata, labels):
        count, _, length = signals.shape
        return signals.masked_fill(self.segments(count, length, self.selected(count)), 0.0), metadata, labels, self.unchanged_labels(labels)

AUGMENTATIONS = {
    "crop": (RandomCrop, RandomCropConfig),
    "mixup": (Mixup, MixupConfig),
    "cutmix": (CutMix, CutMixConfig),
    "rotation": (VectorRotation, VectorRotationConfig),
    "time_mask": (TimeMask, TimeMaskConfig)
}

class AugmentationFactory:
    @staticmethod
    def params(name):
        if name is None:
            return {}
        return {"augmentation": name} | {f"augmentation_{key}": value for key, value in asdict(AUGMENTATIONS[name][1]()).items()}

    @staticmethod
    def validate(name, architecture):
        if name is not None and AUGMENTATIONS[name][0].changes_length and FIXED_LENGTH_FIELD in ArchitectureFactory.default_config(architecture):
            raise ValueError(f"{architecture} has a fixed input length and cannot be trained with {name}")

    @staticmethod
    def build(name, normalizer, seed):
        if name is None:
            return NoAugmentation()
        augmentation_class, config_class = AUGMENTATIONS[name]
        return augmentation_class(config_class(), normalizer, seed)
