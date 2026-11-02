from dataclasses import asdict
from src.architectures.attention import Transformer1d, Transformer1dConfig
from src.architectures.base import EcgClassifier
from src.architectures.convolutional import (
    Cnn1d,
    Cnn1dConfig,
    Fcn,
    FcnConfig,
    Inception1d,
    Inception1dConfig,
    ResNet1d,
    ResNet1dConfig,
    SeResNet1d,
    SeResNet1dConfig,
    XResNet1d,
    XResNet1dConfig
)
from src.architectures.recurrent import CnnBiLstm, CnnBiLstmConfig
from src.architectures.state_space import S4D, S4DConfig
from src.architectures.temporal import Tcn, TcnConfig
from src.config import LEAD_NAMES, METADATA_COLUMNS, SUPERCLASSES

ARCHITECTURES = {
    "cnn1d": (Cnn1d, Cnn1dConfig),
    "fcn": (Fcn, FcnConfig),
    "resnet1d": (ResNet1d, ResNet1dConfig),
    "xresnet1d": (XResNet1d, XResNet1dConfig),
    "se_resnet1d": (SeResNet1d, SeResNet1dConfig),
    "inception1d": (Inception1d, Inception1dConfig),
    "tcn": (Tcn, TcnConfig),
    "cnn_bilstm": (CnnBiLstm, CnnBiLstmConfig),
    "transformer1d": (Transformer1d, Transformer1dConfig),
    "s4d": (S4D, S4DConfig)
}

class ArchitectureFactory:
    @staticmethod
    def default_config(name):
        return asdict(ARCHITECTURES[name][1]())

    @staticmethod
    def build(name, head_dropout, overrides=None):
        backbone_class, config_class = ARCHITECTURES[name]
        config = config_class(**{key: tuple(value) if isinstance(value, list) else value for key, value in (overrides or {}).items()})
        model = EcgClassifier(backbone_class(len(LEAD_NAMES), config), len(SUPERCLASSES), len(METADATA_COLUMNS), head_dropout)
        return model, asdict(config)

    @staticmethod
    def from_checkpoint(checkpoint):
        model, _ = ArchitectureFactory.build(checkpoint["model_name"], checkpoint["head_dropout"], checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()