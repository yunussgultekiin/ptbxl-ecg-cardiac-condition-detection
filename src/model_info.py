import argparse
import inspect
import json
from pathlib import Path
import torch
import torchinfo
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from src.architectures import ARCHITECTURES, ArchitectureFactory
from src.config import LEAD_NAMES, METADATA_COLUMNS, PROJECT_ROOT, SIGNAL_LENGTH, SUPERCLASSES, ProjectPaths, TrainingConfig

INFO_DEVICE = "cpu"
TABLE_DEPTH = 4
LAYER_DEPTH = 20
INFO_BATCH_SIZE = 1
BYTES_PER_MEGABYTE = 1e6
FLOPS_PER_MULT_ADD = 2
JSON_INDENT = 2
INLINE_LIST_WIDTH = 100

class CompactJson:
    @staticmethod
    def scalar(value):
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def inline(value):
        if not isinstance(value, list) or any(isinstance(item, dict) for item in value):
            return None
        text = CompactJson.scalar(value)
        return text if len(text) <= INLINE_LIST_WIDTH else None

    @staticmethod
    def dumps(value, depth=0):
        inline = CompactJson.inline(value)
        if inline is not None:
            return inline
        if not isinstance(value, (dict, list)) or not value:
            return CompactJson.scalar(value)
        indent = " " * JSON_INDENT * (depth + 1)
        if isinstance(value, dict):
            items, brackets = [f"{CompactJson.scalar(key)}: {CompactJson.dumps(item, depth + 1)}" for key, item in value.items()], "{}"
        else:
            items, brackets = [CompactJson.dumps(item, depth + 1) for item in value], "[]"
        return f"{brackets[0]}\n" + ",\n".join(indent + item for item in items) + f"\n{' ' * JSON_INDENT * depth}{brackets[1]}"

class ModelInfo:
    def __init__(self, name, head_dropout):
        self.name = name
        self.head_dropout = head_dropout
        self.model, self.config = ArchitectureFactory.build(name, head_dropout)
        self.paths = {id(module): path or name for path, module in self.model.named_modules()}

    @property
    def input_sizes(self):
        return {"signals": [INFO_BATCH_SIZE, len(LEAD_NAMES), SIGNAL_LENGTH], "metadata": [INFO_BATCH_SIZE, len(METADATA_COLUMNS)]}

    @property
    def backbone_file(self):
        return inspect.getfile(type(self.model.backbone))

    @property
    def input_data(self):
        return [torch.zeros(size) for size in self.input_sizes.values()]

    def mult_adds(self, statistics):
        # torchinfo misses matmuls, sequence-wise linears and parametrized convolutions, while the
        # PyTorch flop counter misses LSTM kernels, so LSTM layers keep the torchinfo count
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            self.model(*self.input_data)
        recurrent = sum(layer.macs for layer in statistics.summary_list if isinstance(layer.module, nn.LSTM))
        return counter.get_total_flops() // FLOPS_PER_MULT_ADD + recurrent

    def statistics(self, depth):
        statistics = torchinfo.summary(self.model, input_data=self.input_data, depth=depth, device=INFO_DEVICE, verbose=0)
        statistics.total_mult_adds = self.mult_adds(statistics)
        return statistics

    def layer_record(self, layer):
        return {
            "name": layer.var_name,
            "path": self.paths.get(id(layer.module), layer.var_name),
            "class_name": layer.class_name,
            "depth": layer.depth,
            "input_size": layer.input_size,
            "output_size": layer.output_size,
            "num_parameters": layer.num_params,
            "trainable": layer.trainable_params > 0
        }

    def build(self):
        table = self.statistics(TABLE_DEPTH)
        layers = self.statistics(LAYER_DEPTH)
        return {
            "file_name": f"{self.name}.json",
            "artifact_type": "model_info",
            "model_name": self.name,
            "class_name": type(self.model).__name__,
            "backbone_class_name": type(self.model.backbone).__name__,
            "model_family": inspect.getmodulename(self.backbone_file),
            "origin": "custom backbone + ConcatPool + age/sex metadata + linear head, trained from scratch",
            "source_module": Path(self.backbone_file).relative_to(PROJECT_ROOT).as_posix(),
            "task_type": "multilabel_classification",
            "classes": list(SUPERCLASSES),
            "representation": "1d",
            "weight_strategy": "from_scratch",
            "weights": None,
            "config": self.config,
            "head_dropout": self.head_dropout,
            "input_size": self.input_sizes,
            "output_size": table.summary_list[0].output_size,
            "total_parameters": table.total_params,
            "trainable_parameters": table.trainable_params,
            "non_trainable_parameters": table.total_params - table.trainable_params,
            "total_mult_adds": table.total_mult_adds,
            "estimated_input_size_mb": table.total_input / BYTES_PER_MEGABYTE,
            "estimated_forward_backward_size_mb": table.total_output_bytes / BYTES_PER_MEGABYTE,
            "estimated_parameter_size_mb": table.total_param_bytes / BYTES_PER_MEGABYTE,
            "estimated_total_size_mb": (table.total_input + table.total_output_bytes + table.total_param_bytes) / BYTES_PER_MEGABYTE,
            "torch_version": torch.__version__,
            "torchinfo_version": torchinfo.__version__,
            "architecture": str(table).splitlines(),
            "layers": [self.layer_record(layer) for layer in layers.summary_list]
        }

class ModelInfoWriter:
    def __init__(self, paths):
        self.directory = paths.architecture_reports_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, info):
        path = self.directory / info["file_name"]
        path.write_text(CompactJson.dumps(info), encoding="utf-8")
        return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES))
    selected = parser.parse_args().architecture
    paths = ProjectPaths()
    writer = ModelInfoWriter(paths)
    for name in [selected] if selected else list(ARCHITECTURES):
        info = ModelInfo(name, TrainingConfig().head_dropout).build()
        path = writer.write(info)
        print(f"{name}: {info['total_parameters']:,} parameters | {info['total_mult_adds']:,} mult-adds | {paths.relative(path)}")