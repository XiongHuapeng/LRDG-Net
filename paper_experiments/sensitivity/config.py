"""Final compact sensitivity configuration.

This file matches the *actual* completed sensitivity grid:
- voxel size: 0.10, 0.20, 0.30, 0.40 m at radius=1;
- context radius: 1, 2, 3 at voxel size=0.20 m.

Sensitivity remains a one-seed secondary experiment and is not used for selecting
the already-frozen main LRDG-Net configuration.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from config import PATHS as MAIN_PATHS, TRAIN as MAIN_TRAIN, TEST as MAIN_TEST, RESEARCH as MAIN_RESEARCH, MODEL as MAIN_MODEL


@dataclass(frozen=True)
class PathConfig:
    lidaroc_data_root: str = MAIN_PATHS.lidaroc_data_root
    workspace_root: str = MAIN_PATHS.workspace_root
    @property
    def cache_root(self): return str(Path(self.workspace_root) / "cache" / "sensitivity")
    @property
    def model_root(self): return str(Path(self.workspace_root) / "models" / "sensitivity")
    @property
    def evaluation_root(self): return str(Path(self.workspace_root) / "evaluations" / "sensitivity")


@dataclass(frozen=True)
class DataConfig:
    target_classes: Tuple[str, ...] = ("clean", "water", "dust", "mud", "oil")
    intensity_scale: float = 255.0
    relative_std_eps: float = 1e-3
    normalization_eps: float = 1e-6
    overwrite_cache: bool = False


@dataclass(frozen=True)
class ModelConfig:
    geometry_in_channels: int = MAIN_MODEL.geometry_in_channels
    degradation_in_channels: int = MAIN_MODEL.degradation_in_channels
    geometry_hidden: Tuple[int, int] = MAIN_MODEL.geometry_hidden
    degradation_hidden: Tuple[int, int] = MAIN_MODEL.degradation_hidden
    instance_channels: int = MAIN_MODEL.instance_channels
    classifier_hidden: int = MAIN_MODEL.classifier_hidden
    dropout: float = MAIN_MODEL.dropout
    num_classes: int = MAIN_MODEL.num_classes


@dataclass(frozen=True)
class TrainConfig:
    val_ratio: float = MAIN_TRAIN.val_ratio
    split_seed: int = MAIN_TRAIN.split_seed
    epochs: int = MAIN_TRAIN.epochs
    batch_size: int = MAIN_TRAIN.batch_size
    learning_rate: float = MAIN_TRAIN.learning_rate
    weight_decay: float = MAIN_TRAIN.weight_decay
    early_stopping_patience: int = MAIN_TRAIN.early_stopping_patience
    num_workers: int = MAIN_TRAIN.num_workers
    device: str = MAIN_TRAIN.device
    overwrite_models: bool = False


@dataclass(frozen=True)
class TestConfig:
    batch_size: int = MAIN_TEST.batch_size
    num_workers: int = MAIN_TEST.num_workers


@dataclass(frozen=True)
class SensitivityConfig:
    domains: Tuple[str, ...] = MAIN_RESEARCH.domains
    seeds: Tuple[int, ...] = (42,)
    base_voxel_size: float = 0.20
    base_context_radius: int = 1
    voxel_sizes: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    context_radii: Tuple[int, ...] = (1, 2, 3)


PATHS = PathConfig(); DATA = DataConfig(); MODEL = ModelConfig(); TRAIN = TrainConfig(); TEST = TestConfig(); SENS = SensitivityConfig()


def ensure_dirs():
    for p in (PATHS.cache_root, PATHS.model_root, PATHS.evaluation_root):
        Path(p).mkdir(parents=True, exist_ok=True)
