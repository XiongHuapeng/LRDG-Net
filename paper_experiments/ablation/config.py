"""Final LRDG-Net G/A/R/G+A/G+R ablation configuration.

Only the five representation-level variants reported in the manuscript are included here.
All artifacts stay inside the main workspace under ``cache/ablation``,
``models/ablation`` and ``evaluations/ablation``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from config import PATHS as MAIN_PATHS, RESEARCH as MAIN_RESEARCH, TRAIN as MAIN_TRAIN, TEST as MAIN_TEST


@dataclass(frozen=True)
class PathConfig:
    lidaroc_data_root: str = MAIN_PATHS.lidaroc_data_root
    workspace_root: str = MAIN_PATHS.workspace_root
    @property
    def cache_root(self) -> str: return str(Path(self.workspace_root) / "cache" / "ablation")
    @property
    def model_root(self) -> str: return str(Path(self.workspace_root) / "models" / "ablation")
    @property
    def evaluation_root(self) -> str: return str(Path(self.workspace_root) / "evaluations" / "ablation")


@dataclass(frozen=True)
class DataConfig:
    target_classes: Tuple[str, ...] = ("clean", "water", "dust", "mud", "oil")
    voxel_size: float = 0.20
    context_radius_voxels: int = 1
    intensity_scale: float = 255.0
    relative_std_eps: float = 1e-3
    normalization_eps: float = 1e-6
    overwrite_cache: bool = False


@dataclass(frozen=True)
class ModelConfig:
    variants: Tuple[str, ...] = ("E0_G", "E1_A", "E2_R", "E3_GA", "E4_GR")
    geometry_in_channels: int = 12
    absolute_degradation_channels: int = 3
    relative_degradation_channels: int = 4
    geometry_hidden: Tuple[int, int] = (32, 32)
    degradation_hidden: Tuple[int, int] = (16, 32)
    instance_channels: int = 64
    classifier_hidden: int = 64
    dropout: float = 0.20
    num_classes: int = 2


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
class ResearchConfig:
    domains: Tuple[str, ...] = MAIN_RESEARCH.domains
    seeds: Tuple[int, ...] = MAIN_RESEARCH.seeds


PATHS = PathConfig(); DATA = DataConfig(); MODEL = ModelConfig(); TRAIN = TrainConfig(); TEST = TestConfig(); RESEARCH = ResearchConfig()


def ensure_dirs():
    for p in (PATHS.cache_root, PATHS.model_root, PATHS.evaluation_root):
        Path(p).mkdir(parents=True, exist_ok=True)
