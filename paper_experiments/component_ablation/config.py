"""R-only component-ablation configuration used by the JEI experiments.

All user-editable data/workspace paths are inherited from the repository-level
``config.py``. Scientific hyperparameters mirror the representation-level ablation
protocol; only the selected subset of the four R components changes.
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
    def cache_root(self) -> str:
        return str(Path(self.workspace_root) / "cache" / "component_ablation")

    @property
    def model_root(self) -> str:
        return str(Path(self.workspace_root) / "models" / "component_ablation")

    @property
    def evaluation_root(self) -> str:
        return str(Path(self.workspace_root) / "evaluations" / "component_ablation")


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
    variants: Tuple[str, ...] = (
        "R_FULL",
        "R_ONLY_DELTA_I",
        "R_ONLY_LOG_STD_RATIO",
        "R_ONLY_LOCAL_STD",
        "R_ONLY_REL_LOG_DENSITY",
        "R_WO_DELTA_I",
        "R_WO_LOG_STD_RATIO",
        "R_WO_LOCAL_STD",
        "R_WO_REL_LOG_DENSITY",
    )
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


PATHS = PathConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
TEST = TestConfig()
RESEARCH = ResearchConfig()


def ensure_dirs() -> None:
    for path in (PATHS.cache_root, PATHS.model_root, PATHS.evaluation_root):
        Path(path).mkdir(parents=True, exist_ok=True)
