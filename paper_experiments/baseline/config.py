"""Unified baseline configuration.

Only the main project ``config.py`` owns user-editable paths. Baseline artifacts are
stored under the same LRDG_Net_Research workspace:

- cache/baselines/
- models/baselines/
- evaluations/baseline_cross_domain/

The baseline protocol is preserved from the validated baseline project.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from config import PATHS as MAIN_PATHS, RESEARCH as MAIN_RESEARCH, TRAIN as MAIN_TRAIN


@dataclass(frozen=True)
class PathConfig:
    lidaroc_data_root: str = MAIN_PATHS.lidaroc_data_root
    workspace_root: str = MAIN_PATHS.workspace_root
    lrdg_workspace_root: str = MAIN_PATHS.workspace_root

    @property
    def cache_root(self): return str(Path(self.workspace_root) / "cache" / "baselines")
    @property
    def model_root(self): return str(Path(self.workspace_root) / "models" / "baselines")
    @property
    def eval_root(self): return str(Path(self.workspace_root) / "evaluations")


@dataclass(frozen=True)
class DataConfig:
    target_classes: Tuple[str, ...] = ("clean", "water", "dust", "mud", "oil")
    num_points: int = 4096
    xyz_scale_m: float = 80.0
    intensity_scale: float = 255.0
    global_count_reference: int = 200000
    rangeview_h: int = 128
    rangeview_w: int = 1024
    rangeview_fov_up_deg: float = 15.0
    rangeview_fov_down_deg: float = -25.0
    rangeview_min_range_m: float = 0.1
    autogran_voxel_size: float = 0.20
    roi_x: Tuple[float, float] = (0.0, 80.0)
    roi_y: Tuple[float, float] = (-5.5, 10.0)
    overwrite_cache: bool = False


@dataclass(frozen=True)
class TrainConfig:
    val_ratio: float = 0.20
    split_seed: int = 42
    seeds: Tuple[int, ...] = MAIN_RESEARCH.seeds
    domains: Tuple[str, ...] = MAIN_RESEARCH.domains
    max_epochs: int = 60
    patience: int = 10
    num_workers: int = 0
    device: str = MAIN_TRAIN.device
    overwrite_models: bool = False


@dataclass(frozen=True)
class BaselineConfig:
    # Canonical IDs stay stable in result files; display names are defined below.
    baselines: Tuple[str, ...] = (
        "globalstats",
        "rangenet",
        "pointnet",
        "pointnet2_ref",
        "dgcnn",
        "pointnext",
        "autogran",
    )
    generic_lr: float = 1e-3
    generic_weight_decay: float = 1e-4
    globalstats_batch_size: int = 64
    pointnet_batch_size: int = 16
    pointnet2_batch_size: int = 16
    dgcnn_batch_size: int = 16
    pointnext_batch_size: int = 16
    dgcnn_k: int = 20
    rangenet21_batch_size: int = 2
    pointnext_width: int = 32
    pointnext_nsample: int = 32
    pointnext_radii_m: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    pointnext_optimizer: str = "adamw"
    autogran_lr: float = 0.005
    autogran_weight_decay: float = 5e-4
    autogran_batch_size: int = 16
    autogran_max_epochs: int = 20

    # Paper-facing names. RangeNet and PointNeXt are adapted frame-classification
    # implementations, not claims of exact official reproduction.
    display_names = {
        "globalstats": "Global-Statistics MLP",
        "rangenet": "RangeNet-style",
        "pointnet": "PointNet",
        "pointnet2_ref": "PointNet++",
        "dgcnn": "DGCNN",
        "pointnext": "PointNeXt-S-style",
        "autogran": "AutoGrAN",
        "LRDG-Net": "LRDG-Net",
    }


PATHS = PathConfig()
DATA = DataConfig()
TRAIN = TrainConfig()
BASELINES = BaselineConfig()


def ensure_dirs():
    for p in (PATHS.cache_root, PATHS.model_root, PATHS.eval_root):
        Path(p).mkdir(parents=True, exist_ok=True)
