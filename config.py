"""
LRDG-Net unified research configuration / LRDG-Net 统一科研配置。

Normal use:
1) Configure paths through environment variables or the fields in this file.
2) Run the numbered ``run_*.py`` entry points.
3) The experiment scripts do not require command-line arguments.

Scientific constraints kept by default:
- Paper-aligned LRDG-Net architecture and descriptors.
- Binary task only: clean=0, contaminated=1.
- AT128 levels 1/2/3 are all mapped to contaminated; they are analysis strata, not prediction classes.
- Normalization is fitted only on LIDAROC training sessions.
- AT128 is never used for normalization, checkpoint selection, or threshold tuning.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PathConfig:
    # ===================== ONLY EDIT THESE PATHS / 正常使用主要修改这里 =====================
    # LIDAROC 原始 .bin 数据根目录。
    lidaroc_data_root: str = os.getenv("LRDG_LIDAROC_ROOT", str(PROJECT_ROOT / "data" / "lidaroc"))

    # 本项目所有 cache / models / evaluations 的统一工作目录。
    workspace_root: str = os.getenv("LRDG_WORKSPACE_ROOT", str(PROJECT_ROOT / "outputs"))

    # 自建 AT128 PCAP 数据目录，例如包含 8m_0.pcap ... 20m_3.pcap。
    at128_pcap_root: str = os.getenv("LRDG_AT128_PCAP_ROOT", str(PROJECT_ROOT / "data" / "at128" / "pcap"))

    # 当前 AT128P 对应的角度标定文件。
    at128_angle_calibration: str = os.getenv("LRDG_AT128_CALIBRATION", str(PROJECT_ROOT / "data" / "at128" / "AT128P_AngleCalibration.dat"))
    # ======================================================================================

    @property
    def cache_root(self) -> str:
        return str(Path(self.workspace_root) / "cache" / "lidaroc")

    @property
    def model_root(self) -> str:
        return str(Path(self.workspace_root) / "models")

    @property
    def evaluation_root(self) -> str:
        return str(Path(self.workspace_root) / "evaluations")

    @property
    def at128_work_root(self) -> str:
        return str(Path(self.workspace_root) / "at128")


@dataclass(frozen=True)
class DataConfig:
    # Binary-class protocol used in the manuscript.
    target_classes: Tuple[str, ...] = ("clean", "water", "dust", "mud", "oil")
    merge_mud_subtypes: bool = True

    # Paper descriptor settings. Do not change for the main reported experiments.
    voxel_size: float = 0.20
    context_radius_voxels: int = 1
    intensity_scale: float = 255.0
    relative_std_eps: float = 1e-3
    normalization_eps: float = 1e-6

    # Cache safety: False reuses compatible cache; True rebuilds it.
    overwrite_cache: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "LRDG-Net"
    geometry_in_channels: int = 12
    degradation_in_channels: int = 4
    geometry_hidden: Tuple[int, int] = (32, 32)
    degradation_hidden: Tuple[int, int] = (16, 32)
    instance_channels: int = 64
    classifier_hidden: int = 64
    dropout: float = 0.20
    num_classes: int = 2
    pooling: str = "mean"


@dataclass(frozen=True)
class TrainConfig:
    # Training protocol used in the manuscript experiments.
    val_ratio: float = 0.20
    split_seed: int = 42
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    num_workers: int = 0
    device: str = os.getenv("LRDG_DEVICE", "cuda")
    checkpoint_rule: str = "macro_f1_then_loss"

    # Safer research default: do not delete completed runs unless explicitly requested.
    overwrite_models: bool = False


@dataclass(frozen=True)
class TestConfig:
    batch_size: int = 12
    num_workers: int = 0


@dataclass(frozen=True)
class ResearchConfig:
    # Internal LIDAROC domain-generalization experiment.
    domains: Tuple[str, ...] = ("5m", "10m", "20m")
    seeds: Tuple[int, ...] = (42, 2026, 3407)

    # ``run_03_train_all_lidaroc.py`` trains one model per seed using all three domains.
    all_source_domains: Tuple[str, ...] = ("5m", "10m", "20m")


@dataclass(frozen=True)
class AT128Config:
    # Fixed conversion protocol for the external dataset.
    return_policy: str = "strongest"  # strongest / last / both
    udp_port: int = 2368
    min_complete_ratio: float = 0.80
    overwrite_preparation: bool = False
    overwrite_evaluation: bool = False

    # Frame-level diagnostic probability threshold. The manuscript's primary
    # independent-sequence direct-transfer decision instead averages frame
    # logit margins and uses the fixed boundary M(S) >= 0.
    threshold: float = 0.50

    # For run_05_test_at128.py only.
    # "all_source": automatically use LRDG_ALL_seed<model_seed>
    # "single_source": automatically use LRDG_SRC_<source>_seed<model_seed>
    # "custom": use custom_model_dir below (also supports legacy experiment folders).
    model_mode: str = "all_source"
    model_seed: int = 42
    single_source_domain: str = "5m"
    custom_model_dir: str = r""




@dataclass(frozen=True)
class TransferConfig:
    """AT128 transfer study / AT128 迁移实验。

    Purpose: test whether the LIDAROC-pretrained LRDG representation is useful on AT128.
    The split unit is the independent PCAP recording, never an individual frame.
    """
    folds: int = 5
    split_seed: int = 42
    seeds: Tuple[int, ...] = (42, 2026, 3407)

    # Frozen LIDAROC encoder + a new linear binary classifier on AT128 training folds.
    run_frozen_linear_probe: bool = True
    linear_c: float = 1.0
    linear_max_iter: int = 3000

    # Control: same LRDG-Net trained from random initialization on AT128 training folds.
    run_at128_from_scratch: bool = True
    inner_val_ratio: float = 0.20

    # Reuse completed transfer outputs unless intentionally rebuilding them.
    overwrite: bool = False


@dataclass(frozen=True)
class AT128CalibrationConfig:
    """Final clean-reference AT128 calibration protocol.

    The workflow uses outer leave-one-distance-out evaluation. For each held-out
    distance, K=1 exhausts every clean recording from the remaining distances and K=3
    exhausts every 3-clean-recording combination. No calibration distance is manually
    selected.
    """
    calibration_budgets: Tuple[int, ...] = (1, 3)
    exhaustive_calibration_choices: bool = True
    scale_eps: float = 1e-6
    overwrite: bool = False


@dataclass(frozen=True)
class SingleInferenceConfig:
    model_dir: str = r""
    bin_path: str = r""


PATHS = PathConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
TEST = TestConfig()
RESEARCH = ResearchConfig()
AT128 = AT128Config()
TRANSFER = TransferConfig()
AT128_CALIBRATION = AT128CalibrationConfig()
SINGLE_INFERENCE = SingleInferenceConfig()


def ensure_project_dirs() -> None:
    for path in (
        PATHS.cache_root,
        PATHS.model_root,
        PATHS.evaluation_root,
        PATHS.at128_work_root,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
