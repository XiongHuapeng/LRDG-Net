"""
LRDG-Net source-training-only normalization。
LRDG-Net 仅使用 source-training split 拟合标准化参数。
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import DATA

NORMALIZATION_FILENAME = "normalization_stats.json"


@dataclass
class NormalizationStats:
    geometry_mean: list
    geometry_std: list
    degradation_mean: list
    degradation_std: list
    source_train_frames: int
    geometry_instances: int
    degradation_instances: int
    fit_scope: str = "source_training_split_only"
    target_domain_used: bool = False

    def normalize_geometry(self, array: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.geometry_mean, dtype=np.float32)
        std = np.asarray(self.geometry_std, dtype=np.float32)
        return ((array - mean) / std).astype(np.float32)

    def normalize_degradation(self, array: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.degradation_mean, dtype=np.float32)
        std = np.asarray(self.degradation_std, dtype=np.float32)
        return ((array - mean) / std).astype(np.float32)


def _stream_mean_std(paths, key: str, eps: float):
    total_n = 0
    total_sum = None
    total_sq = None
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            x = npz[key].astype(np.float64)
        if x.size == 0:
            continue
        if total_sum is None:
            total_sum = np.zeros(x.shape[1], dtype=np.float64)
            total_sq = np.zeros(x.shape[1], dtype=np.float64)
        total_n += x.shape[0]
        total_sum += x.sum(axis=0)
        total_sq += np.square(x).sum(axis=0)
    if total_n == 0 or total_sum is None:
        raise RuntimeError(f"No instances found for normalization key={key} / 无实例可拟合标准化。")
    mean = total_sum / total_n
    var = np.maximum(total_sq / total_n - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)
    return mean.tolist(), std.tolist(), int(total_n)


def fit_normalization_stats(train_dataframe: pd.DataFrame) -> NormalizationStats:
    paths = train_dataframe["cache_path"].tolist()
    g_mean, g_std, g_n = _stream_mean_std(paths, "geometry", DATA.normalization_eps)
    d_mean, d_std, d_n = _stream_mean_std(paths, "relative_degradation", DATA.normalization_eps)
    return NormalizationStats(
        geometry_mean=g_mean,
        geometry_std=g_std,
        degradation_mean=d_mean,
        degradation_std=d_std,
        source_train_frames=len(train_dataframe),
        geometry_instances=g_n,
        degradation_instances=d_n,
    )


def save_normalization_stats(stats: NormalizationStats, path) -> None:
    Path(path).write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False), encoding="utf-8")


def load_normalization_stats(path) -> NormalizationStats:
    return NormalizationStats(**json.loads(Path(path).read_text(encoding="utf-8")))
