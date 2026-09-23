from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from paper_experiments.component_ablation.config import DATA
from paper_experiments.component_ablation.src.models.r_component import VARIANT_SPECS, R_COMPONENT_NAMES

NORMALIZATION_FILENAME = "normalization_stats.json"


@dataclass
class NormalizationStats:
    degradation_mean: list
    degradation_std: list
    selected_indices: list
    selected_components: list
    source_train_frames: int
    degradation_instances: int
    fit_scope: str = "source_training_split_only"
    target_domain_used: bool = False

    def normalize_degradation(self, x):
        return (
            (x - np.asarray(self.degradation_mean, np.float32))
            / np.asarray(self.degradation_std, np.float32)
        ).astype(np.float32)


def _stream_selected(paths, indices, eps):
    n = 0
    s = None
    ss = None
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            full = z["relative_degradation"].astype(np.float64)
        x = full[:, indices]
        if s is None:
            s = np.zeros(x.shape[1], dtype=np.float64)
            ss = np.zeros(x.shape[1], dtype=np.float64)
        n += x.shape[0]
        s += x.sum(0)
        ss += np.square(x).sum(0)
    if n == 0:
        raise RuntimeError("No relative-degradation instances found.")
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0)
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)
    return mean.tolist(), std.tolist(), int(n)


def fit_normalization_stats(df: pd.DataFrame, variant: str):
    variant = variant.upper()
    if variant not in VARIANT_SPECS:
        raise ValueError(variant)
    indices = tuple(VARIANT_SPECS[variant]["indices"])
    dm, ds, dn = _stream_selected(df["cache_path"].tolist(), indices, DATA.normalization_eps)
    components = [R_COMPONENT_NAMES[i] for i in indices]
    return NormalizationStats(
        degradation_mean=dm,
        degradation_std=ds,
        selected_indices=list(indices),
        selected_components=components,
        source_train_frames=len(df),
        degradation_instances=dn,
    )


def save_normalization_stats(stats, path):
    Path(path).write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False), encoding="utf-8")


def load_normalization_stats(path):
    return NormalizationStats(**json.loads(Path(path).read_text(encoding="utf-8")))
