from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from paper_experiments.ablation.config import DATA
from paper_experiments.ablation.src.models.ablation import VARIANT_SPECS

NORMALIZATION_FILENAME = "normalization_stats.json"


@dataclass
class NormalizationStats:
    geometry_mean: list
    geometry_std: list
    degradation_mean: list
    degradation_std: list
    degradation_kind: str
    source_train_frames: int
    geometry_instances: int
    degradation_instances: int
    fit_scope: str = "source_training_split_only"
    target_domain_used: bool = False

    def normalize_geometry(self, x):
        if not self.geometry_mean:
            return x.astype(np.float32)
        return ((x - np.asarray(self.geometry_mean, np.float32)) / np.asarray(self.geometry_std, np.float32)).astype(np.float32)

    def normalize_degradation(self, x):
        if self.degradation_kind == "none":
            return x.astype(np.float32)
        return ((x - np.asarray(self.degradation_mean, np.float32)) / np.asarray(self.degradation_std, np.float32)).astype(np.float32)


def _stream(paths, key, eps):
    n = 0
    s = None
    ss = None
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            x = z[key].astype(np.float64)
        if s is None:
            s = np.zeros(x.shape[1])
            ss = np.zeros(x.shape[1])
        n += x.shape[0]
        s += x.sum(0)
        ss += np.square(x).sum(0)
    if n == 0:
        raise RuntimeError(f"No instances for {key}")
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0)
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)
    return mean.tolist(), std.tolist(), int(n)


def fit_normalization_stats(df: pd.DataFrame, variant: str):
    variant = variant.upper()
    spec = VARIANT_SPECS[variant]
    paths = df["cache_path"].tolist()

    if spec["geometry"]:
        gm, gs, gn = _stream(paths, "geometry", DATA.normalization_eps)
    else:
        gm, gs, gn = [], [], 0

    kind = spec["degradation"]
    if kind is None:
        dm, ds, dn = [], [], 0
        norm_kind = "none"
    elif kind == "absolute":
        dm, ds, dn = _stream(paths, "absolute_degradation", DATA.normalization_eps)
        norm_kind = "absolute"
    elif kind == "relative":
        dm, ds, dn = _stream(paths, "relative_degradation", DATA.normalization_eps)
        norm_kind = "relative"
    else:
        raise RuntimeError(kind)

    return NormalizationStats(gm, gs, dm, ds, norm_kind, len(df), gn, dn)


def save_normalization_stats(stats, path):
    Path(path).write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False), encoding="utf-8")


def load_normalization_stats(path):
    return NormalizationStats(**json.loads(Path(path).read_text(encoding="utf-8")))
