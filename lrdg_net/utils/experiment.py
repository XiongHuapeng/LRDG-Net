"""Experiment specifications, paths, status, and reproducibility snapshots."""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

from config import DATA, MODEL, PATHS, TRAIN
from lrdg_net.data.normalization import NORMALIZATION_FILENAME
from lrdg_net.utils.io import save_json

TRAIN_OUTPUTS = {
    "config.json",
    "split_sessions.csv",
    NORMALIZATION_FILENAME,
    "best_model.pt",
    "training_history.csv",
}


@dataclass(frozen=True)
class TrainingSpec:
    """A frozen training definition independent of any evaluation target."""

    train_domains: Tuple[str, ...]
    seed: int
    protocol: str  # single_source / all_source

    def __post_init__(self):
        domains = tuple(str(x) for x in self.train_domains)
        if not domains:
            raise ValueError("train_domains cannot be empty")
        object.__setattr__(self, "train_domains", domains)
        if self.protocol not in {"single_source", "all_source"}:
            raise ValueError("protocol must be 'single_source' or 'all_source'")
        if self.protocol == "single_source" and len(domains) != 1:
            raise ValueError("single_source protocol requires exactly one domain")

    @property
    def name(self) -> str:
        if self.protocol == "single_source":
            return f"LRDG_SRC_{self.train_domains[0]}_seed{self.seed}"
        return f"LRDG_ALL_seed{self.seed}"



def single_source_spec(domain: str, seed: int) -> TrainingSpec:
    return TrainingSpec((domain,), int(seed), "single_source")



def all_source_spec(domains, seed: int) -> TrainingSpec:
    return TrainingSpec(tuple(domains), int(seed), "all_source")



def model_directory(spec: TrainingSpec) -> Path:
    group = "single_source" if spec.protocol == "single_source" else "all_source"
    return Path(PATHS.model_root) / group / spec.name



def lidaroc_evaluation_directory(spec: TrainingSpec, target_domain: str) -> Path:
    return Path(PATHS.evaluation_root) / "lidaroc_cross_domain" / spec.name / f"to_{target_domain}"



def at128_evaluation_directory(model_dir: Path) -> Path:
    return Path(PATHS.evaluation_root) / "at128_external" / Path(model_dir).name



def model_status(model_dir: Path) -> str:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return "missing"
    files = {p.name for p in model_dir.iterdir() if p.is_file()}
    if TRAIN_OUTPUTS.issubset(files):
        return "trained"
    return "incomplete"



def reset_directory(path: Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)



def save_training_snapshot(spec: TrainingSpec, model_dir: Path) -> dict:
    snapshot = {
        "experiment_name": spec.name,
        "model_name": MODEL.model_name,
        "training_protocol": spec.protocol,
        "train_domains": list(spec.train_domains),
        "seed": int(spec.seed),
        "paths": {
            "lidaroc_data_root": PATHS.lidaroc_data_root,
            "cache_root": PATHS.cache_root,
            "model_root": PATHS.model_root,
            "evaluation_root": PATHS.evaluation_root,
        },
        "data": asdict(DATA),
        "model": asdict(MODEL),
        "train": asdict(TRAIN),
        "protocol": {
            "task": "binary clean vs contaminated",
            "split_level": "complete session",
            "split_strata": ["domain", "raw_class", "severity"],
            "evaluation_target_used_for_training_or_normalization": False,
            "absolute_xyz_classifier_input": False,
            "absolute_degradation_classifier_input": False,
            "geometry": "fine/context centered covariance normalized by voxel_size^2",
            "degradation": "fine-to-context relative degradation",
            "pooling": "mean",
            "normalization": "training-split-only instance-weighted Z-score for geometry and relative degradation",
            "checkpoint": "maximize validation Macro-F1; ties -> lower validation loss",
        },
    }
    save_json(snapshot, Path(model_dir) / "config.json")
    return snapshot
