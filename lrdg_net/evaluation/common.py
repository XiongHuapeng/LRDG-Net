"""Shared evaluation helpers."""
from pathlib import Path
import json

import torch

from config import TRAIN
from lrdg_net.data.normalization import NORMALIZATION_FILENAME, NormalizationStats, load_normalization_stats
from lrdg_net.training.trainer import build_model
from lrdg_net.utils.reproducibility import resolve_device


def load_stats_compat(model_dir: Path) -> NormalizationStats:
    """Load current normalization stats while tolerating legacy JSON files with extra fields."""
    path = Path(model_dir) / NORMALIZATION_FILENAME
    try:
        return load_normalization_stats(path)
    except TypeError:
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(NormalizationStats.__dataclass_fields__.keys())
        return NormalizationStats(**{k: v for k, v in payload.items() if k in allowed})


def load_frozen_model(model_dir: Path):
    model_dir = Path(model_dir)
    for required in ("best_model.pt", NORMALIZATION_FILENAME):
        if not (model_dir / required).exists():
            raise FileNotFoundError(f"Missing {required}: {model_dir}")
    device = resolve_device(TRAIN.device)
    model = build_model().to(device)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, load_stats_compat(model_dir), device
