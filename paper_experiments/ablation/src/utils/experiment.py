from dataclasses import asdict
from pathlib import Path
import json, shutil
from paper_experiments.ablation.config import PATHS, DATA, MODEL, TRAIN


def model_dir(variant, source, seed):
    return Path(PATHS.model_root) / variant / f"{variant}_MEAN_SRC_{source}_seed{seed}"


def eval_dir(variant, source, seed, target):
    return Path(PATHS.evaluation_root) / variant / f"{variant}_MEAN_SRC_{source}_seed{seed}" / f"to_{target}"


def reset_dir(p):
    if Path(p).exists():
        shutil.rmtree(p)
    Path(p).mkdir(parents=True, exist_ok=True)


def status(p):
    p = Path(p)
    req = ["config.json", "split_sessions.csv", "normalization_stats.json", "best_model.pt", "training_history.csv"]
    if not p.exists():
        return "missing"
    return "trained" if all((p / x).exists() for x in req) else "incomplete"


def save_snapshot(variant, source, seed, p):
    snap = {
        "variant": variant,
        "source_domain": source,
        "seed": seed,
        "data": asdict(DATA),
        "model": asdict(MODEL),
        "train": asdict(TRAIN),
        "protocol": {
            "task": "binary clean vs contaminated",
            "split_level": "complete session",
            "target_domain_used_for_training_or_normalization": False,
            "pooling_fixed_across_ablation": "mean_only",
            "parameter_count_forced_equal": False,
            "E0_G": "12D centered covariance geometry only",
            "E1_A": "3D absolute degradation only",
            "E2_R": "4D local-relative degradation only",
            "E3_GA": "12D centered covariance geometry + 3D absolute degradation",
            "E4_GR": "12D centered covariance geometry + 4D local-relative degradation (full LRDG-Net)",
        },
    }
    Path(p, "config.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap
