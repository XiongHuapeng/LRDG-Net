from dataclasses import asdict
from pathlib import Path
import json
import shutil
from paper_experiments.component_ablation.config import PATHS, DATA, MODEL, TRAIN
from paper_experiments.component_ablation.src.models.r_component import VARIANT_SPECS, R_COMPONENT_NAMES


def model_dir(variant, source, seed):
    return Path(PATHS.model_root) / variant / f"{variant}_MEAN_SRC_{source}_seed{seed}"


def eval_dir(variant, source, seed, target):
    return (
        Path(PATHS.evaluation_root)
        / variant
        / f"{variant}_MEAN_SRC_{source}_seed{seed}"
        / f"to_{target}"
    )


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
    spec = VARIANT_SPECS[variant]
    indices = tuple(spec["indices"])
    snap = {
        "variant": variant,
        "study": spec["study"],
        "paper_label": spec["paper_label"],
        "source_domain": source,
        "seed": seed,
        "selected_indices": list(indices),
        "selected_components": [R_COMPONENT_NAMES[i] for i in indices],
        "data": asdict(DATA),
        "model": asdict(MODEL),
        "train": asdict(TRAIN),
        "protocol": {
            "task": "binary clean vs contaminated",
            "representation": "R-only local-relative degradation",
            "geometry_branch_used": False,
            "feature_subset_mode": "true_input_dimensionality_no_zero_mask",
            "all_variants_trained_from_scratch": True,
            "R_FULL_rerun_in_same_project": True,
            "split_level": "complete session",
            "target_domain_used_for_training_or_normalization": False,
            "normalization_fit": "selected R components; source training split only",
            "pooling": "mean_only",
            "parameter_count_forced_equal": False,
        },
    }
    Path(p, "config.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap
