from pathlib import Path
import pandas as pd
from paper_experiments.component_ablation.config import DATA, MODEL, PATHS, RESEARCH, ensure_dirs
from paper_experiments.component_ablation.src.data.metadata import scan_raw_dataset
from paper_experiments.component_ablation.src.data.dataset import build_feature_cache, MANIFEST_FILENAME, SESSION_MANIFEST_FILENAME
from paper_experiments.component_ablation.src.training.trainer import train_variant
from paper_experiments.component_ablation.src.evaluation.lidaroc import evaluate_variant
from paper_experiments.component_ablation.src.models.r_component import VARIANT_SPECS, R_COMPONENT_NAMES


def preprocess():
    ensure_dirs()
    raw = scan_raw_dataset(PATHS.lidaroc_data_root, DATA.target_classes)
    man = build_feature_cache(raw, PATHS.cache_root, DATA)
    man.to_csv(Path(PATHS.cache_root) / MANIFEST_FILENAME, index=False, encoding="utf-8-sig")
    man[["domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid"]].drop_duplicates().to_csv(
        Path(PATHS.cache_root) / SESSION_MANIFEST_FILENAME, index=False, encoding="utf-8-sig"
    )
    print(f"Done: {len(man)} frames, {man.session_uid.nunique()} sessions")


def build_summaries(runs: pd.DataFrame, root: Path):
    direction = runs.groupby(["variant", "study", "paper_label", "train_source", "test_domain"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        fnr_mean=("fnr", "mean"),
        fpr_mean=("fpr", "mean"),
        runs=("macro_f1", "count"),
    )
    direction.to_csv(root / "r_component_direction_summary.csv", index=False, encoding="utf-8-sig")

    model = direction.groupby(["variant", "study", "paper_label"], as_index=False).agg(
        mean_macro_f1=("macro_f1_mean", "mean"),
        worst_macro_f1=("macro_f1_mean", "min"),
        direction_std=("macro_f1_mean", "std"),
        mean_fnr=("fnr_mean", "mean"),
        mean_fpr=("fpr_mean", "mean"),
    )
    aux = runs.groupby("variant", as_index=False).agg(
        input_dim=("input_dim", "first"),
        parameter_count=("parameter_count", "first"),
    )
    model = model.merge(aux, on="variant", how="left")

    order = {v: i for i, v in enumerate(MODEL.variants)}
    model["_order"] = model["variant"].map(order)
    model = model.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    cols = [
        "variant", "study", "paper_label", "input_dim", "mean_macro_f1", "worst_macro_f1",
        "direction_std", "mean_fnr", "mean_fpr", "parameter_count",
    ]
    model = model[cols]
    model.to_csv(root / "r_component_model_summary.csv", index=False, encoding="utf-8-sig")
    model.to_csv(root / "paper_r_component_summary.csv", index=False, encoding="utf-8-sig")

    ref = model[model.variant == "R_FULL"].iloc[0]
    effects = model.copy()
    effects["delta_mean_f1_vs_R_FULL"] = effects["mean_macro_f1"] - float(ref.mean_macro_f1)
    effects["delta_worst_f1_vs_R_FULL"] = effects["worst_macro_f1"] - float(ref.worst_macro_f1)
    effects.to_csv(root / "r_component_effects_vs_full.csv", index=False, encoding="utf-8-sig")

    defs = []
    for v in MODEL.variants:
        spec = VARIANT_SPECS[v]
        inds = tuple(spec["indices"])
        defs.append({
            "variant": v,
            "study": spec["study"],
            "paper_label": spec["paper_label"],
            "input_dim": len(inds),
            "selected_indices": ";".join(map(str, inds)),
            "selected_components": ";".join(R_COMPONENT_NAMES[i] for i in inds),
            "description": spec["description"],
            "geometry_branch": "none",
            "pooling": "mean",
            "zero_mask_used": False,
        })
    pd.DataFrame(defs).to_csv(root / "r_component_variant_definitions.csv", index=False, encoding="utf-8-sig")

    return model


def run_all():
    ensure_dirs()
    rows = []
    for variant in MODEL.variants:
        for source in RESEARCH.domains:
            for seed in RESEARCH.seeds:
                train_variant(variant, source, seed)
                for target in RESEARCH.domains:
                    if target != source:
                        rows.append(evaluate_variant(variant, source, seed, target))

    runs = pd.DataFrame(rows)
    root = Path(PATHS.evaluation_root)
    runs.to_csv(root / "r_component_runs.csv", index=False, encoding="utf-8-sig")
    model = build_summaries(runs, root)
    print("\nPaper summary:\n", model.to_string(index=False))
    return model


def rebuild_summaries_from_runs():
    root = Path(PATHS.evaluation_root)
    p = root / "r_component_runs.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run run_02_r_component_ablation_six_direction.py first.")
    runs = pd.read_csv(p)
    model = build_summaries(runs, root)
    print("\nRebuilt paper summary:\n", model.to_string(index=False))
    return model
