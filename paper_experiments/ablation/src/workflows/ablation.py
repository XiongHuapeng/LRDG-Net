from pathlib import Path
import pandas as pd
from paper_experiments.ablation.config import DATA, MODEL, PATHS, RESEARCH, ensure_dirs
from paper_experiments.ablation.src.data.metadata import scan_raw_dataset
from paper_experiments.ablation.src.data.dataset import build_feature_cache, MANIFEST_FILENAME, SESSION_MANIFEST_FILENAME
from paper_experiments.ablation.src.training.trainer import train_variant
from paper_experiments.ablation.src.evaluation.lidaroc import evaluate_variant
from paper_experiments.ablation.src.models.ablation import VARIANT_SPECS


def preprocess():
    ensure_dirs()
    raw = scan_raw_dataset(PATHS.lidaroc_data_root, DATA.target_classes)
    man = build_feature_cache(raw, PATHS.cache_root, DATA)
    man.to_csv(Path(PATHS.cache_root) / MANIFEST_FILENAME, index=False, encoding="utf-8-sig")
    man[["domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid"]].drop_duplicates().to_csv(
        Path(PATHS.cache_root) / SESSION_MANIFEST_FILENAME, index=False, encoding="utf-8-sig"
    )
    print(f"Done: {len(man)} frames, {man.session_uid.nunique()} sessions")


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
    runs.to_csv(root / "ablation_runs.csv", index=False, encoding="utf-8-sig")

    direction = runs.groupby(["variant", "train_source", "test_domain"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        fnr_mean=("fnr", "mean"),
        fpr_mean=("fpr", "mean"),
        runs=("macro_f1", "count"),
    )
    direction.to_csv(root / "ablation_direction_summary.csv", index=False, encoding="utf-8-sig")

    model = direction.groupby("variant", as_index=False).agg(
        mean_macro_f1=("macro_f1_mean", "mean"),
        worst_macro_f1=("macro_f1_mean", "min"),
        direction_std=("macro_f1_mean", "std"),
        mean_fnr=("fnr_mean", "mean"),
        mean_fpr=("fpr_mean", "mean"),
    )
    params = runs.groupby("variant")["parameter_count"].first().reset_index()
    model = model.merge(params, on="variant", how="left")

    labels = {
        "E0_G": "Geometry only",
        "E1_A": "Absolute degradation only",
        "E2_R": "Relative degradation only",
        "E3_GA": "Geometry + absolute degradation",
        "E4_GR": "Geometry + relative degradation (LRDG-Net)",
    }
    model["model"] = model["variant"].map(labels)
    model = model[["variant", "model", "mean_macro_f1", "worst_macro_f1", "direction_std", "mean_fnr", "mean_fpr", "parameter_count"]]
    model.to_csv(root / "ablation_model_summary.csv", index=False, encoding="utf-8-sig")
    model.to_csv(root / "paper_ablation_summary.csv", index=False, encoding="utf-8-sig")

    defs = []
    for v in MODEL.variants:
        spec = VARIANT_SPECS[v]
        defs.append({
            "variant": v,
            "model": labels[v],
            "geometry": "centered_12D" if spec["geometry"] else "none",
            "degradation": spec["degradation"] or "none",
            "pooling": "mean",
        })
    pd.DataFrame(defs).to_csv(root / "ablation_variant_definitions.csv", index=False, encoding="utf-8-sig")

    print("\nPaper summary:\n", model.to_string(index=False))
    return model
