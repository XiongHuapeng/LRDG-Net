"""High-level LIDAROC workflows for direct experiment execution."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import DATA, PATHS, RESEARCH, TRAIN, ensure_project_dirs
from lrdg_net.data.dataset import MANIFEST_FILENAME, SESSION_MANIFEST_FILENAME, build_feature_cache
from lrdg_net.data.metadata import scan_raw_dataset
from lrdg_net.evaluation.lidaroc import evaluate_lidaroc_domain
from lrdg_net.training.trainer import run_training
from lrdg_net.utils.experiment import all_source_spec, single_source_spec


def preprocess_lidaroc() -> Path:
    ensure_project_dirs()
    print("[1/2] Scan LIDAROC / 扫描 LIDAROC ...")
    raw = scan_raw_dataset(PATHS.lidaroc_data_root, DATA.target_classes)
    print(raw.groupby(["domain", "pollution_type"]).size())

    print("\n[2/2] Build LRDG-Net descriptors / 构建 LRDG-Net 描述子缓存 ...")
    manifest = build_feature_cache(raw, PATHS.cache_root, DATA)
    manifest_path = Path(PATHS.cache_root) / MANIFEST_FILENAME
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    sessions = manifest[[
        "domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid"
    ]].drop_duplicates()
    sessions.to_csv(Path(PATHS.cache_root) / SESSION_MANIFEST_FILENAME, index=False, encoding="utf-8-sig")

    print("\nPoint / instance statistics / 点数与局部实例统计:")
    print(manifest.groupby("domain")[["point_count", "instance_count"]].agg(["mean", "median", "min", "max"]))
    print(f"\nDone / 完成: {len(manifest)} frames, {manifest['session_uid'].nunique()} sessions")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def run_internal_cross_domain() -> pd.DataFrame:
    """
    Train only 3 domains × seeds = 9 models, then evaluate each model on both unseen LIDAROC domains.

    This replaces the old 18-direction retraining pattern. Since target data is never loaded during
    training, training a source+seed model once and evaluating it on two targets is scientifically
    equivalent while removing redundant training.
    """
    ensure_project_dirs()
    rows = []
    total_models = len(RESEARCH.domains) * len(RESEARCH.seeds)
    model_index = 0
    for source in RESEARCH.domains:
        for seed in RESEARCH.seeds:
            model_index += 1
            spec = single_source_spec(source, seed)
            print("\n" + "=" * 110)
            print(f"[Model {model_index}/{total_models}] {spec.name}")
            print("=" * 110)
            run_training(spec)
            for target in RESEARCH.domains:
                if target == source:
                    continue
                metrics = evaluate_lidaroc_domain(spec, target, overwrite=TRAIN.overwrite_models)
                rows.append(metrics)

    summary = pd.DataFrame(rows)
    summary_path = Path(PATHS.evaluation_root) / "lidaroc_cross_domain" / "cross_domain_runs.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    direction = summary.groupby(["train_source", "test_domain"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        fnr_mean=("fnr", "mean"),
        fnr_std=("fnr", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
        runs=("macro_f1", "count"),
    )
    direction_path = summary_path.parent / "cross_domain_direction_summary.csv"
    direction.to_csv(direction_path, index=False, encoding="utf-8-sig")

    model_summary = pd.DataFrame([{
        "model_name": "LRDG-Net",
        "direction_macro_f1_mean": float(direction["macro_f1_mean"].mean()),
        "worst_direction_macro_f1": float(direction["macro_f1_mean"].min()),
        "direction_macro_f1_std": float(direction["macro_f1_mean"].std()),
        "direction_fnr_mean": float(direction["fnr_mean"].mean()),
        "direction_fpr_mean": float(direction["fpr_mean"].mean()),
        "directions": int(len(direction)),
    }])
    model_summary.to_csv(
        summary_path.parent / "cross_domain_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\nCross-domain summary / 六方向汇总: {summary_path}")
    print(direction.to_string(index=False))
    print("\nModel summary / 模型汇总:")
    print(model_summary.to_string(index=False))
    return summary


def train_all_lidaroc() -> list[Path]:
    """Train the final all-LIDAROC model for each configured seed; no AT128 data are loaded."""
    ensure_project_dirs()
    model_dirs = []
    for index, seed in enumerate(RESEARCH.seeds, start=1):
        spec = all_source_spec(RESEARCH.all_source_domains, seed)
        print("\n" + "=" * 110)
        print(f"[All-LIDAROC {index}/{len(RESEARCH.seeds)}] {spec.name}")
        print("=" * 110)
        model_dirs.append(run_training(spec))
    return model_dirs
