"""Utility: rebuild LIDAROC cross-domain summary from saved metrics."""
from pathlib import Path
import json
import pandas as pd

from config import PATHS


def main():
    root = Path(PATHS.evaluation_root) / "lidaroc_cross_domain"
    rows = []
    for path in sorted(root.glob("LRDG_SRC_*_seed*/to_*/test_metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No internal cross-domain test_metrics.json under {root}")

    df = pd.DataFrame(rows)
    df.to_csv(root / "cross_domain_runs.csv", index=False, encoding="utf-8-sig")
    agg = df.groupby(["train_source", "test_domain"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        fnr_mean=("fnr", "mean"),
        fnr_std=("fnr", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
        runs=("macro_f1", "count"),
    )
    agg.to_csv(root / "cross_domain_direction_summary.csv", index=False, encoding="utf-8-sig")

    model_summary = pd.DataFrame([{
        "model_name": "LRDG-Net",
        "direction_macro_f1_mean": float(agg["macro_f1_mean"].mean()),
        "worst_direction_macro_f1": float(agg["macro_f1_mean"].min()),
        "direction_macro_f1_std": float(agg["macro_f1_mean"].std()),
        "direction_fnr_mean": float(agg["fnr_mean"].mean()),
        "direction_fpr_mean": float(agg["fpr_mean"].mean()),
        "directions": int(len(agg)),
    }])
    model_summary.to_csv(root / "cross_domain_model_summary.csv", index=False, encoding="utf-8-sig")
    print(agg.to_string(index=False))
    print("\n", model_summary.to_string(index=False))


if __name__ == "__main__":
    main()
