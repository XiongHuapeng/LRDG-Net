"""Combined all-LIDAROC -> AT128 research workflow."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import AT128, PATHS, RESEARCH, TRAIN, ensure_project_dirs
from lrdg_net.evaluation.at128 import evaluate_at128
from lrdg_net.training.trainer import run_training
from lrdg_net.utils.experiment import all_source_spec, model_directory, single_source_spec
from lrdg_net.workflows.at128 import prepare_at128


def resolve_configured_model_dir() -> Path:
    mode = AT128.model_mode.lower().strip()
    if mode == "all_source":
        return model_directory(all_source_spec(RESEARCH.all_source_domains, AT128.model_seed))
    if mode == "single_source":
        return model_directory(single_source_spec(AT128.single_source_domain, AT128.model_seed))
    if mode == "custom":
        if not AT128.custom_model_dir:
            raise ValueError("AT128.custom_model_dir is empty while model_mode='custom'.")
        return Path(AT128.custom_model_dir)
    raise ValueError("AT128.model_mode must be 'all_source', 'single_source', or 'custom'.")


def test_configured_model_on_at128() -> dict:
    ensure_project_dirs()
    model_dir = resolve_configured_model_dir()
    return evaluate_at128(model_dir, overwrite=AT128.overwrite_evaluation)


def run_all_lidaroc_to_at128() -> pd.DataFrame:
    """
    Current main external-validation workflow:
    1) prepare AT128 once;
    2) train LRDG-Net on all LIDAROC domains for each configured seed;
    3) freeze each model and evaluate on AT128 without target normalization/threshold tuning.
    """
    ensure_project_dirs()

    # Train/freeze all LIDAROC models before touching the AT128 preparation workflow.
    # This ordering makes the external-validation protocol explicit in the code path.
    frozen = []
    for index, seed in enumerate(RESEARCH.seeds, start=1):
        spec = all_source_spec(RESEARCH.all_source_domains, seed)
        print("\n" + "=" * 110)
        print(f"[All-LIDAROC training {index}/{len(RESEARCH.seeds)}] {spec.name}")
        print("=" * 110)
        frozen.append((spec, run_training(spec)))

    prepare_at128()

    rows = []
    for index, (spec, model_dir) in enumerate(frozen, start=1):
        print("\n" + "=" * 110)
        print(f"[Frozen AT128 test {index}/{len(frozen)}] {spec.name}")
        print("=" * 110)
        metrics = evaluate_at128(model_dir, overwrite=AT128.overwrite_evaluation)
        rows.append({
            "model": spec.name,
            "seed": int(spec.seed),
            **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
        })

    summary = pd.DataFrame(rows)
    out = Path(PATHS.evaluation_root) / "at128_external" / "all_lidaroc_to_at128_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nAll-LIDAROC -> AT128 summary / 汇总: {out}")
    return summary
