"""Static consistency checks for the final LRDG-Net Python project.

The checks are data-free: they validate paper-critical configuration, reference-result
snapshots, required project files, and accidental developer-path leakage without training models.
"""
from pathlib import Path
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA, MODEL, RESEARCH, TRAIN, AT128_CALIBRATION
from paper_experiments.ablation.config import MODEL as ABL_MODEL
from paper_experiments.component_ablation.config import MODEL as COMP_MODEL
from paper_experiments.sensitivity.config import SENS
from paper_experiments.baseline.config import BASELINES

REF = ROOT / "reference_results"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    # Paper-critical configuration.
    assert DATA.voxel_size == 0.20
    assert DATA.context_radius_voxels == 1
    assert DATA.intensity_scale == 255.0
    assert DATA.relative_std_eps == 1e-3
    assert MODEL.model_name == "LRDG-Net"
    assert MODEL.geometry_in_channels == 12
    assert MODEL.degradation_in_channels == 4
    assert MODEL.geometry_hidden == (32, 32)
    assert MODEL.degradation_hidden == (16, 32)
    assert MODEL.instance_channels == 64
    assert MODEL.classifier_hidden == 64
    assert MODEL.dropout == 0.20
    assert TRAIN.epochs == 60
    assert TRAIN.batch_size == 8
    assert TRAIN.learning_rate == 1e-3
    assert TRAIN.weight_decay == 1e-4
    assert TRAIN.early_stopping_patience == 10
    assert RESEARCH.seeds == (42, 2026, 3407)

    # Paper experiment variants.
    assert ABL_MODEL.variants == ("E0_G", "E1_A", "E2_R", "E3_GA", "E4_GR")
    assert COMP_MODEL.variants == (
        "R_FULL",
        "R_ONLY_DELTA_I",
        "R_ONLY_LOG_STD_RATIO",
        "R_ONLY_LOCAL_STD",
        "R_ONLY_REL_LOG_DENSITY",
        "R_WO_DELTA_I",
        "R_WO_LOG_STD_RATIO",
        "R_WO_LOCAL_STD",
        "R_WO_REL_LOG_DENSITY",
    )
    assert SENS.voxel_sizes == (0.10, 0.20, 0.30, 0.40)
    assert SENS.context_radii == (1, 2, 3)
    assert AT128_CALIBRATION.calibration_budgets == (1, 3)
    assert AT128_CALIBRATION.exhaustive_calibration_choices is True
    assert BASELINES.baselines == (
        "globalstats", "rangenet", "pointnet", "pointnet2_ref", "dgcnn", "pointnext", "autogran"
    )

    # Final Python-project structure.
    for rel in (
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        "requirements-optional.txt",
        "assets/lrdg_net_overview.png",
        "lrdg_net/models/lrdg_net.py",
        "lrdg_net/cli.py",
        "tests/test_model_spec.py",
        "tests/test_descriptors.py",
        "tests/test_at128_calibration.py",
        "tests/test_reference_results.py",
    ):
        require(ROOT / rel)
    assert not (ROOT / "src").exists(), "Generic root src package should have been renamed to lrdg_net"

    # Reference result snapshots used for manuscript traceability.
    files = [
        REF / "lidaroc_cross_domain" / "cross_domain_model_summary.csv",
        REF / "baseline_cross_domain" / "paper_baseline_comparison.csv",
        REF / "ablation" / "paper_ablation_summary.csv",
        REF / "sensitivity" / "paper_sensitivity_summary.csv",
        REF / "at128_external" / "at128_external_metrics.csv",
        REF / "at128_calibration" / "paper_cross_sensor_summary.csv",
        REF / "at128_calibration" / "protocol.json",
    ]
    for path in files:
        require(path)

    cross = pd.read_csv(REF / "lidaroc_cross_domain" / "cross_domain_model_summary.csv")
    row = cross.loc[cross.model_name == "LRDG-Net"].iloc[0]
    assert abs(float(row.direction_macro_f1_mean) - 0.9885686531972887) < 1e-12
    assert abs(float(row.worst_direction_macro_f1) - 0.964922445902355) < 1e-12
    assert abs(float(row.direction_macro_f1_std) - 0.01771629794336035) < 1e-12

    abl = pd.read_csv(REF / "ablation" / "paper_ablation_summary.csv")
    assert set(abl["variant"]) == set(ABL_MODEL.variants)
    full = abl.loc[abl.variant == "E4_GR"].iloc[0]
    assert int(full.parameter_count) == 10898

    cal = pd.read_csv(REF / "at128_calibration" / "paper_cross_sensor_summary.csv")
    k0 = cal.loc[cal.setting == "K0_raw"].iloc[0]
    k3 = cal.loc[cal.setting == "K3_clean_norm"].iloc[0]
    assert abs(float(k0.macro_f1_mean) - 0.4285714285714286) < 1e-12
    assert abs(float(k3.macro_f1_mean) - 0.9999333999333999) < 1e-12
    assert abs(float(k3.worst_distance_macro_f1) - 0.9991341991341992) < 1e-12

    cal_protocol = json.loads((REF / "at128_calibration" / "protocol.json").read_text(encoding="utf-8"))
    assert cal_protocol.get("protocol_id") == "at128_clean_reference_lodo_exhaustive_v1"

    # Project hygiene: reject developer-local Windows home paths anywhere in text files.
    windows_user_path = re.compile(r"[A-Za-z]:\\Users\\[^\\]+\\")
    text_suffixes = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yml", ".yaml", ".cff"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        assert windows_user_path.search(text) is None, f"Local user path leaked in {path}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Submission-stage repository" not in readme
    assert "actions/workflows" not in readme

    print("Project consistency check PASSED")
    print("Main model: LRDG-Net, 12D geometry + 4D local relative degradation")
    print("Parameters: 10,898")
    print("Seeds:", RESEARCH.seeds)
    print("Representation ablation:", ABL_MODEL.variants)
    print("Component ablation variants:", len(COMP_MODEL.variants))
    print("Sensitivity voxel sizes:", SENS.voxel_sizes, "radii:", SENS.context_radii)
    print("AT128 calibration protocol:", cal_protocol.get("protocol_id"))


if __name__ == "__main__":
    main()
