"""Utility: inspect configured paths and required files without training."""
from pathlib import Path
from config import PATHS


def _status(label, path):
    p = Path(path)
    print(f"{label:<28} {'OK' if p.exists() else 'MISSING':<8} {p}")


def main():
    print("Configured project paths / 当前配置路径")
    print("=" * 100)
    _status("LIDAROC data root", PATHS.lidaroc_data_root)
    _status("Workspace root", PATHS.workspace_root)
    _status("AT128 PCAP root", PATHS.at128_pcap_root)
    _status("AT128 calibration", PATHS.at128_angle_calibration)
    print("-" * 100)
    print(f"LIDAROC cache -> {PATHS.cache_root}")
    print(f"Models         -> {PATHS.model_root}")
    print(f"Evaluations    -> {PATHS.evaluation_root}")
    print(f"AT128 work     -> {PATHS.at128_work_root}")


if __name__ == "__main__":
    main()
