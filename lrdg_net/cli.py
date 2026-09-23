"""Command-line interface for the main LRDG-Net workflows.

The numbered ``run_*.py`` scripts remain the canonical paper-reproduction entry points.
This CLI is a convenience layer for the installed Python project.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from config import DATA
from lrdg_net.data.local_descriptors import extract_local_descriptors
from lrdg_net.training.trainer import build_model


def _smoke() -> None:
    """Run a small data-free architecture/descriptor check."""
    rng = np.random.default_rng(0)
    xyz = rng.uniform(low=(-0.5, -0.5, -0.5), high=(0.5, 0.5, 0.5), size=(2048, 3)).astype(np.float32)
    intensity = rng.integers(0, 256, size=(len(xyz), 1)).astype(np.float32)
    points = np.concatenate([xyz, intensity], axis=1)
    desc = extract_local_descriptors(
        points,
        DATA.voxel_size,
        DATA.context_radius_voxels,
        DATA.intensity_scale,
        DATA.relative_std_eps,
    )
    model = build_model()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    geometry = torch.from_numpy(desc["geometry"])
    degradation = torch.from_numpy(desc["relative_degradation"])
    bag_index = torch.zeros(len(geometry), dtype=torch.long)
    logits = model(geometry, degradation, bag_index, batch_size=1)
    if geometry.shape[1] != 12 or degradation.shape[1] != 4 or tuple(logits.shape) != (1, 2) or params != 10898:
        raise RuntimeError("LRDG-Net smoke check failed")
    print(f"LRDG-Net smoke check passed: {params:,} parameters, logits={tuple(logits.shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lrdg-net",
        description="LRDG-Net paper-aligned experiment utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="Run a data-free descriptor/model smoke test.")
    sub.add_parser("verify", help="Run static paper/repository consistency checks.")
    sub.add_parser("preprocess-lidaroc", help="Build the LIDAROC feature cache.")
    sub.add_parser("cross-domain", help="Run the six directed LIDAROC transfer experiment.")
    sub.add_parser("train-all", help="Train all-LIDAROC source models for the paper seeds.")
    sub.add_parser("prepare-at128", help="Decode and prepare AT128 recordings.")
    sub.add_parser("test-at128", help="Evaluate the configured frozen model on AT128.")
    sub.add_parser("external-at128", help="Run the all-LIDAROC to AT128 frozen external test.")
    sub.add_parser("calibrate-at128", help="Run clean-reference AT128 operating-point calibration.")
    transfer = sub.add_parser("transfer-at128", help="Run supplementary target-supervised AT128 transfer controls.")
    transfer.add_argument("--split-mode", choices=("pcap", "distance"), default="pcap")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        _smoke()
    elif args.command == "verify":
        from tools.verify_project import main as verify_main
        verify_main()
    elif args.command == "preprocess-lidaroc":
        from lrdg_net.workflows.lidaroc import preprocess_lidaroc
        preprocess_lidaroc()
    elif args.command == "cross-domain":
        from lrdg_net.workflows.lidaroc import run_internal_cross_domain
        run_internal_cross_domain()
    elif args.command == "train-all":
        from lrdg_net.workflows.lidaroc import train_all_lidaroc
        train_all_lidaroc()
    elif args.command == "prepare-at128":
        from lrdg_net.workflows.at128 import prepare_at128
        prepare_at128()
    elif args.command == "test-at128":
        from lrdg_net.workflows.external import test_configured_model_on_at128
        test_configured_model_on_at128()
    elif args.command == "external-at128":
        from lrdg_net.workflows.external import run_all_lidaroc_to_at128
        run_all_lidaroc_to_at128()
    elif args.command == "calibrate-at128":
        from lrdg_net.workflows.at128_calibration import run_at128_calibration
        run_at128_calibration()
    elif args.command == "transfer-at128":
        from lrdg_net.workflows.transfer import run_at128_transfer_study
        run_at128_transfer_study(split_mode=args.split_mode)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
