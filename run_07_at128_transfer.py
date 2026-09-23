"""Experiment entry point: minimal AT128 transfer study.

Run this after:
- run_03_train_all_lidaroc.py
- run_04_prepare_at128.py

No terminal arguments are required.
"""
from lrdg_net.workflows.transfer import run_at128_transfer_study

if __name__ == "__main__":
    run_at128_transfer_study()
