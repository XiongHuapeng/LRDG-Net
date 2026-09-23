"""Experiment entry point: final exhaustive AT128 clean-reference calibration.

Prerequisites:
- run_01_preprocess_lidaroc.py
- run_03_train_all_lidaroc.py
- run_06_all_lidaroc_to_at128.py

Protocol:
- outer leave-one-distance-out across all AT128 distance blocks;
- K=1 exhausts every other-distance clean PCAP;
- K=3 exhausts every 3-clean-PCAP subset from the remaining distances;
- no contaminated target label is used for calibration;
- no LRDG-Net retraining and no AT128 re-inference.
"""
from lrdg_net.workflows.at128_calibration import run_at128_calibration


if __name__ == "__main__":
    run_at128_calibration()
