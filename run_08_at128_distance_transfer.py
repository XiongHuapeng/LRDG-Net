"""Experiment entry point: stricter AT128 transfer control grouped by distance.

Scientific purpose:
- Keep the LRDG-Net architecture unchanged.
- Keep the two v2 transfer methods unchanged.
- Change only the outer split protocol so that 0/1/2/3 recordings from the same
  distance are always assigned to the same fold.

Prerequisites:
- All-LIDAROC models already trained (run_03_train_all_lidaroc.py).
- AT128 cache already prepared (run_04_prepare_at128.py).

No command-line arguments are required.
"""
from lrdg_net.workflows.transfer import run_at128_transfer_study


if __name__ == "__main__":
    run_at128_transfer_study(split_mode="distance")
