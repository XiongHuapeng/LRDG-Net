"""Experiment entry point: current main study, All-LIDAROC training -> frozen AT128 external test."""
from lrdg_net.workflows.external import run_all_lidaroc_to_at128

if __name__ == "__main__":
    run_all_lidaroc_to_at128()
