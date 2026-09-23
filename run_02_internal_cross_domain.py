"""Experiment entry point: 9 trainings + 18 LIDAROC cross-domain evaluations."""
from lrdg_net.workflows.lidaroc import run_internal_cross_domain

if __name__ == "__main__":
    run_internal_cross_domain()
