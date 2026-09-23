"""Experiment entry point: train final LRDG-Net on 5m+10m+20m LIDAROC for all configured seeds."""
from lrdg_net.workflows.lidaroc import train_all_lidaroc

if __name__ == "__main__":
    train_all_lidaroc()
