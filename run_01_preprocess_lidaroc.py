"""Experiment entry point: LIDAROC .bin -> frozen LRDG descriptor cache."""
from lrdg_net.workflows.lidaroc import preprocess_lidaroc

if __name__ == "__main__":
    preprocess_lidaroc()
