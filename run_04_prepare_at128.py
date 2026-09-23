"""Experiment entry point: AT128 PCAP -> complete-frame XYZI -> frozen LRDG descriptor cache."""
from lrdg_net.workflows.at128 import prepare_at128

if __name__ == "__main__":
    prepare_at128()
