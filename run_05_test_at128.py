"""Experiment entry point: evaluate the model selected in config.AT128 on prepared AT128 data."""
from lrdg_net.workflows.external import test_configured_model_on_at128

if __name__ == "__main__":
    test_configured_model_on_at128()
