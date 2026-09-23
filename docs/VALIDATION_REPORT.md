# Validation report

This report records the data-free checks performed on the final Python project. It is not a substitute for rerunning the complete experiments on the raw datasets.

## Checks

- Python bytecode compilation for `config.py`, `lrdg_net/`, `paper_experiments/`, `tools/`, and `tests/`.
- Unit tests for the 10,898-parameter architecture, 12D/4D descriptor formulas, the `>=` calibrated decision boundary, and stored manuscript-result snapshots.
- Main LRDG-Net descriptor/model smoke test.
- Representation-ablation smoke test for `G/A/R/G+A/G+R`.
- Sensitivity-model smoke test.
- R-component-ablation smoke test for all nine variants.
- Static consistency checks for paper-critical settings, reference-result snapshots, required Python project files, and accidental local path leakage.
- Baseline smoke checks for all methods that do not require the optional `torch-geometric` package.
- Editable package installation and `lrdg-net` command-line smoke/verification commands.

## Limitations

The raw LIDAROC and Hesai AT128 datasets were not included in the supplied source package, so full training and full end-to-end numerical reproduction were not rerun. Existing numerical snapshots under `reference_results/` are retained only as traceability artifacts and are not used by training code.

AutoGrAN requires the optional `torch-geometric` dependency. Its forward smoke test is skipped when that package is unavailable.
