# Reproducibility checklist

Before running a full experiment:

1. Install `requirements.txt` and verify the PyTorch/CUDA environment.
2. Configure the raw-data paths through environment variables or `config.py`.
3. Run `python run_00_smoke_test.py`.
4. Run the experiment-specific smoke test (`run_10`, `run_14`, `run_17`, or `run_19`).
5. Run `python tools/verify_project.py` to check canonical protocol settings and bundled result snapshots.

## Canonical settings

- Binary labels: clean=0, contaminated=1.
- Domains: `5m`, `10m`, `20m`.
- Main seeds: `42`, `2026`, `3407`.
- Source validation ratio: 0.20 at session level.
- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: 8.
- Maximum epochs: 60.
- Early-stopping patience: 10.
- Full-model local scale: voxel size 0.20 m, context radius 1.
- Full-model pooling: mean.

## Leakage controls

The main cross-domain pipeline fits normalization only on the source training split. Target-domain data do not participate in early stopping or checkpoint selection. AT128 strict external evaluation uses source-trained normalization and checkpoints.

## Result snapshots

`reference_results/` contains read-only snapshots bundled with the source project. They are used by `tools/verify_project.py` only for consistency checking. They are never inputs to training.

## Determinism

`lrdg_net/utils/reproducibility.py` fixes Python, NumPy, and PyTorch seeds and uses deterministic cuDNN settings when available. Exact bitwise equality across GPU architectures, CUDA/cuDNN versions, or third-party graph-library versions is not guaranteed; paper-level comparisons should use the saved metric summaries and the specified seeds/protocol.
