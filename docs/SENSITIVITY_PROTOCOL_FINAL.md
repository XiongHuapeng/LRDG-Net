# Final Sensitivity Protocol

This document defines the local-scale sensitivity protocol used in the manuscript.

## Purpose

Sensitivity is a compact secondary experiment. It examines the two structural parameters of the LRDG descriptor while leaving the final model configuration frozen.

## Fixed evaluation protocol

- LIDAROC acquisition domains: 5m / 10m / 20m.
- Six source→target directions.
- Binary clean vs contaminated task.
- Seed: 42 only.
- Same source Train/Val isolation and source-only normalization as the main experiment.

## One-factor-at-a-time settings

### Voxel-size study

Context radius fixed to 1 voxel:

```text
v = 0.10, 0.20, 0.30, 0.40 m
```

### Context study

Voxel size fixed to 0.20 m:

```text
radius = 1, 2, 3
context side = 3, 5, 7 voxels
```

The final paper setting remains:

```text
v = 0.20 m
radius = 1
```

The sensitivity experiment is not used to retroactively select a new optimum; it shows that the final setting lies in a broad stable performance region.
