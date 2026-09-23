# LRDG-Net model specification

This document records the architecture and input representation used for the main manuscript experiments.

## Input

Each frame is an `N x 4` float32 point array:

```text
[x, y, z, intensity]
```

Absolute coordinates are used for voxel partitioning and local statistics, but are not directly passed to the classifier.

## Local voxel context

The main configuration uses voxel side length `v = 0.20 m` and context radius `r = 1`. The cubic context therefore contains

```text
q = (2r + 1)^3 = 27
```

grid locations. Empty grid locations contribute zero observed points but remain part of `q` in the occupancy statistic.

## Centered geometry `G` (12D)

For the center voxel and its context, compute population covariance after centering by the corresponding centroid. Normalize covariance by `v^2`, retain the six independent symmetric entries in order

```text
xx, xy, xz, yy, yz, zz
```

and concatenate center + context values to obtain 12 dimensions.

## Local relative degradation `R` (4D)

Intensity mean and standard deviation are divided by 255. The four components are:

```text
Delta I = mean_i/255 - mean_Ci/255
R_sigma = log((std_i/255 + 1e-3) / (std_Ci/255 + 1e-3))
V_I     = std_i/255
R_rho   = log(1+n_i) - log(1+n_Ci/q)
```

The term "local relative degradation" describes the center-context organization of the representation; `V_I` is an absolute local variability statistic.

## Network

```text
G: 12 -> 32 -> 32
R:  4 -> 16 -> 32
          concat -> 64
                    |
              fusion 64 -> 64
                    |
            mean over voxels
                    |
             classifier 64 -> 64 -> 2
```

Hidden feature-encoder layers use `Linear -> LayerNorm -> GELU`, with dropout after the first hidden layer. The fusion layer uses `Linear -> LayerNorm -> GELU -> Dropout`. The classifier is `Linear -> GELU -> Dropout -> Linear`. Dropout probability is `0.2`.

Trainable parameter count: **10,898**.

## Training and model selection

- Loss: two-class cross entropy.
- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `8`.
- Maximum epochs: `60`.
- Early-stopping patience: `10`.
- Training seeds: `42`, `2026`, `3407`.
- Source train/validation split is performed by complete acquisition sequence/session.
- Input normalization is fitted only on source training data.
- Checkpoint selection uses source validation only; target-domain samples are not used for early stopping or checkpoint selection.

## Cross-sensor operating-point calibration

For each frame, the logit margin is `m = z_1 - z_0`. Source and target clean-reference center/scale estimates use median and Gaussian-consistent MAD (`1.4826 * MAD`). Sequence scores average normalized frame margins. The calibrated decision follows the manuscript convention:

```text
contaminated iff sequence_score >= tau_S
```

The implementation contains deterministic fallbacks if a MAD scale is numerically degenerate; these are numerical safeguards rather than an additional learned calibration model.
