# R-only component-ablation protocol

## Purpose

This experiment isolates the four components of the final 4D local-relative degradation representation `R` and evaluates both **single-component sufficiency** and **conditional necessity/redundancy**. The geometry branch is excluded from every configuration so that centered geometry cannot compensate for deleted degradation information.

## Descriptor components

Canonical order (matching `RELATIVE_DEGRADATION_SCHEMA`):

1. `ΔI = mean_i/255 - mean_context/255`
2. `Rσ = log((std_i/255 + 1e-3)/(std_context/255 + 1e-3))`
3. `V_I = std_i/255`
4. `Rρ = log1p(n_i) - log1p(n_context/27)`

## Configurations

Reference:

- `R_FULL = [ΔI, Rσ, V_I, Rρ]`

Single-component sufficiency:

- `R_ONLY_DELTA_I = [ΔI]`
- `R_ONLY_LOG_STD_RATIO = [Rσ]`
- `R_ONLY_LOCAL_STD = [V_I]`
- `R_ONLY_REL_LOG_DENSITY = [Rρ]`

Leave-one-component-out:

- `R_WO_DELTA_I = [Rσ, V_I, Rρ]`
- `R_WO_LOG_STD_RATIO = [ΔI, V_I, Rρ]`
- `R_WO_LOCAL_STD = [ΔI, Rσ, Rρ]`
- `R_WO_REL_LOG_DENSITY = [ΔI, Rσ, V_I]`

## Controlled retraining

Every configuration is trained from scratch using the same source-domain session split, seeds, optimizer, learning rate, weight decay, batch size, epoch limit, early stopping, mean bag pooling, and six source→target evaluations. Only the selected feature subset changes.

For each variant/source training split, normalization is fit only on the selected R components from source-training frames. Target-domain samples are not used for normalization.

## Network capacity

The R-only branch keeps the hidden widths:

```text
input_dim -> 16 -> 32 -> 64 -> mean pooling -> 64 -> 2
```

The true input dimensionality is 4, 3, or 1. No all-zero dummy channels are retained. `R_FULL` has **7,250 trainable parameters**; parameter-count differences for 3D/1D subsets are therefore expected and are reported.

## Metrics and interpretation

For each configuration, the code reports mean Macro-F1 across six directed transfers, worst-direction Macro-F1, descriptive standard deviation across direction means, mean FPR, mean FNR, and parameter count. Direction metrics are first averaged over seeds 42, 2026, and 3407.

Single-component experiments estimate whether a statistic is individually sufficient for discrimination. Leave-one-out experiments estimate whether it provides non-redundant information conditional on the remaining three statistics. Neither analysis is a causal or Shapley-style importance decomposition.

`R_FULL` is rerun through the same code path as all eight subset variants and serves as the reference for component-wise differences.
