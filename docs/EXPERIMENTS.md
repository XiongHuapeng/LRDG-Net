# Experiment entry points

This document identifies the canonical scripts corresponding to the final JEI manuscript. Generated artifacts are written under `config.PATHS.workspace_root` (default: `outputs/`).

## 1. Main LRDG-Net on LIDAROC

```bash
python run_01_preprocess_lidaroc.py
python run_02_internal_cross_domain.py
```

`run_02` trains each source domain (`5m`, `10m`, `20m`) independently for seeds `42`, `2026`, and `3407`, and evaluates on the other two domains. This yields 9 training runs and 18 directed evaluations.

## 2. All-source training and strict AT128 external test

```bash
python run_03_train_all_lidaroc.py
python run_04_prepare_at128.py
python run_06_all_lidaroc_to_at128.py
```

The AT128 test is frozen with respect to network parameters, source normalization, and checkpoint selection. AT128 levels 1/2/3 are contaminated metadata strata, not separate prediction classes.

`run_05_test_at128.py` is a convenience evaluator for a model selected in `config.AT128`.

## 3. Clean-reference AT128 operating-point calibration

```bash
python run_09_at128_calibration.py
```

The final protocol uses leave-one-distance-out evaluation and exhaustive clean-reference calibration choices for budgets K=1 and K=3. Network parameters remain fixed and contaminated target labels are not used to fit the calibration reference.

## 4. Baseline comparison

```bash
python run_10_baseline_check.py
python run_11_baseline_preprocess.py
python run_12_baseline_cross_domain.py
python run_13_baseline_paper_table.py
```

Canonical baseline IDs:

```text
globalstats
rangenet
pointnet
pointnet2_ref
dgcnn
pointnext
autogran
```

## 5. Representation-level ablation

```bash
python run_14_ablation_check.py
python run_15_ablation_preprocess.py
python run_16_ablation_cross_domain.py
```

Variants:

```text
E0_G   centered geometry only
E1_A   absolute degradation only
E2_R   local relative degradation only
E3_GA  geometry + absolute degradation
E4_GR  geometry + local relative degradation (full LRDG-Net)
```

## 6. Local-scale sensitivity

```bash
python run_17_sensitivity_check.py
python run_18_sensitivity.py
```

The OFAT settings are voxel sizes `0.10, 0.20, 0.30, 0.40 m` at radius 1, and context radii `1, 2, 3` at voxel size `0.20 m`. This study uses seed 42 as a trend/sensitivity analysis.

## 7. Four-component R ablation

```bash
python run_19_component_ablation_check.py
python run_20_component_ablation_preprocess.py
python run_21_component_ablation_cross_domain.py
python run_22_component_ablation_paper_table.py
```

The four canonical R components are:

1. `ΔI`: center-context mean-intensity difference;
2. `Rσ`: log ratio of center/context intensity dispersion;
3. `V_I`: center-voxel local intensity standard deviation;
4. `Rρ`: relative log occupancy/density.

The component study contains one full-R reference, four single-component sufficiency models, and four leave-one-component-out models. Geometry is excluded from this experiment by design.

## 8. Supplementary target-supervised AT128 controls

```bash
python run_07_at128_transfer.py
python run_08_at128_distance_transfer.py
```

These experiments use AT128 target labels during model/classifier development. They are controls for representation transfer and should not be interpreted as strict external generalization.
