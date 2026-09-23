# AT128 clean-reference operating-point calibration protocol

## Purpose

This is the manuscript's cross-sensor operating-point calibration experiment. It tests whether a frozen All-LIDAROC LRDG-Net can be deployed on a different LiDAR using only a small amount of **known-clean target-sensor data**, without contaminated AT128 labels and without network retraining.

The evaluation avoids choosing a favorable calibration distance manually. Every AT128 distance is used once as an unseen outer test block, and every eligible clean-reference choice from the remaining distances is evaluated.

## Inputs

The workflow reuses the frame-level outputs produced by `run_06_all_lidaroc_to_at128.py`:

```text
<workspace_root>/evaluations/at128_external/
├── LRDG_ALL_seed42/at128_frame_predictions.csv
├── LRDG_ALL_seed2026/at128_frame_predictions.csv
└── LRDG_ALL_seed3407/at128_frame_predictions.csv
```

No AT128 point-cloud inference is rerun during calibration.

## 1. Source clean-score reference

For each `LRDG_ALL_seed*` checkpoint, define the frame logit margin

\[
m_t = z_{t,1} - z_{t,0}.
\]

Using only **clean frames in the source training split**, estimate

\[
c_S = \operatorname{median}_{t\in C_S}(m_t),
\]

\[
s_S = 1.4826\,\operatorname{median}_{t\in C_S}|m_t-c_S|.
\]

The normalized source frame score is

\[
\widetilde m_t^S = \frac{m_t-c_S}{s_S}.
\]

Normalized frame scores are mean-pooled within each source validation acquisition sequence. A single source threshold `tau_S` is selected on source validation sequences by maximizing sequence-level Macro-F1. Ties prefer lower FPR and then smaller `|tau_S|`.

AT128 never participates in source threshold selection.

## 2. Outer leave-one-distance-out evaluation

AT128 contains 13 distance blocks, from 8 m through 20 m. For each held-out distance `d`:

```text
Test block = {d_0, d_1, d_2, d_3}
```

All four sequences at that distance are locked as test data. No sequence from the held-out distance is used as a clean reference.

Across all 13 folds, all 52 AT128 sequences are evaluated once as members of a held-out distance block.

## 3. K0 — direct transfer without target clean references

`K0_raw` implements the manuscript's uncalibrated sequence decision:

1. compute each frame's logit margin;
2. mean-pool margins over all frames in the sequence;
3. predict contaminated when the mean margin is `>= 0`.

No target calibration data are used. Mean frame probability is retained in exported files as a diagnostic only. Earlier repository exports also contained a `0.50` probability threshold diagnostic; for the archived 52-sequence predictions, it gives the same K0 labels, but the manuscript decision rule is the mean-logit-margin boundary above.

## 4. K1 — exhaustive one-clean-sequence calibration

For a held-out distance `d`, each of the 12 clean sequences from the other distances is used separately as the target clean-reference set:

\[
K=1:\quad 12\ \text{reference choices per outer fold}.
\]

With 13 outer folds and 3 source-model seeds, this gives

\[
13\times12\times3=468
\]

K1 evaluations.

## 5. K3 — exhaustive three-clean-sequence calibration

For each held-out distance, every three-sequence subset of the 12 remaining clean sequences is evaluated:

\[
\binom{12}{3}=220\ \text{reference choices per outer fold}.
\]

Across 13 outer folds and 3 source-model seeds:

\[
13\times220\times3=8580
\]

K3 evaluations.

These repeated calibration runs reuse the same physical acquisition sequences and are not interpreted as 8580 independent physical replicates.

## 6. Target clean normalization and sequence decision

For each target clean-reference subset `C_T`, pool its clean-reference frames and estimate

\[
c_T = \operatorname{median}_{t\in C_T}(m_t),
\]

\[
s_T = 1.4826\,\operatorname{median}_{t\in C_T}|m_t-c_T|.
\]

For a held-out sequence `S`, average target-normalized frame margins:

\[
S(S)=\frac{1}{|S|}\sum_{t\in S}\frac{m_t-c_T}{s_T}.
\]

The sequence prediction is

\[
\hat y_S = \mathbb{I}[S(S)\ge\tau_S].
\]

The release code uses this same `>=` convention, including equality cases.

No contaminated AT128 sample is used to estimate `c_T`, `s_T`, or `tau_S`. The network and original source-domain input normalization remain unchanged.

## 7. Summary statistics

For K1/K3, results are first equally averaged over eligible clean-reference choices and the three source-model seeds within each held-out distance. The manuscript summary then equally weights the 13 held-out distance means and reports:

- mean Macro-F1;
- worst distance-wise mean Macro-F1;
- FPR;
- FNR.

The repository also exports additional descriptive diagnostics by seed, distance, and reference choice. Those diagnostics must not be interpreted as additional independent physical test sets.

## Outputs

```text
<workspace_root>/evaluations/at128_calibration/
├── paper_cross_sensor_summary.csv
├── at128_calibration_runs.csv
├── at128_calibration_by_distance.csv
├── at128_calibration_by_seed.csv
├── at128_calibration_pcap_predictions.csv
├── source_reference_summary.csv
├── target_clean_calibration_summary.csv
├── calibration_choice_robustness.csv
├── calibration_failures.csv
├── protocol.json
├── distance_response_clean_norm.png
└── source_reference/
   ├── LRDG_ALL_seed42.json
   ├── ..._clean_train_scores.csv
   └── ..._validation_session_scores.csv
```

## Scientific constraints

- Frozen LRDG-Net checkpoints.
- Original LIDAROC input normalization remains unchanged.
- Source clean statistics use source training only.
- Source threshold selection uses source validation only.
- Target center/scale use known-clean AT128 sequences only.
- No contaminated target label is used for calibration.
- No network parameter update.
- No AT128 re-inference during calibration.
- Complete held-out-distance separation between target clean references and test sequences.
- Exhaustive K1/K3 reference choices; no manually selected calibration distance.
