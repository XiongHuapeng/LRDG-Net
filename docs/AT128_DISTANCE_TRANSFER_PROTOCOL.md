# AT128 Distance-Grouped Transfer Control

## Purpose

This is the final stricter control after the PCAP-grouped transfer experiment reached 100% for both methods.
It does **not** introduce a new model or a new task.

The only change is the split protocol:

- Primary supplementary transfer control: independent PCAP-grouped 5-fold.
- This control: **distance-grouped 5-fold**.

For every distance `d`, the four recordings

```text
{d}m_0
{d}m_1
{d}m_2
{d}m_3
```

must stay in the same fold. Therefore, when one distance is in the test fold, the training data contain no recording from that distance.

## Task

Binary only:

- level 0 -> clean (0)
- levels 1/2/3 -> contaminated (1)

No contamination-severity prediction is performed.

## Methods

Exactly the same two methods as v2:

1. Frozen All-LIDAROC LRDG representation + AT128 linear classifier.
2. AT128 LRDG-Net trained from random initialization.

No LRDG-Net architecture, descriptor, pooling, loss, or label definition is changed.

## Execution

After the All-LIDAROC models and AT128 cache already exist, run only:

```text
run_08_at128_distance_transfer.py
```

No terminal arguments are needed.

## Output

A new directory is used so that the previous PCAP-grouped result is never overwritten:

```text
<workspace_root>/evaluations/at128_transfer_distance_grouped/
```

Important files:

```text
at128_distance_transfer_folds.csv
fold_summary.csv
at128_transfer_summary.csv
at128_transfer_aggregate.csv
frozen_linear_probe/
at128_from_scratch/
```

## Interpretation

This control answers only one question:

> Does the 100% supervised AT128 result remain when the test distances are unseen during training?

After this experiment, no additional AT128 model expansion is planned unless the result itself provides a clear scientific reason.
