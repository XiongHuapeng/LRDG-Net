# Experiment protocol

## Task

Binary LiDAR cover-contamination recognition:

```text
0 = clean
1 = contaminated
```

### LIDAROC

```text
clean -> 0
water / dust / mud / oil film / other contamination -> 1
```

### Hesai AT128

```text
*_0.pcap -> 0
*_1.pcap -> 1
*_2.pcap -> 1
*_3.pcap -> 1
```

AT128 levels 1/2/3 identify preparation conditions only. They are not model classes and are not treated as a quantitative contamination scale.

## Protocol A — Six directed LIDAROC cross-acquisition transfers

The manuscript treats `5m`, `10m`, and `20m` as three acquisition domains. For each source domain and seed in `{42, 2026, 3407}`:

1. split source acquisitions into train/validation sets at the acquisition-session level;
2. fit input normalization on source training samples only;
3. train with source training data and use only source validation data for early stopping/model selection;
4. freeze the selected model;
5. evaluate independently on each of the other two domains.

This gives 9 source trainings and 18 seed-specific source→target evaluations, summarized as the six directed transfers:

```text
5 -> 10, 5 -> 20, 10 -> 5, 10 -> 20, 20 -> 5, 20 -> 10
```

For each direction, metrics are averaged over the three seeds. The six direction means are then equally weighted for the manuscript mean, worst-direction value, and descriptive standard deviation.

## Protocol B — All-LIDAROC source models for external validation

For the AT128 experiment, source models are trained on the union of the `5m`, `10m`, and `20m` LIDAROC source data, independently for seeds `42`, `2026`, and `3407`.

Source train/validation separation remains at the acquisition-session level. Input normalization is fitted on source training data only, and early stopping/checkpoint selection use source validation data only. AT128 is not used for training, source normalization fitting, early stopping, or checkpoint selection.

## Protocol C — AT128 direct cross-sensor transfer

The paper's primary external-validation unit is one complete AT128 acquisition sequence (one PCAP recording). The direct-transfer decision is:

1. decode the PCAP to complete-frame float32 XYZI samples;
2. extract the same frozen 12D centered-geometry + 4D local-relative-degradation descriptors;
3. apply the LIDAROC-trained normalization and frozen LIDAROC checkpoint;
4. compute each frame logit margin `m_t = z_contaminated - z_clean`;
5. average frame margins within the sequence to obtain `M(S)`;
6. predict contaminated when `M(S) >= 0`.

The evaluator also stores frame-level probabilities and a fixed `0.50` frame-level probability diagnostic. That diagnostic is **not** the manuscript's primary sequence-level direct-transfer decision rule.

### Strict external-validation constraints

AT128 labels are not used to:

- fit source/input normalization;
- select a source checkpoint;
- change the architecture or descriptor definition;
- tune the direct-transfer boundary;
- fine-tune or update the network.

The manuscript reports direct transfer at the sequence level and separately reports ranking metrics (AUROC and average precision) to distinguish score ordering from fixed-operating-point classification.

## Protocol D — Clean-reference operating-point calibration

Clean-reference calibration is a separate post-processing protocol and does not update the network.

- Source clean-training frame margins determine source median `c_S` and MAD scale `s_S`.
- A normalized source sequence threshold `tau_S` is selected using source validation sessions only.
- For each target evaluation fold, one (`K1`) or three (`K3`) known-clean target sequences from other distances estimate target median `c_T` and MAD scale `s_T`.
- Held-out target sequence frame margins are normalized by `(c_T, s_T)`, averaged within the sequence, and compared with the fixed source threshold using `S(S) >= tau_S`.
- Contaminated target labels are used only for final evaluation.

The AT128 evaluation uses leave-one-distance-out testing and exhaustive eligible K1/K3 clean-reference choices; it does not manually select a favorable calibration distance.

## Independent evaluation unit

One `.pcap` is one independent AT128 acquisition sequence. Frames within the same recording are repeated temporal observations and are not treated as independent physical replicates.
