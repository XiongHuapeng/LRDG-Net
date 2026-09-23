# Code-to-manuscript alignment

This repository was organized against the manuscript **“LiDAR cover contamination recognition using a local relative degradation representation.”**

| Manuscript item | Repository implementation |
|---|---|
| 12D centered geometry + 4D local relative degradation | `lrdg_net/data/local_descriptors.py` |
| Full LRDG-Net, 10,898 trainable parameters | `lrdg_net/models/lrdg_net.py` |
| Session-level source Train/Val separation | `lrdg_net/data/splitter.py` |
| Source-only normalization | `lrdg_net/data/normalization.py`, `lrdg_net/training/trainer.py` |
| Six directed LIDAROC transfers | `lrdg_net/workflows/lidaroc.py`, `run_02_internal_cross_domain.py` |
| Global statistics / RangeNet-style / PointNet / PointNet++ / DGCNN / PointNeXt-S-style / AutoGrAN | `paper_experiments/baseline/` |
| G, A, R, G+A, G+R ablation | `paper_experiments/ablation/` |
| Single-component + leave-one-component-out R study | `paper_experiments/component_ablation/` |
| Voxel-size/context-radius sensitivity | `paper_experiments/sensitivity/` |
| AT128 PCAP decoding and preparation | `lrdg_net/data/at128_pcap.py`, `lrdg_net/workflows/at128.py` |
| Strict frozen LIDAROC → AT128 evaluation | `run_06_all_lidaroc_to_at128.py` |
| Uncalibrated AT128 sequence decision: mean frame logit margin ≥ 0 | `lrdg_net/evaluation/at128.py`, `lrdg_net/workflows/at128_calibration.py` (`K0_raw`) |
| Clean-reference operating-point calibration | `lrdg_net/workflows/at128_calibration.py`, `run_09_at128_calibration.py` |

## Claims intentionally separated in code

The repository preserves the distinction used in the manuscript between:

- **strict external evaluation**, where AT128 is not used for training, source normalization, or checkpoint selection;
- **clean-reference operating-point calibration**, where only known-clean target sequences establish the score reference and the network remains frozen; and
- **target-supervised transfer controls**, which use AT128 labels and are therefore supplementary rather than strict external-validation evidence.
