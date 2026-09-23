# Data directory

Raw datasets are intentionally not included in this repository.

## LIDAROC

Set `LRDG_LIDAROC_ROOT` or place the dataset under `data/lidaroc/`.
The code recursively scans `*.bin` files and expects the filename pattern used by the experiments:

```text
<class_id>_<class_name>_<level_id>_<severity>_<session_id>_<frame_id>.bin
```

The acquisition domain (`5m`, `10m`, or `20m`) is inferred from the directory path. Each `.bin` file must contain float32 XYZI points (`x, y, z, intensity`). `clean` is mapped to class 0; `water`, `dust`, `mud`, and `oil` are mapped to class 1. `muddrop`/`muduniform` aliases are normalized to `mud`.

Train/validation splitting is performed at the complete acquisition-session level; target-domain samples are not used for source normalization, checkpoint selection, or early stopping.

## Hesai AT128 external data

Set `LRDG_AT128_PCAP_ROOT` or place classic-PCAP recordings under `data/at128/pcap/`. The decoder expects recording names of the form:

```text
<distance>m_<level>.pcap
```

where `level=0` is clean and `level=1/2/3` are contaminated analysis strata. The task remains binary.

The AT128 angle-calibration file can be specified through `LRDG_AT128_CALIBRATION`; the default location is `data/at128/AT128P_AngleCalibration.dat`.

The AT128 recordings used in the manuscript are not redistributed here. Their inclusion depends on the authors' data-release rights.
