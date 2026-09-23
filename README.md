# LRDG-Net

Official Python implementation of **LRDG-Net** for LiDAR cover contamination recognition.

This repository provides the code used for the paper experiments, including the main LIDAROC cross-acquisition experiments, ablation studies, and Hesai AT128 external validation with clean-reference operating-point calibration.

<p align="center">
  <img src="assets/lrdg_net_overview.png" width="850" alt="LRDG-Net overview">
</p>

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment and install the project:

```bash
python -m pip install --upgrade pip
pip install -e .
```

If you need a specific CPU/CUDA build of PyTorch, install PyTorch first and then run:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

AutoGrAN additionally requires:

```bash
pip install -r requirements-optional.txt
```

Quick check:

```bash
python -m lrdg_net smoke
python -m unittest discover -s tests -v
```

## Data

Raw datasets are not included in this repository.

### LIDAROC

**LIDAROC is a public LiDAR cover-contamination dataset.**

Place the dataset under:

```text
data/lidaroc/
```

or set:

```bash
LRDG_LIDAROC_ROOT=/path/to/lidaroc
```

Each scan is expected to be a float32 `N x 4` XYZI `.bin` file. The acquisition domains `5m`, `10m`, and `20m` are inferred from the directory structure.

Please cite the original LIDAROC paper when using the dataset:

```bibtex
@ARTICLE{10613519,
  author={Jati, Grafika and Molan, Martin and Barchi, Francesco and Bartolini, Andrea and Mercurio, Giuseppe and Acquaviva, Andrea},
  journal={IEEE Sensors Letters},
  title={LIDAROC: Realistic LiDAR Cover Contamination Dataset for Enhancing Autonomous Vehicle Perception Reliability},
  year={2024},
  volume={8},
  number={9},
  pages={1-4},
  keywords={Laser radar;Contamination;Object detection;Automobiles;Pedestrians;Accuracy;Sensor phenomena and characterization;Sensor phenomena;anomaly;autonomous vehicle;contamination;dataset;LiDAR corruption;object detection benchmark;perception robustness testing;sensor},
  doi={10.1109/LSENS.2024.3434624}
}
```

### Hesai AT128 external-validation dataset

The AT128 dataset used for the external-validation experiments is available on Zenodo:

https://zenodo.org/records/22858748

Place the PCAP recordings under:

```text
data/at128/pcap/
```

and place the AT128 angle-calibration file at:

```text
data/at128/AT128P_AngleCalibration.dat
```

Alternatively:

```bash
LRDG_AT128_PCAP_ROOT=/path/to/at128_pcaps
LRDG_AT128_CALIBRATION=/path/to/AT128P_AngleCalibration.dat
```

The expected PCAP naming format is:

```text
<distance>m_<level>.pcap
```

where `level=0` is clean and `level=1/2/3` are contaminated conditions.

See `data/README.md` for the detailed data format.

## Paper configuration

The default settings in `config.py` follow the paper:

```text
Voxel size:              0.20 m
Context radius:          1
Optimizer:               AdamW
Learning rate:           1e-3
Weight decay:            1e-4
Batch size:              8
Maximum epochs:          60
Early-stopping patience: 10
Seeds:                   42, 2026, 3407
```

Generated caches, checkpoints, and evaluation files are written to `outputs/`.

To use CPU:

```bash
LRDG_DEVICE=cpu
```

## Reproduce the paper

### Main LIDAROC cross-acquisition experiment

Preprocess LIDAROC:

```bash
python run_01_preprocess_lidaroc.py
```

Run the six directed transfers among `5m`, `10m`, and `20m`:

```bash
python run_02_internal_cross_domain.py
```

### AT128 external validation

Train the source models using all three LIDAROC acquisition domains:

```bash
python run_03_train_all_lidaroc.py
```

Prepare the AT128 recordings:

```bash
python run_04_prepare_at128.py
```

Run the strict LIDAROC-to-AT128 external-validation experiment:

```bash
python run_06_all_lidaroc_to_at128.py
```

Run clean-reference operating-point calibration:

```bash
python run_09_at128_calibration.py
```

### Ablation and supplementary experiments

```bash
# Comparison baselines
python run_11_baseline_preprocess.py
python run_12_baseline_cross_domain.py
python run_13_baseline_paper_table.py

# Representation-level ablation
python run_15_ablation_preprocess.py
python run_16_ablation_cross_domain.py

# Local-scale sensitivity
python run_18_sensitivity.py

# Degradation-component ablation
python run_20_component_ablation_preprocess.py
python run_21_component_ablation_cross_domain.py
python run_22_component_ablation_paper_table.py
```

Detailed protocols are available under `docs/`.

## Project structure

```text
LRDG-Net/
├── lrdg_net/              # main model, data, training and evaluation code
├── paper_experiments/     # baselines and ablations
├── docs/                  # experiment protocols
├── tests/                 # data-free tests
├── tools/                 # utility scripts
├── data/                  # dataset location and instructions
├── config.py              # paths and paper configuration
├── requirements.txt
├── pyproject.toml
└── run_00_...run_22_*.py  # paper experiment entry points
```

## Citation

If you use this code, please cite the LRDG-Net paper and the datasets used in your experiments. Repository citation metadata are provided in `CITATION.cff`.
