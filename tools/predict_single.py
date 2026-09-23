"""Utility: predict one float32 XYZI .bin frame."""
from pathlib import Path

import numpy as np
import torch

from config import DATA, SINGLE_INFERENCE
from lrdg_net.data.local_descriptors import extract_local_descriptors, load_bin
from lrdg_net.evaluation.common import load_frozen_model


def main():
    model_dir = Path(SINGLE_INFERENCE.model_dir)
    bin_path = Path(SINGLE_INFERENCE.bin_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found / 模型目录不存在: {model_dir}")
    if not bin_path.exists():
        raise FileNotFoundError(f"bin_path not found / 点云不存在: {bin_path}")

    points = load_bin(str(bin_path))
    d = extract_local_descriptors(
        points,
        DATA.voxel_size,
        DATA.context_radius_voxels,
        DATA.intensity_scale,
        DATA.relative_std_eps,
    )
    model, _, stats, device = load_frozen_model(model_dir)
    geometry = torch.from_numpy(stats.normalize_geometry(d["geometry"])).to(device)
    degradation = torch.from_numpy(stats.normalize_degradation(d["relative_degradation"])).to(device)
    bag_index = torch.zeros(geometry.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(geometry, degradation, bag_index, batch_size=1)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        margin = float((logits[0, 1] - logits[0, 0]).item())
    pred = int(np.argmax(probs))

    print(f"File / 文件: {bin_path}")
    print(f"Points / 点数: {len(points):,}")
    print(f"Occupied voxels / 非空体素: {len(geometry):,}")
    print(f"Prediction / 预测: {'contaminated' if pred else 'clean'}")
    print(f"P(clean)={probs[0]:.8f}, P(contaminated)={probs[1]:.8f}")
    print(f"Logit margin (contaminated-clean)={margin:.8f}")


if __name__ == "__main__":
    main()
