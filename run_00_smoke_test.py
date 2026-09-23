"""Experiment entry point: data-free integrity test."""
import numpy as np
import torch

from config import DATA
from lrdg_net.data.local_descriptors import extract_local_descriptors
from lrdg_net.training.trainer import build_model


def main():
    rng = np.random.default_rng(0)
    xyz1 = np.column_stack([
        rng.uniform(0, 8, 5000), rng.uniform(-3, 3, 5000), rng.normal(0, 0.03, 5000)
    ])
    xyz2 = np.column_stack([
        rng.uniform(0, 8, 4000), rng.normal(2.0, 0.03, 4000), rng.uniform(0, 2, 4000)
    ])
    xyz = np.vstack([xyz1, xyz2]).astype(np.float32)
    intensity = rng.integers(0, 256, size=(len(xyz), 1), endpoint=False).astype(np.float32)
    points = np.concatenate([xyz, intensity], axis=1)

    descriptors = extract_local_descriptors(
        points,
        DATA.voxel_size,
        DATA.context_radius_voxels,
        DATA.intensity_scale,
        DATA.relative_std_eps,
    )
    assert descriptors["geometry"].shape[1] == 12
    assert descriptors["relative_degradation"].shape[1] == 4

    model = build_model()
    geometry = torch.from_numpy(descriptors["geometry"])
    degradation = torch.from_numpy(descriptors["relative_degradation"])
    bag_index = torch.zeros(len(geometry), dtype=torch.long)
    logits, aux = model(geometry, degradation, bag_index, batch_size=1, return_aux=True)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 10898, f"Expected 10,898 parameters, got {params}"

    print("LRDG-Net smoke test PASSED / 烟雾测试通过")
    print(f"geometry={tuple(geometry.shape)}, degradation={tuple(degradation.shape)}")
    print(f"embedding={tuple(aux['embedding'].shape)}, logits={tuple(logits.shape)}")
    print(f"parameters={params:,}")


if __name__ == "__main__":
    main()
