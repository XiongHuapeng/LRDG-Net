"""Data-free integrity test for the four-component R ablation."""
import numpy as np
import torch
from paper_experiments.component_ablation.config import MODEL, DATA
from paper_experiments.component_ablation.src.data.local_descriptors import extract_local_descriptors, RELATIVE_DEGRADATION_SCHEMA
from paper_experiments.component_ablation.src.models.r_component import build_r_component_model, VARIANT_SPECS, R_COMPONENT_NAMES


def main():
    assert tuple(RELATIVE_DEGRADATION_SCHEMA) == tuple(R_COMPONENT_NAMES)
    rng = np.random.default_rng(42)
    points = np.column_stack([
        rng.normal(size=(3000, 3)).astype(np.float32),
        rng.uniform(0, 255, size=3000).astype(np.float32),
    ])
    desc = extract_local_descriptors(points, DATA.voxel_size, DATA.context_radius_voxels, DATA.intensity_scale, DATA.relative_std_eps)
    full = desc["relative_degradation"]
    assert full.shape[1] == 4
    counts = {}
    for variant in MODEL.variants:
        indices = tuple(VARIANT_SPECS[variant]["indices"])
        x_np = full[:, indices]
        model = build_r_component_model(variant, MODEL)
        x = torch.from_numpy(x_np)
        bag_index = torch.zeros(x.shape[0], dtype=torch.long)
        logits, aux = model(x, bag_index, 1, return_aux=True)
        counts[variant] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert tuple(logits.shape) == (1, 2)
        assert tuple(aux["embedding"].shape) == (1, 64)
    assert counts["R_FULL"] == 7250
    assert counts["R_ONLY_DELTA_I"] < counts["R_WO_DELTA_I"] < counts["R_FULL"]
    print("Component-ablation smoke test PASSED")
    for key in MODEL.variants:
        print(f"{key}: {counts[key]:,} parameters")


if __name__ == "__main__":
    main()
