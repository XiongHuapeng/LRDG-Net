import numpy as np
import torch
from paper_experiments.ablation.config import MODEL, DATA
from paper_experiments.ablation.src.data.local_descriptors import extract_local_descriptors
from paper_experiments.ablation.src.models.ablation import build_ablation_model, VARIANT_SPECS

if __name__ == "__main__":
    rng=np.random.default_rng(42)
    pts=np.column_stack([rng.normal(size=(3000,3)).astype(np.float32),rng.uniform(0,255,size=3000).astype(np.float32)])
    d=extract_local_descriptors(pts,DATA.voxel_size,DATA.context_radius_voxels,DATA.intensity_scale,DATA.relative_std_eps)
    counts={}
    for v in MODEL.variants:
        model=build_ablation_model(v,MODEL); spec=VARIANT_SPECS[v]; n=len(d["geometry"])
        g=torch.from_numpy(d["geometry"]) if spec["geometry"] else torch.empty((n,0),dtype=torch.float32)
        if spec["degradation"]=="absolute": x=torch.from_numpy(d["absolute_degradation"])
        elif spec["degradation"]=="relative": x=torch.from_numpy(d["relative_degradation"])
        else: x=torch.empty((n,0),dtype=torch.float32)
        y,_=model(g,x,torch.zeros(n,dtype=torch.long),1,return_aux=True)
        counts[v]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(v,counts[v],tuple(y.shape))
    assert counts["E4_GR"]==10898
    print("Final ablation smoke test passed.")
