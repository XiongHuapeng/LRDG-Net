from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from paper_experiments.component_ablation.config import DATA, MODEL
from paper_experiments.component_ablation.src.data.local_descriptors import extract_local_descriptors, load_bin, RELATIVE_DEGRADATION_SCHEMA
from paper_experiments.component_ablation.src.models.r_component import VARIANT_SPECS, R_COMPONENT_NAMES

MANIFEST_FILENAME = "manifest.csv"
SESSION_MANIFEST_FILENAME = "session_manifest.csv"
CACHE_VERSION = "lrdg_r_only_component_ablation_v1"
META_COLS = [
    "sample_uid", "source_path", "domain", "raw_class", "pollution_type",
    "severity", "session_id", "frame_id", "session_uid",
]


def _signature(cfg):
    return {
        "cache_version": CACHE_VERSION,
        "voxel_size": cfg.voxel_size,
        "context_radius_voxels": cfg.context_radius_voxels,
        "intensity_scale": cfg.intensity_scale,
        "relative_std_eps": cfg.relative_std_eps,
        "relative_degradation_schema": list(RELATIVE_DEGRADATION_SCHEMA),
        "canonical_component_names": list(R_COMPONENT_NAMES),
        "pooling": "mean_only",
        "variants": list(MODEL.variants),
        "feature_subset_mode": "true_input_dimensionality_no_zero_mask",
    }


def build_feature_cache(raw_dataframe, cache_root, data_config=DATA):
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    feat = root / "features"
    sig = root / "preprocess_config.json"
    cur = _signature(data_config)

    if data_config.overwrite_cache and feat.exists():
        shutil.rmtree(feat)
    if sig.exists() and not data_config.overwrite_cache:
        old = json.loads(sig.read_text(encoding="utf-8"))
        if old != cur:
            raise RuntimeError(
                "Cache config changed. Use a new workspace_root (recommended) or set "
                "DATA.overwrite_cache=True once."
            )

    feat.mkdir(parents=True, exist_ok=True)
    sig.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for row in tqdm(raw_dataframe.to_dict("records"), desc="LRDG R-only component preprocess"):
        p = feat / (row["sample_uid"].replace("|", "__") + ".npz")
        if p.exists() and not data_config.overwrite_cache:
            with np.load(p, allow_pickle=False) as z:
                n = int(z["relative_degradation"].shape[0])
                pc = int(z["point_count"])
        else:
            pts = load_bin(row["source_path"])
            desc = extract_local_descriptors(
                pts,
                data_config.voxel_size,
                data_config.context_radius_voxels,
                data_config.intensity_scale,
                data_config.relative_std_eps,
            )
            r = desc["relative_degradation"]
            n = r.shape[0]
            pc = pts.shape[0]
            np.savez_compressed(
                p,
                relative_degradation=r,
                point_count=np.asarray(pc, np.int64),
            )
        rows.append({**row, "cache_path": str(p.resolve()), "point_count": pc, "instance_count": n})
    return pd.DataFrame(rows)


class RComponentDataset(Dataset):
    def __init__(self, dataframe, stats, variant):
        self.df = dataframe.reset_index(drop=True)
        self.stats = stats
        self.variant = variant.upper()
        if self.variant not in VARIANT_SPECS:
            raise ValueError(f"Unsupported variant: {self.variant}")
        self.indices = tuple(VARIANT_SPECS[self.variant]["indices"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with np.load(r.cache_path, allow_pickle=False) as z:
            full = z["relative_degradation"].astype(np.float32)
        x = full[:, self.indices]
        if self.stats is not None:
            x = self.stats.normalize_degradation(x)
        meta = {k: str(r[k]) for k in META_COLS}
        return torch.from_numpy(x), torch.tensor(int(r.label), dtype=torch.long), meta


def collate_bags(batch):
    xs, ys, ms = zip(*batch)
    counts = [x.shape[0] for x in xs]
    x = torch.cat(xs, 0)
    y = torch.stack(ys)
    bi = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)], 0)
    meta = {k: [m[k] for m in ms] for k in ms[0]}
    meta["instance_count"] = counts
    return x, bi, y, meta


def make_loader(df, batch_size, num_workers, shuffle, stats, variant):
    return DataLoader(
        RComponentDataset(df, stats, variant),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )
