from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from paper_experiments.ablation.config import DATA, MODEL
from paper_experiments.ablation.src.data.local_descriptors import *
from paper_experiments.ablation.src.models.ablation import VARIANT_SPECS

MANIFEST_FILENAME = "manifest.csv"
SESSION_MANIFEST_FILENAME = "session_manifest.csv"
CACHE_VERSION = "lrdg_ablation_g_a_r_ga_gr_v1"
META_COLS = ["sample_uid", "source_path", "domain", "raw_class", "pollution_type", "severity", "session_id", "frame_id", "session_uid"]


def _signature(cfg):
    return {
        "cache_version": CACHE_VERSION,
        "voxel_size": cfg.voxel_size,
        "context_radius_voxels": cfg.context_radius_voxels,
        "intensity_scale": cfg.intensity_scale,
        "relative_std_eps": cfg.relative_std_eps,
        "geometry_schema": GEOMETRY_SCHEMA,
        "absolute_degradation_schema": ABSOLUTE_DEGRADATION_SCHEMA,
        "relative_degradation_schema": RELATIVE_DEGRADATION_SCHEMA,
        "pooling": "mean_only",
        "variants": list(MODEL.variants),
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
                "Cache config changed. Use a new workspace_root (recommended) or set DATA.overwrite_cache=True once."
            )

    feat.mkdir(parents=True, exist_ok=True)
    sig.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    rows = []
    for row in tqdm(raw_dataframe.to_dict("records"), desc="LRDG five-variant ablation preprocess"):
        p = feat / (row["sample_uid"].replace("|", "__") + ".npz")
        if p.exists() and not data_config.overwrite_cache:
            with np.load(p, allow_pickle=False) as z:
                n = int(z["geometry"].shape[0])
                pc = int(z["point_count"])
        else:
            pts = load_bin(row["source_path"])
            d = extract_local_descriptors(
                pts,
                data_config.voxel_size,
                data_config.context_radius_voxels,
                data_config.intensity_scale,
                data_config.relative_std_eps,
            )
            n = d["geometry"].shape[0]
            pc = pts.shape[0]
            np.savez_compressed(
                p,
                geometry=d["geometry"],
                absolute_degradation=d["absolute_degradation"],
                relative_degradation=d["relative_degradation"],
                point_count=np.asarray(pc, np.int64),
            )
        rows.append({**row, "cache_path": str(p.resolve()), "point_count": pc, "instance_count": n})
    return pd.DataFrame(rows)


class AblationDataset(Dataset):
    def __init__(self, dataframe, stats, variant):
        self.df = dataframe.reset_index(drop=True)
        self.stats = stats
        self.variant = variant.upper()
        if self.variant not in VARIANT_SPECS:
            raise ValueError(f"Unsupported variant: {self.variant}")
        self.spec = VARIANT_SPECS[self.variant]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with np.load(r.cache_path, allow_pickle=False) as z:
            if self.spec["geometry"]:
                g = z["geometry"].astype(np.float32)
            else:
                # Keep instance count aligned with degradation without providing geometry information.
                n = z["absolute_degradation"].shape[0]
                g = np.empty((n, 0), np.float32)

            kind = self.spec["degradation"]
            if kind is None:
                d = np.empty((len(g), 0), np.float32)
            elif kind == "absolute":
                d = z["absolute_degradation"].astype(np.float32)
            elif kind == "relative":
                d = z["relative_degradation"].astype(np.float32)
            else:
                raise RuntimeError(kind)

        if self.stats is not None:
            g = self.stats.normalize_geometry(g)
            d = self.stats.normalize_degradation(d)
        meta = {k: str(r[k]) for k in META_COLS}
        return torch.from_numpy(g), torch.from_numpy(d), torch.tensor(int(r.label), dtype=torch.long), meta


def collate_bags(batch):
    gs, ds, ys, ms = zip(*batch)
    counts = [x.shape[0] for x in gs]
    g = torch.cat(gs, 0)
    d = torch.cat(ds, 0)
    y = torch.stack(ys)
    bi = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)], 0)
    meta = {k: [m[k] for m in ms] for k in ms[0]}
    meta["instance_count"] = counts
    return g, d, bi, y, meta


def make_loader(df, batch_size, num_workers, shuffle, stats, variant):
    return DataLoader(
        AblationDataset(df, stats, variant),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )
