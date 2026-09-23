"""
LRDG-Net 正式缓存、Dataset 与 variable-instance bag collate。
Final LRDG-Net cache, Dataset, and variable-instance bag collation.
"""
from pathlib import Path
import json
import shutil
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DATA
from lrdg_net.data.local_descriptors import (
    GEOMETRY_SCHEMA,
    RELATIVE_DEGRADATION_SCHEMA,
    extract_local_descriptors,
    load_bin,
)
from lrdg_net.data.normalization import NormalizationStats

MANIFEST_FILENAME = "manifest.csv"
SESSION_MANIFEST_FILENAME = "session_manifest.csv"
CACHE_VERSION = "lidaroc_lrdg_net_final_v1"
META_COLS = [
    "sample_uid", "source_path", "domain", "raw_class", "pollution_type",
    "severity", "session_id", "frame_id", "session_uid",
]


def _cache_signature(data_config) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "voxel_size": data_config.voxel_size,
        "context_radius_voxels": data_config.context_radius_voxels,
        "intensity_scale": data_config.intensity_scale,
        "relative_std_eps": data_config.relative_std_eps,
        "geometry_schema": GEOMETRY_SCHEMA,
        "relative_degradation_schema": RELATIVE_DEGRADATION_SCHEMA,
        "absolute_xyz_classifier_input": False,
        "absolute_degradation_classifier_input": False,
        "pooling": "mean",
    }


def _prepare_cache(cache_root: str, data_config) -> Path:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    sig_path = root / "preprocess_config.json"
    current = _cache_signature(data_config)

    if data_config.overwrite_cache:
        if (root / "features").exists():
            shutil.rmtree(root / "features")
        for filename in (MANIFEST_FILENAME, SESSION_MANIFEST_FILENAME, "preprocess_config.json"):
            p = root / filename
            if p.exists():
                p.unlink()
    elif sig_path.exists():
        old = json.loads(sig_path.read_text(encoding="utf-8"))
        if old != current:
            raise RuntimeError(
                "LRDG-Net cache configuration changed. Set DATA.overwrite_cache=True and rerun preprocess.py. / "
                "LRDG-Net 缓存配置已变化，请将 DATA.overwrite_cache=True 后重新预处理。"
            )

    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return feature_dir


def build_feature_cache(raw_dataframe: pd.DataFrame, cache_root: str, data_config=DATA) -> pd.DataFrame:
    feature_dir = _prepare_cache(cache_root, data_config)
    rows = []
    for row in tqdm(raw_dataframe.to_dict("records"), desc="LRDG-Net preprocess"):
        safe_uid = row["sample_uid"].replace("|", "__")
        cache_path = feature_dir / f"{safe_uid}.npz"

        if cache_path.exists() and not data_config.overwrite_cache:
            with np.load(cache_path, allow_pickle=False) as npz:
                n_voxels = int(npz["geometry"].shape[0])
                point_count = int(npz["point_count"])
        else:
            points = load_bin(row["source_path"])
            descriptors = extract_local_descriptors(
                points=points,
                voxel_size=data_config.voxel_size,
                context_radius_voxels=data_config.context_radius_voxels,
                intensity_scale=data_config.intensity_scale,
                relative_std_eps=data_config.relative_std_eps,
            )
            n_voxels = int(descriptors["geometry"].shape[0])
            point_count = int(points.shape[0])
            np.savez_compressed(
                cache_path,
                geometry=descriptors["geometry"],
                relative_degradation=descriptors["relative_degradation"],
                fine_count=descriptors["fine_count"],
                context_count=descriptors["context_count"],
                point_count=np.asarray(point_count, dtype=np.int64),
            )
        rows.append({**row, "cache_path": str(cache_path.resolve()), "point_count": point_count, "instance_count": n_voxels})
    return pd.DataFrame(rows)


class LRDGNetDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, normalization_stats: Optional[NormalizationStats]):
        self.dataframe = dataframe.reset_index(drop=True)
        self.normalization_stats = normalization_stats

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with np.load(row.cache_path, allow_pickle=False) as npz:
            geometry = npz["geometry"].astype(np.float32)
            degradation = npz["relative_degradation"].astype(np.float32)
        if self.normalization_stats is not None:
            geometry = self.normalization_stats.normalize_geometry(geometry)
            degradation = self.normalization_stats.normalize_degradation(degradation)
        meta = {key: str(row[key]) for key in META_COLS}
        return torch.from_numpy(geometry), torch.from_numpy(degradation), torch.tensor(int(row.label), dtype=torch.long), meta


def collate_bags(batch):
    geometries, degradations, labels, metas = zip(*batch)
    instance_counts = [x.shape[0] for x in geometries]
    if any(c <= 0 for c in instance_counts):
        raise RuntimeError("Empty frame after descriptor extraction / 描述子提取后出现空帧。")
    geometry = torch.cat(geometries, dim=0)
    degradation = torch.cat(degradations, dim=0)
    label = torch.stack(labels)
    bag_index = torch.cat([
        torch.full((count,), bag_id, dtype=torch.long)
        for bag_id, count in enumerate(instance_counts)
    ], dim=0)
    keys = metas[0].keys()
    meta = {key: [m[key] for m in metas] for key in keys}
    meta["instance_count"] = instance_counts
    return geometry, degradation, bag_index, label, meta


def make_loader(
    dataframe: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    normalization_stats: Optional[NormalizationStats],
):
    return DataLoader(
        LRDGNetDataset(dataframe, normalization_stats),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )
