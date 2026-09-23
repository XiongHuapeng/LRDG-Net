from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
from paper_experiments.baseline.config import DATA, PATHS


def read_bin(path):
  x = np.fromfile(path, dtype=np.float32)
  if x.size % 4:
    raise ValueError(f"Invalid XYZI bin: {path}")
  x = x.reshape(-1, 4)
  return x[np.isfinite(x).all(axis=1)]


def stable_seed(text: str) -> int:
  return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def sample_point_tensor(points, sample_uid):
  """Shared deterministic point input for all raw-point baselines."""
  n = DATA.num_points
  if len(points) == 0:
    raise ValueError("Empty point cloud")
  rng = np.random.default_rng(stable_seed(sample_uid))
  if len(points) >= n:
    idx = rng.choice(len(points), n, replace=False)
  else:
    idx = rng.choice(len(points), n, replace=True)
  p = points[idx].astype(np.float32)
  # Raw-point baselines use XYZ in the original metric scale (meters).
  # Only LiDAR intensity is normalized to [approximately] 0-1 by the fixed 255 scale.
  p[:, 3] /= float(DATA.intensity_scale)
  return p


def global_statistics(points):
  """1D global aggregation adapted to available XYZI.

  Each normalized channel contributes:
   min, mean, std, q25, q50, q75, max -> 4*7 = 28 values
  plus one fixed-scale log point-count statistic -> 29D.

  No target-domain statistics are fitted.
  """
  if len(points) == 0:
    raise ValueError("Empty point cloud")
  p = points.astype(np.float64, copy=True)
  p[:, :3] /= float(DATA.xyz_scale_m)
  p[:, 3] /= float(DATA.intensity_scale)
  feats = []
  for c in range(4):
    v = p[:, c]
    q25, q50, q75 = np.quantile(v, [0.25, 0.50, 0.75])
    feats.extend([
      float(np.min(v)), float(np.mean(v)), float(np.std(v)),
      float(q25), float(q50), float(q75), float(np.max(v)),
    ])
  denom = math.log1p(float(DATA.global_count_reference))
  count_feature = math.log1p(float(len(points))) / max(denom, 1e-12)
  feats.append(float(count_feature))
  return np.asarray(feats, dtype=np.float32)


def rangeview_shape():
  return int(DATA.rangeview_h), int(DATA.rangeview_w)


def build_sparse_rangeview(points):
  """RangeNet-style spherical projection.

  Each occupied range-image pixel stores five channels:
   [range/80, x/80, y/80, z/80, intensity/255]

  The vertical FoV is fixed from the RS-Ruby sensor specification. The
  horizontal axis covers 360 degrees as in RangeNet++; LIDAROC's already
  cropped/front-facing points simply occupy a subset of those columns.

  If multiple 3D points project to the same pixel, the nearest point is kept,
  matching the visible-surface logic of a range image.
  """
  h, w = rangeview_shape()
  xyz = points[:, :3].astype(np.float64, copy=False)
  depth = np.linalg.norm(xyz, axis=1)
  valid = np.isfinite(depth) & (depth > float(DATA.rangeview_min_range_m))
  if not np.any(valid):
    return np.zeros((0,), np.int32), np.zeros((0, 5), np.float16)

  p = points[valid]
  xyz = xyz[valid]
  depth = depth[valid]
  yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
  pitch = np.arcsin(np.clip(xyz[:, 2] / depth, -1.0, 1.0))

  fov_up = np.deg2rad(float(DATA.rangeview_fov_up_deg))
  fov_down = np.deg2rad(float(DATA.rangeview_fov_down_deg))
  vm = (pitch >= fov_down) & (pitch <= fov_up)
  if not np.any(vm):
    return np.zeros((0,), np.int32), np.zeros((0, 5), np.float16)

  p = p[vm]
  xyz = xyz[vm]
  depth = depth[vm]
  yaw = yaw[vm]
  pitch = pitch[vm]

  proj_x = 0.5 * (1.0 - yaw / np.pi)
  proj_y = (fov_up - pitch) / max(fov_up - fov_down, 1e-12)
  px = np.floor(proj_x * w).astype(np.int32)
  py = np.floor(proj_y * h).astype(np.int32)
  px = np.clip(px, 0, w - 1)
  py = np.clip(py, 0, h - 1)
  flat = py * w + px

  # Farthest first; later writes overwrite earlier ones, so nearest survives.
  order = np.argsort(depth)[::-1]
  dense = np.zeros((h * w, 5), dtype=np.float32)
  occupied = np.zeros((h * w,), dtype=bool)
  scale = float(DATA.xyz_scale_m)
  vals = np.column_stack([
    depth / scale,
    xyz[:, 0] / scale,
    xyz[:, 1] / scale,
    xyz[:, 2] / scale,
    p[:, 3] / float(DATA.intensity_scale),
  ]).astype(np.float32)
  dense[flat[order]] = vals[order]
  occupied[flat] = True
  occ = np.flatnonzero(occupied).astype(np.int32)
  return occ, dense[occ].astype(np.float16)


def crop_autogran(points):
  x0, x1 = DATA.roi_x
  y0, y1 = DATA.roi_y
  m = (
    (points[:, 0] >= x0) & (points[:, 0] <= x1) &
    (points[:, 1] >= y0) & (points[:, 1] <= y1)
  )
  return points[m]


def voxel_average(points):
  vox = np.floor(points[:, :3] / DATA.autogran_voxel_size).astype(np.int32)
  uniq, inv, cnt = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
  sums = np.zeros((len(uniq), 4), np.float64)
  for c in range(4):
    np.add.at(sums[:, c], inv, points[:, c])
  return (sums / cnt[:, None]).astype(np.float32)


def build_autogran_data(points):
  try:
    from torch_geometric.data import Data
    from torch_geometric.utils import coalesce
  except Exception as e:
    raise RuntimeError("AutoGrAN preprocessing requires torch-geometric.") from e
  p = crop_autogran(points)
  if len(p) < 2:
    raise ValueError("Too few points after AutoGrAN ROI")
  nodes = voxel_average(p)
  if len(nodes) < 2:
    raise ValueError("Too few voxels")
  x = torch.from_numpy(nodes)
  pos = x[:, :3]
  edges = []
  for d in range(3):
    order = torch.argsort(pos[:, d])
    a, b = order[:-1], order[1:]
    edges.append(torch.stack([a, b]))
  edge_index = coalesce(torch.cat(edges, dim=1), num_nodes=x.size(0))
  return Data(x=x, edge_index=edge_index)


def build_cache(raw_df: pd.DataFrame):
  point_root = Path(PATHS.cache_root) / "points4096_rawxyz"
  stat_root = Path(PATHS.cache_root) / "globalstats29"
  range_root = Path(PATHS.cache_root) / f"rangeview_{DATA.rangeview_h}x{DATA.rangeview_w}"
  graph_root = Path(PATHS.cache_root) / "autogran"
  for root in (point_root, stat_root, range_root, graph_root):
    root.mkdir(parents=True, exist_ok=True)

  rows = []
  total = len(raw_df)
  for i, r in raw_df.iterrows():
    sid = str(r.sample_uid)
    key = hashlib.sha1(sid.encode()).hexdigest()[:20]
    pp = point_root / f"{key}.npy"
    sp = stat_root / f"{key}.npy"
    rp = range_root / f"{key}.npz"
    gp = graph_root / f"{key}.pt"
    pts = read_bin(r.source_path)

    if DATA.overwrite_cache or not pp.exists():
      np.save(pp, sample_point_tensor(pts, sid))
    if DATA.overwrite_cache or not sp.exists():
      np.save(sp, global_statistics(pts))
    if DATA.overwrite_cache or not rp.exists():
      flat_idx, values = build_sparse_rangeview(pts)
      np.savez_compressed(rp, flat_idx=flat_idx, values=values)
    if DATA.overwrite_cache or not gp.exists():
      torch.save(build_autogran_data(pts), gp)

    row = r.to_dict()
    row.update(
      point_cache=str(pp.resolve()),
      globalstats_cache=str(sp.resolve()),
      rangeview_cache=str(rp.resolve()),
      graph_cache=str(gp.resolve()),
      point_count=len(pts),
    )
    rows.append(row)
    if (i + 1) % 250 == 0 or i + 1 == total:
      print(f"Cached {i + 1}/{total}")

  out = pd.DataFrame(rows)
  Path(PATHS.cache_root).mkdir(parents=True, exist_ok=True)
  out.to_csv(Path(PATHS.cache_root) / "manifest.csv", index=False, encoding="utf-8-sig")
  h, w = rangeview_shape()
  cfg = {
    "num_points": DATA.num_points,
    "raw_point_xyz_mode": "raw_meters_no_global_scaling",
    "raw_point_intensity_scale": DATA.intensity_scale,
    "xyz_scale_m": DATA.xyz_scale_m,
    "intensity_scale": DATA.intensity_scale,
    "globalstats_dim": 29,
    "global_count_reference": DATA.global_count_reference,
    "rangeview_hw": [h, w],
    "rangeview_fov_up_deg": DATA.rangeview_fov_up_deg,
    "rangeview_fov_down_deg": DATA.rangeview_fov_down_deg,
    "rangeview_channels": ["range_over_80", "x_over_80", "y_over_80", "z_over_80", "intensity_over_255"],
    "autogran_voxel_size": DATA.autogran_voxel_size,
    "roi_x": DATA.roi_x,
    "roi_y": DATA.roi_y,
  }
  (Path(PATHS.cache_root) / "preprocess_config.json").write_text(
    json.dumps(cfg, indent=2), encoding="utf-8"
  )
  return out
