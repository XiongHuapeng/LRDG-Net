"""Local descriptors for the LRDG-Net ablation study.

The cache contains all descriptors needed by the five natural ablation variants:
- 12D centered covariance geometry;
- 3D absolute degradation;
- 4D local-relative degradation used by the final LRDG-Net.
"""
from itertools import product
from typing import Dict, Tuple
import numpy as np

GEOMETRY_SCHEMA = [
    "fine_cov_xx", "fine_cov_xy", "fine_cov_xz", "fine_cov_yy", "fine_cov_yz", "fine_cov_zz",
    "context_cov_xx", "context_cov_xy", "context_cov_xz", "context_cov_yy", "context_cov_yz", "context_cov_zz",
]
ABSOLUTE_DEGRADATION_SCHEMA = [
    "mean_intensity_div255", "std_intensity_div255", "log1p_local_point_count"
]
RELATIVE_DEGRADATION_SCHEMA = [
    "delta_mean_intensity_div255",
    "log_std_ratio",
    "local_std_intensity_div255",
    "relative_log_density",
]


def load_bin(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    if a.size == 0 or a.size % 4 != 0:
        raise ValueError(f"Invalid XYZI .bin: {path}, values={a.size}")
    x = a.reshape(-1, 4)
    if not np.isfinite(x).all():
        raise ValueError(f"NaN/Inf found: {path}")
    return x


def _stats(points: np.ndarray, voxel_size: float):
    xyz = points[:, :3].astype(np.float64, copy=False)
    intensity = points[:, 3].astype(np.float64, copy=False)
    idx = np.floor(xyz / voxel_size).astype(np.int32)
    vox, inv = np.unique(idx, axis=0, return_inverse=True)
    m = len(vox)
    count = np.bincount(inv, minlength=m).astype(np.float64)
    sum_xyz = np.stack([np.bincount(inv, weights=xyz[:, k], minlength=m) for k in range(3)], axis=1)
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    sum_outer = np.stack([
        np.bincount(inv, weights=xyz[:, a] * xyz[:, b], minlength=m) for a, b in pairs
    ], axis=1)
    sum_i = np.bincount(inv, weights=intensity, minlength=m)
    sum_i2 = np.bincount(inv, weights=intensity * intensity, minlength=m)
    return vox, count, sum_xyz, sum_outer, sum_i, sum_i2


def _context_sum(vox: np.ndarray, arrays, radius: int):
    u = vox.astype(np.int64, copy=False)
    margin = radius + 1
    min_v = u.min(axis=0) - margin
    max_v = u.max(axis=0) + margin
    dims = max_v - min_v + 1
    sy, sz = int(dims[1]), int(dims[2])
    keys = ((u[:, 0] - min_v[0]) * sy + (u[:, 1] - min_v[1])) * sz + (u[:, 2] - min_v[2])
    outputs = [np.zeros_like(a, dtype=np.float64) for a in arrays]
    m = len(u)
    for dx, dy, dz in product(range(-radius, radius + 1), repeat=3):
        q = u + np.array([dx, dy, dz], dtype=np.int64)
        qkeys = ((q[:, 0] - min_v[0]) * sy + (q[:, 1] - min_v[1])) * sz + (q[:, 2] - min_v[2])
        pos = np.searchsorted(keys, qkeys)
        mask = pos < m
        if not np.any(mask):
            continue
        t = np.nonzero(mask)[0]
        s = pos[mask]
        matched = keys[s] == qkeys[mask]
        if not np.any(matched):
            continue
        t, s = t[matched], s[matched]
        for out, arr in zip(outputs, arrays):
            out[t] += arr[s]
    return outputs


def _cov(count, sum_xyz, sum_outer):
    n = np.maximum(count[:, None], 1.0)
    mean = sum_xyz / n
    exx = sum_outer / n
    c = np.empty_like(exx)
    c[:, 0] = exx[:, 0] - mean[:, 0] * mean[:, 0]
    c[:, 1] = exx[:, 1] - mean[:, 0] * mean[:, 1]
    c[:, 2] = exx[:, 2] - mean[:, 0] * mean[:, 2]
    c[:, 3] = exx[:, 3] - mean[:, 1] * mean[:, 1]
    c[:, 4] = exx[:, 4] - mean[:, 1] * mean[:, 2]
    c[:, 5] = exx[:, 5] - mean[:, 2] * mean[:, 2]
    c[:, 0] = np.maximum(c[:, 0], 0.0)
    c[:, 3] = np.maximum(c[:, 3], 0.0)
    c[:, 5] = np.maximum(c[:, 5], 0.0)
    return c


def _mean_std(count, s, ss) -> Tuple[np.ndarray, np.ndarray]:
    n = np.maximum(count, 1.0)
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


def extract_local_descriptors(points, voxel_size, context_radius_voxels, intensity_scale, relative_std_eps) -> Dict[str, np.ndarray]:
    if context_radius_voxels < 1:
        raise ValueError("context_radius_voxels must be >= 1")

    vox, count, sx, so, si, si2 = _stats(points, voxel_size)
    cc, csx, cso, csi, csi2 = _context_sum(vox, [count, sx, so, si, si2], context_radius_voxels)

    fine_cov = _cov(count, sx, so)
    context_cov = _cov(cc, csx, cso)
    scale = max(voxel_size * voxel_size, 1e-12)
    geometry = np.concatenate([fine_cov / scale, context_cov / scale], axis=1)

    mean_i, std_i = _mean_std(count, si, si2)
    cmean_i, cstd_i = _mean_std(cc, csi, csi2)
    mean_i_n = mean_i / intensity_scale
    std_i_n = std_i / intensity_scale
    cmean_i_n = cmean_i / intensity_scale
    cstd_i_n = cstd_i / intensity_scale

    absolute = np.stack([
        mean_i_n,
        std_i_n,
        np.log1p(count),
    ], axis=1)

    context_side = 2 * context_radius_voxels + 1
    context_volume = float(context_side ** 3)
    context_mean_voxel_count = cc / context_volume
    relative = np.stack([
        mean_i_n - cmean_i_n,
        np.log((std_i_n + relative_std_eps) / (cstd_i_n + relative_std_eps)),
        std_i_n,
        np.log1p(count) - np.log1p(context_mean_voxel_count),
    ], axis=1)

    out = {
        "geometry": geometry.astype(np.float32),
        "absolute_degradation": absolute.astype(np.float32),
        "relative_degradation": relative.astype(np.float32),
    }
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise FloatingPointError(f"Non-finite descriptor: {k}")
    return out
