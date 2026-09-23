"""
LRDG-Net 正式局部描述子提取。
Final local descriptor extraction for LRDG-Net.

核心约束 / Core constraints:
- absolute XYZ 不作为分类特征；仅用于 voxelization 和 centered covariance。
  Absolute XYZ is never a classifier feature; it is used only for voxelization and centered covariance.
- geometry = fine voxel covariance + 3D context covariance (12D).
- degradation = fine-to-context relative intensity/density descriptor (4D).
- The main descriptor extractor returns only the geometry and local-relative degradation inputs used by LRDG-Net.
  Absolute-degradation features are implemented only in the dedicated representation-ablation package.
"""
from itertools import product
from typing import Dict, Tuple
import numpy as np


GEOMETRY_SCHEMA = [
    "fine_cov_xx", "fine_cov_xy", "fine_cov_xz", "fine_cov_yy", "fine_cov_yz", "fine_cov_zz",
    "context_cov_xx", "context_cov_xy", "context_cov_xz", "context_cov_yy", "context_cov_yz", "context_cov_zz",
]
RELATIVE_DEGRADATION_SCHEMA = [
    "delta_mean_intensity_div255",
    "log_std_ratio",
    "local_std_intensity_div255",
    "relative_log_density",
]


def load_bin(path: str) -> np.ndarray:
    array = np.fromfile(path, dtype=np.float32)
    if array.size == 0 or array.size % 4 != 0:
        raise ValueError(f"Invalid XYZI .bin / 非法 XYZI 文件: {path}, values={array.size}")
    points = array.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"NaN/Inf found / 存在 NaN/Inf: {path}")
    return points


def _voxel_sufficient_statistics(points: np.ndarray, voxel_size: float):
    xyz = points[:, :3].astype(np.float64, copy=False)
    intensity = points[:, 3].astype(np.float64, copy=False)
    voxel_index = np.floor(xyz / voxel_size).astype(np.int32)

    unique_voxels, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    m = len(unique_voxels)
    count = np.bincount(inverse, minlength=m).astype(np.float64)

    sum_xyz = np.stack([
        np.bincount(inverse, weights=xyz[:, k], minlength=m) for k in range(3)
    ], axis=1)
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    sum_outer = np.stack([
        np.bincount(inverse, weights=xyz[:, a] * xyz[:, b], minlength=m)
        for a, b in pairs
    ], axis=1)
    sum_intensity = np.bincount(inverse, weights=intensity, minlength=m)
    sum_intensity_sq = np.bincount(inverse, weights=intensity * intensity, minlength=m)
    return unique_voxels, count, sum_xyz, sum_outer, sum_intensity, sum_intensity_sq


def _context_sum_by_sparse_offsets(voxel_index: np.ndarray, arrays, radius: int):
    """Vectorized sparse box-neighborhood summation over occupied voxels."""
    u = voxel_index.astype(np.int64, copy=False)
    margin = radius + 1
    min_v = u.min(axis=0) - margin
    max_v = u.max(axis=0) + margin
    dims = max_v - min_v + 1
    stride_y = int(dims[1])
    stride_z = int(dims[2])

    keys = ((u[:, 0] - min_v[0]) * stride_y + (u[:, 1] - min_v[1])) * stride_z + (u[:, 2] - min_v[2])
    if np.any(keys[1:] < keys[:-1]):
        raise RuntimeError("Internal voxel key order error / voxel key 排序异常。")

    outputs = [np.zeros_like(a, dtype=np.float64) for a in arrays]
    m = len(u)
    for dx, dy, dz in product(range(-radius, radius + 1), repeat=3):
        q = u + np.array([dx, dy, dz], dtype=np.int64)
        qkeys = ((q[:, 0] - min_v[0]) * stride_y + (q[:, 1] - min_v[1])) * stride_z + (q[:, 2] - min_v[2])
        pos = np.searchsorted(keys, qkeys)
        in_range = pos < m
        if not np.any(in_range):
            continue
        target_idx = np.nonzero(in_range)[0]
        source_idx = pos[in_range]
        matched = keys[source_idx] == qkeys[in_range]
        if not np.any(matched):
            continue
        target_idx = target_idx[matched]
        source_idx = source_idx[matched]
        for out, arr in zip(outputs, arrays):
            out[target_idx] += arr[source_idx]
    return outputs


def _covariance_from_sufficient(count, sum_xyz, sum_outer) -> np.ndarray:
    n = np.maximum(count[:, None], 1.0)
    mean = sum_xyz / n
    exx = sum_outer / n
    cov = np.empty_like(exx)
    cov[:, 0] = exx[:, 0] - mean[:, 0] * mean[:, 0]
    cov[:, 1] = exx[:, 1] - mean[:, 0] * mean[:, 1]
    cov[:, 2] = exx[:, 2] - mean[:, 0] * mean[:, 2]
    cov[:, 3] = exx[:, 3] - mean[:, 1] * mean[:, 1]
    cov[:, 4] = exx[:, 4] - mean[:, 1] * mean[:, 2]
    cov[:, 5] = exx[:, 5] - mean[:, 2] * mean[:, 2]
    cov[:, 0] = np.maximum(cov[:, 0], 0.0)
    cov[:, 3] = np.maximum(cov[:, 3], 0.0)
    cov[:, 5] = np.maximum(cov[:, 5], 0.0)
    return cov


def _mean_std(count, sum_value, sum_sq) -> Tuple[np.ndarray, np.ndarray]:
    n = np.maximum(count, 1.0)
    mean = sum_value / n
    var = np.maximum(sum_sq / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


def extract_local_descriptors(
    points: np.ndarray,
    voxel_size: float,
    context_radius_voxels: int,
    intensity_scale: float,
    relative_std_eps: float,
) -> Dict[str, np.ndarray]:
    """Extract the frozen 12D geometry + 4D relative-degradation descriptors."""
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected Nx4 XYZI / 需要 Nx4 XYZI, got {points.shape}")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive / voxel_size 必须大于 0")
    if context_radius_voxels < 1:
        raise ValueError("context_radius_voxels must be >=1 / context radius 必须 >=1")

    vox, count, sum_xyz, sum_outer, sum_i, sum_i2 = _voxel_sufficient_statistics(points, voxel_size)
    context_count, context_xyz, context_outer, context_i, context_i2 = _context_sum_by_sparse_offsets(
        vox, [count, sum_xyz, sum_outer, sum_i, sum_i2], context_radius_voxels
    )

    fine_cov = _covariance_from_sufficient(count, sum_xyz, sum_outer)
    context_cov = _covariance_from_sufficient(context_count, context_xyz, context_outer)

    # Manuscript Eq. (5): covariance normalized by voxel_size^2.
    geom_scale = max(voxel_size * voxel_size, 1e-12)
    geometry = np.concatenate([fine_cov / geom_scale, context_cov / geom_scale], axis=1)

    mean_i, std_i = _mean_std(count, sum_i, sum_i2)
    context_mean_i, context_std_i = _mean_std(context_count, context_i, context_i2)
    mean_i_n = mean_i / intensity_scale
    std_i_n = std_i / intensity_scale
    context_mean_i_n = context_mean_i / intensity_scale
    context_std_i_n = context_std_i / intensity_scale

    context_side = 2 * context_radius_voxels + 1
    context_volume = float(context_side ** 3)
    context_mean_voxel_count = context_count / context_volume

    relative_degradation = np.stack([
        mean_i_n - context_mean_i_n,
        np.log((std_i_n + relative_std_eps) / (context_std_i_n + relative_std_eps)),
        std_i_n,
        np.log1p(count) - np.log1p(context_mean_voxel_count),
    ], axis=1)

    outputs = {
        "geometry": geometry.astype(np.float32),
        "relative_degradation": relative_degradation.astype(np.float32),
        "fine_count": count.astype(np.float32),
        "context_count": context_count.astype(np.float32),
    }
    for name, array in outputs.items():
        if not np.isfinite(array).all():
            raise FloatingPointError(f"Non-finite descriptor / 描述子存在非有限值: {name}")
    return outputs
