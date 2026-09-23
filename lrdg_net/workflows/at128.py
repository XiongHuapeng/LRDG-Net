"""AT128 preparation workflow used by the experiment entry points."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pandas as pd

from config import AT128, DATA, PATHS
from lrdg_net.data.at128_pcap import (
    discover_pcaps,
    export_recording,
    load_at128p_calibration,
    validate_c3_bins,
    write_manifest,
)
from lrdg_net.data.dataset import MANIFEST_FILENAME, SESSION_MANIFEST_FILENAME, build_feature_cache


def _to_lrdg_manifest(conversion_rows) -> pd.DataFrame:
    """Map converted AT128 frames to the same cached-feature contract used by LRDG-Net."""
    rows = []
    for r in conversion_rows:
        if r["label"] == "":
            raise ValueError(
                f"Cannot infer label from {r['recording_id']}. Expected '<distance>m_<level>.pcap' with level 0/1/2/3."
            )
        level = int(r["condition_level"])
        label = int(r["label"])
        frame_id = f"{int(r['frame_index']):04d}"
        recording_id = str(r["recording_id"])
        raw_class = "clean" if label == 0 else "mixed_contamination"
        severity = "clean" if label == 0 else f"level{level}"
        session_uid = f"AT128|{recording_id}"
        sample_uid = f"{session_uid}|frame_{frame_id}"
        rows.append({
            "source_path": str(r["bin_path"]),
            "sample_id": f"{recording_id}_frame_{frame_id}",
            "label": label,
            "domain": "AT128",
            "raw_class": raw_class,
            "pollution_type": raw_class,
            "severity": severity,
            "session_id": recording_id,
            "frame_id": frame_id,
            "session_uid": session_uid,
            "sample_uid": sample_uid,
            # AT128-only metadata; never classifier inputs.
            "recording_id": recording_id,
            "distance_m": float(r["distance_m"]),
            "condition_level": level,
            "source_pcap": str(r["source_pcap"]),
            "mirror_id": int(r["mirror_id"]),
            "packet_count": int(r["packet_count"]),
            "capture_start_s": float(r["capture_start_s"]),
            "capture_end_s": float(r["capture_end_s"]),
            "duration_s": float(r["duration_s"]),
            "return_policy": str(r["return_policy"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["distance_m", "condition_level", "recording_id", "frame_id"]
    ).reset_index(drop=True)


def prepare_at128() -> Path:
    pcap_root = Path(PATHS.at128_pcap_root)
    calib_path = Path(PATHS.at128_angle_calibration)
    work_root = Path(PATHS.at128_work_root)
    bin_root = work_root / "bin"
    cache_root = work_root / "cache"
    conversion_manifest = bin_root / "at128_conversion_manifest.csv"

    if not pcap_root.exists():
        raise FileNotFoundError(f"AT128 PCAP root not found / PCAP路径不存在: {pcap_root}")
    if not calib_path.exists():
        raise FileNotFoundError(f"AT128 calibration not found / 标定文件不存在: {calib_path}")

    if AT128.overwrite_preparation and work_root.exists():
        shutil.rmtree(work_root)

    bin_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    pcaps = discover_pcaps(pcap_root)
    if not pcaps:
        raise RuntimeError(f"No .pcap found under {pcap_root}")
    calib = load_at128p_calibration(calib_path)

    print("\n" + "=" * 100)
    print("AT128 preparation / AT128 数据准备")
    print(f"PCAP recordings / PCAP录制数: {len(pcaps)}")
    print(f"Return policy / 回波策略: {AT128.return_policy}")
    print("Label mapping / 标签映射: 0 -> clean, 1/2/3 -> contaminated")
    print("Condition levels 1/2/3 are metadata only; no severity classification is created.")
    print("=" * 100)

    if conversion_manifest.exists() and not AT128.overwrite_preparation:
        print(f"Reuse converted BIN manifest / 复用已转换BIN: {conversion_manifest}")
        conversion_rows = pd.read_csv(conversion_manifest).to_dict("records")
        missing = [r["bin_path"] for r in conversion_rows if not Path(r["bin_path"]).exists()]
        if missing:
            raise FileNotFoundError(
                "Conversion manifest references missing BIN files. Set AT128.overwrite_preparation=True. "
                f"First missing: {missing[0]}"
            )
    else:
        conversion_rows = []
        for index, pcap in enumerate(pcaps, start=1):
            print("\n" + "-" * 92)
            print(f"[{index}/{len(pcaps)}] {pcap.name}")
            conversion_rows.extend(export_recording(
                pcap_path=pcap,
                output_root=bin_root,
                calib=calib,
                return_policy=AT128.return_policy,
                min_complete_ratio=AT128.min_complete_ratio,
                keep_partial=False,
                udp_port=AT128.udp_port,
            ))
        write_manifest(conversion_rows, conversion_manifest)
        validate_c3_bins(conversion_rows)

    raw = _to_lrdg_manifest(conversion_rows)
    raw_manifest_path = work_root / "at128_raw_manifest.csv"
    raw.to_csv(raw_manifest_path, index=False, encoding="utf-8-sig")

    # Use the same paper-aligned descriptor extraction as for LIDAROC.
    # No AT128 normalization is fitted here.
    data_cfg = replace(DATA, overwrite_cache=AT128.overwrite_preparation)
    manifest_path = cache_root / MANIFEST_FILENAME
    if manifest_path.exists() and not AT128.overwrite_preparation:
        print(f"Reuse AT128 descriptor cache / 复用AT128描述子缓存: {manifest_path}")
        manifest = pd.read_csv(manifest_path, dtype={"session_id": str, "frame_id": str})
    else:
        print("\nBuilding AT128 LRDG descriptors / 构建AT128 LRDG描述子 ...")
        manifest = build_feature_cache(raw, str(cache_root), data_cfg)
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        session_cols = [
            "domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid",
            "recording_id", "distance_m", "condition_level", "source_pcap",
        ]
        manifest[session_cols].drop_duplicates().to_csv(
            cache_root / SESSION_MANIFEST_FILENAME, index=False, encoding="utf-8-sig"
        )

    print("\n" + "=" * 100)
    print("AT128 ready / AT128 数据准备完成")
    print(f"Independent PCAP recordings / 独立录制: {manifest['session_uid'].nunique()}")
    print(f"Complete frames / 完整帧: {len(manifest)}")
    print(manifest.groupby(["raw_class", "condition_level"]).size())
    print(f"Descriptor manifest / 描述子清单: {manifest_path}")
    print("AT128 normalization fitted: NO / 未使用AT128拟合归一化")
    print("=" * 100)
    return manifest_path
