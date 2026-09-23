"""
AT128P classic-PCAP decoder for the LRDG-Net research project.
AT128P 原始 PCAP -> 完整帧 XYZI 的内部解析模块。

Scientific/data contract:
- Uses the per-device AT128P angle-calibration ``.dat`` file.
- Exports float32 [x, y, z, reflectivity] frames compatible with the LRDG-Net descriptor code.
- Keeps one selected return per firing/channel by default (configured elsewhere as ``strongest``).
- Drops incomplete first/last frame segments in the standard workflow.
- Does not use target ROI CSV files, labels, normalization, or model predictions during decoding.

This is an internal module. Researchers normally run ``run_04_prepare_at128.py``.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Constants / 常量
# ----------------------------
AT128_SOP = b"\xEE\xFF"
EXPECTED_CHANNELS = 128
EXPECTED_BLOCKS = 2
EXPECTED_UDP_PAYLOAD = 1118
PRE_HEADER_BYTES = 6
UDP_HEADER_BYTES = 6
BODY_OFFSET = PRE_HEADER_BYTES + UDP_HEADER_BYTES  # 12
BLOCK_CHANNEL_BYTES = 128 * 4  # 512
BLOCK_BYTES = 2 + 1 + BLOCK_CHANNEL_BYTES  # azimuth(2) + fine azimuth(1) + channels(512) = 515
ANGLE_BINS = 180  # 0,2,...,358 deg

NAME_RE = re.compile(r"^(?P<distance>\d+(?:\.\d+)?)m_(?P<level>[0-3])$", re.IGNORECASE)


@dataclass
class AngleCalibration:
    protocol_major: int
    protocol_minor: int
    channels: int
    mirrors: int
    frame_number: int
    frame_config: Tuple[int, ...]
    resolution_deg: float
    start_deg: np.ndarray          # [M]
    end_deg: np.ndarray            # [M]
    azimuth_offset_deg: np.ndarray # [N]
    elevation_deg: np.ndarray      # [N]
    azimuth_adjust: np.ndarray     # int8 [N,180]
    elevation_adjust: np.ndarray   # int8 [N,180]


@dataclass
class PacketMeasurement:
    timestamp: float
    encoder_deg: float
    mirror_id: int
    xyz_i: np.ndarray  # float32 [K,4]


@dataclass
class FrameSegment:
    mirror_id: int
    packets: List[PacketMeasurement]

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def start_time(self) -> float:
        return self.packets[0].timestamp

    @property
    def end_time(self) -> float:
        return self.packets[-1].timestamp

    @property
    def points(self) -> np.ndarray:
        arrays = [p.xyz_i for p in self.packets if len(p.xyz_i)]
        if not arrays:
            return np.empty((0, 4), dtype=np.float32)
        return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


# ----------------------------
# Calibration / 角度标定
# ----------------------------
def load_at128p_calibration(path: Path) -> AngleCalibration:
    data = path.read_bytes()
    if len(data) < 64 or data[:2] != AT128_SOP:
        raise ValueError(f"Invalid AT128P angle calibration file: {path}")

    # AT128P .dat format is little-endian unless otherwise specified.
    pos = 2
    protocol_major = data[pos]
    protocol_minor = data[pos + 1]
    channels = data[pos + 2]
    mirrors = data[pos + 3]
    frame_number = data[pos + 4]
    pos += 5

    frame_config = tuple(int(v) for v in data[pos: pos + 8])
    pos += 8
    resolution = float(data[pos])
    pos += 1

    expected_size = 48 + 8 * mirrors + 368 * channels
    if len(data) != expected_size:
        raise ValueError(
            f"Unexpected calibration size: got {len(data)}, expected {expected_size} "
            f"for N={channels}, M={mirrors}."
        )

    start_raw = np.frombuffer(data, dtype="<u4", count=mirrors, offset=pos).copy(); pos += 4 * mirrors
    end_raw = np.frombuffer(data, dtype="<u4", count=mirrors, offset=pos).copy(); pos += 4 * mirrors
    az_offset_raw = np.frombuffer(data, dtype="<i4", count=channels, offset=pos).copy(); pos += 4 * channels
    elevation_raw = np.frombuffer(data, dtype="<i4", count=channels, offset=pos).copy(); pos += 4 * channels
    az_adjust = np.frombuffer(data, dtype=np.int8, count=channels * ANGLE_BINS, offset=pos).copy().reshape(channels, ANGLE_BINS)
    pos += channels * ANGLE_BINS
    el_adjust = np.frombuffer(data, dtype=np.int8, count=channels * ANGLE_BINS, offset=pos).copy().reshape(channels, ANGLE_BINS)
    pos += channels * ANGLE_BINS

    stored_hash = data[pos:pos + 32]
    calc_hash = hashlib.sha256(data[:pos]).digest()
    if stored_hash != calc_hash:
        raise ValueError("Angle calibration SHA-256 check failed; file may be corrupted.")

    unit = resolution / 25600.0
    return AngleCalibration(
        protocol_major=protocol_major,
        protocol_minor=protocol_minor,
        channels=channels,
        mirrors=mirrors,
        frame_number=frame_number,
        frame_config=frame_config,
        resolution_deg=resolution,
        start_deg=start_raw.astype(np.float64) * unit,
        end_deg=end_raw.astype(np.float64) * unit,
        azimuth_offset_deg=az_offset_raw.astype(np.float64) * unit,
        elevation_deg=elevation_raw.astype(np.float64) * unit,
        azimuth_adjust=az_adjust,
        elevation_adjust=el_adjust,
    )


def _angle_in_interval(angle: float, start: float, end: float) -> bool:
    """Circular half-open interval [start,end). / 环形角度区间判断。"""
    angle %= 360.0; start %= 360.0; end %= 360.0
    if start < end:
        return start <= angle < end
    return angle >= start or angle < end


def mirror_id_from_encoder(encoder_deg: float, calib: AngleCalibration) -> int:
    for m in range(calib.mirrors):
        if _angle_in_interval(encoder_deg, float(calib.start_deg[m]), float(calib.end_deg[m])):
            return m
    raise RuntimeError(f"Encoder angle {encoder_deg:.6f} deg does not belong to any mirror interval.")


def _interp_adjust(table: np.ndarray, encoder_deg: float, resolution_deg: float) -> np.ndarray:
    """Linear interpolation of channel adjustment table at 2-degree encoder bins."""
    a = encoder_deg % 360.0
    pos = a / 2.0
    left = int(math.floor(pos)) % ANGLE_BINS
    frac = pos - math.floor(pos)
    right = (left + 1) % ANGLE_BINS
    # Stored adjustment unit = Resolution * 0.01 degree.
    scale = resolution_deg * 0.01
    return ((1.0 - frac) * table[:, left] + frac * table[:, right]) * scale


def corrected_angles(encoder_deg: float, mirror_id: int, calib: AngleCalibration) -> Tuple[np.ndarray, np.ndarray]:
    """Return horizontal/elevation angles in degrees for all 128 channels."""
    # Circular encoder displacement from the mirror start.
    delta = (encoder_deg - float(calib.start_deg[mirror_id])) % 360.0
    az_adj = _interp_adjust(calib.azimuth_adjust, encoder_deg, calib.resolution_deg)
    el_adj = _interp_adjust(calib.elevation_adjust, encoder_deg, calib.resolution_deg)

    # Official AT128P angle-correction formulas.
    h_deg = 2.0 * delta - calib.azimuth_offset_deg + az_adj
    v_deg = calib.elevation_deg + el_adj
    return h_deg, v_deg


# ----------------------------
# PCAP reader / PCAP读取
# ----------------------------
def iter_classic_pcap(path: Path) -> Iterator[Tuple[float, bytes]]:
    """Minimal classic-PCAP reader. Supports micro/nanosecond timestamps and both endian modes."""
    with path.open("rb") as f:
        gh = f.read(24)
        if len(gh) != 24:
            raise ValueError(f"PCAP global header too short: {path}")
        magic = gh[:4]
        if magic == b"\xd4\xc3\xb2\xa1":
            endian, ts_scale = "<", 1e-6
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian, ts_scale = ">", 1e-6
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian, ts_scale = "<", 1e-9
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian, ts_scale = ">", 1e-9
        else:
            raise ValueError(f"Unsupported PCAP magic {magic.hex()}; pcapng is not supported by this script.")

        network = struct.unpack_from(endian + "I", gh, 20)[0]
        if network != 1:
            raise ValueError(f"Only Ethernet LINKTYPE=1 is supported; got {network}.")

        ph_fmt = endian + "IIII"
        while True:
            ph = f.read(16)
            if not ph:
                break
            if len(ph) != 16:
                raise ValueError("Truncated PCAP packet header.")
            ts_sec, ts_frac, incl_len, _orig_len = struct.unpack(ph_fmt, ph)
            packet = f.read(incl_len)
            if len(packet) != incl_len:
                raise ValueError("Truncated PCAP packet data.")
            yield ts_sec + ts_frac * ts_scale, packet


def extract_udp_payload(ethernet_frame: bytes, target_port: int = 2368) -> Optional[bytes]:
    """Extract IPv4/UDP payload. Handles a single 802.1Q VLAN tag if present."""
    if len(ethernet_frame) < 14:
        return None
    eth_type = struct.unpack("!H", ethernet_frame[12:14])[0]
    ip_off = 14
    if eth_type == 0x8100 and len(ethernet_frame) >= 18:  # VLAN
        eth_type = struct.unpack("!H", ethernet_frame[16:18])[0]
        ip_off = 18
    if eth_type != 0x0800 or len(ethernet_frame) < ip_off + 20:
        return None

    ver_ihl = ethernet_frame[ip_off]
    if (ver_ihl >> 4) != 4:
        return None
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or len(ethernet_frame) < ip_off + ihl + 8:
        return None
    if ethernet_frame[ip_off + 9] != 17:  # UDP
        return None

    udp_off = ip_off + ihl
    src_port, dst_port, udp_len, _checksum = struct.unpack("!HHHH", ethernet_frame[udp_off:udp_off + 8])
    if dst_port != target_port:
        return None
    if udp_len < 8:
        return None
    payload_end = min(len(ethernet_frame), udp_off + udp_len)
    return ethernet_frame[udp_off + 8:payload_end]


# ----------------------------
# AT128P packet decoding / 数据包解码
# ----------------------------
def parse_block(payload: bytes, block_index: int, channels: int) -> Tuple[float, np.ndarray, np.ndarray]:
    off = BODY_OFFSET + block_index * BLOCK_BYTES
    az_low = struct.unpack_from("<H", payload, off)[0]
    az_fine = payload[off + 2]
    encoder_deg = az_low * 0.01 + az_fine * (0.01 / 256.0)
    base = off + 3

    raw = np.frombuffer(payload, dtype=np.uint8, count=channels * 4, offset=base).reshape(channels, 4)
    # 2-byte distance is little-endian.
    distance_raw = raw[:, 0].astype(np.uint16) | (raw[:, 1].astype(np.uint16) << 8)
    reflectivity = raw[:, 2].astype(np.uint8)
    return encoder_deg, distance_raw, reflectivity


def _select_one_return(
    d1: np.ndarray, r1: np.ndarray,
    d2: np.ndarray, r2: np.ndarray,
    policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if policy == "last":
        return d1, r1
    if policy == "strongest":
        # In Last+Strongest dual-return mode, Block 1 is Last and Block 2 is Strongest;
        # when Last is itself Strongest, Block 2 may contain the second strongest return.
        # Selecting the larger reflectivity therefore yields one strongest-like return per channel.
        choose2 = r2 > r1
        d = np.where(choose2, d2, d1)
        r = np.where(choose2, r2, r1)
        return d, r
    raise ValueError(policy)


def ranges_to_xyzi(
    distance_raw: np.ndarray,
    reflectivity: np.ndarray,
    distance_unit_mm: float,
    h_deg: np.ndarray,
    v_deg: np.ndarray,
) -> np.ndarray:
    valid = distance_raw > 0
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float32)

    d = distance_raw[valid].astype(np.float64) * (distance_unit_mm / 1000.0)
    h = np.deg2rad(h_deg[valid])
    v = np.deg2rad(v_deg[valid])
    cv = np.cos(v)

    # AT128P: Y-axis is 0 deg, clockwise (top view) is positive.
    # Vehicle convention: +X forward, +Y left, +Z up.
    x = d * cv * np.sin(h)
    y = d * cv * np.cos(h)
    z = d * np.sin(v)
    intensity = reflectivity[valid].astype(np.float64)
    return np.column_stack((x, y, z, intensity)).astype(np.float32)


def decode_measurements(
    pcap_path: Path,
    calib: AngleCalibration,
    return_policy: str = "strongest",
    udp_port: int = 2368,
) -> List[PacketMeasurement]:
    measurements: List[PacketMeasurement] = []
    seen_point_packets = 0

    for ts, ethernet in iter_classic_pcap(pcap_path):
        payload = extract_udp_payload(ethernet, target_port=udp_port)
        if payload is None or len(payload) != EXPECTED_UDP_PAYLOAD or payload[:2] != AT128_SOP:
            continue

        seen_point_packets += 1
        if len(payload) < BODY_OFFSET + 2 * BLOCK_BYTES:
            raise ValueError(f"Short AT128P payload in {pcap_path}")

        channels = int(payload[6])
        blocks = int(payload[7])
        distance_unit_mm = float(payload[9])
        returns_per_firing = int(payload[10])
        if channels != calib.channels or channels != EXPECTED_CHANNELS:
            raise ValueError(f"Channel mismatch: UDP={channels}, calibration={calib.channels}")
        if blocks != EXPECTED_BLOCKS:
            raise ValueError(f"Expected 2 blocks/packet for this AT128P converter; got {blocks}")
        if distance_unit_mm <= 0:
            raise ValueError("Invalid distance unit in UDP header.")

        a1, d1, r1 = parse_block(payload, 0, channels)
        a2, d2, r2 = parse_block(payload, 1, channels)

        if return_policy in ("strongest", "last"):
            # The current dataset is dual-return: paired blocks should share an encoder angle.
            if returns_per_firing != 2 or abs(((a2 - a1 + 180.0) % 360.0) - 180.0) > 0.02:
                raise ValueError(
                    f"'{return_policy}' expects paired dual-return blocks with same azimuth, "
                    f"but packet has returns={returns_per_firing}, az1={a1:.5f}, az2={a2:.5f}."
                )
            d, r = _select_one_return(d1, r1, d2, r2, return_policy)
            mirror_id = mirror_id_from_encoder(a1, calib)
            h, v = corrected_angles(a1, mirror_id, calib)
            xyzi = ranges_to_xyzi(d, r, distance_unit_mm, h, v)
            measurements.append(PacketMeasurement(ts, a1, mirror_id, xyzi))

        elif return_policy == "both":
            # Keep both blocks. This doubles the maximum sampling support and can affect LRDG-Net occupancy features.
            for a, d, r in ((a1, d1, r1), (a2, d2, r2)):
                mirror_id = mirror_id_from_encoder(a, calib)
                h, v = corrected_angles(a, mirror_id, calib)
                xyzi = ranges_to_xyzi(d, r, distance_unit_mm, h, v)
                measurements.append(PacketMeasurement(ts, a, mirror_id, xyzi))
        else:
            raise ValueError(f"Unknown return policy: {return_policy}")

    if seen_point_packets == 0:
        raise RuntimeError(f"No AT128P point-cloud UDP packets found in {pcap_path}")
    return measurements


def segment_frames(measurements: Sequence[PacketMeasurement], return_policy: str) -> List[FrameSegment]:
    if not measurements:
        return []
    segments: List[FrameSegment] = []
    current: List[PacketMeasurement] = []
    current_mirror: Optional[int] = None

    # In 'both' mode each PCAP packet contributes two same-mirror PacketMeasurements;
    # frame segmentation by mirror transition remains valid.
    for m in measurements:
        if current_mirror is None:
            current_mirror = m.mirror_id
        if m.mirror_id != current_mirror:
            segments.append(FrameSegment(current_mirror, current))
            current = []
            current_mirror = m.mirror_id
        current.append(m)
    if current:
        segments.append(FrameSegment(int(current_mirror), current))
    return segments


def select_complete_frames(segments: Sequence[FrameSegment], min_ratio: float) -> List[FrameSegment]:
    """Drop start/end partial captures using packet-count consistency, not label information."""
    if not segments:
        return []
    counts = np.asarray([s.packet_count for s in segments], dtype=np.float64)
    # Robust reference: median of the upper half, so short partial segments do not pull it down.
    reference = float(np.median(counts[counts >= np.median(counts)]))
    threshold = max(1.0, min_ratio * reference)
    return [s for s in segments if s.packet_count >= threshold]


# ----------------------------
# Dataset export / 数据集输出
# ----------------------------
def parse_recording_name(path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], str]:
    m = NAME_RE.match(path.stem)
    if not m:
        return None, None, None, "unknown"
    distance_m = float(m.group("distance"))
    level = int(m.group("level"))
    label = 0 if level == 0 else 1
    state = "clean" if label == 0 else "contaminated"
    return distance_m, level, label, state


def discover_pcaps(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pcap":
            raise ValueError("Input file must be .pcap / 输入文件必须是 .pcap")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(input_path.rglob("*.pcap"), key=lambda p: p.name.lower())


def export_recording(
    pcap_path: Path,
    output_root: Path,
    calib: AngleCalibration,
    return_policy: str,
    min_complete_ratio: float,
    keep_partial: bool,
    udp_port: int,
) -> List[dict]:
    distance_m, level, label, state = parse_recording_name(pcap_path)
    recording_id = pcap_path.stem

    print(f"\n[PCAP] {pcap_path.name}")
    measurements = decode_measurements(pcap_path, calib, return_policy=return_policy, udp_port=udp_port)
    segments = segment_frames(measurements, return_policy)
    frames = list(segments) if keep_partial else select_complete_frames(segments, min_complete_ratio)

    if not frames:
        raise RuntimeError(f"No complete frame detected in {pcap_path.name}")

    class_dir = state if state != "unknown" else "unknown"
    rec_dir = output_root / class_dir / recording_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for idx, frame in enumerate(frames):
        points = frame.points
        if len(points) == 0:
            continue
        bin_path = rec_dir / f"frame_{idx:04d}.bin"
        points.astype(np.float32, copy=False).tofile(bin_path)

        rows.append({
            "recording_id": recording_id,
            "source_pcap": str(pcap_path.resolve()),
            "distance_m": "" if distance_m is None else distance_m,
            "condition_level": "" if level is None else level,
            "pollution_type": "clean" if label == 0 else ("mixed_contamination" if label == 1 else "unknown"),
            "label": "" if label is None else label,
            "binary_state": state,
            "frame_index": idx,
            "mirror_id": frame.mirror_id,
            "packet_count": frame.packet_count,
            "point_count": len(points),
            "capture_start_s": f"{frame.start_time:.9f}",
            "capture_end_s": f"{frame.end_time:.9f}",
            "duration_s": f"{frame.end_time - frame.start_time:.9f}",
            "return_policy": return_policy,
            "bin_path": str(bin_path.resolve()),
        })

    all_counts = [s.packet_count for s in segments]
    kept_counts = [s.packet_count for s in frames]
    point_counts = [int(r["point_count"]) for r in rows]
    print(f"  raw frame segments : {len(segments)} | packet counts={all_counts[:3]}...{all_counts[-3:] if len(all_counts)>3 else []}")
    print(f"  complete frames    : {len(rows)}")
    if kept_counts:
        print(f"  packets/frame      : {min(kept_counts)}..{max(kept_counts)}")
    if point_counts:
        print(f"  points/frame       : {min(point_counts):,}..{max(point_counts):,} (mean={np.mean(point_counts):,.1f})")
    return rows


def write_manifest(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "recording_id", "source_pcap", "distance_m", "condition_level", "pollution_type",
        "label", "binary_state", "frame_index", "mirror_id", "packet_count", "point_count",
        "capture_start_s", "capture_end_s", "duration_s", "return_policy", "bin_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_c3_bins(rows: Sequence[dict]) -> None:
    """Check the LRDG-Net raw input contract: float32 Nx4 finite XYZI."""
    if not rows:
        raise RuntimeError("No frames were exported.")
    bad = []
    for row in rows:
        p = Path(row["bin_path"])
        arr = np.fromfile(p, dtype=np.float32)
        if arr.size == 0 or arr.size % 4 != 0:
            bad.append((str(p), "size"))
            continue
        pts = arr.reshape(-1, 4)
        if not np.isfinite(pts).all():
            bad.append((str(p), "nonfinite"))
    if bad:
        raise RuntimeError(f"LRDG-Net input compatibility validation failed: {bad[:5]}")



# No CLI entry point by design. / 正式版不提供命令行入口。
