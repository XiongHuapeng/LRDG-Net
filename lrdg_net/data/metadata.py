"""
LIDAROC 文件扫描与元数据解析。
LIDAROC file scanning and metadata parsing.

与 AutoGrAN / HT-VGNet M4 的文件名协议保持一致。
The filename protocol is kept consistent with the existing AutoGrAN / HT-VGNet M4 projects.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence
import re
import pandas as pd

DOMAIN_RE = re.compile(r"(?<!\d)(5|10|20)\s*m(?!\d)", re.IGNORECASE)


def normalize_class_name(name: str) -> str:
    value = str(name).strip().lower()
    if value in {"muddrop", "muduniform", "mud_drop", "mud_uniform"}:
        return "mud"
    return value


def infer_domain(path: Path) -> Optional[str]:
    match = DOMAIN_RE.search(str(path).replace("\\", "/"))
    return f"{match.group(1)}m" if match else None


def parse_lidaroc_filename(path: Path) -> Optional[Dict[str, str]]:
    """
    预期文件名 / Expected filename:
        classid_class_levelid_severity_session_frame.bin
    """
    parts = path.stem.split("_")
    if len(parts) < 6:
        return None
    domain = infer_domain(path)
    if domain is None:
        return None

    raw_class = parts[1].strip()
    pollution_type = normalize_class_name(raw_class)
    severity = parts[3].strip().lower()
    session_id = str(parts[-2]).strip()
    frame_id = str(parts[-1]).strip()

    session_uid = f"{domain}|{raw_class.lower()}|{severity}|session_{session_id}"
    sample_uid = f"{session_uid}|frame_{frame_id}"

    return {
        "domain": domain,
        "raw_class": raw_class,
        "pollution_type": pollution_type,
        "severity": severity,
        "session_id": session_id,
        "frame_id": frame_id,
        "session_uid": session_uid,
        "sample_uid": sample_uid,
    }


def scan_raw_dataset(data_root: str, target_classes: Sequence[str]) -> pd.DataFrame:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found / 数据集路径不存在: {root}")

    target_set = {normalize_class_name(x) for x in target_classes}
    rows = []
    for path in sorted(root.rglob("*.bin")):
        meta = parse_lidaroc_filename(path)
        if meta is None or meta["pollution_type"] not in target_set:
            continue
        rows.append({
            "source_path": str(path.resolve()),
            "sample_id": path.stem,
            "label": 0 if meta["pollution_type"] == "clean" else 1,
            **meta,
        })

    if not rows:
        raise RuntimeError(
            "No valid LIDAROC .bin files found for target_classes. / "
            "未找到符合当前类别配置的 LIDAROC .bin。"
        )

    return pd.DataFrame(rows).sort_values(
        ["domain", "raw_class", "severity", "session_id", "frame_id", "source_path"]
    ).reset_index(drop=True)
