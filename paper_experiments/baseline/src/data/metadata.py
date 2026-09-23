from pathlib import Path
from typing import Dict, Optional, Sequence
import re
import pandas as pd

DOMAIN_RE = re.compile(r"(?<!\d)(5|10|20)\s*m(?!\d)", re.IGNORECASE)

def normalize_class_name(name: str) -> str:
  v = str(name).strip().lower()
  if v in {"muddrop", "muduniform", "mud_drop", "mud_uniform"}: return "mud"
  return v

def infer_domain(path: Path) -> Optional[str]:
  m = DOMAIN_RE.search(str(path).replace("\\", "/"))
  return f"{m.group(1)}m" if m else None

def parse_lidaroc_filename(path: Path) -> Optional[Dict[str, str]]:
  parts = path.stem.split("_")
  if len(parts) < 6: return None
  domain = infer_domain(path)
  if domain is None: return None
  raw_class = parts[1].strip()
  pollution_type = normalize_class_name(raw_class)
  severity = parts[3].strip().lower()
  session_id = str(parts[-2]).strip()
  frame_id = str(parts[-1]).strip()
  session_uid = f"{domain}|{raw_class.lower()}|{severity}|session_{session_id}"
  return dict(domain=domain, raw_class=raw_class, pollution_type=pollution_type,
        severity=severity, session_id=session_id, frame_id=frame_id,
        session_uid=session_uid, sample_uid=f"{session_uid}|frame_{frame_id}")

def scan_raw_dataset(data_root: str, target_classes: Sequence[str]) -> pd.DataFrame:
  root = Path(data_root)
  if not root.exists(): raise FileNotFoundError(f"LIDAROC root not found: {root}")
  targets = {normalize_class_name(x) for x in target_classes}
  rows=[]
  for p in sorted(root.rglob("*.bin")):
    m=parse_lidaroc_filename(p)
    if m is None or m["pollution_type"] not in targets: continue
    rows.append({"source_path":str(p.resolve()), "sample_id":p.stem,
           "label":0 if m["pollution_type"]=="clean" else 1, **m})
  if not rows: raise RuntimeError("No valid LIDAROC .bin files found.")
  return pd.DataFrame(rows).sort_values(["domain","raw_class","severity","session_id","frame_id"]).reset_index(drop=True)
