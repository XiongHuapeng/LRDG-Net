"""
Session-level splitting utilities / Session级数据划分工具。

The split is always performed on complete sessions. Multiple LIDAROC domains are supported,
which allows the final all-LIDAROC model to use 5m+10m+20m while preserving the same
within-stratum session-level protocol used in the manuscript.
"""
from typing import Tuple
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid"
}


def strict_session_train_val_split(
    dataframe: pd.DataFrame,
    val_ratio: float,
    split_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns / manifest 缺列: {sorted(missing)}")
    if dataframe.empty:
        raise RuntimeError("Empty training dataframe / 训练数据为空。")

    rng = np.random.default_rng(split_seed)
    train_parts, val_parts = [], []

    # Domain is part of the stratum so that all-source training preserves each domain's
    # class/severity session composition rather than mixing sessions across domains.
    for _, group in dataframe.groupby(["domain", "raw_class", "severity"], sort=True):
        sessions = group["session_uid"].drop_duplicates().astype(str).to_numpy()
        if len(sessions) < 2:
            # Never fall back to frame-level splitting; this prevents session leakage.
            train_parts.append(group)
            continue

        shuffled = sessions.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        val_uids = set(shuffled[:n_val].tolist())
        is_val = group["session_uid"].astype(str).isin(val_uids)
        train_parts.append(group[~is_val])
        val_parts.append(group[is_val])

    if not val_parts:
        raise RuntimeError(
            "No session-level validation split can be created without frame leakage. / "
            "无法在不切分 session 的条件下建立验证集。"
        )

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1.0, random_state=split_seed
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(
        frac=1.0, random_state=split_seed
    ).reset_index(drop=True)

    assert_no_session_overlap(train_df, val_df, "Train", "Val")
    return train_df, val_df


def assert_no_session_overlap(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str) -> None:
    overlap = set(a["session_uid"].astype(str)) & set(b["session_uid"].astype(str))
    if overlap:
        raise RuntimeError(
            f"Session leakage between {a_name} and {b_name} / session 泄漏: {sorted(overlap)[:5]}"
        )


def save_train_val_sessions(train_df: pd.DataFrame, val_df: pd.DataFrame, path) -> None:
    cols = ["domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid"]
    pieces = []
    for split_name, frame in (("train", train_df), ("val", val_df)):
        x = frame[cols].drop_duplicates().copy()
        x.insert(0, "split", split_name)
        pieces.append(x)
    pd.concat(pieces, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
