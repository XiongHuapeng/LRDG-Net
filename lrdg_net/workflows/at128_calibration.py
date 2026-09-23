"""AT128 clean-reference cross-sensor calibration with exhaustive LODO evaluation.

This workflow is a post-processing study on top of the original strict AT128 external
validation. It reuses the real frame-level predictions produced by
``run_06_all_lidaroc_to_at128.py`` and never retrains LRDG-Net or reruns AT128 point-cloud
inference.

Final protocol
--------------
1. Each All-LIDAROC checkpoint remains frozen.
2. Source clean center/scale are estimated only from that model's LIDAROC training clean
   frames.
3. The normalized decision threshold is selected only from that model's LIDAROC
   validation sessions (Macro-F1, tie -> lower FPR, then lower |threshold|).
4. AT128 calibration uses only known-clean recordings (``condition_level == 0``). No
   contaminated AT128 label is used to fit center/scale or threshold, and no network
   parameter is updated.
5. Evaluation is outer leave-one-distance-out (LODO): for each AT128 distance, all four
   recordings at that distance are held out as test.
6. K=1 exhaustively evaluates every eligible clean recording from the other 12
   distances. K=3 exhaustively evaluates every 3-clean-recording subset from the other
   12 distances (C(12,3)=220 choices per outer distance for the current dataset).
7. Calibrated frame scores are averaged within each PCAP, preserving the original
   project's temporal aggregation philosophy.
8. Paper metrics first aggregate over calibration choices and source seeds *within each
   held-out distance*, then summarize the 13 held-out distance blocks. Calibration-choice
   robustness is reported separately; calibration combinations are not treated as
   independent physical test sets.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from config import AT128, AT128_CALIBRATION, PATHS, RESEARCH, TEST, ensure_project_dirs
from lrdg_net.data.dataset import make_loader
from lrdg_net.evaluation.common import load_frozen_model
from lrdg_net.utils.experiment import all_source_spec, model_directory
from lrdg_net.utils.io import save_json


PROTOCOL_ID = "at128_clean_reference_lodo_exhaustive_v1"
REQUIRED_AT128_COLUMNS = {
    "sample_uid", "true_label", "prob_contaminated", "logit_margin",
    "recording_id", "distance_m", "condition_level", "frame_id",
}
SETTING_ORDER = ("K0_raw", "K1_clean_norm", "K3_clean_norm")
SETTING_K = {"K0_raw": 0, "K1_clean_norm": 1, "K3_clean_norm": 3}




def _validate_protocol_config() -> None:
    budgets = tuple(int(x) for x in AT128_CALIBRATION.calibration_budgets)
    if budgets != (1, 3):
        raise RuntimeError(
            f"Final AT128 calibration protocol requires calibration_budgets=(1, 3), got {budgets}."
        )
    if not bool(AT128_CALIBRATION.exhaustive_calibration_choices):
        raise RuntimeError("Final AT128 calibration protocol requires exhaustive_calibration_choices=True.")

def _output_root() -> Path:
    return Path(PATHS.evaluation_root) / "at128_calibration"


def _read_manifest() -> pd.DataFrame:
    path = Path(PATHS.cache_root) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"LIDAROC cache manifest not found: {path}\n"
            "Run run_01_preprocess_lidaroc.py first."
        )
    return pd.read_csv(path, dtype={"session_id": str, "frame_id": str})


def _read_split_sessions(model_dir: Path) -> pd.DataFrame:
    path = Path(model_dir) / "split_sessions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split_sessions.csv: {path}")
    return pd.read_csv(path, dtype={"session_id": str})


def _split_frames(model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _read_manifest()
    splits = _read_split_sessions(model_dir)
    train_u = set(splits.loc[splits["split"] == "train", "session_uid"].astype(str))
    val_u = set(splits.loc[splits["split"] == "val", "session_uid"].astype(str))
    train_df = manifest[manifest["session_uid"].astype(str).isin(train_u)].reset_index(drop=True)
    val_df = manifest[manifest["session_uid"].astype(str).isin(val_u)].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise RuntimeError(f"Empty source train/val split under {model_dir}")
    overlap = set(train_df.session_uid.astype(str)) & set(val_df.session_uid.astype(str))
    if overlap:
        raise RuntimeError(f"Source train/val session leakage: {sorted(overlap)[:5]}")
    return train_df, val_df


@torch.no_grad()
def _predict_source_frames(model_dir: Path, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return frame-level source margins using the frozen model and its own normalization."""
    model, _, stats, device = load_frozen_model(model_dir)
    loader = make_loader(dataframe, TEST.batch_size, TEST.num_workers, False, stats)
    rows = []
    for geometry, degradation, bag_index, labels, meta in loader:
        geometry = geometry.to(device)
        degradation = degradation.to(device)
        bag_index = bag_index.to(device)
        labels = labels.to(device)
        logits = model(geometry, degradation, bag_index, labels.shape[0])
        lg = logits.detach().cpu().numpy()
        yy = labels.detach().cpu().numpy()
        for i in range(len(yy)):
            rows.append({
                "sample_uid": str(meta["sample_uid"][i]),
                "session_uid": str(meta["session_uid"][i]),
                "domain": str(meta["domain"][i]),
                "raw_class": str(meta["raw_class"][i]),
                "severity": str(meta["severity"][i]),
                "true_label": int(yy[i]),
                "logit_margin": float(lg[i, 1] - lg[i, 0]),
            })
    return pd.DataFrame(rows)


def _robust_center_scale(values, eps: float) -> tuple[float, float, str]:
    """Median + Gaussian-consistent MAD, with deterministic degenerate fallbacks."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise RuntimeError("At least two finite clean frame scores are required for scale estimation.")
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.482602218505602 * mad
    method = "1.4826*MAD"
    if not np.isfinite(scale) or scale <= eps:
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scale = float((q75 - q25) / 1.3489795003921634)
        method = "IQR/1.349"
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.std(x, ddof=1))
        method = "sample_std"
    if not np.isfinite(scale) or scale <= eps:
        scale = 1.0
        method = "unit_fallback"
    return center, scale, method


def _confusion_rates(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def _binary_macro_f1(y_true, y_pred) -> float:
    rates = _confusion_rates(y_true, y_pred)
    tn, fp, fn, tp = rates["tn"], rates["fp"], rates["fn"], rates["tp"]
    den0 = 2 * tn + fp + fn
    den1 = 2 * tp + fp + fn
    f10 = (2.0 * tn / den0) if den0 else 0.0
    f11 = (2.0 * tp / den1) if den1 else 0.0
    return float((f10 + f11) / 2.0)


def _ranking_metrics(y_true, score) -> dict:
    """Small-sample AUROC/AUPRC without repeated sklearn estimator overhead."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if np.unique(y).size < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    pos = s[y == 1]
    neg = s[y == 0]
    pair = pos[:, None] - neg[None, :]
    auroc = float((np.sum(pair > 0) + 0.5 * np.sum(pair == 0)) / pair.size)

    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    cum_pos = np.cumsum(ys == 1)
    pos_idx = np.flatnonzero(ys == 1)
    if pos_idx.size:
        precisions = cum_pos[pos_idx] / (pos_idx + 1)
        auprc = float(np.mean(precisions))
    else:
        auprc = np.nan
    return {"auroc": auroc, "auprc": auprc}

def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if unique.size == 1:
        return np.asarray([unique[0]], dtype=np.float64)
    mid = (unique[:-1] + unique[1:]) / 2.0
    span = max(float(unique[-1] - unique[0]), 1.0)
    return np.concatenate(([unique[0] - span], mid, [unique[-1] + span]))


def _select_source_threshold(session_df: pd.DataFrame) -> dict:
    """Source-only threshold selection: Macro-F1, tie -> lower FPR -> lower |threshold|."""
    y = session_df["true_label"].to_numpy(dtype=int)
    scores = session_df["normalized_score"].to_numpy(dtype=float)
    best = None
    for threshold in _threshold_candidates(scores):
        pred = (scores >= threshold).astype(int)
        mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
        rates = _confusion_rates(y, pred)
        candidate = {
            "threshold": float(threshold),
            "macro_f1": mf1,
            "fpr": rates["fpr"],
            "fnr": rates["fnr"],
        }
        key = (candidate["macro_f1"], -candidate["fpr"], -abs(candidate["threshold"]))
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1]


def _build_source_reference(seed: int, overwrite: bool = False) -> dict:
    out_root = _output_root()
    source_dir = out_root / "source_reference"
    source_dir.mkdir(parents=True, exist_ok=True)
    json_path = source_dir / f"LRDG_ALL_seed{seed}.json"
    train_csv = source_dir / f"LRDG_ALL_seed{seed}_clean_train_scores.csv"
    val_csv = source_dir / f"LRDG_ALL_seed{seed}_validation_session_scores.csv"
    if json_path.exists() and not overwrite:
        return json.loads(json_path.read_text(encoding="utf-8"))

    spec = all_source_spec(RESEARCH.all_source_domains, seed)
    model_dir = model_directory(spec)
    train_df, val_df = _split_frames(model_dir)
    train_clean_df = train_df[train_df["label"].astype(int) == 0].reset_index(drop=True)
    if train_clean_df.empty:
        raise RuntimeError(f"No clean LIDAROC training frames for {spec.name}")

    train_clean_scores = _predict_source_frames(model_dir, train_clean_df)
    val_scores = _predict_source_frames(model_dir, val_df)
    center, scale, scale_method = _robust_center_scale(
        train_clean_scores["logit_margin"].to_numpy(), AT128_CALIBRATION.scale_eps
    )
    val_scores["normalized_score"] = (val_scores["logit_margin"] - center) / scale

    # Session is the independent source recording unit, analogous to one AT128 PCAP.
    session = val_scores.groupby("session_uid", as_index=False).agg(
        true_label=("true_label", "first"),
        domain=("domain", "first"),
        raw_class=("raw_class", "first"),
        severity=("severity", "first"),
        frames=("sample_uid", "count"),
        normalized_score=("normalized_score", "mean"),
    )
    threshold_info = _select_source_threshold(session)

    train_clean_scores.to_csv(train_csv, index=False, encoding="utf-8-sig")
    session.to_csv(val_csv, index=False, encoding="utf-8-sig")
    payload = {
        "model": spec.name,
        "seed": int(seed),
        "model_dir": str(model_dir),
        "source_clean_scope": "LIDAROC training split, clean frames only",
        "source_threshold_scope": "LIDAROC validation sessions only",
        "center": center,
        "scale": scale,
        "scale_method": scale_method,
        "clean_train_frames": int(len(train_clean_scores)),
        "clean_train_sessions": int(train_clean_df["session_uid"].nunique()),
        "validation_sessions": int(len(session)),
        "threshold_normalized": threshold_info["threshold"],
        "validation_macro_f1": threshold_info["macro_f1"],
        "validation_fpr": threshold_info["fpr"],
        "validation_fnr": threshold_info["fnr"],
        "target_domain_used": False,
    }
    save_json(payload, json_path)
    return payload


def _read_at128_frame_predictions(seed: int) -> pd.DataFrame:
    path = Path(PATHS.evaluation_root) / "at128_external" / f"LRDG_ALL_seed{seed}" / "at128_frame_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing original AT128 frame predictions: {path}\n"
            "Run run_06_all_lidaroc_to_at128.py first."
        )
    df = pd.read_csv(path, dtype={"frame_id": str})
    missing = REQUIRED_AT128_COLUMNS.difference(df.columns)
    if missing:
        raise RuntimeError(f"AT128 prediction CSV missing columns: {sorted(missing)}")
    if len(df) <= df["recording_id"].nunique():
        raise RuntimeError(f"Expected true frame-level predictions but got {len(df)} rows: {path}")
    return df


def _available_distances(frame_df: pd.DataFrame) -> tuple[float, ...]:
    distances = tuple(sorted(float(x) for x in frame_df["distance_m"].dropna().unique()))
    if len(distances) < 4:
        raise RuntimeError("At least four AT128 distance blocks are required for K=3 LODO calibration.")
    return distances


def _validate_at128_structure(frame_df: pd.DataFrame) -> None:
    """Validate one known-clean recording per distance and complete held-out blocks."""
    for distance, g in frame_df.groupby("distance_m", sort=True):
        clean = g[g["condition_level"].astype(int) == 0]
        clean_pcaps = clean["recording_id"].nunique()
        if clean_pcaps != 1:
            raise RuntimeError(
                f"Expected exactly one condition_level=0 clean PCAP at {distance} m, found {clean_pcaps}."
            )
        if (clean["true_label"].astype(int) != 0).any():
            raise RuntimeError(f"Known-clean recording at {distance} m contains non-clean labels.")
        labels_per_pcap = g.groupby("recording_id")["true_label"].nunique()
        if (labels_per_pcap != 1).any():
            raise RuntimeError(f"A recording at {distance} m contains mixed labels.")


def _calibration_choices(distances: tuple[float, ...], test_distance: float, k: int) -> tuple[tuple[float, ...], ...]:
    eligible = tuple(d for d in distances if d != float(test_distance))
    if k == 1:
        return tuple((d,) for d in eligible)
    if k == 3:
        return tuple(combinations(eligible, 3))
    raise ValueError(f"Unsupported calibration budget: K={k}")


def _format_distances(distances) -> str:
    def _fmt(x: float) -> str:
        x = float(x)
        return str(int(x)) if x.is_integer() else f"{x:g}"
    return ";".join(_fmt(x) for x in distances)


def _prepare_at128_seed(frame_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Pre-aggregate test PCAP means and cache clean frame margins by distance.

    This makes exhaustive K1/K3 evaluation fast without changing the mathematics:
    mean((m-c)/s) == (mean(m)-c)/s.
    """
    pcap = frame_df.groupby("recording_id", as_index=False).agg(
        true_label=("true_label", "first"),
        distance_m=("distance_m", "first"),
        condition_level=("condition_level", "first"),
        frames=("sample_uid", "count"),
        prob_contaminated=("prob_contaminated", "mean"),
        logit_margin=("logit_margin", "mean"),
    )
    clean_by_distance: dict[float, np.ndarray] = {}
    for distance, g in frame_df[frame_df["condition_level"].astype(int) == 0].groupby("distance_m", sort=True):
        clean_by_distance[float(distance)] = g["logit_margin"].to_numpy(dtype=np.float64)
    return pcap, clean_by_distance


def _aggregate_raw_k0(pcap_base: pd.DataFrame, test_distance: float) -> pd.DataFrame:
    pcap = pcap_base[pcap_base["distance_m"].astype(float) == float(test_distance)].copy()
    pcap["decision_score"] = pcap["logit_margin"]
    pcap["pred_label"] = (pcap["decision_score"] >= 0.0).astype(int)
    return pcap.reset_index(drop=True)


def _aggregate_calibrated(
    pcap_base: pd.DataFrame,
    clean_margin_by_distance: dict[float, np.ndarray],
    source_ref: dict,
    clean_distances: tuple[float, ...],
    test_distance: float,
) -> tuple[pd.DataFrame, dict]:
    if float(test_distance) in set(float(x) for x in clean_distances):
        raise RuntimeError("Calibration and held-out test distance overlap.")

    arrays = []
    for distance in clean_distances:
        key = float(distance)
        if key not in clean_margin_by_distance:
            raise RuntimeError(f"Missing known-clean AT128 calibration frames at {key} m")
        arrays.append(clean_margin_by_distance[key])
    cal_values = np.concatenate(arrays)
    center, scale, method = _robust_center_scale(cal_values, AT128_CALIBRATION.scale_eps)

    pcap = pcap_base[pcap_base["distance_m"].astype(float) == float(test_distance)].copy()
    pcap["raw_logit_margin"] = pcap["logit_margin"].astype(float)
    pcap["normalized_score"] = (pcap["raw_logit_margin"] - center) / scale
    threshold = float(source_ref["threshold_normalized"])
    pcap["decision_score"] = pcap["normalized_score"] - threshold
    pcap["pred_label"] = (pcap["decision_score"] >= 0.0).astype(int)
    info = {
        "target_clean_distances_m": [float(x) for x in clean_distances],
        "target_clean_pcaps": int(len(clean_distances)),
        "target_clean_frames": int(sum(len(x) for x in arrays)),
        "target_center": float(center),
        "target_scale": float(scale),
        "target_scale_method": method,
        "source_threshold_normalized": threshold,
        "target_contaminated_labels_used_for_calibration": False,
        "network_parameters_updated": False,
    }
    return pcap.reset_index(drop=True), info

def _metrics_for_pcap(pcap: pd.DataFrame) -> dict:
    y = pcap["true_label"].to_numpy(dtype=int)
    pred = pcap["pred_label"].to_numpy(dtype=int)
    m = {
        "macro_f1": _binary_macro_f1(y, pred),
        "recordings": int(len(pcap)),
    }
    m.update(_confusion_rates(y, pred))
    score_col = "decision_score" if "decision_score" in pcap else "logit_margin"
    m.update(_ranking_metrics(y, pcap[score_col].to_numpy(dtype=float)))
    return m


def _q05(values) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.quantile(x, 0.05)) if x.size else np.nan


def _aggregate_by_distance(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, distance), g in run_df.groupby(["setting", "test_distance_m"], sort=True):
        rows.append({
            "setting": setting,
            "test_distance_m": float(distance),
            "runs": int(len(g)),
            "source_seeds": int(g["seed"].nunique()),
            "unique_calibration_choices": int(g["calibration_distances_m"].nunique()),
            "macro_f1_mean": float(g["macro_f1"].mean()),
            "macro_f1_std": float(g["macro_f1"].std(ddof=1)) if len(g) > 1 else 0.0,
            "macro_f1_p05": _q05(g["macro_f1"]),
            "macro_f1_worst_combo": float(g["macro_f1"].min()),
            "perfect_run_rate": float(np.mean(np.isclose(g["macro_f1"].to_numpy(float), 1.0))),
            "fpr_mean": float(g["fpr"].mean()),
            "fnr_mean": float(g["fnr"].mean()),
            "auroc_mean": float(g["auroc"].mean()),
        })
    return pd.DataFrame(rows)


def _aggregate_by_seed(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, seed), g in run_df.groupby(["setting", "seed"], sort=True):
        rows.append({
            "setting": setting,
            "seed": int(seed),
            "runs": int(len(g)),
            "held_out_distances": int(g["test_distance_m"].nunique()),
            "macro_f1_mean": float(g["macro_f1"].mean()),
            "macro_f1_std": float(g["macro_f1"].std(ddof=1)) if len(g) > 1 else 0.0,
            "macro_f1_p05": _q05(g["macro_f1"]),
            "macro_f1_worst_combo": float(g["macro_f1"].min()),
            "perfect_run_rate": float(np.mean(np.isclose(g["macro_f1"].to_numpy(float), 1.0))),
            "fpr_mean": float(g["fpr"].mean()),
            "fnr_mean": float(g["fnr"].mean()),
            "auroc_mean": float(g["auroc"].mean()),
        })
    return pd.DataFrame(rows)


def _paper_summary(run_df: pd.DataFrame, distance_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight the 13 held-out distance blocks for the main paper summary."""
    rows = []
    for setting in SETTING_ORDER:
        g = run_df[run_df["setting"] == setting]
        d = distance_df[distance_df["setting"] == setting]
        s = seed_df[seed_df["setting"] == setting]
        if g.empty or d.empty or s.empty:
            raise RuntimeError(f"Missing calibration results for {setting}")
        is_calibrated = SETTING_K[setting] > 0
        rows.append({
            "setting": setting,
            "clean_calibration_pcaps": int(SETTING_K[setting]),
            "target_contaminated_labels_used": False,
            "network_retraining": False,
            "held_out_test_distances": int(d["test_distance_m"].nunique()),
            "source_models": int(g["seed"].nunique()),
            "evaluation_runs": int(len(g)),
            "macro_f1_mean": float(d["macro_f1_mean"].mean()),
            "macro_f1_std_across_distances": float(d["macro_f1_mean"].std(ddof=1)),
            "worst_seed_macro_f1": float(s["macro_f1_mean"].min()),
            "worst_distance_macro_f1": float(d["macro_f1_mean"].min()),
            "fpr_mean": float(d["fpr_mean"].mean()),
            "fnr_mean": float(d["fnr_mean"].mean()),
            "auroc_mean": float(d["auroc_mean"].mean()),
            "calibration_choice_success_rate": (
                float(np.mean(np.isclose(g["macro_f1"].to_numpy(float), 1.0))) if is_calibrated else np.nan
            ),
            "combo_macro_f1_p05": _q05(g["macro_f1"]) if is_calibrated else np.nan,
            "worst_combo_macro_f1": float(g["macro_f1"].min()) if is_calibrated else np.nan,
        })
    return pd.DataFrame(rows)


def _choice_robustness(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in ("K1_clean_norm", "K3_clean_norm"):
        g = run_df[run_df["setting"] == setting]
        rows.append({
            "setting": setting,
            "clean_calibration_pcaps": SETTING_K[setting],
            "runs": int(len(g)),
            "source_models": int(g["seed"].nunique()),
            "held_out_distances": int(g["test_distance_m"].nunique()),
            "perfect_run_rate": float(np.mean(np.isclose(g["macro_f1"].to_numpy(float), 1.0))),
            "macro_f1_mean": float(g["macro_f1"].mean()),
            "macro_f1_p05": _q05(g["macro_f1"]),
            "macro_f1_worst_combo": float(g["macro_f1"].min()),
            "fpr_mean": float(g["fpr"].mean()),
            "fnr_mean": float(g["fnr"].mean()),
        })
    return pd.DataFrame(rows)


def _plot_distance_response(predictions: pd.DataFrame, output_path: Path, setting: str = "K1_clean_norm") -> None:
    """Diagnostic plot: mean +/- std across source seeds and exhaustive calibration choices."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skip plot because matplotlib is unavailable: {exc}")
        return
    x = predictions[predictions["setting"] == setting].copy()
    if x.empty:
        return
    summary = x.groupby(["distance_m", "condition_level"], as_index=False).agg(
        score_mean=("decision_score", "mean"),
        score_std=("decision_score", "std"),
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for level, g in summary.groupby("condition_level", sort=True):
        g = g.sort_values("distance_m")
        ax.errorbar(
            g["distance_m"], g["score_mean"], yerr=g["score_std"].fillna(0.0),
            marker="o", label=f"Condition {int(level)}",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.set_xlabel("Held-out distance (m)")
    ax.set_ylabel("Clean-normalized decision score")
    ax.set_title("AT128 response under exhaustive K=1 clean-reference calibration")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _protocol_is_current(out_root: Path) -> bool:
    path = out_root / "protocol.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("protocol_id") == PROTOCOL_ID


def run_at128_calibration(overwrite: bool | None = None) -> pd.DataFrame:
    """Run strict K0 plus exhaustive K1/K3 LODO clean-reference calibration."""
    ensure_project_dirs()
    _validate_protocol_config()
    overwrite = AT128_CALIBRATION.overwrite if overwrite is None else bool(overwrite)
    out_root = _output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "paper_cross_sensor_summary.csv"

    if summary_path.exists() and not overwrite and _protocol_is_current(out_root):
        print(f"Reuse current AT128 calibration outputs: {summary_path}")
        return pd.read_csv(summary_path)
    if summary_path.exists() and not overwrite and not _protocol_is_current(out_root):
        print("Existing at128_calibration outputs use an older protocol; rebuilding final exhaustive LODO results.")

    run_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    source_rows: list[dict] = []
    calibration_rows: list[dict] = []
    reference_distances: tuple[float, ...] | None = None

    for seed in RESEARCH.seeds:
        # Source reference files from the previous calibration run remain valid and are reused
        # unless overwrite=True. They contain no AT128-derived source threshold tuning.
        source_ref = _build_source_reference(int(seed), overwrite=overwrite)
        source_rows.append(source_ref)
        frame_df = _read_at128_frame_predictions(int(seed))
        _validate_at128_structure(frame_df)
        distances = _available_distances(frame_df)
        pcap_base, clean_margin_by_distance = _prepare_at128_seed(frame_df)
        if reference_distances is None:
            reference_distances = distances
        elif tuple(reference_distances) != tuple(distances):
            raise RuntimeError("AT128 available distances differ across source-model seeds.")

        for test_distance in distances:
            # K0: strict raw transfer. One run per source seed and held-out distance.
            pcap = _aggregate_raw_k0(pcap_base, test_distance)
            met = _metrics_for_pcap(pcap)
            run_id = f"K0_raw|seed{seed}|test{_format_distances((test_distance,))}"
            run_rows.append({
                "run_id": run_id,
                "setting": "K0_raw",
                "seed": int(seed),
                "test_distance_m": float(test_distance),
                "clean_calibration_pcaps": 0,
                "calibration_distances_m": "",
                **met,
                "target_clean_frames": 0,
                "target_center": np.nan,
                "target_scale": np.nan,
                "target_scale_method": "none",
                "source_threshold_normalized": np.nan,
                "target_contaminated_labels_used_for_calibration": False,
                "network_parameters_updated": False,
            })
            pp = pcap.copy()
            pp.insert(0, "calibration_distances_m", "")
            pp.insert(0, "test_distance_m", float(test_distance))
            pp.insert(0, "seed", int(seed))
            pp.insert(0, "setting", "K0_raw")
            pp.insert(0, "run_id", run_id)
            all_predictions.append(pp)

            # K=1 and K=3: exhaustive clean-reference choices from all other distances.
            for setting, k in (("K1_clean_norm", 1), ("K3_clean_norm", 3)):
                choices = _calibration_choices(distances, test_distance, k)
                for choice_index, clean_distances in enumerate(choices):
                    pcap, info = _aggregate_calibrated(
                        pcap_base, clean_margin_by_distance, source_ref, clean_distances, test_distance
                    )
                    met = _metrics_for_pcap(pcap)
                    choice_str = _format_distances(clean_distances)
                    run_id = (
                        f"{setting}|seed{seed}|test{_format_distances((test_distance,))}|cal{choice_str}"
                    )
                    run_rows.append({
                        "run_id": run_id,
                        "setting": setting,
                        "seed": int(seed),
                        "test_distance_m": float(test_distance),
                        "clean_calibration_pcaps": int(k),
                        "calibration_choice_index": int(choice_index),
                        "calibration_distances_m": choice_str,
                        **met,
                        **info,
                    })
                    calibration_rows.append({
                        "run_id": run_id,
                        "setting": setting,
                        "seed": int(seed),
                        "test_distance_m": float(test_distance),
                        "clean_calibration_pcaps": int(k),
                        "calibration_choice_index": int(choice_index),
                        "calibration_distances_m": choice_str,
                        **info,
                    })
                    pp = pcap.copy()
                    pp.insert(0, "calibration_distances_m", choice_str)
                    pp.insert(0, "test_distance_m", float(test_distance))
                    pp.insert(0, "seed", int(seed))
                    pp.insert(0, "setting", setting)
                    pp.insert(0, "run_id", run_id)
                    all_predictions.append(pp)

    run_df = pd.DataFrame(run_rows)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    source_df = pd.DataFrame(source_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    distance_df = _aggregate_by_distance(run_df)
    seed_df = _aggregate_by_seed(run_df)
    paper = _paper_summary(run_df, distance_df, seed_df)
    robustness = _choice_robustness(run_df)
    failures = run_df[
        run_df["setting"].isin(["K1_clean_norm", "K3_clean_norm"])
        & (~np.isclose(run_df["macro_f1"].to_numpy(float), 1.0))
    ].copy()

    run_df.to_csv(out_root / "at128_calibration_runs.csv", index=False, encoding="utf-8-sig")
    distance_df.to_csv(out_root / "at128_calibration_by_distance.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(out_root / "at128_calibration_by_seed.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_root / "at128_calibration_pcap_predictions.csv", index=False, encoding="utf-8-sig")
    source_df.to_csv(out_root / "source_reference_summary.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(out_root / "target_clean_calibration_summary.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(out_root / "calibration_choice_robustness.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(out_root / "calibration_failures.csv", index=False, encoding="utf-8-sig")
    paper.to_csv(summary_path, index=False, encoding="utf-8-sig")

    distances = tuple(reference_distances or ())
    save_json({
        "protocol_id": PROTOCOL_ID,
        "protocol": "AT128 clean-reference robust location-scale calibration with exhaustive outer LODO evaluation",
        "config": asdict(AT128_CALIBRATION),
        "source_reference": "LIDAROC training clean frames",
        "source_threshold_selection": "LIDAROC validation sessions; maximize Macro-F1; tie -> lower FPR -> lower |threshold|",
        "target_calibration": "known-clean AT128 frame margins only",
        "outer_evaluation": "leave one complete AT128 distance block out; test all condition 0/1/2/3 PCAPs at that distance",
        "k1_calibration": "exhaust all one-clean-PCAP choices from non-test distances",
        "k3_calibration": "exhaust all three-clean-PCAP combinations from non-test distances",
        "target_contaminated_labels_used_for_calibration": False,
        "network_retraining": False,
        "at128_inference_rerun": False,
        "k0_pcap_decision": "mean frame logit margin >= 0",
        "pcap_aggregation": "mean calibrated frame score",
        "available_distances_m": [float(x) for x in distances],
        "held_out_distance_blocks": int(len(distances)),
        "k1_choices_per_outer_fold": int(len(distances) - 1) if distances else 0,
        "k3_choices_per_outer_fold": int(len(tuple(combinations(distances[1:], 3)))) if len(distances) >= 4 else 0,
        "statistical_note": "calibration choices are repeated evaluations of the same held-out physical distance block and are not treated as independent test sets",
    }, out_root / "protocol.json")

    # Remove the obsolete fixed-choice figure from the previous protocol if present.
    old_plot = out_root / "distance_response_K3_clean_norm.png"
    if old_plot.exists():
        old_plot.unlink()
    _plot_distance_response(pred_df, out_root / "distance_response_clean_norm.png", setting="K1_clean_norm")

    print("\n" + "=" * 118)
    print("AT128 exhaustive clean-reference cross-sensor calibration finished")
    print("Outer evaluation: leave one complete distance block out (all four PCAPs are test).")
    print("K=1: every other-distance clean PCAP is evaluated; K=3: every three-clean-PCAP subset is evaluated.")
    print("No contaminated target label is used for calibration; no network update; no AT128 re-inference.")
    print(f"Held-out distances: {distances}")
    print("\nPaper summary:")
    print(paper.to_string(index=False))
    print("\nCalibration-choice robustness:")
    print(robustness.to_string(index=False))
    print(f"\nNon-perfect calibrated runs: {len(failures)}")
    print(f"Outputs: {out_root}")
    print("=" * 118)
    return paper
