"""Frozen LRDG-Net evaluation on the independently collected AT128 dataset."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from config import AT128, PATHS, TEST
from lrdg_net.data.dataset import make_loader
from lrdg_net.evaluation.common import load_frozen_model
from lrdg_net.evaluation.metrics import classification_metrics, confusion_matrix_counts
from lrdg_net.evaluation.visualization import save_confusion_matrix
from lrdg_net.utils.experiment import at128_evaluation_directory
from lrdg_net.utils.io import save_json


def _binary_metrics(y_true, y_pred) -> tuple[dict, np.ndarray]:
    base = classification_metrics(y_true, y_pred)
    cm = confusion_matrix_counts(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    base.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "clean_recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "contaminated_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    })
    return base, cm


def _ranking_metrics(y_true, score) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
    }


@torch.no_grad()
def evaluate_at128(model_dir: Path, overwrite: bool = True) -> dict:
    """Evaluate one frozen model; never fit target normalization or tune the threshold."""
    model_dir = Path(model_dir)
    manifest_path = Path(PATHS.at128_work_root) / "cache" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run run_04_prepare_at128.py first / 请先运行 run_04_prepare_at128.py: {manifest_path}"
        )

    result_dir = at128_evaluation_directory(model_dir)
    metrics_path = result_dir / "at128_frame_metrics.json"
    if metrics_path.exists() and not overwrite:
        print(f"Reuse AT128 evaluation / 复用AT128评估: {result_dir}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, dtype={"session_id": str, "frame_id": str})
    model, checkpoint, stats, device = load_frozen_model(model_dir)
    loader = make_loader(manifest, TEST.batch_size, TEST.num_workers, False, stats)

    records = []
    loss_sum, n = 0.0, 0
    for geometry, degradation, bag_index, labels, meta in loader:
        geometry = geometry.to(device, non_blocking=True)
        degradation = degradation.to(device, non_blocking=True)
        bag_index = bag_index.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(geometry, degradation, bag_index, labels.shape[0])
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        pred = (probs[:, 1] >= AT128.threshold).long()

        loss_sum += float(F.cross_entropy(logits, labels).item()) * labels.numel()
        n += labels.numel()

        yy = labels.cpu().numpy()
        pp = pred.cpu().numpy()
        pr = probs.cpu().numpy()
        lp = log_probs.cpu().numpy()
        lg = logits.cpu().numpy()
        for i in range(len(yy)):
            records.append({
                "sample_uid": meta["sample_uid"][i],
                "true_label": int(yy[i]),
                "pred_label": int(pp[i]),
                "prob_clean": float(pr[i, 0]),
                "prob_contaminated": float(pr[i, 1]),
                # Preserve non-saturated scores for scientific diagnosis.
                "log_prob_clean": float(lp[i, 0]),
                "log_prob_contaminated": float(lp[i, 1]),
                "logit_clean": float(lg[i, 0]),
                "logit_contaminated": float(lg[i, 1]),
                "logit_margin": float(lg[i, 1] - lg[i, 0]),
                "instance_count": int(meta["instance_count"][i]),
            })

    pred_df = pd.DataFrame(records)
    metadata_cols = [
        "sample_uid", "recording_id", "distance_m", "condition_level", "source_pcap",
        "session_uid", "frame_id", "raw_class", "severity", "point_count", "instance_count",
    ]
    keep_meta = manifest[metadata_cols].copy().rename(columns={"instance_count": "cached_instance_count"})
    pred_df = pred_df.merge(keep_meta, on="sample_uid", how="left", validate="one_to_one")

    frame_metrics, frame_cm = _binary_metrics(pred_df.true_label, pred_df.pred_label)
    frame_metrics.update(_ranking_metrics(pred_df.true_label, pred_df.logit_margin))
    frame_metrics.update({
        "evaluation_unit": "frame",
        "threshold": float(AT128.threshold),
        "loss": loss_sum / max(n, 1),
        "frames": int(len(pred_df)),
        "independent_recordings": int(pred_df.recording_id.nunique()),
        "checkpoint": str((model_dir / "best_model.pt").resolve()),
        "normalization_source": str((model_dir / "normalization_stats.json").resolve()),
        "target_normalization_fitted": False,
        "target_threshold_tuned": False,
        "ranking_score": "logit_margin = logit_contaminated - logit_clean",
        "best_epoch": int(checkpoint.get("epoch", -1)),
    })

    # Manuscript direct-transfer sequence rule: compute the frame margin
    # m_t = z_{t,1} - z_{t,0}, average it within each recording, and
    # classify as contaminated when the sequence mean margin is >= 0.
    capture = pred_df.groupby("recording_id", as_index=False).agg(
        true_label=("true_label", "first"),
        distance_m=("distance_m", "first"),
        condition_level=("condition_level", "first"),
        frames=("sample_uid", "count"),
        prob_contaminated=("prob_contaminated", "mean"),
        prob_contaminated_std=("prob_contaminated", "std"),
        logit_margin=("logit_margin", "mean"),
        point_count=("point_count", "mean"),
        instance_count=("cached_instance_count", "mean"),
    )
    capture["pred_label"] = (capture["logit_margin"] >= 0.0).astype(int)
    capture_metrics, capture_cm = _binary_metrics(capture.true_label, capture.pred_label)
    capture_metrics.update(_ranking_metrics(capture.true_label, capture.logit_margin))
    capture_metrics.update({
        "evaluation_unit": "independent_pcap_recording",
        "aggregation": "mean frame logit margin",
        "threshold": 0.0,
        "recordings": int(len(capture)),
        "target_threshold_tuned": False,
    })

    # Level 1/2/3 are analysis strata only, not model targets.
    level_rows = []
    for level, group in pred_df.groupby("condition_level", sort=True):
        level = int(level)
        if level == 0:
            level_rows.append({
                "condition_level": level,
                "binary_ground_truth": "clean",
                "frames": len(group),
                "clean_recall": float((group.pred_label == 0).mean()),
                "fpr": float((group.pred_label == 1).mean()),
                "contaminated_recall": np.nan,
                "fnr": np.nan,
            })
        else:
            level_rows.append({
                "condition_level": level,
                "binary_ground_truth": "contaminated",
                "frames": len(group),
                "clean_recall": np.nan,
                "fpr": np.nan,
                "contaminated_recall": float((group.pred_label == 1).mean()),
                "fnr": float((group.pred_label == 0).mean()),
            })
    by_level = pd.DataFrame(level_rows)

    distance_rows = []
    for distance, group in pred_df.groupby("distance_m", sort=True):
        clean = group[group.true_label == 0]
        contam = group[group.true_label == 1]
        distance_rows.append({
            "distance_m": float(distance),
            "frames": len(group),
            "clean_frames": len(clean),
            "contaminated_frames": len(contam),
            "fpr": float((clean.pred_label == 1).mean()) if len(clean) else np.nan,
            "fnr": float((contam.pred_label == 0).mean()) if len(contam) else np.nan,
            "clean_recall": float((clean.pred_label == 0).mean()) if len(clean) else np.nan,
            "contaminated_recall": float((contam.pred_label == 1).mean()) if len(contam) else np.nan,
        })
    by_distance = pd.DataFrame(distance_rows)

    pred_df.to_csv(result_dir / "at128_frame_predictions.csv", index=False, encoding="utf-8-sig")
    capture.to_csv(result_dir / "at128_pcap_predictions.csv", index=False, encoding="utf-8-sig")
    by_level.to_csv(result_dir / "at128_by_condition_level.csv", index=False, encoding="utf-8-sig")
    by_distance.to_csv(result_dir / "at128_by_distance.csv", index=False, encoding="utf-8-sig")
    save_json(frame_metrics, result_dir / "at128_frame_metrics.json")
    save_json(capture_metrics, result_dir / "at128_pcap_metrics.json")
    save_confusion_matrix(
        frame_cm,
        result_dir / "at128_frame_confusion.csv",
        result_dir / "at128_frame_confusion.png",
        "AT128 frame-level external test",
    )
    save_confusion_matrix(
        capture_cm,
        result_dir / "at128_pcap_confusion.csv",
        result_dir / "at128_pcap_confusion.png",
        "AT128 PCAP-level external test",
    )

    print("\n" + "=" * 100)
    print("Frozen LRDG-Net -> AT128 external evaluation finished / AT128 外部测试完成")
    print(f"Model / 模型: {model_dir}")
    print(f"Frames / 帧: {len(pred_df)}")
    print(f"Independent PCAPs / 独立录制: {len(capture)}")
    print("\nFrame-level metrics / 帧级指标:")
    print(json.dumps(frame_metrics, ensure_ascii=False, indent=2))
    print("\nPCAP-level metrics / 录制级指标:")
    print(json.dumps(capture_metrics, ensure_ascii=False, indent=2))
    print(f"\nOutputs / 输出: {result_dir}")
    print("AT128 normalization fitted: NO / 未使用AT128拟合归一化")
    print("AT128 threshold tuned: NO / 未使用AT128标签调阈值")
    print("=" * 100)
    return frame_metrics
