"""Independent LIDAROC target-domain evaluation for a frozen trained model."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from config import PATHS, TEST
from lrdg_net.data.dataset import make_loader
from lrdg_net.data.splitter import assert_no_session_overlap
from lrdg_net.evaluation.common import load_frozen_model
from lrdg_net.evaluation.metrics import classification_metrics, confusion_matrix_counts, pollution_metrics
from lrdg_net.evaluation.visualization import save_confusion_matrix
from lrdg_net.utils.experiment import TrainingSpec, lidaroc_evaluation_directory
from lrdg_net.utils.io import save_json


@torch.no_grad()
def evaluate_lidaroc_domain(spec: TrainingSpec, target_domain: str, overwrite: bool = False) -> dict:
    if target_domain in spec.train_domains:
        raise ValueError(
            f"target_domain={target_domain} is part of training domains {spec.train_domains}; "
            "cross-domain evaluation requires an unseen target domain."
        )

    model_dir = Path(PATHS.model_root) / (
        "single_source" if spec.protocol == "single_source" else "all_source"
    ) / spec.name
    result_dir = lidaroc_evaluation_directory(spec, target_domain)
    metrics_path = result_dir / "test_metrics.json"
    if metrics_path.exists() and not overwrite:
        import json
        print(f"Reuse evaluation / 复用评估结果: {result_dir}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(Path(PATHS.cache_root) / "manifest.csv", dtype={"session_id": str, "frame_id": str})
    splits = pd.read_csv(model_dir / "split_sessions.csv", dtype={"session_id": str})
    train_u = set(splits.loc[splits["split"] == "train", "session_uid"].astype(str))
    val_u = set(splits.loc[splits["split"] == "val", "session_uid"].astype(str))
    train_df = manifest[manifest["session_uid"].astype(str).isin(train_u)]
    val_df = manifest[manifest["session_uid"].astype(str).isin(val_u)]
    test_df = manifest[manifest["domain"].astype(str) == str(target_domain)].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError(f"No samples for LIDAROC target domain: {target_domain}")
    assert_no_session_overlap(train_df, test_df, "Train", "Test")
    assert_no_session_overlap(val_df, test_df, "Val", "Test")

    model, checkpoint, stats, device = load_frozen_model(model_dir)
    loader = make_loader(test_df, TEST.batch_size, TEST.num_workers, False, stats)

    ys, ps, records = [], [], []
    loss_sum, n = 0.0, 0
    for geometry, degradation, bag_index, labels, meta in loader:
        geometry = geometry.to(device)
        degradation = degradation.to(device)
        bag_index = bag_index.to(device)
        labels = labels.to(device)
        logits = model(geometry, degradation, bag_index, labels.shape[0])
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)
        loss_sum += float(F.cross_entropy(logits, labels).item()) * labels.numel()
        n += labels.numel()

        yy = labels.cpu().numpy()
        pp = pred.cpu().numpy()
        pr = probs.cpu().numpy()
        lp = log_probs.cpu().numpy()
        lg = logits.cpu().numpy()
        ys.extend(yy.tolist())
        ps.extend(pp.tolist())
        for i in range(len(yy)):
            records.append({
                **{k: meta[k][i] for k in meta if k != "instance_count"},
                "instance_count": int(meta["instance_count"][i]),
                "true_label": int(yy[i]),
                "pred_label": int(pp[i]),
                "prob_clean": float(pr[i, 0]),
                "prob_contaminated": float(pr[i, 1]),
                "log_prob_clean": float(lp[i, 0]),
                "log_prob_contaminated": float(lp[i, 1]),
                "logit_clean": float(lg[i, 0]),
                "logit_contaminated": float(lg[i, 1]),
                "logit_margin": float(lg[i, 1] - lg[i, 0]),
            })

    metrics = classification_metrics(ys, ps)
    cm = confusion_matrix_counts(ys, ps)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "loss": loss_sum / max(n, 1),
        "model_name": "LRDG-Net",
        "training_protocol": spec.protocol,
        "train_domains": list(spec.train_domains),
        "train_source": spec.train_domains[0] if len(spec.train_domains) == 1 else "+".join(spec.train_domains),
        "test_domain": str(target_domain),
        "train_seed": int(spec.seed),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_macro_f1": float(checkpoint["val_macro_f1"]),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive": int(fp),
        "true_negative": int(tn),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "clean_recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "target_domain_used_for_training_or_normalization": False,
    })

    records_df = pd.DataFrame(records)
    save_json(metrics, metrics_path)
    records_df.to_csv(result_dir / "prediction_records.csv", index=False, encoding="utf-8-sig")
    pollution_metrics(records_df).to_csv(result_dir / "pollution_metrics.csv", index=False, encoding="utf-8-sig")
    pollution_metrics(records_df, group_columns=("raw_class", "severity")).to_csv(
        result_dir / "pollution_metrics_raw_class.csv", index=False, encoding="utf-8-sig"
    )
    save_confusion_matrix(
        cm,
        result_dir / "confusion_matrix.csv",
        result_dir / "confusion_matrix.png",
        f"LRDG-Net target={target_domain}",
    )
    test_df[["domain", "raw_class", "pollution_type", "severity", "session_id", "session_uid"]].drop_duplicates().to_csv(
        result_dir / "test_sessions.csv", index=False, encoding="utf-8-sig"
    )
    return metrics
