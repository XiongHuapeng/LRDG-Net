"""
Unified LRDG-Net trainer / 统一训练器。

The scientific model follows the manuscript. The implementation separates
"training a model" from "evaluating a target domain", so one source+seed model is trained once
and may then be evaluated on multiple target domains without redundant retraining.
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from config import MODEL, PATHS, TRAIN
from lrdg_net.data.dataset import make_loader
from lrdg_net.data.normalization import NORMALIZATION_FILENAME, fit_normalization_stats, save_normalization_stats
from lrdg_net.data.splitter import save_train_val_sessions, strict_session_train_val_split
from lrdg_net.evaluation.metrics import classification_metrics
from lrdg_net.models.lrdg_net import LRDGNet
from lrdg_net.training.checkpoint import is_better_checkpoint, save_best_checkpoint
from lrdg_net.utils.experiment import TrainingSpec, model_directory, model_status, reset_directory, save_training_snapshot
from lrdg_net.utils.reproducibility import resolve_device, set_training_seed


def build_model() -> LRDGNet:
    return LRDGNet(
        geometry_in_channels=MODEL.geometry_in_channels,
        degradation_in_channels=MODEL.degradation_in_channels,
        geometry_hidden=MODEL.geometry_hidden,
        degradation_hidden=MODEL.degradation_hidden,
        instance_channels=MODEL.instance_channels,
        classifier_hidden=MODEL.classifier_hidden,
        dropout=MODEL.dropout,
        num_classes=MODEL.num_classes,
    )


def _move_batch(batch, device):
    geometry, degradation, bag_index, labels, meta = batch
    return (
        geometry.to(device, non_blocking=True),
        degradation.to(device, non_blocking=True),
        bag_index.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        meta,
    )


def _train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, n = 0.0, 0
    for batch in loader:
        geometry, degradation, bag_index, labels, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(geometry, degradation, bag_index, labels.shape[0])
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * labels.numel()
        n += labels.numel()
    return loss_sum / max(n, 1)


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        geometry, degradation, bag_index, labels, _ = _move_batch(batch, device)
        logits = model(geometry, degradation, bag_index, labels.shape[0])
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        loss_sum += float(loss.item()) * labels.numel()
        n += labels.numel()
    metrics = classification_metrics(ys, ps)
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics


def _load_lidaroc_manifest() -> pd.DataFrame:
    manifest_path = Path(PATHS.cache_root) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run run_01_preprocess_lidaroc.py first / 请先运行 run_01_preprocess_lidaroc.py: {manifest_path}"
        )
    return pd.read_csv(manifest_path, dtype={"session_id": str, "frame_id": str})


def run_training(spec: TrainingSpec, overwrite: bool | None = None) -> Path:
    """Train one frozen-protocol LRDG-Net model and return its model directory."""
    overwrite = TRAIN.overwrite_models if overwrite is None else bool(overwrite)
    model_dir = model_directory(spec)
    status = model_status(model_dir)

    if overwrite:
        reset_directory(model_dir)
    elif status == "trained":
        print(f"Reuse trained model / 复用已训练模型: {model_dir}")
        return model_dir
    elif status == "incomplete":
        raise RuntimeError(
            f"Incomplete model directory / 模型目录不完整: {model_dir}\n"
            "Set TRAIN.overwrite_models=True if you intentionally want to rebuild it."
        )
    else:
        model_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_lidaroc_manifest()
    source = manifest[manifest["domain"].astype(str).isin(spec.train_domains)].reset_index(drop=True)
    found_domains = tuple(sorted(source["domain"].astype(str).unique()))
    expected_domains = tuple(sorted(spec.train_domains))
    if found_domains != expected_domains:
        raise RuntimeError(f"Training domains mismatch: expected={expected_domains}, found={found_domains}")

    train_df, val_df = strict_session_train_val_split(source, TRAIN.val_ratio, TRAIN.split_seed)
    save_train_val_sessions(train_df, val_df, model_dir / "split_sessions.csv")

    # Strict rule: fit normalization only on the training split.
    stats = fit_normalization_stats(train_df)
    save_normalization_stats(stats, model_dir / NORMALIZATION_FILENAME)
    snapshot = save_training_snapshot(spec, model_dir)

    set_training_seed(spec.seed)
    train_loader = make_loader(train_df, TRAIN.batch_size, TRAIN.num_workers, True, stats)
    val_loader = make_loader(val_df, TRAIN.batch_size, TRAIN.num_workers, False, stats)
    device = resolve_device(TRAIN.device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay
    )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 100)
    print(f"Model / 模型            : {spec.name}")
    print(f"Protocol / 协议         : {spec.protocol}")
    print(f"Training domains / 训练域: {', '.join(spec.train_domains)}")
    print(f"Device / 设备           : {device}")
    print(f"Train frames/sessions   : {len(train_df)} / {train_df['session_uid'].nunique()}")
    print(f"Val frames/sessions     : {len(val_df)} / {val_df['session_uid'].nunique()}")
    print(f"Parameters / 参数量     : {parameter_count:,}")
    print(f"Normalization instances : {stats.geometry_instances:,}")
    print("No evaluation target is loaded during training. / 训练阶段不读取任何评估目标域。")
    print("=" * 100)

    best_f1, best_loss, stale, history = -1.0, float("inf"), 0, []
    for epoch in range(1, TRAIN.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        val = _validate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_macro_f1": val["macro_f1"],
        })
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"val_loss={val['loss']:.5f} | val_macro_f1={val['macro_f1']:.5f}"
        )
        if is_better_checkpoint(val["macro_f1"], val["loss"], best_f1, best_loss):
            best_f1, best_loss, stale = float(val["macro_f1"]), float(val["loss"]), 0
            save_best_checkpoint(
                model_dir / "best_model.pt", model, optimizer, epoch, best_f1, best_loss, snapshot
            )
        else:
            stale += 1
            if stale >= TRAIN.early_stopping_patience:
                print("Early stopping / 提前停止。")
                break

    pd.DataFrame(history).to_csv(
        model_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"Training finished / 训练完成. Best Val Macro-F1={best_f1:.6f}, "
        f"loss={best_loss:.6f}\nModel dir: {model_dir}"
    )
    return model_dir
