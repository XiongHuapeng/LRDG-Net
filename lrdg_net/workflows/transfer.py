"""Minimal AT128 transfer study / 最小化 AT128 迁移实验。

Scientific question only:
Does the LIDAROC-pretrained LRDG representation remain useful on the independently collected AT128 data?

Two experiments are implemented with exactly the same PCAP-level 5-fold split:
1) Frozen LIDAROC LRDG feature extractor + AT128 linear classifier.
2) AT128-only LRDG-Net trained from random initialization (control).

Important:
- A PCAP is the independent split unit; frames from one PCAP never cross train/test.
- Labels remain binary: level 0 -> clean; levels 1/2/3 -> contaminated.
- This module does not modify the LRDG-Net architecture or descriptors.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from config import AT128, PATHS, RESEARCH, TEST, TRAIN, TRANSFER, ensure_project_dirs
from lrdg_net.data.dataset import make_loader
from lrdg_net.data.normalization import fit_normalization_stats, save_normalization_stats
from lrdg_net.evaluation.common import load_frozen_model
from lrdg_net.evaluation.metrics import classification_metrics, confusion_matrix_counts
from lrdg_net.models.lrdg_net import LRDGNet
from lrdg_net.training.checkpoint import is_better_checkpoint
from lrdg_net.training.trainer import build_model
from lrdg_net.utils.experiment import all_source_spec, model_directory
from lrdg_net.utils.io import save_json
from lrdg_net.utils.reproducibility import resolve_device, set_training_seed


def _load_manifest() -> pd.DataFrame:
    path = Path(PATHS.at128_work_root) / "cache" / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"AT128 cache not found: {path}\nRun run_04_prepare_at128.py first / 请先运行 run_04_prepare_at128.py。"
        )
    df = pd.read_csv(path, dtype={"session_id": str, "frame_id": str})
    required = {"sample_uid", "recording_id", "label", "distance_m", "condition_level", "cache_path"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"AT128 manifest missing columns / AT128 manifest 缺列: {sorted(missing)}")
    return df


def _recording_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for recording_id, g in manifest.groupby("recording_id", sort=True):
        labels = g["label"].astype(int).unique()
        if len(labels) != 1:
            raise RuntimeError(f"Recording has mixed labels: {recording_id}")
        rows.append({
            "recording_id": str(recording_id),
            "label": int(labels[0]),
            "distance_m": float(g["distance_m"].iloc[0]),
            "condition_level": int(g["condition_level"].iloc[0]),
            "frames": int(len(g)),
        })
    table = pd.DataFrame(rows).sort_values(["distance_m", "condition_level", "recording_id"]).reset_index(drop=True)
    if table["label"].nunique() != 2:
        raise RuntimeError("AT128 transfer requires both clean and contaminated recordings.")
    return table


def _build_or_load_folds(
    manifest: pd.DataFrame,
    output_root: Path,
    split_mode: str = "pcap",
) -> pd.DataFrame:
    """Build the outer 5-fold protocol.

    ``pcap`` keeps the original v2 experiment: each independent recording is a group.
    ``distance`` is the stricter follow-up experiment: all four recordings from the
    same distance (0/1/2/3) are assigned to the same test fold.
    """
    if split_mode not in {"pcap", "distance"}:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    filename = (
        "at128_transfer_folds.csv"
        if split_mode == "pcap"
        else "at128_distance_transfer_folds.csv"
    )
    path = output_root / filename
    recordings = _recording_table(manifest)

    if path.exists() and not TRANSFER.overwrite:
        folds = pd.read_csv(path)
        expected = set(recordings["recording_id"].astype(str))
        found = set(folds["recording_id"].astype(str))
        valid = expected == found and folds["fold"].nunique() == TRANSFER.folds
        if valid and split_mode == "distance":
            # Scientific safety check: a distance must never appear in two folds.
            valid = bool((folds.groupby("distance_m")["fold"].nunique() == 1).all())
        if valid:
            return folds

    recordings["fold"] = -1
    if split_mode == "pcap":
        splitter = StratifiedKFold(
            n_splits=TRANSFER.folds,
            shuffle=True,
            random_state=TRANSFER.split_seed,
        )
        for fold, (_, test_idx) in enumerate(
            splitter.split(recordings, recordings["label"]),
            start=1,
        ):
            recordings.loc[test_idx, "fold"] = fold
    else:
        # There are 13 distance groups (8--20 m). Each group contains one clean
        # recording and three contaminated recordings, so splitting the distance
        # groups directly also preserves both binary classes in every outer fold.
        distances = np.array(sorted(recordings["distance_m"].unique()), dtype=float)
        if len(distances) < TRANSFER.folds:
            raise RuntimeError(
                f"Need at least {TRANSFER.folds} unique distances, got {len(distances)}."
            )
        splitter = KFold(
            n_splits=TRANSFER.folds,
            shuffle=True,
            random_state=TRANSFER.split_seed,
        )
        for fold, (_, test_idx) in enumerate(splitter.split(distances), start=1):
            test_distances = set(distances[test_idx].tolist())
            recordings.loc[recordings["distance_m"].isin(test_distances), "fold"] = fold

        if not (recordings.groupby("distance_m")["fold"].nunique() == 1).all():
            raise RuntimeError("Distance-grouped split failed: one distance crosses folds.")

    if (recordings["fold"] < 1).any():
        raise RuntimeError("Failed to assign all AT128 recordings to folds.")
    for fold in range(1, TRANSFER.folds + 1):
        labels = recordings.loc[recordings["fold"] == fold, "label"]
        if labels.nunique() != 2:
            raise RuntimeError(f"Fold {fold} does not contain both binary classes.")

    recordings.to_csv(path, index=False, encoding="utf-8-sig")
    return recordings


def _binary_metrics(y_true, y_pred, score=None) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    m = classification_metrics(y_true, y_pred)
    cm = confusion_matrix_counts(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    m.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "clean_recall": float(tn / (tn + fp)) if tn + fp else 0.0,
        "contaminated_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
    })
    if score is not None and len(np.unique(y_true)) == 2:
        score = np.asarray(score, dtype=np.float64)
        m["auroc"] = float(roc_auc_score(y_true, score))
        m["auprc"] = float(average_precision_score(y_true, score))
    return m


def _aggregate_pcap(frame_predictions: pd.DataFrame) -> pd.DataFrame:
    capture = frame_predictions.groupby("recording_id", as_index=False).agg(
        true_label=("true_label", "first"),
        fold=("fold", "first"),
        distance_m=("distance_m", "first"),
        condition_level=("condition_level", "first"),
        frames=("sample_uid", "count"),
        score=("score", "mean"),
        prob_contaminated=("prob_contaminated", "mean"),
    )
    capture["pred_label"] = (capture["prob_contaminated"] >= 0.5).astype(int)
    return capture


@torch.no_grad()
def _extract_pretrained_embeddings(model_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    model, _, stats, device = load_frozen_model(model_dir)
    loader = make_loader(manifest, TEST.batch_size, TEST.num_workers, False, stats)
    rows = []
    for geometry, degradation, bag_index, labels, meta in loader:
        geometry = geometry.to(device, non_blocking=True)
        degradation = degradation.to(device, non_blocking=True)
        bag_index = bag_index.to(device, non_blocking=True)
        logits, aux = model(geometry, degradation, bag_index, labels.shape[0], return_aux=True)
        embedding = aux["embedding"].cpu().numpy()
        for i, uid in enumerate(meta["sample_uid"]):
            row = {"sample_uid": uid, "true_label": int(labels[i].item())}
            for j in range(embedding.shape[1]):
                row[f"e{j:02d}"] = float(embedding[i, j])
            rows.append(row)
    out = pd.DataFrame(rows)
    meta_cols = ["sample_uid", "recording_id", "distance_m", "condition_level"]
    return out.merge(manifest[meta_cols], on="sample_uid", how="left", validate="one_to_one")


def _run_frozen_linear_probe(
    seed: int,
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    output_root: Path,
    split_unit: str,
) -> dict:
    model_dir = model_directory(all_source_spec(RESEARCH.all_source_domains, seed))
    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(
            f"Missing All-LIDAROC model for seed {seed}: {model_dir}\n"
            "Run run_03_train_all_lidaroc.py first."
        )
    result_dir = output_root / "frozen_linear_probe" / f"seed{seed}"
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists() and not TRANSFER.overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    result_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _extract_pretrained_embeddings(model_dir, manifest)
    fold_map = folds.set_index("recording_id")["fold"].to_dict()
    embeddings["fold"] = embeddings["recording_id"].map(fold_map).astype(int)
    feature_cols = [c for c in embeddings.columns if c.startswith("e")]

    all_records, fold_rows = [], []
    for fold in range(1, TRANSFER.folds + 1):
        train_df = embeddings[embeddings["fold"] != fold]
        test_df = embeddings[embeddings["fold"] == fold]
        clf = LogisticRegression(
            C=TRANSFER.linear_c,
            max_iter=TRANSFER.linear_max_iter,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(train_df[feature_cols].to_numpy(), train_df["true_label"].to_numpy())
        prob = clf.predict_proba(test_df[feature_cols].to_numpy())[:, 1]
        score = clf.decision_function(test_df[feature_cols].to_numpy())
        pred = (prob >= 0.5).astype(int)
        rec = test_df[["sample_uid", "recording_id", "distance_m", "condition_level", "true_label", "fold"]].copy()
        rec["score"] = score
        rec["prob_contaminated"] = prob
        rec["pred_label"] = pred
        all_records.append(rec)
        fm = _binary_metrics(rec.true_label, rec.pred_label, rec.score)
        fm.update({"fold": fold, "test_recordings": int(rec.recording_id.nunique()), "test_frames": int(len(rec))})
        fold_rows.append(fm)

    frame = pd.concat(all_records, ignore_index=True).sort_values(["fold", "recording_id", "sample_uid"])
    pcap = _aggregate_pcap(frame)
    frame_metrics = _binary_metrics(frame.true_label, frame.pred_label, frame.score)
    pcap_metrics = _binary_metrics(pcap.true_label, pcap.pred_label, pcap.score)
    metrics = {
        "method": "frozen_lidaroc_lrdg_plus_linear_probe",
        "seed": int(seed),
        "split_unit": split_unit,
        "folds": int(TRANSFER.folds),
        "pretrained_model": str(model_dir),
        "encoder_frozen": True,
        "at128_used_to_train_linear_classifier": True,
        "frame": frame_metrics,
        "pcap": pcap_metrics,
    }
    frame.to_csv(result_dir / "oof_frame_predictions.csv", index=False, encoding="utf-8-sig")
    pcap.to_csv(result_dir / "oof_pcap_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(result_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(metrics, metrics_path)
    return metrics


def _split_inner_train_val(
    outer_train: pd.DataFrame,
    seed: int,
    fold: int,
    split_mode: str = "pcap",
):
    """Create the inner train/val split used only for scratch checkpoint selection."""
    recordings = _recording_table(outer_train)
    if split_mode == "distance":
        distances = np.array(sorted(recordings["distance_m"].unique()), dtype=float)
        train_distances, val_distances = train_test_split(
            distances,
            test_size=TRANSFER.inner_val_ratio,
            random_state=int(seed + fold),
        )
        train_distances = set(map(float, train_distances))
        val_distances = set(map(float, val_distances))
        if train_distances & val_distances:
            raise RuntimeError("Inner distance split overlap detected.")
        inner_train = outer_train[outer_train["distance_m"].astype(float).isin(train_distances)].reset_index(drop=True)
        inner_val = outer_train[outer_train["distance_m"].astype(float).isin(val_distances)].reset_index(drop=True)
    else:
        train_ids, val_ids = train_test_split(
            recordings["recording_id"].to_numpy(),
            test_size=TRANSFER.inner_val_ratio,
            random_state=int(seed + fold),
            stratify=recordings["label"].to_numpy(),
        )
        train_ids, val_ids = set(map(str, train_ids)), set(map(str, val_ids))
        inner_train = outer_train[outer_train["recording_id"].astype(str).isin(train_ids)].reset_index(drop=True)
        inner_val = outer_train[outer_train["recording_id"].astype(str).isin(val_ids)].reset_index(drop=True)

    if inner_train.empty or inner_val.empty:
        raise RuntimeError("Inner train/val split produced an empty partition.")
    return inner_train, inner_val


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
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        geometry, degradation, bag_index, labels, _ = _move_batch(batch, device)
        logits = model(geometry, degradation, bag_index, labels.shape[0])
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        loss_sum += float(loss.item()) * labels.numel()
        n += labels.numel()
    m = classification_metrics(ys, ps)
    m["loss"] = loss_sum / max(n, 1)
    return m


@torch.no_grad()
def _predict_scratch(model, loader, device, fold: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for geometry, degradation, bag_index, labels, meta in loader:
        geometry = geometry.to(device, non_blocking=True)
        degradation = degradation.to(device, non_blocking=True)
        bag_index = bag_index.to(device, non_blocking=True)
        logits = model(geometry, degradation, bag_index, labels.shape[0])
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        for i, uid in enumerate(meta["sample_uid"]):
            rows.append({
                "sample_uid": uid,
                "true_label": int(labels[i].item()),
                "score": float(margin[i]),
                "prob_contaminated": float(prob[i]),
                "pred_label": int(pred[i]),
                "fold": int(fold),
            })
    return pd.DataFrame(rows)


def _run_at128_from_scratch(
    seed: int,
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    output_root: Path,
    split_mode: str,
    split_unit: str,
) -> dict:
    result_dir = output_root / "at128_from_scratch" / f"seed{seed}"
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists() and not TRANSFER.overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    result_dir.mkdir(parents=True, exist_ok=True)

    fold_map = folds.set_index("recording_id")["fold"].to_dict()
    data = manifest.copy()
    data["fold"] = data["recording_id"].map(fold_map).astype(int)
    device = resolve_device(TRAIN.device)
    all_records, fold_rows = [], []

    for fold in range(1, TRANSFER.folds + 1):
        fold_dir = result_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_train = data[data["fold"] != fold].reset_index(drop=True)
        outer_test = data[data["fold"] == fold].reset_index(drop=True)
        inner_train, inner_val = _split_inner_train_val(outer_train, seed, fold, split_mode)

        # From-scratch control is target-supervised by definition: normalization is fitted only
        # on the AT128 inner-training recordings of this fold.
        stats = fit_normalization_stats(inner_train)
        stats.fit_scope = "at128_outer_train_inner_train_only"
        stats.target_domain_used = True
        save_normalization_stats(stats, fold_dir / "normalization_stats.json")

        split_rows = []
        for name, df in (("train", inner_train), ("val", inner_val), ("test", outer_test)):
            rec_table = _recording_table(df)
            for row in rec_table.itertuples(index=False):
                split_rows.append({
                    "recording_id": str(row.recording_id),
                    "distance_m": float(row.distance_m),
                    "condition_level": int(row.condition_level),
                    "label": int(row.label),
                    "split": name,
                })
        pd.DataFrame(split_rows).to_csv(fold_dir / "recording_split.csv", index=False, encoding="utf-8-sig")

        set_training_seed(seed + fold)
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay)
        train_loader = make_loader(inner_train, TRAIN.batch_size, TRAIN.num_workers, True, stats)
        val_loader = make_loader(inner_val, TRAIN.batch_size, TRAIN.num_workers, False, stats)

        best_f1, best_loss, stale = -1.0, float("inf"), 0
        best_state = None
        history = []
        for epoch in range(1, TRAIN.epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, device)
            val = _validate(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_macro_f1": val["macro_f1"]})
            if is_better_checkpoint(val["macro_f1"], val["loss"], best_f1, best_loss):
                best_f1, best_loss, stale = float(val["macro_f1"]), float(val["loss"]), 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= TRAIN.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("AT128 scratch training failed to produce a checkpoint.")
        model.load_state_dict(best_state, strict=True)
        torch.save({"model_state": best_state, "seed": seed, "fold": fold, "best_val_macro_f1": best_f1, "best_val_loss": best_loss}, fold_dir / "best_model.pt")
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")

        test_loader = make_loader(outer_test, TEST.batch_size, TEST.num_workers, False, stats)
        rec = _predict_scratch(model, test_loader, device, fold)
        meta_cols = ["sample_uid", "recording_id", "distance_m", "condition_level"]
        rec = rec.merge(outer_test[meta_cols], on="sample_uid", how="left", validate="one_to_one")
        all_records.append(rec)
        fm = _binary_metrics(rec.true_label, rec.pred_label, rec.score)
        fm.update({"fold": fold, "test_recordings": int(rec.recording_id.nunique()), "test_frames": int(len(rec)), "best_val_macro_f1": best_f1})
        fold_rows.append(fm)

    frame = pd.concat(all_records, ignore_index=True).sort_values(["fold", "recording_id", "sample_uid"])
    pcap = _aggregate_pcap(frame)
    frame_metrics = _binary_metrics(frame.true_label, frame.pred_label, frame.score)
    pcap_metrics = _binary_metrics(pcap.true_label, pcap.pred_label, pcap.score)
    metrics = {
        "method": "at128_lrdg_from_scratch",
        "seed": int(seed),
        "split_unit": split_unit,
        "folds": int(TRANSFER.folds),
        "lidaroc_pretraining_used": False,
        "at128_used_for_training": True,
        "frame": frame_metrics,
        "pcap": pcap_metrics,
    }
    frame.to_csv(result_dir / "oof_frame_predictions.csv", index=False, encoding="utf-8-sig")
    pcap.to_csv(result_dir / "oof_pcap_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(result_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(metrics, metrics_path)
    return metrics


def _flatten_summary(metrics: dict) -> dict:
    row = {"method": metrics["method"], "seed": metrics["seed"]}
    for prefix in ("frame", "pcap"):
        for key in ("accuracy", "macro_f1", "fpr", "fnr", "clean_recall", "contaminated_recall", "auroc", "auprc"):
            if key in metrics[prefix]:
                row[f"{prefix}_{key}"] = metrics[prefix][key]
    return row


def run_at128_transfer_study(split_mode: str = "pcap") -> pd.DataFrame:
    """Run the AT128 transfer study through the provided entry point.

    Parameters
    ----------
    split_mode:
        ``"pcap"`` reproduces the v2 PCAP-grouped 5-fold experiment.
        ``"distance"`` is the stricter follow-up: all 0/1/2/3 recordings from the
        same distance stay in the same fold.
    """
    if split_mode not in {"pcap", "distance"}:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    ensure_project_dirs()
    manifest = _load_manifest()
    if split_mode == "pcap":
        output_root = Path(PATHS.evaluation_root) / "at128_transfer"
        split_unit = "independent_pcap_recording"
        protocol_label = "PCAP-grouped 5-fold"
    else:
        output_root = Path(PATHS.evaluation_root) / "at128_transfer_distance_grouped"
        split_unit = "distance_group_all_four_recordings"
        protocol_label = "distance-grouped 5-fold"

    output_root.mkdir(parents=True, exist_ok=True)
    folds = _build_or_load_folds(manifest, output_root, split_mode=split_mode)

    print("\n" + "=" * 100)
    print("AT128 transfer study / AT128 迁移实验")
    print(f"Independent PCAPs / 独立PCAP: {folds['recording_id'].nunique()}")
    print(f"Unique distances / 独立距离: {folds['distance_m'].nunique()}")
    print(f"CV protocol / 交叉验证: {protocol_label}")
    if split_mode == "distance":
        print("Constraint / 约束: 同一距离的 0/1/2/3 四个 PCAP 永远处于同一 fold。")
    print("Task / 任务: clean vs contaminated only")
    print("Methods / 方法: Frozen LIDAROC LRDG + Linear, and AT128 LRDG From Scratch")
    print("=" * 100)

    all_metrics = []
    for seed in TRANSFER.seeds:
        if TRANSFER.run_frozen_linear_probe:
            print(f"\n[Frozen linear probe] seed={seed}")
            all_metrics.append(
                _run_frozen_linear_probe(seed, manifest, folds, output_root, split_unit)
            )
        if TRANSFER.run_at128_from_scratch:
            print(f"\n[AT128 from scratch] seed={seed}")
            all_metrics.append(
                _run_at128_from_scratch(
                    seed, manifest, folds, output_root, split_mode, split_unit
                )
            )

    summary = pd.DataFrame([_flatten_summary(x) for x in all_metrics])
    summary.to_csv(output_root / "at128_transfer_summary.csv", index=False, encoding="utf-8-sig")

    aggregate_rows = []
    for method, group in summary.groupby("method", sort=True):
        row = {"method": method, "seeds": int(len(group))}
        for col in summary.columns:
            if col in {"method", "seed"}:
                continue
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_root / "at128_transfer_aggregate.csv", index=False, encoding="utf-8-sig")

    fold_summary = (
        folds.groupby("fold", as_index=False)
        .agg(
            distances=("distance_m", lambda x: ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in sorted(set(map(float, x))))),
            distance_count=("distance_m", "nunique"),
            recordings=("recording_id", "nunique"),
            clean_recordings=("label", lambda x: int((x == 0).sum())),
            contaminated_recordings=("label", lambda x: int((x == 1).sum())),
        )
    )
    fold_summary.to_csv(output_root / "fold_summary.csv", index=False, encoding="utf-8-sig")

    protocol = {
        "research_question": "Does the LIDAROC-pretrained LRDG representation transfer to AT128 under unseen-distance testing?"
        if split_mode == "distance"
        else "Does the LIDAROC-pretrained LRDG representation transfer to AT128?",
        "task": "binary clean vs contaminated",
        "condition_levels": "AT128 levels 1/2/3 are all contaminated; no severity prediction",
        "split_mode": split_mode,
        "split_unit": split_unit,
        "cv_folds": TRANSFER.folds,
        "fold_split_seed": TRANSFER.split_seed,
        "methods": [m["method"] for m in all_metrics],
        "important": "This is a target-supervised transfer experiment, not zero-shot external validation.",
    }
    save_json(protocol, output_root / "protocol.json")

    print("\nTransfer study finished / 迁移实验完成")
    print(summary.to_string(index=False))
    print(f"\nOutputs / 输出: {output_root}")
    return summary
