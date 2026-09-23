"""Optional diagnostic: clean-frame LIDAROC domain linear probe on a frozen model embedding.

Set ``SINGLE_INFERENCE.model_dir`` in config.py to the frozen model directory before running.
This diagnostic never changes training, normalization, checkpoint selection, or AT128 evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from config import PATHS, SINGLE_INFERENCE, TEST
from lrdg_net.data.dataset import make_loader
from lrdg_net.evaluation.common import load_frozen_model
from lrdg_net.utils.io import save_json


@torch.no_grad()
def _extract_embeddings(model, loader, device):
    rows, embeddings = [], []
    for geometry, degradation, bag_index, labels, meta in loader:
        _, aux = model(
            geometry.to(device), degradation.to(device), bag_index.to(device), labels.shape[0], return_aux=True
        )
        embeddings.append(aux["embedding"].cpu().numpy())
        for i in range(len(labels)):
            rows.append({
                "sample_uid": meta["sample_uid"][i],
                "domain": meta["domain"][i],
                "session_uid": meta["session_uid"][i],
            })
    return pd.DataFrame(rows), np.concatenate(embeddings, axis=0)


def _session_split(frame, fraction=0.34, seed=42):
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(frame), dtype=bool)
    test_mask = np.zeros(len(frame), dtype=bool)
    for domain, group in frame.groupby("domain", sort=True):
        sessions = np.array(sorted(group["session_uid"].unique()))
        if len(sessions) < 2:
            raise RuntimeError(f"Domain {domain} has <2 clean sessions")
        rng.shuffle(sessions)
        n_test = max(1, int(round(len(sessions) * fraction)))
        n_test = min(n_test, len(sessions) - 1)
        test_sessions = set(sessions[:n_test])
        idx = group.index.to_numpy()
        local_test = group["session_uid"].isin(test_sessions).to_numpy()
        test_mask[idx] = local_test
        train_mask[idx] = ~local_test
    return train_mask, test_mask


def main():
    model_dir = Path(SINGLE_INFERENCE.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError("Set SINGLE_INFERENCE.model_dir in config.py first.")
    manifest = pd.read_csv(Path(PATHS.cache_root) / "manifest.csv", dtype={"session_id": str, "frame_id": str})
    clean = manifest[
        (manifest["pollution_type"].astype(str) == "clean")
        & (manifest["domain"].astype(str).isin(["5m", "10m", "20m"]))
    ].reset_index(drop=True)

    model, _, stats, device = load_frozen_model(model_dir)
    loader = make_loader(clean, TEST.batch_size, TEST.num_workers, False, stats)
    frame, embedding = _extract_embeddings(model, loader, device)
    train_mask, test_mask = _session_split(frame)
    label_map = {d: i for i, d in enumerate(sorted(frame["domain"].unique()))}
    y = frame["domain"].map(label_map).to_numpy()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(embedding[train_mask], y[train_mask])
    pred = clf.predict(embedding[test_mask])
    result = {
        "model_dir": str(model_dir.resolve()),
        "probe_task": "clean-frame LIDAROC domain classification",
        "domains": label_map,
        "train_frames": int(train_mask.sum()),
        "test_frames": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y[test_mask], pred)),
        "macro_f1": float(f1_score(y[test_mask], pred, average="macro", zero_division=0)),
        "note": "diagnostic only; never used for model selection",
    }
    save_json(result, model_dir / "domain_probe_metrics.json")
    print(result)


if __name__ == "__main__":
    main()
