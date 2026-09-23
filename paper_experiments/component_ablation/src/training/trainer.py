from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from paper_experiments.component_ablation.config import MODEL, PATHS, TRAIN
from paper_experiments.component_ablation.src.data.dataset import make_loader
from paper_experiments.component_ablation.src.data.normalization import fit_normalization_stats, save_normalization_stats
from paper_experiments.component_ablation.src.data.splitter import strict_session_train_val_split, save_train_val_sessions
from paper_experiments.component_ablation.src.evaluation.metrics import classification_metrics
from paper_experiments.component_ablation.src.models.r_component import build_r_component_model
from paper_experiments.component_ablation.src.training.checkpoint import is_better_checkpoint, save_best_checkpoint
from paper_experiments.component_ablation.src.utils.experiment import model_dir, status, reset_dir, save_snapshot
from paper_experiments.component_ablation.src.utils.reproducibility import resolve_device, set_training_seed


def _manifest():
    p = Path(PATHS.cache_root) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError("Run run_01_preprocess.py first.")
    return pd.read_csv(p, dtype={"session_id": str, "frame_id": str})


def _move(batch, dev):
    x, bi, y, meta = batch
    return x.to(dev), bi.to(dev), y.to(dev), meta


def train_variant(variant, source, seed):
    out = model_dir(variant, source, seed)
    st = status(out)
    if TRAIN.overwrite_models:
        reset_dir(out)
    elif st == "trained":
        print(f"Reuse model: {out}")
        return out
    elif st == "incomplete":
        raise RuntimeError(f"Incomplete model dir: {out}. Set overwrite_models=True to rebuild.")
    else:
        out.mkdir(parents=True, exist_ok=True)

    df = _manifest()
    src = df[df.domain.astype(str) == str(source)].reset_index(drop=True)
    tr, va = strict_session_train_val_split(src, TRAIN.val_ratio, TRAIN.split_seed)
    save_train_val_sessions(tr, va, out / "split_sessions.csv")
    stats = fit_normalization_stats(tr, variant)
    save_normalization_stats(stats, out / "normalization_stats.json")
    snap = save_snapshot(variant, source, seed, out)

    set_training_seed(seed)
    dev = resolve_device(TRAIN.device)
    model = build_r_component_model(variant, MODEL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay)
    tl = make_loader(tr, TRAIN.batch_size, TRAIN.num_workers, True, stats, variant)
    vl = make_loader(va, TRAIN.batch_size, TRAIN.num_workers, False, stats, variant)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n{variant} | source={source} seed={seed} | input_dim={len(stats.selected_indices)} "
        f"| params={params:,} | train={len(tr)} val={len(va)}"
    )

    best_f1, best_loss, stale = -1.0, float("inf"), 0
    hist = []
    for ep in range(1, TRAIN.epochs + 1):
        model.train()
        loss_sum = n = 0
        for batch in tl:
            x, bi, y, _ = _move(batch, dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x, bi, y.shape[0])
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y.numel()
            n += y.numel()

        model.eval()
        ys, ps = [], []
        vloss = vn = 0
        with torch.no_grad():
            for batch in vl:
                x, bi, y, _ = _move(batch, dev)
                logits = model(x, bi, y.shape[0])
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(1)
                ys += y.cpu().tolist()
                ps += pred.cpu().tolist()
                vloss += float(loss.item()) * y.numel()
                vn += y.numel()

        met = classification_metrics(ys, ps)
        trloss = loss_sum / max(n, 1)
        val_loss = vloss / max(vn, 1)
        hist.append({
            "epoch": ep,
            "train_loss": trloss,
            "val_loss": val_loss,
            "val_macro_f1": met["macro_f1"],
        })
        print(
            f"Epoch {ep:03d} | train_loss={trloss:.5f} | val_loss={val_loss:.5f} "
            f"| val_macro_f1={met['macro_f1']:.5f}"
        )
        if is_better_checkpoint(met["macro_f1"], val_loss, best_f1, best_loss):
            best_f1, best_loss, stale = float(met["macro_f1"]), float(val_loss), 0
            save_best_checkpoint(out / "best_model.pt", model, opt, ep, best_f1, best_loss, snap)
        else:
            stale += 1
            if stale >= TRAIN.early_stopping_patience:
                print("Early stopping.")
                break

    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    return out
