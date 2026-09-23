from pathlib import Path
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from paper_experiments.baseline.config import BASELINES, PATHS, TRAIN
from paper_experiments.baseline.src.data.datasets import PointDataset, GraphDataset, GlobalStatsDataset, RangeViewDataset
from paper_experiments.baseline.src.data.splitter import strict_session_train_val_split
from paper_experiments.baseline.src.evaluation.metrics import compute_metrics
from paper_experiments.baseline.src.models.factory import make_model
from paper_experiments.baseline.src.utils.repro import seed_all


def device():
  return torch.device(TRAIN.device if TRAIN.device == "cpu" or torch.cuda.is_available() else "cpu")


def params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loader(name, df, train):
  if name == "autogran":
    from torch_geometric.loader import DataLoader as PyGLoader
    return PyGLoader(
      GraphDataset(df), batch_size=BASELINES.autogran_batch_size,
      shuffle=train, num_workers=TRAIN.num_workers,
      drop_last=bool(train),
    )

  if name == "globalstats":
    ds = GlobalStatsDataset(df); bs = BASELINES.globalstats_batch_size
  elif name == "rangenet":
    ds = RangeViewDataset(df); bs = BASELINES.rangenet21_batch_size
  else:
    ds = PointDataset(df)
    if name == "pointnet": bs = BASELINES.pointnet_batch_size
    elif name == "pointnet2_ref": bs = BASELINES.pointnet2_batch_size
    elif name == "dgcnn": bs = BASELINES.dgcnn_batch_size
    elif name == "pointnext": bs = BASELINES.pointnext_batch_size
    else: raise ValueError(name)

  return DataLoader(
    ds, batch_size=bs, shuffle=train, num_workers=TRAIN.num_workers,
    drop_last=bool(train), pin_memory=False,
  )


def _step(model, batch, name, dev):
  if name == "autogran":
    batch = batch.to(dev)
    return model(batch), batch.y.view(-1)
  x, y, _ = batch
  x = x.to(dev, non_blocking=False)
  y = y.to(dev, non_blocking=False)
  return model(x), y


def evaluate(model, loader, name, dev):
  model.eval()
  ys, ps = [], []
  loss_sum = 0.0
  n = 0
  ce = torch.nn.CrossEntropyLoss()
  with torch.no_grad():
    for b in loader:
      logits, y = _step(model, b, name, dev)
      loss = ce(logits, y)
      pred = logits.argmax(1)
      ys.extend(y.cpu().tolist())
      ps.extend(pred.cpu().tolist())
      loss_sum += loss.detach().item() * len(y)
      n += len(y)
  if n == 0:
    raise RuntimeError(f"Empty evaluation loader for {name}")
  m = compute_metrics(ys, ps)
  m["loss"] = loss_sum / n
  return m


def _optimizer(name, model):
  # Keep the training policy of the previously frozen baseline project.
  # Only AutoGrAN and PointNeXt retain their pre-existing method-specific optimizer choices.
  if name == "autogran":
    return torch.optim.Adam(
      model.parameters(), lr=BASELINES.autogran_lr,
      weight_decay=BASELINES.autogran_weight_decay,
    )
  if name == "pointnext" and BASELINES.pointnext_optimizer.lower() == "adamw":
    return torch.optim.AdamW(
      model.parameters(), lr=BASELINES.generic_lr,
      weight_decay=BASELINES.generic_weight_decay,
    )
  # GlobalStats / RangeNet / PointNet / PointNet++ / DGCNN
  # all follow the same generic baseline optimizer.
  return torch.optim.Adam(
    model.parameters(), lr=BASELINES.generic_lr,
    weight_decay=BASELINES.generic_weight_decay,
  )


def _max_epochs(name):
  return int(BASELINES.autogran_max_epochs if name == "autogran" else TRAIN.max_epochs)

def train_one(name, source, seed, manifest):
  seed_all(seed)
  dev = device()
  out = Path(PATHS.model_root) / name / f"SRC_{source}_seed{seed}"
  out.mkdir(parents=True, exist_ok=True)
  best = out / "best_model.pt"
  done = out / "done.json"
  if done.exists() and best.exists() and not TRAIN.overwrite_models:
    print(f"SKIP existing: {out}")
    return out

  sdf = manifest[manifest.domain == source].reset_index(drop=True)
  tr, va = strict_session_train_val_split(sdf, TRAIN.val_ratio, TRAIN.split_seed)
  split = pd.concat([tr.assign(split="train"), va.assign(split="val")])
  split[["split", "session_uid", "domain", "raw_class", "severity"]].drop_duplicates().to_csv(
    out / "split_sessions.csv", index=False
  )

  model = make_model(name, BASELINES.dgcnn_k).to(dev)
  opt = _optimizer(name, model)
  max_epochs = _max_epochs(name)
  ce = torch.nn.CrossEntropyLoss()
  tl = _loader(name, tr, True)
  vl = _loader(name, va, False)

  history = []
  best_f = -1.0
  best_l = 1e30
  stale = 0
  for ep in range(1, max_epochs + 1):
    model.train()
    loss_sum = 0.0
    n = 0
    for b in tl:
      opt.zero_grad(set_to_none=True)
      logits, y = _step(model, b, name, dev)
      loss = ce(logits, y)
      loss.backward()
      opt.step()
      loss_sum += loss.detach().item() * len(y)
      n += len(y)

    vm = evaluate(model, vl, name, dev)
    improved = (vm["macro_f1"] > best_f + 1e-12) or (
      abs(vm["macro_f1"] - best_f) <= 1e-12 and vm["loss"] < best_l
    )
    if improved:
      best_f = vm["macro_f1"]
      best_l = vm["loss"]
      stale = 0
      torch.save({
        "state_dict": model.state_dict(), "model": name,
        "source": source, "seed": seed, "params": params(model),
      }, best)
    else:
      stale += 1

    history.append({
      "epoch": ep,
      "lr": float(opt.param_groups[0]["lr"]),
      "train_loss": loss_sum / max(n, 1),
      **{f"val_{k}": v for k, v in vm.items()},
    })
    print(f"[{name} {source} seed={seed}] epoch {ep:02d} valF1={vm['macro_f1']:.4f} stale={stale}")
    if stale >= TRAIN.patience:
      break

  pd.DataFrame(history).to_csv(out / "history.csv", index=False)
  done.write_text(json.dumps({
    "best_val_macro_f1": best_f,
    "best_val_loss": best_l,
    "params": params(model),
    "model": name,
    "source": source,
    "seed": seed,
  }, indent=2), encoding="utf-8")
  return out


def test_one(name, source, target, seed, manifest):
  dev = device()
  model_dir = Path(PATHS.model_root) / name / f"SRC_{source}_seed{seed}"
  ck = torch.load(model_dir / "best_model.pt", map_location=dev, weights_only=False)
  model = make_model(name, BASELINES.dgcnn_k).to(dev)
  model.load_state_dict(ck["state_dict"])
  df = manifest[manifest.domain == target].reset_index(drop=True)
  m = evaluate(model, _loader(name, df, False), name, dev)
  return {
    "model": name, "train_source": source, "test_domain": target,
    "seed": seed, "params": ck.get("params", params(model)), **m,
  }
