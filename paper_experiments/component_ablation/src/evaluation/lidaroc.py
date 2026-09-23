from pathlib import Path
import json
import pandas as pd
import torch
from paper_experiments.component_ablation.config import MODEL, PATHS, TEST, TRAIN
from paper_experiments.component_ablation.src.data.dataset import make_loader
from paper_experiments.component_ablation.src.data.normalization import load_normalization_stats
from paper_experiments.component_ablation.src.evaluation.metrics import classification_metrics, confusion_matrix_counts
from paper_experiments.component_ablation.src.models.r_component import build_r_component_model, VARIANT_SPECS
from paper_experiments.component_ablation.src.utils.experiment import model_dir, eval_dir
from paper_experiments.component_ablation.src.utils.reproducibility import resolve_device


@torch.no_grad()
def evaluate_variant(variant, source, seed, target):
    out = eval_dir(variant, source, seed, target)
    mp = out / "test_metrics.json"
    if mp.exists() and not TRAIN.overwrite_models:
        return json.loads(mp.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    md = model_dir(variant, source, seed)
    ck = torch.load(md / "best_model.pt", map_location="cpu", weights_only=False)
    stats = load_normalization_stats(md / "normalization_stats.json")
    dev = resolve_device(TRAIN.device)
    model = build_r_component_model(variant, MODEL).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()

    df = pd.read_csv(Path(PATHS.cache_root) / "manifest.csv", dtype={"session_id": str, "frame_id": str})
    test = df[df.domain.astype(str) == str(target)].reset_index(drop=True)
    loader = make_loader(test, TEST.batch_size, TEST.num_workers, False, stats, variant)

    ys, ps = [], []
    for x, bi, y, _ in loader:
        x, bi = x.to(dev), bi.to(dev)
        logits = model(x, bi, y.shape[0])
        ys += y.tolist()
        ps += logits.argmax(1).cpu().tolist()

    met = classification_metrics(ys, ps)
    cm = confusion_matrix_counts(ys, ps)
    tn, fp, fn, tp = cm.ravel()
    spec = VARIANT_SPECS[variant]
    met.update({
        "variant": variant,
        "study": spec["study"],
        "paper_label": spec["paper_label"],
        "train_source": source,
        "test_domain": target,
        "train_seed": seed,
        "input_dim": len(spec["indices"]),
        "macro_f1": float(met["macro_f1"]),
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    mp.write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding="utf-8")
    return met
