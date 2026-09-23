"""Unified baseline experiment entry points."""
from pathlib import Path
import pandas as pd
import torch
from paper_experiments.baseline.config import BASELINES, DATA, PATHS, TRAIN, ensure_dirs
from paper_experiments.baseline.src.data.metadata import scan_raw_dataset
from paper_experiments.baseline.src.data.preprocess import build_cache
from paper_experiments.baseline.src.models.factory import make_model
from paper_experiments.baseline.src.training.runner import train_one, test_one

KEYS = ["model", "train_source", "test_domain", "seed"]


def smoke_test():
    print("LIDAROC:", PATHS.lidaroc_data_root)
    print("Unified workspace:", PATHS.workspace_root)
    print("Baseline cache:", PATHS.cache_root)
    print("Baseline models:", PATHS.model_root)
    print("Baseline evaluations:", Path(PATHS.eval_root) / "baseline_cross_domain")
    if not Path(PATHS.lidaroc_data_root).exists():
        print("WARNING: edit PATHS.lidaroc_data_root in root config.py before preprocessing.")

    def params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    m = make_model("globalstats").eval()
    with torch.no_grad(): y = m(torch.randn(4, 29))
    print("Global-Statistics MLP", tuple(y.shape), "params", params(m))

    h, w = int(DATA.rangeview_h), int(DATA.rangeview_w)
    m = make_model("rangenet").eval()
    with torch.no_grad(): y = m(torch.randn(2, 5, h, w))
    print("RangeNet-style", tuple(y.shape), "params", params(m))

    x = torch.randn(2, min(DATA.num_points, 256), 4)
    for name, label in [
        ("pointnet", "PointNet"),
        ("pointnet2_ref", "PointNet++"),
        ("dgcnn", "DGCNN"),
        ("pointnext", "PointNeXt-S-style"),
    ]:
        k = min(BASELINES.dgcnn_k, x.size(1) - 1)
        m = make_model(name, k=k).eval()
        with torch.no_grad(): y = m(x)
        print(label, tuple(y.shape), "params", params(m))
    try:
        m = make_model("autogran")
        print("AutoGrAN params", params(m))
    except ModuleNotFoundError as exc:
        print("AutoGrAN check skipped until torch-geometric is installed:", exc)
    print("Frozen baseline IDs:", BASELINES.baselines)


def preprocess():
    ensure_dirs()
    raw = scan_raw_dataset(PATHS.lidaroc_data_root, DATA.target_classes)
    print(raw.groupby(["domain", "pollution_type"]).size())
    out = build_cache(raw)
    print("Done:", len(out), "frames; manifest:", Path(PATHS.cache_root) / "manifest.csv")


def _save_and_summarize(rows, out):
    runs = pd.DataFrame(list(rows.values()))
    if runs.empty:
        return pd.DataFrame()
    runs = runs[runs["model"].isin(BASELINES.baselines)].copy()
    runs = runs.sort_values(KEYS).reset_index(drop=True)
    runs.to_csv(out / "baseline_runs.csv", index=False, encoding="utf-8-sig")

    direction = runs.groupby(["model", "train_source", "test_domain"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        fnr_mean=("fnr", "mean"),
        fpr_mean=("fpr", "mean"),
        params=("params", "first"),
        runs=("seed", "count"),
    )
    direction.to_csv(out / "baseline_direction_summary.csv", index=False, encoding="utf-8-sig")

    summary = direction.groupby("model", as_index=False).agg(
        mean_macro_f1=("macro_f1_mean", "mean"),
        worst_macro_f1=("macro_f1_mean", "min"),
        direction_std=("macro_f1_mean", "std"),
        mean_fnr=("fnr_mean", "mean"),
        mean_fpr=("fpr_mean", "mean"),
        params=("params", "first"),
    )
    summary.to_csv(out / "baseline_model_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def run_cross_domain():
    ensure_dirs()
    mp = Path(PATHS.cache_root) / "manifest.csv"
    if not mp.exists():
        raise FileNotFoundError("Run run_11_baseline_preprocess.py first.")
    manifest = pd.read_csv(mp)
    out = Path(PATHS.eval_root) / "baseline_cross_domain"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    runs_path = out / "baseline_runs.csv"
    if runs_path.exists():
        old = pd.read_csv(runs_path)
        for r in old.to_dict("records"):
            rows[tuple(r[k] for k in KEYS)] = r
        print(f"Loaded {len(rows)} existing baseline evaluation rows.")

    total = len(BASELINES.baselines) * len(TRAIN.domains) * len(TRAIN.seeds)
    counter = 0
    for name in BASELINES.baselines:
        for source in TRAIN.domains:
            for seed in TRAIN.seeds:
                counter += 1
                print(f"\n=== [{counter}/{total}] {name} source={source} seed={seed} ===")
                train_one(name, source, seed, manifest)
                for target in TRAIN.domains:
                    if target == source:
                        continue
                    r = test_one(name, source, target, seed, manifest)
                    rows[tuple(r[k] for k in KEYS)] = r
                    print(r)
                    _save_and_summarize(rows, out)
    summary = _save_and_summarize(rows, out)
    print("\nBASELINE SUMMARY\n", summary.to_string(index=False))
    return summary


def make_paper_table():
    base = Path(PATHS.eval_root) / "baseline_cross_domain"
    bd = pd.read_csv(base / "baseline_direction_summary.csv")
    bd["direction"] = bd.train_source.astype(str) + "→" + bd.test_domain.astype(str)
    piv = bd.pivot(index="model", columns="direction", values="macro_f1_mean").reset_index()
    sm = pd.read_csv(base / "baseline_model_summary.csv")
    table = piv.merge(sm, on="model", how="left")

    lr = Path(PATHS.lrdg_workspace_root) / "evaluations" / "lidaroc_cross_domain" / "cross_domain_direction_summary.csv"
    if lr.exists():
        d = pd.read_csv(lr)
        d["direction"] = d.train_source.astype(str) + "→" + d.test_domain.astype(str)
        row = {"model": "LRDG-Net"}
        for _, r in d.iterrows(): row[r.direction] = r.macro_f1_mean
        row.update(
            mean_macro_f1=float(d.macro_f1_mean.mean()),
            worst_macro_f1=float(d.macro_f1_mean.min()),
            direction_std=float(d.macro_f1_mean.std()),
            mean_fnr=float(d.fnr_mean.mean()),
            mean_fpr=float(d.fpr_mean.mean()),
            params=10898,
        )
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        print("NOTE: final LRDG-Net result not found:", lr)

    table["model"] = table["model"].map(lambda x: BASELINES.display_names.get(x, x))
    order = [
        "Global-Statistics MLP", "RangeNet-style", "PointNet", "PointNet++", "DGCNN",
        "PointNeXt-S-style", "AutoGrAN", "LRDG-Net",
    ]
    cols = [
        "model", "5m→10m", "5m→20m", "10m→5m", "10m→20m", "20m→5m", "20m→10m",
        "mean_macro_f1", "worst_macro_f1", "direction_std", "mean_fnr", "mean_fpr", "params",
    ]
    for c in cols:
        if c not in table.columns: table[c] = float("nan")
    rank = {name: i for i, name in enumerate(order)}
    table = table[cols]
    table["_order"] = table.model.map(lambda x: rank.get(x, 999))
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out = base / "paper_baseline_comparison.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print("\nSaved:", out)
    return table
