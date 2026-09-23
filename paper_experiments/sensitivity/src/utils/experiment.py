from pathlib import Path
import json,shutil
from paper_experiments.sensitivity.config import PATHS
def sid(v,r): return f"v{v:.2f}_r{r}".replace(".","p")
def model_dir(v,r,source,seed): return Path(PATHS.model_root)/sid(v,r)/f"LRDG_{sid(v,r)}_SRC_{source}_seed{seed}"
def eval_dir(v,r,source,seed,target): return Path(PATHS.evaluation_root)/sid(v,r)/f"LRDG_{sid(v,r)}_SRC_{source}_seed{seed}"/f"to_{target}"
def status(p):
    p=Path(p);req=["config.json","split_sessions.csv","normalization_stats.json","best_model.pt","training_history.csv"]
    if not p.exists():return "missing"
    return "trained" if all((p/x).exists() for x in req) else "incomplete"
def reset(p):
    if Path(p).exists():shutil.rmtree(p)
    Path(p).mkdir(parents=True,exist_ok=True)
def snapshot(v,r,source,seed,p):
    s={"model":"LRDG-Net","voxel_size":v,"context_radius_voxels":r,"context_side_voxels":2*r+1,"source_domain":source,"seed":seed,"target_used":False,"pooling":"mean","descriptor":"final 12D Gaussian + 4D relative degradation"}
    Path(p,"config.json").write_text(json.dumps(s,indent=2),encoding="utf-8");return s
