from pathlib import Path
import pandas as pd
from paper_experiments.sensitivity.config import DATA,PATHS,SENS,ensure_dirs
from paper_experiments.sensitivity.src.data.metadata import scan_raw_dataset
from paper_experiments.sensitivity.src.data.dataset import build_feature_cache,setting_id
from paper_experiments.sensitivity.src.training.trainer import train_setting
from paper_experiments.sensitivity.src.evaluation.lidaroc import evaluate_setting

def settings():
    # One-factor-at-a-time, de-duplicated.
    s=[(v,SENS.base_context_radius) for v in SENS.voxel_sizes]
    s += [(SENS.base_voxel_size,r) for r in SENS.context_radii]
    out=[]
    for x in s:
        if x not in out:out.append(x)
    return out

def run_all():
    ensure_dirs();raw=scan_raw_dataset(PATHS.lidaroc_data_root,DATA.target_classes);rows=[]
    print("Sensitivity settings:",[(setting_id(v,r),v,2*r+1) for v,r in settings()])
    for v,r in settings():
        build_feature_cache(raw,PATHS.cache_root,v,r)
        for source in SENS.domains:
            for seed in SENS.seeds:
                train_setting(v,r,source,seed)
                for target in SENS.domains:
                    if target!=source:rows.append(evaluate_setting(v,r,source,seed,target))
    runs=pd.DataFrame(rows);root=Path(PATHS.evaluation_root);runs.to_csv(root/"sensitivity_runs.csv",index=False,encoding="utf-8-sig")
    direction=runs.groupby(["setting","voxel_size","context_radius","context_side","train_source","test_domain"],as_index=False).agg(macro_f1_mean=("macro_f1","mean"),fnr_mean=("fnr","mean"),fpr_mean=("fpr","mean"))
    direction.to_csv(root/"sensitivity_direction_summary.csv",index=False,encoding="utf-8-sig")
    summary=direction.groupby(["setting","voxel_size","context_radius","context_side"],as_index=False).agg(mean_macro_f1=("macro_f1_mean","mean"),worst_macro_f1=("macro_f1_mean","min"),direction_std=("macro_f1_mean","std"),mean_fnr=("fnr_mean","mean"),mean_fpr=("fpr_mean","mean"))
    summary["is_final_setting"]=(summary.voxel_size==SENS.base_voxel_size)&(summary.context_radius==SENS.base_context_radius)
    summary.to_csv(root/"paper_sensitivity_summary.csv",index=False,encoding="utf-8-sig")
    print("\nPaper sensitivity summary:\n",summary.to_string(index=False));return summary
