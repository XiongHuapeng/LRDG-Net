from dataclasses import dataclass,asdict
from pathlib import Path
import json,numpy as np,pandas as pd
from paper_experiments.sensitivity.config import DATA

@dataclass
class NormalizationStats:
    geometry_mean:list; geometry_std:list; degradation_mean:list; degradation_std:list
    source_train_frames:int; geometry_instances:int; degradation_instances:int
    fit_scope:str="source_training_split_only"; target_domain_used:bool=False
    def normalize_geometry(self,x): return ((x-np.asarray(self.geometry_mean,np.float32))/np.asarray(self.geometry_std,np.float32)).astype(np.float32)
    def normalize_degradation(self,x): return ((x-np.asarray(self.degradation_mean,np.float32))/np.asarray(self.degradation_std,np.float32)).astype(np.float32)

def _stream(paths,key):
    n=0;s=None;ss=None
    for p in paths:
        with np.load(p,allow_pickle=False) as z: x=z[key].astype(np.float64)
        if s is None: s=np.zeros(x.shape[1]);ss=np.zeros(x.shape[1])
        n+=x.shape[0];s+=x.sum(0);ss+=np.square(x).sum(0)
    mean=s/n;var=np.maximum(ss/n-mean*mean,0);std=np.sqrt(var);std=np.where(std<DATA.normalization_eps,1.0,std)
    return mean.tolist(),std.tolist(),int(n)
def fit_normalization_stats(df):
    paths=df.cache_path.tolist();gm,gs,gn=_stream(paths,"geometry");dm,ds,dn=_stream(paths,"relative_degradation")
    return NormalizationStats(gm,gs,dm,ds,len(df),gn,dn)
def save_normalization_stats(s,p): Path(p).write_text(json.dumps(asdict(s),indent=2),encoding="utf-8")
def load_normalization_stats(p): return NormalizationStats(**json.loads(Path(p).read_text(encoding="utf-8")))
