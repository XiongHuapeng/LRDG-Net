from pathlib import Path
import json,shutil
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from paper_experiments.sensitivity.config import DATA
from paper_experiments.sensitivity.src.data.local_descriptors import extract_local_descriptors,load_bin

META_COLS=["sample_uid","source_path","domain","raw_class","pollution_type","severity","session_id","frame_id","session_uid"]

def setting_id(voxel_size,context_radius): return f"v{voxel_size:.2f}_r{context_radius}".replace(".","p")

def build_feature_cache(raw_dataframe,cache_root,voxel_size,context_radius):
    root=Path(cache_root)/setting_id(voxel_size,context_radius);root.mkdir(parents=True,exist_ok=True);feat=root/"features";sig=root/"preprocess_config.json"
    cur={"voxel_size":voxel_size,"context_radius_voxels":context_radius,"intensity_scale":DATA.intensity_scale,"relative_std_eps":DATA.relative_std_eps,"formula":"paper_LRDG_v1"}
    if DATA.overwrite_cache and feat.exists(): shutil.rmtree(feat)
    if sig.exists() and not DATA.overwrite_cache and json.loads(sig.read_text(encoding="utf-8"))!=cur: raise RuntimeError(f"Cache signature mismatch: {root}")
    feat.mkdir(parents=True,exist_ok=True);sig.write_text(json.dumps(cur,indent=2),encoding="utf-8")
    rows=[]
    for row in tqdm(raw_dataframe.to_dict("records"),desc=f"Sensitivity preprocess {setting_id(voxel_size,context_radius)}"):
        p=feat/(row["sample_uid"].replace("|","__")+".npz")
        if p.exists() and not DATA.overwrite_cache:
            with np.load(p,allow_pickle=False) as z: n=int(z["geometry"].shape[0]);pc=int(z["point_count"])
        else:
            pts=load_bin(row["source_path"]);d=extract_local_descriptors(pts,voxel_size,context_radius,DATA.intensity_scale,DATA.relative_std_eps);n=d["geometry"].shape[0];pc=pts.shape[0]
            np.savez_compressed(p,geometry=d["geometry"],relative_degradation=d["relative_degradation"],point_count=np.asarray(pc,np.int64))
        rows.append({**row,"cache_path":str(p.resolve()),"point_count":pc,"instance_count":n})
    man=pd.DataFrame(rows);man.to_csv(root/"manifest.csv",index=False,encoding="utf-8-sig");return man

class LRDGDataset(Dataset):
    def __init__(self,df,stats): self.df=df.reset_index(drop=True);self.stats=stats
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with np.load(r.cache_path,allow_pickle=False) as z:g=z["geometry"].astype(np.float32);d=z["relative_degradation"].astype(np.float32)
        if self.stats is not None:g=self.stats.normalize_geometry(g);d=self.stats.normalize_degradation(d)
        return torch.from_numpy(g),torch.from_numpy(d),torch.tensor(int(r.label),dtype=torch.long),{k:str(r[k]) for k in META_COLS}
def collate_bags(b):
    gs,ds,ys,ms=zip(*b);counts=[len(x) for x in gs];g=torch.cat(gs);d=torch.cat(ds);y=torch.stack(ys);bi=torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(counts)])
    meta={k:[m[k] for m in ms] for k in ms[0]};meta["instance_count"]=counts;return g,d,bi,y,meta
def make_loader(df,batch_size,num_workers,shuffle,stats):
    return DataLoader(LRDGDataset(df,stats),batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=torch.cuda.is_available(),collate_fn=collate_bags)
