import numpy as np
import pandas as pd

def strict_session_train_val_split(df: pd.DataFrame, val_ratio: float, split_seed: int):
  rng=np.random.default_rng(split_seed); train_parts=[]; val_parts=[]
  for _, g in df.groupby(["domain","raw_class","severity"], sort=True):
    sessions=g["session_uid"].drop_duplicates().astype(str).to_numpy()
    if len(sessions)<2:
      train_parts.append(g); continue
    s=sessions.copy(); rng.shuffle(s)
    n=max(1,int(round(len(s)*val_ratio))); n=min(n,len(s)-1)
    val=set(s[:n].tolist()); mask=g["session_uid"].astype(str).isin(val)
    train_parts.append(g[~mask]); val_parts.append(g[mask])
  if not val_parts: raise RuntimeError("Cannot create session-level validation split.")
  tr=pd.concat(train_parts,ignore_index=True).sample(frac=1,random_state=split_seed).reset_index(drop=True)
  va=pd.concat(val_parts,ignore_index=True).sample(frac=1,random_state=split_seed).reset_index(drop=True)
  overlap=set(tr.session_uid)&set(va.session_uid)
  if overlap: raise RuntimeError(f"Session leakage: {list(overlap)[:3]}")
  return tr,va
