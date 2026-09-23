from pathlib import Path
import pandas as pd,torch
import torch.nn.functional as F
from paper_experiments.sensitivity.config import MODEL,PATHS,TRAIN
from paper_experiments.sensitivity.src.data.dataset import make_loader,setting_id
from paper_experiments.sensitivity.src.data.normalization import fit_normalization_stats,save_normalization_stats
from paper_experiments.sensitivity.src.data.splitter import strict_session_train_val_split,save_train_val_sessions
from paper_experiments.sensitivity.src.evaluation.metrics import classification_metrics
from paper_experiments.sensitivity.src.models.lrdg_net import LRDGNet
from paper_experiments.sensitivity.src.training.checkpoint import better,save
from paper_experiments.sensitivity.src.utils.experiment import model_dir,status,reset,snapshot
from paper_experiments.sensitivity.src.utils.reproducibility import resolve_device,set_training_seed

def build_model(): return LRDGNet(MODEL.geometry_in_channels,MODEL.degradation_in_channels,MODEL.geometry_hidden,MODEL.degradation_hidden,MODEL.instance_channels,MODEL.classifier_hidden,MODEL.dropout,MODEL.num_classes)
def train_setting(v,r,source,seed):
    out=model_dir(v,r,source,seed);st=status(out)
    if TRAIN.overwrite_models:reset(out)
    elif st=="trained": print(f"Reuse model: {out}");return out
    elif st=="incomplete":raise RuntimeError(f"Incomplete model dir: {out}")
    else:out.mkdir(parents=True,exist_ok=True)
    mp=Path(PATHS.cache_root)/setting_id(v,r)/"manifest.csv"
    if not mp.exists():raise FileNotFoundError(f"Missing cache: {mp}")
    df=pd.read_csv(mp,dtype={"session_id":str,"frame_id":str});src=df[df.domain.astype(str)==str(source)].reset_index(drop=True);tr,va=strict_session_train_val_split(src,TRAIN.val_ratio,TRAIN.split_seed);save_train_val_sessions(tr,va,out/"split_sessions.csv")
    stats=fit_normalization_stats(tr);save_normalization_stats(stats,out/"normalization_stats.json");snap=snapshot(v,r,source,seed,out);set_training_seed(seed);dev=resolve_device(TRAIN.device);m=build_model().to(dev);o=torch.optim.AdamW(m.parameters(),lr=TRAIN.learning_rate,weight_decay=TRAIN.weight_decay);tl=make_loader(tr,TRAIN.batch_size,TRAIN.num_workers,True,stats);vl=make_loader(va,TRAIN.batch_size,TRAIN.num_workers,False,stats)
    bf,bl,stale=-1.0,float("inf"),0;hist=[]
    for ep in range(1,TRAIN.epochs+1):
        m.train();ls=n=0
        for g,d,bi,y,_ in tl:
            g,d,bi,y=g.to(dev),d.to(dev),bi.to(dev),y.to(dev);o.zero_grad(set_to_none=True);log=m(g,d,bi,y.shape[0]);loss=F.cross_entropy(log,y);loss.backward();o.step();ls+=loss.item()*y.numel();n+=y.numel()
        m.eval();ys=[];ps=[];vls=vn=0
        with torch.no_grad():
            for g,d,bi,y,_ in vl:
                g,d,bi,y=g.to(dev),d.to(dev),bi.to(dev),y.to(dev);log=m(g,d,bi,y.shape[0]);loss=F.cross_entropy(log,y);ys+=y.cpu().tolist();ps+=log.argmax(1).cpu().tolist();vls+=loss.item()*y.numel();vn+=y.numel()
        met=classification_metrics(ys,ps);trl=ls/max(n,1);vall=vls/max(vn,1);hist.append({"epoch":ep,"train_loss":trl,"val_loss":vall,"val_macro_f1":met["macro_f1"]});print(f"{setting_id(v,r)} {source} seed{seed} | Epoch {ep:03d} | val_macro_f1={met['macro_f1']:.5f}")
        if better(met["macro_f1"],vall,bf,bl):bf,bl,stale=float(met["macro_f1"]),float(vall),0;save(out/"best_model.pt",m,o,ep,bf,bl,snap)
        else:
            stale+=1
            if stale>=TRAIN.early_stopping_patience:break
    pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False,encoding="utf-8-sig");return out
