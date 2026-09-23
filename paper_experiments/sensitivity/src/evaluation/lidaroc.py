from pathlib import Path
import json,pandas as pd,torch
from paper_experiments.sensitivity.config import MODEL,PATHS,TEST,TRAIN
from paper_experiments.sensitivity.src.data.dataset import make_loader,setting_id
from paper_experiments.sensitivity.src.data.normalization import load_normalization_stats
from paper_experiments.sensitivity.src.evaluation.metrics import classification_metrics,confusion_matrix_counts
from paper_experiments.sensitivity.src.models.lrdg_net import LRDGNet
from paper_experiments.sensitivity.src.utils.experiment import model_dir,eval_dir
from paper_experiments.sensitivity.src.utils.reproducibility import resolve_device

def build_model(): return LRDGNet(MODEL.geometry_in_channels,MODEL.degradation_in_channels,MODEL.geometry_hidden,MODEL.degradation_hidden,MODEL.instance_channels,MODEL.classifier_hidden,MODEL.dropout,MODEL.num_classes)
@torch.no_grad()
def evaluate_setting(v,r,source,seed,target):
    out=eval_dir(v,r,source,seed,target);mp=out/"test_metrics.json"
    if mp.exists() and not TRAIN.overwrite_models:return json.loads(mp.read_text(encoding="utf-8"))
    out.mkdir(parents=True,exist_ok=True);md=model_dir(v,r,source,seed);ck=torch.load(md/"best_model.pt",map_location="cpu",weights_only=False);stats=load_normalization_stats(md/"normalization_stats.json");dev=resolve_device(TRAIN.device);m=build_model().to(dev);m.load_state_dict(ck["model_state"]);m.eval();df=pd.read_csv(Path(PATHS.cache_root)/setting_id(v,r)/"manifest.csv",dtype={"session_id":str,"frame_id":str});test=df[df.domain.astype(str)==str(target)].reset_index(drop=True);loader=make_loader(test,TEST.batch_size,TEST.num_workers,False,stats);ys=[];ps=[]
    for g,d,bi,y,_ in loader:g,d,bi=g.to(dev),d.to(dev),bi.to(dev);log=m(g,d,bi,y.shape[0]);ys+=y.tolist();ps+=log.argmax(1).cpu().tolist()
    met=classification_metrics(ys,ps);tn,fp,fn,tp=confusion_matrix_counts(ys,ps).ravel();met.update({"setting":setting_id(v,r),"voxel_size":v,"context_radius":r,"context_side":2*r+1,"train_source":source,"test_domain":target,"train_seed":seed,"fnr":fn/(fn+tp) if fn+tp else 0.0,"fpr":fp/(fp+tn) if fp+tn else 0.0,"parameter_count":10898});mp.write_text(json.dumps(met,indent=2),encoding="utf-8");return met
