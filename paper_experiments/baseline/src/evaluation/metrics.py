import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score

def compute_metrics(y,p):
  y=np.asarray(y); p=np.asarray(p); cm=confusion_matrix(y,p,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
  return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
      "precision_contaminated":float(precision_score(y,p,pos_label=1,zero_division=0)),"recall_contaminated":float(recall_score(y,p,pos_label=1,zero_division=0)),
      "fpr":float(fp/(fp+tn)) if fp+tn else 0.0,"fnr":float(fn/(fn+tp)) if fn+tp else 0.0,
      "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
