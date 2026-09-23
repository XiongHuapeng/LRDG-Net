import torch
def better(f,l,bf,bl): return (f>bf+1e-12) or (abs(f-bf)<=1e-12 and l<bl-1e-12)
def save(path,m,o,e,f,l,s): torch.save({"epoch":e,"val_macro_f1":f,"val_loss":l,"model_state":m.state_dict(),"optimizer_state":o.state_dict(),"experiment_snapshot":s},path)
