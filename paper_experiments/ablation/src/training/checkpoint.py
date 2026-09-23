import torch

def is_better_checkpoint(f1,loss,best_f1,best_loss):
    return (f1>best_f1+1e-12) or (abs(f1-best_f1)<=1e-12 and loss<best_loss-1e-12)

def save_best_checkpoint(path,model,optimizer,epoch,f1,loss,snapshot):
    torch.save({"epoch":epoch,"val_macro_f1":f1,"val_loss":loss,"model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),"experiment_snapshot":snapshot},path)
