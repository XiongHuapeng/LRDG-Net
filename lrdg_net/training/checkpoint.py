"""
Checkpoint 选择规则。
Checkpoint selection rule.

Checkpoint selection follows the paper experiment protocol: maximize validation Macro-F1; ties -> lower validation loss.
"""
from pathlib import Path
import torch


def is_better_checkpoint(current_f1: float, current_loss: float, best_f1: float, best_loss: float, eps: float = 1e-12) -> bool:
    if current_f1 > best_f1 + eps:
        return True
    if abs(current_f1 - best_f1) <= eps and current_loss < best_loss - eps:
        return True
    return False


def save_best_checkpoint(path, model, optimizer, epoch: int, val_macro_f1: float, val_loss: float, experiment_snapshot: dict) -> None:
    torch.save({
        "epoch": int(epoch),
        "val_macro_f1": float(val_macro_f1),
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "experiment_snapshot": experiment_snapshot,
    }, Path(path))
