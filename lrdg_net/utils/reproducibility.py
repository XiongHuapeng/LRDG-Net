"""随机性与设备管理。 / Reproducibility and device helpers."""
import random
import numpy as np
import torch


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.lower().startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU. / CUDA 不可用，自动切换到 CPU。")
    return torch.device("cpu")
