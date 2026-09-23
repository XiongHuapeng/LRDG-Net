import random, numpy as np, torch

def seed_all(seed):
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
