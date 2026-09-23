import math
import numpy as np
import torch
from torch.utils.data import Dataset
from paper_experiments.baseline.config import DATA


class PointDataset(Dataset):
  def __init__(self, df):
    self.df = df.reset_index(drop=True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    r = self.df.iloc[i]
    x = np.load(r.point_cache).astype(np.float32)
    return torch.from_numpy(x), torch.tensor(int(r.label), dtype=torch.long), str(r.sample_uid)


class GlobalStatsDataset(Dataset):
  def __init__(self, df):
    self.df = df.reset_index(drop=True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    r = self.df.iloc[i]
    x = np.load(r.globalstats_cache).astype(np.float32)
    return torch.from_numpy(x), torch.tensor(int(r.label), dtype=torch.long), str(r.sample_uid)


class RangeViewDataset(Dataset):
  def __init__(self, df):
    self.df = df.reset_index(drop=True)
    self.h = int(DATA.rangeview_h)
    self.w = int(DATA.rangeview_w)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    r = self.df.iloc[i]
    z = np.load(r.rangeview_cache)
    idx = z["flat_idx"].astype(np.int64)
    values = z["values"].astype(np.float32)
    x = np.zeros((5, self.h * self.w), dtype=np.float32)
    if len(idx):
      x[:, idx] = values.T
    x = x.reshape(5, self.h, self.w)
    return torch.from_numpy(x), torch.tensor(int(r.label), dtype=torch.long), str(r.sample_uid)


class GraphDataset(Dataset):
  def __init__(self, df):
    self.df = df.reset_index(drop=True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    r = self.df.iloc[i]
    d = torch.load(r.graph_cache, weights_only=False)
    d.y = torch.tensor([int(r.label)], dtype=torch.long)
    d.sample_uid = str(r.sample_uid)
    return d
