import torch.nn as nn


class GlobalStatsMLP(nn.Module):
  """29D whole-frame statistical-vector baseline adapted from TinyLid's 1D paradigm."""
  def __init__(self, in_dim=29, num_classes=2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, 64, bias=False),
      nn.BatchNorm1d(64),
      nn.ReLU(inplace=True),
      nn.Dropout(0.2),
      nn.Linear(64, 32, bias=False),
      nn.BatchNorm1d(32),
      nn.ReLU(inplace=True),
      nn.Dropout(0.2),
      nn.Linear(32, num_classes),
    )

  def forward(self, x):
    return self.net(x)
