import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn

class AutoGrAN(nn.Module):
  """AutoGrAN / Voxel-GAT architecture from Jati et al., ICPE Companion 2024 (582 params)."""
  def __init__(self):
    super().__init__(); self.conv1=GATConv(4,16,heads=4,concat=True,bias=True); self.conv2=GATConv(64,2,heads=1,concat=False,bias=True)
  def forward(self,data):
    x=F.elu(self.conv1(data.x,data.edge_index)); x=self.conv2(x,data.edge_index); return global_mean_pool(x,data.batch)
