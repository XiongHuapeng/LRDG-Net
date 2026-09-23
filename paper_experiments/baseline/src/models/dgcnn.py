import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x,k):
  # x: B,C,N; squared Euclidean via matrix identity
  xx=torch.sum(x*x,dim=1,keepdim=True)
  pair=-xx.transpose(2,1)-xx+2*torch.matmul(x.transpose(2,1),x)
  return pair.topk(k=k,dim=-1).indices

def graph_feature(x,k):
  b,c,n=x.shape; idx=knn(x,k); base=torch.arange(b,device=x.device).view(-1,1,1)*n; idx=(idx+base).reshape(-1)
  xt=x.transpose(2,1).contiguous(); nei=xt.reshape(b*n,c)[idx].view(b,n,k,c); cen=xt.view(b,n,1,c).expand(-1,-1,k,-1)
  return torch.cat([nei-cen,cen],dim=3).permute(0,3,1,2).contiguous()

class DGCNNClassifier(nn.Module):
  """Standard DGCNN-style EdgeConv classification backbone adapted to 4D XYZI input."""
  def __init__(self,k=20,num_classes=2):
    super().__init__(); self.k=k
    def block(ci,co): return nn.Sequential(nn.Conv2d(ci,co,1,bias=False),nn.BatchNorm2d(co),nn.LeakyReLU(0.2))
    self.e1=block(8,64); self.e2=block(128,64); self.e3=block(128,128); self.e4=block(256,256)
    self.emb=nn.Sequential(nn.Conv1d(512,1024,1,bias=False),nn.BatchNorm1d(1024),nn.LeakyReLU(0.2))
    self.f1=nn.Linear(2048,512,bias=False); self.b1=nn.BatchNorm1d(512); self.f2=nn.Linear(512,256); self.b2=nn.BatchNorm1d(256); self.f3=nn.Linear(256,num_classes); self.drop=nn.Dropout(0.5)
  def forward(self,x):
    x=x.transpose(1,2).contiguous()
    x1=self.e1(graph_feature(x,self.k)).max(dim=-1).values
    x2=self.e2(graph_feature(x1,self.k)).max(dim=-1).values
    x3=self.e3(graph_feature(x2,self.k)).max(dim=-1).values
    x4=self.e4(graph_feature(x3,self.k)).max(dim=-1).values
    x=self.emb(torch.cat([x1,x2,x3,x4],dim=1)); x=torch.cat([F.adaptive_max_pool1d(x,1).squeeze(-1),F.adaptive_avg_pool1d(x,1).squeeze(-1)],dim=1)
    x=self.drop(F.leaky_relu(self.b1(self.f1(x)),0.2)); x=self.drop(F.leaky_relu(self.b2(self.f2(x)),0.2)); return self.f3(x)
