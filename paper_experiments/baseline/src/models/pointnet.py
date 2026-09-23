import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
  def __init__(self):
    super().__init__();
    self.c1=nn.Conv1d(3,64,1); self.c2=nn.Conv1d(64,128,1); self.c3=nn.Conv1d(128,1024,1)
    self.b1=nn.BatchNorm1d(64); self.b2=nn.BatchNorm1d(128); self.b3=nn.BatchNorm1d(1024)
    self.f1=nn.Linear(1024,512); self.f2=nn.Linear(512,256); self.f3=nn.Linear(256,9)
    self.fb1=nn.BatchNorm1d(512); self.fb2=nn.BatchNorm1d(256)
    nn.init.zeros_(self.f3.weight); nn.init.zeros_(self.f3.bias)
  def forward(self,x):
    b=x.size(0); x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x))); x=F.relu(self.b3(self.c3(x)))
    x=torch.max(x,2).values; x=F.relu(self.fb1(self.f1(x))); x=F.relu(self.fb2(self.f2(x))); x=self.f3(x)
    eye=torch.eye(3,device=x.device,dtype=x.dtype).view(1,9).repeat(b,1); return (x+eye).view(-1,3,3)

class PointNetClassifier(nn.Module):
  """PointNet classification baseline adapted to XYZI: STN acts on XYZ; intensity is retained."""
  def __init__(self,num_classes=2):
    super().__init__(); self.stn=STN3d()
    self.c1=nn.Conv1d(4,64,1); self.c2=nn.Conv1d(64,64,1); self.c3=nn.Conv1d(64,128,1); self.c4=nn.Conv1d(128,1024,1)
    self.b1=nn.BatchNorm1d(64); self.b2=nn.BatchNorm1d(64); self.b3=nn.BatchNorm1d(128); self.b4=nn.BatchNorm1d(1024)
    self.f1=nn.Linear(1024,512); self.f2=nn.Linear(512,256); self.f3=nn.Linear(256,num_classes)
    self.fb1=nn.BatchNorm1d(512); self.fb2=nn.BatchNorm1d(256); self.drop=nn.Dropout(0.3)
  def forward(self,x):
    # x: B,N,4
    xyz=x[:,:,:3]; inten=x[:,:,3:4]; t=self.stn(xyz.transpose(1,2)); xyz=torch.bmm(xyz,t)
    x=torch.cat([xyz,inten],dim=2).transpose(1,2)
    x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x))); x=F.relu(self.b3(self.c3(x))); x=F.relu(self.b4(self.c4(x)))
    x=torch.max(x,2).values; x=F.relu(self.fb1(self.f1(x))); x=self.drop(x); x=F.relu(self.fb2(self.f2(x))); x=self.drop(x)
    return self.f3(x)
