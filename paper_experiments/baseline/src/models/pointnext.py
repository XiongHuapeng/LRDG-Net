import torch
import torch.nn as nn
from .pointnet2_ref import farthest_point_sample, index_points, query_ball_point


class ResidualSetAbstraction(nn.Module):
  """Residual set-abstraction block following the PointNeXt-S design principle.

  PointNeXt-S uses the shallow [1,1,1,1,1] depth schedule and residual SA blocks.
  This pure-PyTorch adaptation keeps that structure while using fixed physical
  radii suitable for the LIDAROC scene scale.
  """
  def __init__(self, npoint, radius, nsample, in_channels, out_channels):
    super().__init__()
    self.npoint = int(npoint)
    self.radius = float(radius)
    self.nsample = int(nsample)
    self.local = nn.Sequential(
      nn.Conv2d(in_channels + 3, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
    )
    if in_channels == out_channels:
      self.skip = nn.Identity()
    else:
      self.skip = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm1d(out_channels),
      )
    self.act = nn.ReLU(inplace=True)

  def forward(self, xyz, features):
    # xyz [B,N,3] in meters, features [B,N,C]
    fps_idx = farthest_point_sample(xyz, self.npoint)
    new_xyz = index_points(xyz, fps_idx)
    group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, group_idx)
    grouped_features = index_points(features, group_idx)
    rel_xyz = (grouped_xyz - new_xyz.unsqueeze(2)) / max(self.radius, 1e-6)
    local_input = torch.cat([rel_xyz, grouped_features], dim=-1)
    local_input = local_input.permute(0, 3, 2, 1).contiguous() # B,C,K,S
    local = self.local(local_input).max(dim=2).values # B,Cout,S

    center_features = index_points(features, fps_idx).transpose(1, 2).contiguous()
    skip = self.skip(center_features)
    out = self.act(local + skip)
    return new_xyz, out.transpose(1, 2).contiguous()


class PointNeXtSClassifier(nn.Module):
  """PointNeXt-S-style classification baseline adapted to XYZI LiDAR frames.

  Frozen architecture choices:
   - S depth schedule: [1,1,1,1,1]
   - stride schedule: [1,4,4,4,4]
   - width: 32
   - residual set abstraction
   - fixed metric radii: supplied from config (default 2/4/8/16 m)

  This is intentionally self-contained (no OpenPoints CUDA extensions), making it
  reproducible in the same software environment as the other baselines.
  """
  def __init__(self, width=32, nsample=32, radii=(2.0, 4.0, 8.0, 16.0), num_classes=2):
    super().__init__()
    if len(radii) != 4:
      raise ValueError("PointNeXt-S adaptation expects four stage radii.")
    w = int(width)
    self.stem = nn.Sequential(
      nn.Conv1d(4, w, 1, bias=False),
      nn.BatchNorm1d(w),
      nn.ReLU(inplace=True),
    )
    self.sa1 = ResidualSetAbstraction(512, radii[0], nsample, w, 2 * w)
    self.sa2 = ResidualSetAbstraction(128, radii[1], nsample, 2 * w, 4 * w)
    self.sa3 = ResidualSetAbstraction(32, radii[2], nsample, 4 * w, 8 * w)
    self.sa4 = ResidualSetAbstraction(8, radii[3], nsample, 8 * w, 16 * w)
    c = 16 * w
    self.classifier = nn.Sequential(
      nn.Linear(c, 512, bias=False),
      nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
      nn.Linear(512, 256, bias=False),
      nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
      nn.Linear(256, num_classes),
    )

  def forward(self, x):
    # Shared cache is [x, y, z, I/255]; XYZ remains in original meters.
    xyz = x[..., :3]
    f = self.stem(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
    xyz, f = self.sa1(xyz, f)
    xyz, f = self.sa2(xyz, f)
    xyz, f = self.sa3(xyz, f)
    xyz, f = self.sa4(xyz, f)
    g = f.max(dim=1).values
    return self.classifier(g)
