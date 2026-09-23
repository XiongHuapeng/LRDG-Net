# model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
  """
  src: [B, N, C]
  dst: [B, M, C]
  return: [B, N, M]
  """
  return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points, idx):
  """
  points: [B, N, C]
  idx: [B, S] or [B, S, K]
  return: [B, S, C] or [B, S, K, C]
  """
  device = points.device
  B = points.shape[0]

  view_shape = list(idx.shape)
  view_shape[1:] = [1] * (len(view_shape) - 1)

  repeat_shape = list(idx.shape)
  repeat_shape[0] = 1

  batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
  return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
  """
  xyz: [B, N, 3]
  return: centroids_idx [B, npoint]
  """
  device = xyz.device
  B, N, _ = xyz.shape

  centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
  distance = torch.ones(B, N, device=device) * 1e10
  farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
  batch_indices = torch.arange(B, dtype=torch.long, device=device)

  for i in range(npoint):
    centroids[:, i] = farthest
    centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
    dist = torch.sum((xyz - centroid) ** 2, dim=-1)
    mask = dist < distance
    distance[mask] = dist[mask]
    farthest = torch.max(distance, dim=1)[1]

  return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
  """
  radius: float
  nsample: int
  xyz: [B, N, 3]
  new_xyz: [B, S, 3]
  return: group_idx [B, S, nsample]
  """
  device = xyz.device
  B, N, _ = xyz.shape
  _, S, _ = new_xyz.shape

  sqrdists = square_distance(new_xyz, xyz) # [B, S, N]
  group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)

  group_idx[sqrdists > radius * radius] = N
  group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

  first_group = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
  mask = group_idx == N
  group_idx[mask] = first_group[mask]

  return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
  """
  xyz: [B, N, 3]
  points: [B, N, D] or None

  return:
    new_xyz: [B, S, 3]
    new_points: [B, S, nsample, 3 + D] or [B, S, nsample, 3]
  """
  fps_idx = farthest_point_sample(xyz, npoint)     # [B, S]
  new_xyz = index_points(xyz, fps_idx)         # [B, S, 3]
  group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
  grouped_xyz = index_points(xyz, group_idx)      # [B, S, nsample, 3]
  grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

  if points is not None:
    grouped_points = index_points(points, group_idx) # [B, S, nsample, D]
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
  else:
    new_points = grouped_xyz_norm

  return new_xyz, new_points


def sample_and_group_all(xyz, points):
  """
  xyz: [B, N, 3]
  points: [B, N, D] or None
  return:
    new_xyz: [B, 1, 3]
    new_points: [B, 1, N, 3 + D] or [B, 1, N, 3]
  """
  device = xyz.device
  B, N, _ = xyz.shape
  new_xyz = torch.zeros(B, 1, 3, device=device)
  grouped_xyz = xyz.view(B, 1, N, 3)

  if points is not None:
    grouped_points = points.view(B, 1, N, -1)
    new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
  else:
    new_points = grouped_xyz

  return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
  def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
    """
    in_channel: 额外点特征维度 D，不包含 xyz
    实际卷积输入维度 = D + 3
    """
    super().__init__()
    self.npoint = npoint
    self.radius = radius
    self.nsample = nsample
    self.group_all = group_all

    self.mlp_convs = nn.ModuleList()
    self.mlp_bns = nn.ModuleList()

    last_channel = in_channel + 3
    for out_channel in mlp:
      self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
      self.mlp_bns.append(nn.BatchNorm2d(out_channel))
      last_channel = out_channel

  def forward(self, xyz, points):
    """
    xyz: [B, N, 3]
    points: [B, N, D] or None

    return:
      new_xyz: [B, S, 3]
      new_points: [B, D', S]
    """
    if self.group_all:
      new_xyz, new_points = sample_and_group_all(xyz, points)
    else:
      new_xyz, new_points = sample_and_group(
        self.npoint, self.radius, self.nsample, xyz, points
      )

    # new_points: [B, S, nsample, C]
    new_points = new_points.permute(0, 3, 2, 1).contiguous() # [B, C, nsample, S]

    for conv, bn in zip(self.mlp_convs, self.mlp_bns):
      new_points = F.relu(bn(conv(new_points)), inplace=True)

    new_points = torch.max(new_points, 2)[0] # [B, D', S]
    return new_xyz, new_points


class PointNet2SSGClassifier(nn.Module):
  """
  PointNet++ Single-Scale Grouping classification baseline

  输入:
    x: [B, N, 4] = [x, y, z, intensity_norm]

  注意:
    - xyz 使用米制坐标做 FPS / ball query
    - intensity 作为额外点特征
  """

  def __init__(self, num_classes=2, dropout=0.4):
    super().__init__()

    # 只把 intensity 作为额外特征，所以 in_channel=1
    self.sa1 = PointNetSetAbstraction(
      npoint=512,
      radius=1.0,
      nsample=32,
      in_channel=1,
      mlp=[64, 64, 128],
      group_all=False,
    )

    self.sa2 = PointNetSetAbstraction(
      npoint=128,
      radius=2.0,
      nsample=64,
      in_channel=128,
      mlp=[128, 128, 256],
      group_all=False,
    )

    self.sa3 = PointNetSetAbstraction(
      npoint=None,
      radius=None,
      nsample=None,
      in_channel=256,
      mlp=[256, 512, 1024],
      group_all=True,
    )

    self.classifier = nn.Sequential(
      nn.Linear(1024, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(p=dropout),

      nn.Linear(512, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(p=dropout),

      nn.Linear(256, num_classes)
    )

  def forward(self, x):
    # x: [B, N, 4]
    xyz = x[:, :, :3]   # [B, N, 3]
    points = x[:, :, 3:]  # [B, N, 1]

    l1_xyz, l1_points = self.sa1(xyz, points)         # l1_points: [B, 128, 512]
    l1_points = l1_points.transpose(1, 2).contiguous()    # [B, 512, 128]

    l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # [B, 256, 128]
    l2_points = l2_points.transpose(1, 2).contiguous()    # [B, 128, 256]

    l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # [B, 1024, 1]

    x = l3_points.squeeze(-1)                # [B, 1024]
    logits = self.classifier(x)
    return logits


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = PointNet2SSGClassifier(num_classes=2, dropout=0.4).to(device)

  dummy = torch.randn(2, 4096, 4).to(device)
  out = model(dummy)

  print("Input shape :", dummy.shape)
  print("Output shape:", out.shape)