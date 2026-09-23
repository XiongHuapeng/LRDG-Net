from paper_experiments.baseline.config import BASELINES


def make_model(name, k=20):
  n = name.lower()
  if n == "globalstats":
    from .globalstats import GlobalStatsMLP
    return GlobalStatsMLP()
  if n == "rangenet":
    from .rangenet import RangeNetClassifier
    return RangeNetClassifier()
  if n == "pointnet":
    from .pointnet import PointNetClassifier
    return PointNetClassifier()
  if n == "pointnet2_ref":
    from .pointnet2_ref import PointNet2SSGClassifier
    return PointNet2SSGClassifier(num_classes=2, dropout=0.4)
  if n == "dgcnn":
    from .dgcnn import DGCNNClassifier
    return DGCNNClassifier(k=k)
  if n == "pointnext":
    from .pointnext import PointNeXtSClassifier
    return PointNeXtSClassifier(
      width=BASELINES.pointnext_width,
      nsample=BASELINES.pointnext_nsample,
      radii=BASELINES.pointnext_radii_m,
    )
  if n == "autogran":
    from .autogran import AutoGrAN
    return AutoGrAN()
  raise ValueError(name)
