"""
LRDG-Net sensitivity-study model definition aligned with the manuscript.

Architecture:
1) 12D centered covariance geometry -> geometry MLP;
2) 4D fine-to-context relative degradation -> degradation MLP;
3) concatenate -> 64D local-instance embedding;
4) mean aggregation across occupied voxels;
5) binary classifier.

没有 Graph、max pooling、attention 或 MIL gate。
No graph, max pooling, attention, or MIL gate is used.
"""
import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """两层轻量特征编码器。 / Two-layer lightweight feature encoder."""
    def __init__(self, in_channels: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LRDGNet(nn.Module):
    """Paper-aligned LRDG-Net used in the sensitivity study."""
    def __init__(
        self,
        geometry_in_channels: int = 12,
        degradation_in_channels: int = 4,
        geometry_hidden=(32, 32),
        degradation_hidden=(16, 32),
        instance_channels: int = 64,
        classifier_hidden: int = 64,
        dropout: float = 0.20,
        num_classes: int = 2,
    ):
        super().__init__()
        self.geometry_encoder = FeatureMLP(
            geometry_in_channels, geometry_hidden[0], geometry_hidden[1], dropout
        )
        self.degradation_encoder = FeatureMLP(
            degradation_in_channels, degradation_hidden[0], degradation_hidden[1], dropout
        )
        self.instance_fusion = nn.Sequential(
            nn.Linear(geometry_hidden[1] + degradation_hidden[1], instance_channels),
            nn.LayerNorm(instance_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(instance_channels, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    @staticmethod
    def segment_mean(h: torch.Tensor, bag_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        """按帧对可变数量 voxel embedding 做 mean aggregation。"""
        out = h.new_zeros((batch_size, h.shape[1]))
        out.index_add_(0, bag_index, h)
        counts = h.new_zeros((batch_size, 1))
        counts.index_add_(0, bag_index, h.new_ones((h.shape[0], 1)))
        return out / counts.clamp_min(1.0)

    def encode_instances(self, geometry: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
        hg = self.geometry_encoder(geometry)
        hd = self.degradation_encoder(degradation)
        return self.instance_fusion(torch.cat([hg, hd], dim=1))

    def forward(
        self,
        geometry: torch.Tensor,
        degradation: torch.Tensor,
        bag_index: torch.Tensor,
        batch_size: int,
        return_aux: bool = False,
    ):
        h = self.encode_instances(geometry, degradation)
        z = self.segment_mean(h, bag_index, batch_size)
        logits = self.classifier(z)
        if return_aux:
            return logits, {"embedding": z, "instance_embedding": h}
        return logits
