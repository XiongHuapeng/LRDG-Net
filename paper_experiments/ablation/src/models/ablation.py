"""Natural branch-removal / representation-replacement ablation models.

The variants intentionally have different trainable parameter counts when a branch is removed.
This is expected: the scientific control is the training/evaluation protocol, while each
variant is the actual architecture implied by the retained representation branches.
"""
import torch
import torch.nn as nn


VARIANT_SPECS = {
    "E0_G":  {"geometry": True,  "degradation": None},
    "E1_A":  {"geometry": False, "degradation": "absolute"},
    "E2_R":  {"geometry": False, "degradation": "relative"},
    "E3_GA": {"geometry": True,  "degradation": "absolute"},
    "E4_GR": {"geometry": True,  "degradation": "relative"},
}


class FeatureMLP(nn.Module):
    def __init__(self, in_channels, h1, h2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AblationNet(nn.Module):
    def __init__(self, variant, geometry_hidden=(32, 32), degradation_hidden=(16, 32),
                 instance_channels=64, classifier_hidden=64, dropout=0.20, num_classes=2,
                 geometry_in_channels=12, absolute_degradation_channels=3,
                 relative_degradation_channels=4):
        super().__init__()
        variant = variant.upper()
        if variant not in VARIANT_SPECS:
            raise ValueError(f"Unsupported variant: {variant}. Expected one of {tuple(VARIANT_SPECS)}")
        self.variant = variant
        self.spec = VARIANT_SPECS[variant]

        fusion_in = 0
        if self.spec["geometry"]:
            self.geometry_encoder = FeatureMLP(
                geometry_in_channels, geometry_hidden[0], geometry_hidden[1], dropout
            )
            fusion_in += geometry_hidden[1]

        degradation_kind = self.spec["degradation"]
        if degradation_kind is not None:
            deg_in = absolute_degradation_channels if degradation_kind == "absolute" else relative_degradation_channels
            self.degradation_encoder = FeatureMLP(
                deg_in, degradation_hidden[0], degradation_hidden[1], dropout
            )
            fusion_in += degradation_hidden[1]

        self.instance_fusion = nn.Sequential(
            nn.Linear(fusion_in, instance_channels),
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

    def encode_instances(self, geometry, degradation):
        features = []
        if self.spec["geometry"]:
            features.append(self.geometry_encoder(geometry))
        if self.spec["degradation"] is not None:
            features.append(self.degradation_encoder(degradation))
        x = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        return self.instance_fusion(x)

    @staticmethod
    def segment_mean(h, bag_index, batch_size):
        out = h.new_zeros((batch_size, h.shape[1]))
        out.index_add_(0, bag_index, h)
        counts = h.new_zeros((batch_size, 1))
        counts.index_add_(0, bag_index, h.new_ones((h.shape[0], 1)))
        return out / counts.clamp_min(1.0)

    def forward(self, geometry, degradation, bag_index, batch_size, return_aux=False):
        h = self.encode_instances(geometry, degradation)
        z = self.segment_mean(h, bag_index, batch_size)
        logits = self.classifier(z)
        if return_aux:
            return logits, {"embedding": z, "instance_embedding": h}
        return logits


def build_ablation_model(variant: str, model_config):
    return AblationNet(
        variant=variant,
        geometry_hidden=model_config.geometry_hidden,
        degradation_hidden=model_config.degradation_hidden,
        instance_channels=model_config.instance_channels,
        classifier_hidden=model_config.classifier_hidden,
        dropout=model_config.dropout,
        num_classes=model_config.num_classes,
        geometry_in_channels=model_config.geometry_in_channels,
        absolute_degradation_channels=model_config.absolute_degradation_channels,
        relative_degradation_channels=model_config.relative_degradation_channels,
    )
