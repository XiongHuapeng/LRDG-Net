"""R-only feature-subset models for component importance analysis."""
import torch
import torch.nn as nn

# Canonical 4D order must match src.data.local_descriptors.RELATIVE_DEGRADATION_SCHEMA.
R_COMPONENT_NAMES = (
    "delta_mean_intensity_div255",      # ΔI
    "log_std_ratio",                    # R_sigma
    "local_std_intensity_div255",       # V_I
    "relative_log_density",             # R_rho
)

VARIANT_SPECS = {
    "R_FULL": {
        "study": "reference",
        "indices": (0, 1, 2, 3),
        "paper_label": "R (full)",
        "description": "All four local-relative degradation components",
    },
    "R_ONLY_DELTA_I": {
        "study": "single",
        "indices": (0,),
        "paper_label": "ΔI only",
        "description": "Mean-intensity difference only",
    },
    "R_ONLY_LOG_STD_RATIO": {
        "study": "single",
        "indices": (1,),
        "paper_label": "Rσ only",
        "description": "Relative intensity-dispersion ratio only",
    },
    "R_ONLY_LOCAL_STD": {
        "study": "single",
        "indices": (2,),
        "paper_label": "V_I only",
        "description": "Local intensity standard deviation only",
    },
    "R_ONLY_REL_LOG_DENSITY": {
        "study": "single",
        "indices": (3,),
        "paper_label": "Rρ only",
        "description": "Relative log-density only",
    },
    "R_WO_DELTA_I": {
        "study": "leave_one_out",
        "indices": (1, 2, 3),
        "paper_label": "R − ΔI",
        "description": "Full R without mean-intensity difference",
    },
    "R_WO_LOG_STD_RATIO": {
        "study": "leave_one_out",
        "indices": (0, 2, 3),
        "paper_label": "R − Rσ",
        "description": "Full R without relative intensity-dispersion ratio",
    },
    "R_WO_LOCAL_STD": {
        "study": "leave_one_out",
        "indices": (0, 1, 3),
        "paper_label": "R − V_I",
        "description": "Full R without local intensity standard deviation",
    },
    "R_WO_REL_LOG_DENSITY": {
        "study": "leave_one_out",
        "indices": (0, 1, 2),
        "paper_label": "R − Rρ",
        "description": "Full R without relative log-density",
    },
}


class FeatureMLP(nn.Module):
    def __init__(self, in_channels, h1, h2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ROnlyComponentNet(nn.Module):
    """Same R-only architecture as the original E2_R model, with subset-sized input."""
    def __init__(
        self,
        variant,
        degradation_hidden=(16, 32),
        instance_channels=64,
        classifier_hidden=64,
        dropout=0.20,
        num_classes=2,
    ):
        super().__init__()
        variant = variant.upper()
        if variant not in VARIANT_SPECS:
            raise ValueError(f"Unsupported variant: {variant}. Expected one of {tuple(VARIANT_SPECS)}")
        self.variant = variant
        self.spec = VARIANT_SPECS[variant]
        self.input_indices = tuple(self.spec["indices"])
        in_channels = len(self.input_indices)
        if in_channels < 1:
            raise ValueError("R-only model requires at least one component.")

        self.degradation_encoder = FeatureMLP(
            in_channels,
            degradation_hidden[0],
            degradation_hidden[1],
            dropout,
        )
        self.instance_fusion = nn.Sequential(
            nn.Linear(degradation_hidden[1], instance_channels),
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
    def segment_mean(h, bag_index, batch_size):
        out = h.new_zeros((batch_size, h.shape[1]))
        out.index_add_(0, bag_index, h)
        counts = h.new_zeros((batch_size, 1))
        counts.index_add_(0, bag_index, h.new_ones((h.shape[0], 1)))
        return out / counts.clamp_min(1.0)

    def forward(self, degradation, bag_index, batch_size, return_aux=False):
        h = self.degradation_encoder(degradation)
        h = self.instance_fusion(h)
        z = self.segment_mean(h, bag_index, batch_size)
        logits = self.classifier(z)
        if return_aux:
            return logits, {"embedding": z, "instance_embedding": h}
        return logits


def build_r_component_model(variant: str, model_config):
    return ROnlyComponentNet(
        variant=variant,
        degradation_hidden=model_config.degradation_hidden,
        instance_channels=model_config.instance_channels,
        classifier_hidden=model_config.classifier_hidden,
        dropout=model_config.dropout,
        num_classes=model_config.num_classes,
    )
