import unittest

import torch

from config import MODEL
from lrdg_net.training.trainer import build_model


class ModelSpecTest(unittest.TestCase):
    def test_paper_architecture_and_parameter_count(self):
        self.assertEqual(MODEL.geometry_in_channels, 12)
        self.assertEqual(MODEL.degradation_in_channels, 4)
        self.assertEqual(MODEL.geometry_hidden, (32, 32))
        self.assertEqual(MODEL.degradation_hidden, (16, 32))
        self.assertEqual(MODEL.instance_channels, 64)
        self.assertEqual(MODEL.classifier_hidden, 64)
        self.assertEqual(MODEL.dropout, 0.20)

        model = build_model()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 10898)

        geometry = torch.zeros((7, 12), dtype=torch.float32)
        degradation = torch.zeros((7, 4), dtype=torch.float32)
        bag_index = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        logits = model(geometry, degradation, bag_index, batch_size=2)
        self.assertEqual(tuple(logits.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
