import unittest

import numpy as np

from lrdg_net.data.local_descriptors import extract_local_descriptors


class DescriptorFormulaTest(unittest.TestCase):
    def test_single_occupied_voxel_matches_paper_formulas(self):
        # All points lie in one voxel. The context contains the same observed points plus
        # 26 empty grid locations, so center/context intensity statistics are identical.
        points = np.asarray(
            [
                [0.01, 0.01, 0.01, 10.0],
                [0.02, 0.01, 0.01, 20.0],
                [0.01, 0.03, 0.01, 30.0],
            ],
            dtype=np.float32,
        )
        out = extract_local_descriptors(
            points=points,
            voxel_size=0.20,
            context_radius_voxels=1,
            intensity_scale=255.0,
            relative_std_eps=1e-3,
        )
        self.assertEqual(out["geometry"].shape, (1, 12))
        self.assertEqual(out["relative_degradation"].shape, (1, 4))
        np.testing.assert_allclose(out["geometry"][0, :6], out["geometry"][0, 6:], rtol=1e-6, atol=1e-7)

        intensity = points[:, 3].astype(np.float64)
        std_n = intensity.std(ddof=0) / 255.0
        expected = np.asarray(
            [
                0.0,
                0.0,
                std_n,
                np.log1p(3.0) - np.log1p(3.0 / 27.0),
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(out["relative_degradation"][0], expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
