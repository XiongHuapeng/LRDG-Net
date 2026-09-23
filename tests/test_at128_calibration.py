import unittest

import numpy as np
import pandas as pd

from lrdg_net.workflows.at128_calibration import _aggregate_calibrated


class CalibrationDecisionTest(unittest.TestCase):
    def test_equality_at_threshold_is_contaminated(self):
        # Calibration margins have median zero. A test sequence with raw margin equal
        # to that center has normalized score 0. With tau_S=0, manuscript Eq. (21)
        # classifies equality as contaminated via >=.
        pcap = pd.DataFrame(
            {
                "recording_id": ["test"],
                "true_label": [1],
                "distance_m": [20.0],
                "condition_level": [1],
                "frames": [1],
                "prob_contaminated": [0.5],
                "logit_margin": [0.0],
            }
        )
        clean_by_distance = {8.0: np.asarray([-2.0, 0.0, 2.0], dtype=np.float64)}
        result, info = _aggregate_calibrated(
            pcap_base=pcap,
            clean_margin_by_distance=clean_by_distance,
            source_ref={"threshold_normalized": 0.0},
            clean_distances=(8.0,),
            test_distance=20.0,
        )
        self.assertEqual(int(result.loc[0, "pred_label"]), 1)
        self.assertAlmostEqual(float(result.loc[0, "normalized_score"]), 0.0)
        self.assertFalse(info["network_parameters_updated"])
        self.assertFalse(info["target_contaminated_labels_used_for_calibration"])


if __name__ == "__main__":
    unittest.main()
