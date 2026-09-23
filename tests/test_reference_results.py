import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReferenceResultTest(unittest.TestCase):
    def test_main_and_calibration_numbers_match_manuscript_snapshots(self):
        cross = pd.read_csv(ROOT / "reference_results/lidaroc_cross_domain/cross_domain_model_summary.csv")
        row = cross.loc[cross["model_name"] == "LRDG-Net"].iloc[0]
        self.assertAlmostEqual(float(row["direction_macro_f1_mean"]), 0.9885686531972887)
        self.assertAlmostEqual(float(row["worst_direction_macro_f1"]), 0.964922445902355)

        cal = pd.read_csv(ROOT / "reference_results/at128_calibration/paper_cross_sensor_summary.csv")
        k0 = cal.loc[cal["setting"] == "K0_raw"].iloc[0]
        k3 = cal.loc[cal["setting"] == "K3_clean_norm"].iloc[0]
        self.assertAlmostEqual(float(k0["macro_f1_mean"]), 0.4285714285714286)
        self.assertAlmostEqual(float(k3["macro_f1_mean"]), 0.9999333999333999)
        self.assertAlmostEqual(float(k3["worst_distance_macro_f1"]), 0.9991341991341992)


if __name__ == "__main__":
    unittest.main()
