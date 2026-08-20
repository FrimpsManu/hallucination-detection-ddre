import unittest

import numpy as np

from src.baseline_core import (
    BSEDetector,
    cost_based_prediction,
    discretize_document_score,
    discretize_nbc_score,
)
from src.ddre_core import ULSIFDensityRatio


class TestPublishedBSEConventions(unittest.TestCase):
    def test_released_code_uses_two_different_discretizers(self):
        # NBC_feature.py uses int(score / 10); released main.py uses
        # int((score - 0.1) / 10) for retrieved documents.
        self.assertEqual(discretize_nbc_score(10.0), 1)
        self.assertEqual(discretize_document_score(10.0), 0)

    def test_cost_based_decision_matches_wang_code(self):
        c_miss = 28
        c_false_alarm = 96
        threshold = c_miss / (c_miss + c_false_alarm)
        self.assertEqual(cost_based_prediction(threshold - 1e-4, c_miss, c_false_alarm), 0)
        self.assertEqual(cost_based_prediction(threshold + 1e-4, c_miss, c_false_alarm), 1)

    def test_bse_continue_costs_are_finite(self):
        pos = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        neg = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        for mode in ("official", "eq8"):
            detector = BSEDetector(pos, neg, mode=mode)
            value = detector.continue_cost(0.5)
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 0)


class TestULSIF(unittest.TestCase):
    def test_direct_ratio_is_larger_on_factual_like_scores(self):
        rng = np.random.default_rng(7)
        factual = np.clip(rng.normal(78, 7, 80), 0, 100)
        hallucinated = np.clip(rng.normal(22, 7, 80), 0, 100)

        estimator = ULSIFDensityRatio(max_centers=30, random_state=7).fit(
            factual,
            hallucinated,
            folds=4,
        )

        self.assertGreater(estimator.ratio(80), estimator.ratio(20))
        self.assertGreater(estimator.ratio(80), 1.0)


if __name__ == "__main__":
    unittest.main()
