"""Exploratory FACTORY follow-ups (scripts/explore_factory_followups.py), without artifacts."""

import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import explore_factory_followups as X  # noqa: E402
import run_v2_budget as RB  # noqa: E402
from test_run_v2_replay import histogram_everything  # noqa: E402
from test_v2_replay import synthetic  # noqa: E402


class Followups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = synthetic(60, seed=3, max_docs=4)
        for s in cls.prepared:
            s["subclaims"] = s["subclaims"][:1]
        est, sup = histogram_everything()
        cls.curves = RB.budget_curves(cls.prepared, est, sup, RB.build_policies(sup), 4)
        cls.labels = np.array([s["label"] for s in cls.prepared])

    def test_strength_is_label_free_max(self):
        s = X.strength(self.prepared)
        want = [max((max(d["e"], d["k"]) for d in p["subclaims"][0]), default=0.0)
                for p in self.prepared]
        np.testing.assert_array_equal(s, want)

    def test_crossing_fractions(self):
        out = X.crossing(self.curves, self.labels, np.arange(60))
        self.assertEqual(set(out), {"28/96", "14/24"})
        for v in out.values():
            self.assertLessEqual(v["fraction_above"] + v["fraction_below"], 1.0)
            self.assertLess(v["shared_range"][0], v["shared_range"][1])

    def test_pooled_of_identical_halves_equals_mean(self):
        with unittest.mock.patch.object(X.HC, "MAX_DOCS", 4):
            out = X.pooled_vs_mean(self.curves, self.curves, self.labels, self.labels)
        for k, v in out.items():
            if k != "factual_rate":
                self.assertAlmostEqual(v["pooled"], v["weighted_mean_of_datasets"])


if __name__ == "__main__":
    unittest.main()
