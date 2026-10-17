"""Plan-v2 metrics (src/metrics_v2.py)."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from src import metrics_v2 as V
from src.evaluation import wang_pr_auc

Y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
P = np.array([0.9, 0.2, 0.7, 0.4, 0.5, 0.1, 0.8, 0.6, 0.3, 0.95])


class Quality(unittest.TestCase):
    def test_balanced_ap_is_the_mean_of_both_classes(self):
        want = 0.5 * (average_precision_score(Y, P) + average_precision_score(1 - Y, 1 - P))
        self.assertAlmostEqual(V.balanced_ap(Y, P), want, places=12)

    def test_floating_point_noise_does_not_break_ties(self):
        noisy = P.copy()
        noisy[1] = 0.5 + 1e-15   # ties with P[4] = 0.5 once rounded
        tied = P.copy()
        tied[1] = 0.5
        self.assertEqual(V.balanced_ap(Y, noisy), V.balanced_ap(Y, tied))
        self.assertEqual(V.balanced_wang_pr_auc(Y, noisy), V.balanced_wang_pr_auc(Y, tied))

    def test_a_constant_score_is_chance_under_ap_but_not_under_the_trapezoid(self):
        const = np.full(Y.size, 0.5)
        self.assertAlmostEqual(V.balanced_ap(Y, const), 0.5, places=12)
        prevalence = Y.mean()
        self.assertAlmostEqual(V.balanced_wang_pr_auc(Y, const), 0.75, places=12)
        self.assertAlmostEqual(wang_pr_auc(Y, const), (1 + prevalence) / 2, places=12)

    def test_wang_table(self):
        pred = (P > 0.5).astype(int)
        t = V.wang_table(Y, P, pred, np.arange(10), 2 * np.arange(10),
                         passages=[0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        self.assertAlmostEqual(t["nonfactual_pr_auc"], wang_pr_auc(1 - Y, 1 - P), places=12)
        self.assertAlmostEqual(t["factual_pr_auc"], wang_pr_auc(Y, P), places=12)
        self.assertAlmostEqual(t["accuracy"], float(np.mean(pred == Y)), places=12)
        self.assertEqual((t["documents_per_sentence"], t["nli_calls_per_sentence"]), (4.5, 9.0))
        self.assertEqual(t["passages"], 4)
        self.assertIsNotNone(t["pearson"])

    def test_validation(self):
        for y, p in (([], []), ([1, 0], [0.5]), ([1, 2], [0.1, 0.2]), ([1, 0], [0.1, np.nan])):
            with self.assertRaises(V.MetricsError):
                V.balanced_ap(y, p)
        with self.assertRaises(V.MetricsError):
            V.balanced_ap([1, 1], [0.2, 0.3])


class Calibration(unittest.TestCase):
    def test_brier_and_equal_mass_bins(self):
        c = V.calibration(Y, P, bins=3)
        self.assertAlmostEqual(c["brier"], float(np.mean((P - Y) ** 2)), places=12)
        self.assertEqual([b["n"] for b in c["reliability"]], [4, 3, 3])
        means = [b["mean_p"] for b in c["reliability"]]
        self.assertEqual(means, sorted(means))
        ece = sum(b["n"] / 10 * abs(b["mean_p"] - b["fraction_factual"])
                  for b in c["reliability"])
        self.assertAlmostEqual(c["ece"], ece, places=12)

    def test_perfectly_calibrated_bins_have_zero_ece(self):
        y = np.array([0, 0, 0, 1] * 25 + [1, 1, 1, 0] * 25)
        p = np.array([0.25] * 100 + [0.75] * 100)
        self.assertAlmostEqual(V.calibration(y, p, bins=2)["ece"], 0.0, places=12)


class Envelopes(unittest.TestCase):
    def setUp(self):
        # reference (BSE-like): operating point (6, 0.70) is on its curve
        self.ref = V.Envelope([2, 4, 6, 8], [0.60, 0.72, 0.70, 0.74])
        self.m = V.Envelope([1, 3, 5, 9], [0.55, 0.71, 0.73, 0.76])

    def test_step_envelope(self):
        self.assertIsNone(self.ref.quality_at(1.9))
        self.assertEqual(self.ref.quality_at(6.5), 0.72)   # (6, 0.70) dominated by (4, 0.72)
        self.assertEqual(self.ref.cost_for(0.71), 4.0)
        self.assertIsNone(self.ref.cost_for(0.80))

    def test_e_cost_is_net_of_the_references_own_best(self):
        # At q_ref = 0.70 the reference itself needs only 4 (not 6); m needs 3.
        self.assertEqual(V.e_cost(self.m, self.ref, 0.70), 1.0)
        self.assertIsNone(V.e_cost(self.m, self.ref, 0.80))

    def test_e_quality_is_net(self):
        self.assertAlmostEqual(V.e_quality(self.m, self.ref, 6.0), 0.73 - 0.72, places=12)
        self.assertIsNone(V.e_quality(V.Envelope([7], [0.9]), self.ref, 6.0))

    def test_auc_gap_exact(self):
        # shared range [2, 8]; m: 0.55 on [2,3), 0.71 on [3,5), 0.73 on [5,8]
        # ref: 0.60 on [2,4), 0.72 on [4,8]
        area = (1 * (0.55 - 0.60) + 1 * (0.71 - 0.60) + 1 * (0.71 - 0.72)
                + 3 * (0.73 - 0.72))
        self.assertAlmostEqual(V.auc_gap(self.m, self.ref), area / 6, places=12)
        self.assertIsNone(V.auc_gap(V.Envelope([10, 12], [0.8, 0.9]), self.ref))

    def test_reference_must_reach_itself(self):
        with self.assertRaises(V.MetricsError):
            V.e_cost(self.m, self.ref, 0.75)      # m reaches 0.75, the reference never does
        with self.assertRaises(V.MetricsError):
            V.e_quality(self.m, self.ref, 1.5)    # m has a point at 1, the reference none


class Bootstrap(unittest.TestCase):
    def test_pairs_every_statistic_on_the_same_draw(self):
        seen = []

        def stat(idx):
            seen.append(tuple(idx))
            return {"a": float(len(idx)), "b": float(np.sum(idx))}

        clusters = ["x", "x", "y", "z", "z", "z"]
        obs, draws, n_blocks = V.cluster_bootstrap(stat, clusters, 50, seed=7)
        self.assertEqual(n_blocks, 3)
        self.assertEqual(obs, {"a": 6.0, "b": 15.0})
        self.assertEqual(len(draws["a"]), 50)
        for idx in seen[1:]:
            # whole clusters only
            for block in ([0, 1], [2], [3, 4, 5]):
                counts = {i: idx.count(i) for i in block}
                self.assertEqual(len(set(counts.values())), 1)
        _, again, _ = V.cluster_bootstrap(stat, clusters, 50, seed=7)
        self.assertEqual(draws, again)

    def test_not_reached_is_counted_not_imputed(self):
        draws = [0.5, 1.0, None, 2.0, None, -0.5, 1.5, 0.8, 1.2, 0.9]
        s = V.summarise(1.0, draws)
        self.assertEqual((s["not_reached"], s["not_reached_fraction"]), (2, 0.2))
        reached = [v for v in draws if v is not None]
        self.assertAlmostEqual(s["ci_lower"], float(np.percentile(reached, 2.5)), places=12)
        self.assertAlmostEqual(s["p_greater"], (1 + 2) / 10, places=12)
        self.assertAlmostEqual(s["p_less"], (7 + 2) / 10, places=12)
        empty = V.summarise(None, [None, None])
        self.assertIsNone(empty["ci_lower"])
        self.assertEqual(empty["p_greater"], 1.0)

    def test_format_p(self):
        self.assertEqual(V.format_p(0.0, 10000), "< 1e-04")
        self.assertEqual(V.format_p(0.0123, 10000), "0.0123")
        self.assertEqual(V.format_p(None, 10), "n/a")


if __name__ == "__main__":
    unittest.main()
