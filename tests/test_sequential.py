"""Tests for the generic Eq. 8 look-ahead (src/sequential.py, plan v2 §3.2)."""

import math
import random
import unittest

import numpy as np

from src import sequential as S
from src.baseline_core import BSEDetector, bayes_update, stop_cost
from src.mddre_estimators import HistogramEvidence
from src.mddre_replay import bse_subclaim

COSTS = [(28, 96, 1), (14, 24, 1), (28, 96, 0.25), (28, 96, 4), (14, 24, 0.125)]


def random_hist(rng):
    return [rng.randint(1, 60) for _ in range(10)]


class HistogramMatchesBSEEq8(unittest.TestCase):
    """With histogram atoms the engine is BSEDetector(mode="eq8")."""

    def test_expected_risk_and_decision_match_on_a_grid(self):
        rng = random.Random(1)
        for _ in range(20):
            pos, neg = random_hist(rng), random_hist(rng)
            support = S.Support.from_histogram(pos, neg)
            self.assertAlmostEqual(support.mean_ratio, 1.0, places=12)
            for cm, cfa, cr in COSTS:
                bse = BSEDetector(pos, neg, mode="eq8", c_miss=cm, c_false_alarm=cfa,
                                  c_retrieve=cr)
                costs = S.Costs(cm, cfa, cr)
                for p in np.linspace(0.001, 0.999, 211):
                    ours = cr + support.expected_next_stop_risk(p, cm, cfa)
                    self.assertAlmostEqual(ours, bse.continue_cost(p), places=10)
                    margin = S.continue_margin(p, support, costs)
                    if abs(margin) > 1e-9:
                        self.assertEqual(margin > 0, bse.should_continue(p))

    def test_replay_matches_bse_eq8_document_for_document(self):
        rng = random.Random(2)
        for trial in range(30):
            pos, neg = random_hist(rng), random_hist(rng)
            cm, cfa, cr = COSTS[trial % len(COSTS)]
            bse = BSEDetector(pos, neg, mode="eq8", c_miss=cm, c_false_alarm=cfa,
                              c_retrieve=cr)
            hist = HistogramEvidence(pos, neg)
            policy = S.Eq8Policy(S.Support.from_histogram(pos, neg), S.Costs(cm, cfa, cr))
            for _ in range(40):
                scores = [rng.uniform(0, 100) for _ in range(rng.randint(0, 10))]
                spans = [rng.randint(1, 5) for _ in scores]
                docs = [{"e": s} for s in scores]
                want = bse_subclaim([hist.likelihoods(d) for d in docs], spans, bse, 10)
                got = S.run_subclaim([hist.log_ratio(d) for d in docs], spans, policy, 10)
                self.assertEqual(got[1:], want[1:])
                self.assertAlmostEqual(got[0], want[0], places=12)


class ClosedForm(unittest.TestCase):
    def test_weighted_stop_risk_equals_the_posterior_form(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            lr = rng.normal(0, 2, size=rng.integers(1, 40))
            w = rng.uniform(0, 1, size=lr.size)
            support = S.Support(lr, w)
            for p in (0.01, 0.2, 0.5, 0.77, 0.99):
                naive = mass = 0.0
                for r, wi in zip(np.exp(lr), support.weights):
                    pred = p * r + 1 - p
                    naive += wi * pred * stop_cost(p * r / pred, 28, 96)
                    mass += wi * pred
                self.assertAlmostEqual(support.expected_next_stop_risk(p, 28, 96),
                                       naive / mass, places=10)

    def test_grid_evaluation_equals_pointwise(self):
        support = S.Support(np.random.default_rng(4).normal(0, 1.5, 300), np.ones(300))
        ps = np.linspace(0.001, 0.999, 1500)
        grid = support.expected_next_stop_risk_grid(ps, 14, 24)
        for p, g in zip(ps[::37], grid[::37]):
            self.assertAlmostEqual(g, support.expected_next_stop_risk(p, 14, 24), places=12)


class KnownRatio(unittest.TestCase):
    """p_f = N(1, 1), p_h = N(0, 1), so r(x) = exp(x - 1/2) exactly.

    The sample-based look-ahead must match E_next[R_stop(pi')] computed by
    quadrature under the true predictive mixture pi p_f + (1 - pi) p_h.
    """

    def test_sample_support_matches_quadrature(self):
        rng = np.random.default_rng(5)
        xh = rng.normal(0.0, 1.0, 400_000)
        support = S.Support.from_sample(xh - 0.5)
        self.assertAlmostEqual(support.mean_ratio, 1.0, delta=0.01)
        x = np.linspace(-9, 10, 200_001)
        dx = x[1] - x[0]
        phi = lambda m: np.exp(-0.5 * (x - m) ** 2) / math.sqrt(2 * math.pi)
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            pred = p * phi(1.0) + (1 - p) * phi(0.0)
            post = p * phi(1.0) / pred
            risk = np.minimum((1 - post) * 28, post * 96)
            exact = float(np.sum(pred * risk) * dx)
            self.assertAlmostEqual(support.expected_next_stop_risk(p, 28, 96), exact,
                                   delta=0.01 * exact)


class Region(unittest.TestCase):
    def test_region_agrees_with_the_exact_rule(self):
        rng = np.random.default_rng(6)
        support = S.Support(rng.normal(-0.2, 1.3, 800), np.ones(800))
        for cm, cfa, cr in COSTS:
            costs = S.Costs(cm, cfa, cr)
            region = S.ContinueRegion(support, costs)
            for p in rng.uniform(0, 1, 3000):
                exact = S.continue_margin(p, support, costs) > 0
                near = min((abs(p - e) for iv in region.intervals for e in iv), default=1)
                if near > 1e-9:
                    self.assertEqual(region.contains(p), exact)

    def test_histogram_region_is_bses_eq8_interval(self):
        pos, neg = [3, 9, 40, 20, 8, 6, 5, 30, 1, 1], [60, 70, 26, 9, 5, 4, 3, 20, 1, 1]
        region = S.ContinueRegion(S.Support.from_histogram(pos, neg), S.Costs(28, 96, 1))
        bse = BSEDetector(pos, neg, mode="eq8")
        for p in np.linspace(1e-4, 1 - 1e-4, 5001):
            if min(abs(p - e) for iv in region.intervals for e in iv) > 1e-9:
                self.assertEqual(region.contains(p), bse.should_continue(p))

    def test_prohibitive_retrieval_cost_reads_nothing(self):
        support = S.Support.from_histogram([5] * 10, [5] * 10)
        policy = S.Eq8Policy(support, S.Costs(28, 96, 1000))
        self.assertEqual(policy.regions[None].intervals, [])
        self.assertEqual(S.run_subclaim([1.0, 2.0], [3, 3], policy, 10), (0.5, 0, 0))

    def test_uninformative_evidence_reads_nothing(self):
        # r = 1 everywhere: the next posterior equals the current one, so
        # continuing can only add C_retrieve.
        policy = S.Eq8Policy(S.Support([0.0], [1.0]), S.Costs(28, 96, 1e-6))
        self.assertEqual(policy.regions[None].intervals, [])

    def test_a_tie_stops_as_in_wangs_strict_rule(self):
        # r = 1 and C_retrieve = 0: continuing and stopping cost exactly the
        # same. BSE continues only if stopping is strictly dearer.
        policy = S.Eq8Policy(S.Support([0.0], [1.0]), S.Costs(28, 96, 0))
        self.assertEqual(policy.regions[None].intervals, [])
        self.assertFalse(BSEDetector([5] * 10, [5] * 10, mode="eq8",
                                     c_retrieve=0).should_continue(0.5))


class RankConditional(unittest.TestCase):
    def test_policy_uses_the_next_documents_rank(self):
        strong = S.Support([-4.0, 4.0], [0.5, 0.5])   # informative: continue at P0
        flat = S.Support([0.0], [1.0])                # uninformative: stop
        policy = S.Eq8Policy(S.RankedSupport({1: strong, 2: flat, 3: strong}),
                             S.Costs(28, 96, 1))
        self.assertTrue(policy.should_continue(0.5, 1))
        self.assertFalse(policy.should_continue(0.5, 2))
        # A tiny first update leaves P inside rank 1's region, but rank 2 stops.
        self.assertEqual(S.run_subclaim([0.01, 0.01, 0.01], [1, 1, 1], policy, 10)[1], 1)

    def test_missing_rank_fails_closed(self):
        policy = S.Eq8Policy(S.RankedSupport({1: S.Support([-3.0, 3.0], [1, 1])}),
                             S.Costs(28, 96, 1))
        with self.assertRaises(S.SequentialError):
            S.run_subclaim([0.0, 0.0], [1, 1], policy, 10)


class Detection(unittest.TestCase):
    def test_sentence_takes_the_minimum_and_sums_costs(self):
        out = S.sentence([(0.9, 2, 5), (0.3, 1, 2), (0.6, 0, 0)], S.Costs(28, 96, 1))
        self.assertEqual(out["p_factual"], 0.3)
        self.assertEqual((out["documents"], out["nli_calls"]), (3, 7))
        self.assertEqual(out["subclaim_documents"], [2, 1, 0])
        self.assertEqual(out["prediction"], S.decision(0.3, 28, 96))

    def test_decision_threshold_is_wangs(self):
        t = 28 / (28 + 96)
        self.assertEqual(S.decision(t + 1e-9, 28, 96), 1)
        self.assertEqual(S.decision(t - 1e-9, 28, 96), 0)

    def test_update_matches_bayes_rule(self):
        policy = S.Eq8Policy(S.Support([-3.0, 3.0], [1, 1]), S.Costs(28, 96, 0))
        lrs = [0.7, -0.2]
        p, docs, _ = S.run_subclaim(lrs, [1, 1], policy, 2)
        want = 0.5
        for lr in lrs[:docs]:
            want = bayes_update(want, math.exp(lr), 1.0)
        self.assertAlmostEqual(p, want, places=12)

    def test_max_docs_caps_reading(self):
        policy = S.Eq8Policy(S.Support([-3.0, 3.0], [1, 1]), S.Costs(28, 96, 0))
        self.assertEqual(S.run_subclaim([0.0] * 10, [1] * 10, policy, 4)[1], 4)


class FixedK(unittest.TestCase):
    def test_aggregates(self):
        e, lr, sp = [10.0, 80.0, 40.0], [-1.0, 2.0, 0.5], [1, 2, 3]
        self.assertEqual(S.fixed_k_subclaim(e, lr, sp, 2, "max"), (0.8, 2, 3))
        self.assertEqual(S.fixed_k_subclaim(e, lr, sp, 2, "avg"), (0.45, 2, 3))
        score, docs, nli = S.fixed_k_subclaim(e, lr, sp, 3, "nbc")
        self.assertAlmostEqual(score, 1 / (1 + math.exp(-1.5)), places=12)
        self.assertEqual((docs, nli), (3, 6))

    def test_k_beyond_available_and_empty(self):
        self.assertEqual(S.fixed_k_subclaim([50.0], [0.0], [4], 5, "max"), (0.5, 1, 4))
        self.assertEqual(S.fixed_k_subclaim([], [], [], 3, "avg"), (0.5, 0, 0))

    def test_rejects_bad_arguments(self):
        with self.assertRaises(S.SequentialError):
            S.fixed_k_subclaim([1.0], [0.0], [1], 0, "max")
        with self.assertRaises(S.SequentialError):
            S.fixed_k_subclaim([1.0], [0.0], [1], 1, "median")


class Validation(unittest.TestCase):
    def test_support_rejects_bad_atoms(self):
        for lr, w in (([], []), ([1.0], [1.0, 2.0]), ([float("nan")], [1.0]),
                      ([1.0], [-1.0]), ([1.0], [0.0])):
            with self.assertRaises(S.SequentialError):
                S.Support(lr, w)

    def test_histogram_support_rejects_unsmoothed_bins(self):
        with self.assertRaises(S.SequentialError):
            S.Support.from_histogram([0] + [1] * 9, [1] * 10)

    def test_costs_validated(self):
        with self.assertRaises(S.SequentialError):
            S.Costs(0, 96, 1)
        with self.assertRaises(S.SequentialError):
            S.Costs(28, 96, -1)


if __name__ == "__main__":
    unittest.main()
