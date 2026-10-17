"""Backward-induction stopping (src/sequential.py, plan v2 amendment 1)."""

import itertools
import math
import unittest

import numpy as np

from src import sequential as S
from src.baseline_core import BSEDetector, stop_cost

# Wang's NBC histograms (Gate 1, Laplace-smoothed).
WANG_POS = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
WANG_NEG = [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]


def exact_value(p, support, costs, remaining):
    """Optimal expected cost by full enumeration (small supports only)."""
    stop = stop_cost(p, costs.c_miss, costs.c_false_alarm)
    if remaining == 0:
        return stop
    expected = mass = 0.0
    for lr, w in zip(support.log_ratios, support.weights):
        r = math.exp(lr)
        pred = p * r + 1 - p
        nxt = p * r / pred
        expected += w * pred * exact_value(nxt, support, costs, remaining - 1)
        mass += w * pred
    return min(stop, costs.c_retrieve + expected / mass)


def exact_continue(p, support, costs, remaining):
    stop = stop_cost(p, costs.c_miss, costs.c_false_alarm)
    return exact_value(p, support, costs, remaining) < stop - 1e-12


class ReducesToEq8(unittest.TestCase):
    def test_horizon_one_is_the_one_step_rule(self):
        rng = np.random.default_rng(1)
        for _ in range(6):
            support = S.Support(rng.normal(0, 1.5, 25), rng.uniform(0.1, 1, 25))
            for costs in (S.Costs(28, 96, 1), S.Costs(14, 24, 0.25), S.Costs(28, 96, 0.1)):
                bi = S.BackwardInductionPolicy(support, costs, 1)
                one = S.ContinueRegion(support, costs)
                self.assertEqual(len(bi.intervals[0]), len(one.intervals))
                for a, b in zip(bi.intervals[0], one.intervals):
                    self.assertAlmostEqual(a[0], b[0], places=9)
                    self.assertAlmostEqual(a[1], b[1], places=9)


class MatchesFullEnumeration(unittest.TestCase):
    def test_values_and_decisions_on_small_supports(self):
        rng = np.random.default_rng(2)
        for trial in range(8):
            support = S.Support(rng.normal(0, 1.4, 3), rng.uniform(0.2, 1, 3))
            costs = (S.Costs(28, 96, 1), S.Costs(14, 24, 0.5))[trial % 2]
            horizon = 4
            bi = S.BackwardInductionPolicy(support, costs, horizon)
            for p in np.linspace(0.02, 0.98, 49):
                want = exact_value(p, support, costs, horizon)
                got = float(np.interp(S._logit(p), bi.x, bi.values[0]))
                self.assertAlmostEqual(got, want, delta=2e-3 * max(1.0, want))
                stop = stop_cost(p, costs.c_miss, costs.c_false_alarm)
                if abs(want - stop) > 1e-2:
                    self.assertEqual(bi.should_continue(p, 1),
                                     exact_continue(p, support, costs, horizon))


class Structure(unittest.TestCase):
    def setUp(self):
        self.support = S.Support(np.random.default_rng(3).normal(-0.1, 1.2, 400), np.ones(400))

    def test_regions_shrink_as_documents_run_out(self):
        bi = S.BackwardInductionPolicy(self.support, S.Costs(28, 96, 0.5), 8)
        for n in range(7):
            for p in np.linspace(0.01, 0.99, 197):
                if bi.should_continue(p, n + 2):      # stage n + 1
                    self.assertTrue(bi.should_continue(p, n + 1))

    def test_dearer_retrieval_reads_less(self):
        cheap = S.BackwardInductionPolicy(self.support, S.Costs(28, 96, 0.25), 6)
        dear = S.BackwardInductionPolicy(self.support, S.Costs(28, 96, 2.0), 6)
        for p in np.linspace(0.01, 0.99, 197):
            if dear.should_continue(p, 1):
                self.assertTrue(cheap.should_continue(p, 1))

    def test_never_continues_past_the_horizon(self):
        bi = S.BackwardInductionPolicy(self.support, S.Costs(28, 96, 0.0), 3)
        self.assertFalse(bi.should_continue(0.5, 4))
        self.assertEqual(S.run_subclaim([0.0] * 6, [1] * 6, bi, 10)[1], 3)

    def test_ranked_support_uses_the_next_rank(self):
        flat = S.Support([0.0], [1.0])
        strong = S.Support([-4.0, 4.0], [1.0, 1.0])
        ranked = S.RankedSupport({1: flat, 2: strong})
        bi = S.BackwardInductionPolicy(ranked, S.Costs(28, 96, 1.0), 2)
        # The first document is uninformative but opens the way to an
        # informative second one, so the optimal plan reads it; a one-step
        # rule at rank 1 would stop.
        self.assertTrue(bi.should_continue(0.5, 1))
        self.assertFalse(S.ContinueRegion(flat, S.Costs(28, 96, 1.0)).contains(0.5))


class WangsCase(unittest.TestCase):
    """The amendment's finding, kept as a regression test."""

    def test_one_step_eq8_never_reads_at_wangs_costs(self):
        support = S.Support.from_histogram(WANG_POS, WANG_NEG)
        for cr in (1.0, 0.5, 0.1, 0.01):
            self.assertFalse(S.ContinueRegion(support, S.Costs(28, 96, cr)).contains(0.5))
            self.assertFalse(BSEDetector(WANG_POS, WANG_NEG, mode="eq8",
                                         c_retrieve=cr).should_continue(0.5))
        self.assertTrue(BSEDetector(WANG_POS, WANG_NEG, mode="official").should_continue(0.5))

    def test_backward_induction_reads_at_wangs_costs(self):
        support = S.Support.from_histogram(WANG_POS, WANG_NEG)
        bi = S.BackwardInductionPolicy(support, S.Costs(28, 96, 1.0), 10)
        self.assertTrue(bi.should_continue(0.5, 1))


class Compression(unittest.TestCase):
    def test_preserves_mass_and_mean_ratio(self):
        s = S.Support(np.random.default_rng(4).normal(0, 2, 5000), np.ones(5000))
        c = S.compress(s)
        self.assertLess(c.log_ratios.size, s.log_ratios.size)
        self.assertAlmostEqual(float(c.weights.sum()), 1.0, places=12)
        self.assertAlmostEqual(c.mean_ratio, s.mean_ratio, places=10)

    def test_distinct_atoms_are_untouched(self):
        s = S.Support([-1.0, 0.0, 2.0], [0.2, 0.5, 0.3])
        c = S.compress(s)
        self.assertTrue(np.allclose(c.log_ratios, s.log_ratios))
        self.assertTrue(np.allclose(c.weights, s.weights))


class ByHorizon(unittest.TestCase):
    def test_plans_over_the_documents_available(self):
        support = S.Support([-3.0, 3.0], [1.0, 1.0])
        policies = S.PolicyByHorizon(support, S.Costs(28, 96, 0.1), 10)
        self.assertEqual(policies.for_documents(3).horizon, 3)
        self.assertEqual(policies.for_documents(25).horizon, 10)
        self.assertIsNone(policies.for_documents(0))
        self.assertIs(policies.for_documents(3), policies.for_documents(3))
        self.assertEqual(S.run_subclaim_bi([], [], policies, 10), (0.5, 0, 0))
        p, docs, nli = S.run_subclaim_bi([3.0, 3.0], [2, 5], policies, 10)
        self.assertGreater(docs, 0)

    def test_horizon_validated(self):
        with self.assertRaises(S.SequentialError):
            S.BackwardInductionPolicy(S.Support([0.0], [1.0]), S.Costs(), 0)


if __name__ == "__main__":
    unittest.main()
