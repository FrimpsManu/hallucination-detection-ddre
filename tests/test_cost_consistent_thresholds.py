"""Tests for cost-consistent DDRE stopping thresholds (audit finding D-01).

The invariant, and the whole point of the file:

    lower_threshold <= t < upper_threshold,   t = C_M / (C_M + C_FA)

Stopping LOW asserts the posterior will classify NONFACTUAL; stopping HIGH
asserts it will classify FACTUAL. Wang's rule is strict -- factual iff
(1-P)*C_M < P*C_FA -- so P == t classifies NONFACTUAL. That asymmetry is why
``lower <= t`` admits equality while ``upper > t`` does not, and it is checked
in both directions below.

The threshold is derived from the *configured* costs, never hardcoded, so the
CM=14/CFA=24 configuration gets a different and correct search space.

numpy + stdlib only, so CI's numpy-only install runs the whole file.
"""

import math
import unittest

from src.baseline_core import cost_based_prediction
from src.ddre_core import (
    CANDIDATE_LOWER_GRID,
    CANDIDATE_UPPER_GRID,
    DDREDetector,
    cost_consistent_thresholds,
    cost_decision_threshold,
    thresholds_are_cost_consistent,
)

PRIMARY = (28.0, 96.0)
GATE_PRIMARY = (14.0, 24.0)
PRIMARY_THRESHOLD = 28.0 / 124.0
GATE_THRESHOLD = 14.0 / 38.0


class ConstantRatio:
    def __init__(self, value):
        self.value = float(value)

    def ratio(self, score):
        return self.value


class FixedScorer:
    def __init__(self, score):
        self.score = score
        self.calls = 0

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        self.calls += 1
        return self.score, 1


class Doc:
    page_content = "word " * 10


class Sub:
    def __init__(self, n_documents):
        self.text = "claim"
        self.documents = [Doc() for _ in range(n_documents)]


def detector(lower, upper, costs=PRIMARY, ratio=1.0, max_docs=10):
    return DDREDetector(
        ConstantRatio(ratio),
        lower_threshold=lower,
        upper_threshold=upper,
        p0=0.5,
        c_miss=costs[0],
        c_false_alarm=costs[1],
        max_docs=max_docs,
    )


class TestCostDecisionThreshold(unittest.TestCase):
    def test_the_primary_configuration_threshold(self):
        self.assertAlmostEqual(
            cost_decision_threshold(28, 96), 0.2258064516129032, places=15
        )
        self.assertEqual(cost_decision_threshold(28, 96), PRIMARY_THRESHOLD)

    def test_the_gate_primary_configuration_threshold(self):
        self.assertAlmostEqual(
            cost_decision_threshold(14, 24), 0.3684210526315789, places=15
        )

    def test_it_is_the_switch_point_of_the_cost_rule_itself(self):
        # Derived from cost_based_prediction, not asserted alongside it. Both
        # configured cost pairs are covered, plus symmetric ones.
        for c_miss, c_false_alarm in (PRIMARY, GATE_PRIMARY, (1, 1), (50, 50), (1, 9)):
            with self.subTest(costs=(c_miss, c_false_alarm)):
                t = cost_decision_threshold(c_miss, c_false_alarm)
                self.assertEqual(cost_based_prediction(t, c_miss, c_false_alarm), 0)
                self.assertEqual(
                    cost_based_prediction(math.nextafter(t, 1.0), c_miss, c_false_alarm),
                    1,
                )
                self.assertEqual(
                    cost_based_prediction(math.nextafter(t, 0.0), c_miss, c_false_alarm),
                    0,
                )

    def test_the_exact_tie_at_the_threshold_holds_for_both_configured_cost_pairs(self):
        # `lower == t` is permitted because P == t classifies nonfactual. That
        # reasoning needs (1-t)*C_M and t*C_FA to compare equal in float, which
        # they do for CM=28/CFA=96 and CM=14/CFA=24 -- the only cost pairs this
        # experiment uses.
        for c_miss, c_false_alarm in (PRIMARY, GATE_PRIMARY):
            with self.subTest(costs=(c_miss, c_false_alarm)):
                t = cost_decision_threshold(c_miss, c_false_alarm)
                self.assertEqual((1.0 - t) * c_miss, t * c_false_alarm)
                self.assertEqual(cost_based_prediction(t, c_miss, c_false_alarm), 0)

    def test_NOTE_some_other_cost_pairs_have_a_one_ulp_edge_at_the_threshold(self):
        # Documented, not fixed here. For CM=3/CFA=7 the mathematical tie at
        # P == t is broken by a 1-ULP rounding in (1-t)*C_M, so the classifier
        # returns factual exactly at t. That is a property of
        # cost_based_prediction, which this PR does not touch, and it can only
        # bite if a posterior lands on t exactly. If such a cost pair is ever
        # adopted, `lower <= t` must be revisited for it.
        t = cost_decision_threshold(3, 7)
        self.assertNotEqual((1.0 - t) * 3, t * 7)
        self.assertEqual(cost_based_prediction(t, 3, 7), 1)

    def test_it_is_c_miss_over_the_total_not_the_other_way_round(self):
        # C_FA/(C_M+C_FA) would be 0.774 for the primary costs; the rule would
        # then be inverted and every stop would mean the opposite.
        self.assertLess(cost_decision_threshold(28, 96), 0.5)
        self.assertNotAlmostEqual(cost_decision_threshold(28, 96), 96.0 / 124.0)
        self.assertAlmostEqual(
            cost_decision_threshold(96, 28), 96.0 / 124.0
        )  # arguments swap, value swaps

    def test_costs_are_validated_defensively(self):
        for c_miss, c_false_alarm in (
            (float("nan"), 96),
            (28, float("nan")),
            (float("inf"), 96),
            (28, float("inf")),
            (-1, 96),
            (28, -1),
            (0, 0),
            (None, 96),
            ("28", None),
        ):
            with self.subTest(costs=(c_miss, c_false_alarm)):
                with self.assertRaises(ValueError):
                    cost_decision_threshold(c_miss, c_false_alarm)

    def test_a_string_cost_that_parses_is_accepted(self):
        self.assertEqual(cost_decision_threshold("28", "96"), PRIMARY_THRESHOLD)


class TestTheInvariant(unittest.TestCase):
    def test_lower_may_equal_the_threshold(self):
        # P == t classifies nonfactual, which is what a low stop asserts.
        self.assertTrue(
            thresholds_are_cost_consistent(PRIMARY_THRESHOLD, 0.80, PRIMARY_THRESHOLD)
        )

    def test_lower_above_the_threshold_is_rejected(self):
        self.assertFalse(
            thresholds_are_cost_consistent(
                math.nextafter(PRIMARY_THRESHOLD, 1.0), 0.80, PRIMARY_THRESHOLD
            )
        )

    def test_upper_equal_to_the_threshold_is_rejected(self):
        # Wang's rule is strict, so P == t is NOT factual; a high stop there
        # would assert the opposite of what the classifier returns.
        self.assertFalse(
            thresholds_are_cost_consistent(0.10, PRIMARY_THRESHOLD, PRIMARY_THRESHOLD)
        )

    def test_upper_above_the_threshold_is_permitted(self):
        self.assertTrue(
            thresholds_are_cost_consistent(
                0.10, math.nextafter(PRIMARY_THRESHOLD, 1.0), PRIMARY_THRESHOLD
            )
        )


class TestDetectorConstruction(unittest.TestCase):
    def test_the_default_configuration_is_consistent(self):
        built = DDREDetector(ConstantRatio(1.0))
        self.assertEqual(built.cost_decision_threshold, PRIMARY_THRESHOLD)

    def test_every_effective_pair_constructs(self):
        space = cost_consistent_thresholds(*PRIMARY)
        for lower, upper in space["threshold_pairs"]:
            with self.subTest(lower=lower, upper=upper):
                detector(lower, upper)

    def test_an_inconsistent_lower_threshold_is_rejected(self):
        for lower in (0.25, 0.30, 0.35, 0.40):
            with self.subTest(lower=lower):
                with self.assertRaises(ValueError) as caught:
                    detector(lower, 0.80)
                message = str(caught.exception)
                self.assertIn(repr(float(lower)), message)
                self.assertIn(repr(0.80), message)
                self.assertIn(repr(PRIMARY_THRESHOLD), message)
                self.assertIn(repr(28.0), message)
                self.assertIn(repr(96.0), message)

    def test_an_inconsistent_upper_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            detector(0.10, PRIMARY_THRESHOLD)
        with self.assertRaises(ValueError):
            detector(0.10, 0.20)  # entirely below the decision threshold

    def test_lower_exactly_at_the_threshold_constructs(self):
        built = detector(PRIMARY_THRESHOLD, 0.80)
        self.assertEqual(built.lower_threshold, PRIMARY_THRESHOLD)

    def test_the_ordering_guard_still_applies(self):
        for lower, upper in ((0.0, 0.8), (0.8, 0.2), (0.2, 1.0), (0.5, 0.5)):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    detector(lower, upper)

    def test_the_guard_uses_the_configured_costs_not_the_primary_ones(self):
        # 0.35 is invalid at CM=28/CFA=96 and valid at CM=14/CFA=24.
        with self.assertRaises(ValueError):
            detector(0.35, 0.80, costs=PRIMARY)
        detector(0.35, 0.80, costs=GATE_PRIMARY)
        with self.assertRaises(ValueError):
            detector(0.40, 0.80, costs=GATE_PRIMARY)

    def test_bse_is_not_subject_to_this_restriction(self):
        # The guard is DDRE's; BSEDetector's stopping rule is derived from the
        # costs directly and must stay exactly as Wang published it.
        from src.baseline_core import BSEDetector

        built = BSEDetector(
            [1] * 10, [1] * 10, mode="official", p0=0.5, c_miss=28,
            c_false_alarm=96, c_retrieve=1, max_docs=10,
        )
        self.assertFalse(hasattr(built, "lower_threshold"))
        self.assertFalse(hasattr(built, "upper_threshold"))


class TestNoContradictoryStopIsReachable(unittest.TestCase):
    """The defect D-01 described can no longer be produced at all."""

    def stop_low_prediction(self, lower, upper, costs):
        # Drive the posterior just below `lower` with a single document.
        target = lower * 0.9
        ratio = (target / (1.0 - target)) / 1.0  # p0 = 0.5 => prior odds 1
        built = detector(lower, upper, costs=costs, ratio=ratio)
        result = built.detect_subclaim(Sub(10), FixedScorer(20.0))
        return result

    def test_a_low_stop_always_predicts_nonfactual(self):
        for costs in (PRIMARY, GATE_PRIMARY):
            space = cost_consistent_thresholds(*costs)
            for lower, upper in space["threshold_pairs"]:
                with self.subTest(costs=costs, lower=lower, upper=upper):
                    result = self.stop_low_prediction(lower, upper, costs)
                    self.assertEqual(result.documents_used, 1)
                    self.assertLessEqual(result.p_factual, lower)
                    self.assertEqual(
                        result.prediction, 0, "a low stop must classify nonfactual"
                    )

    def test_a_high_stop_always_predicts_factual(self):
        for costs in (PRIMARY, GATE_PRIMARY):
            space = cost_consistent_thresholds(*costs)
            for lower, upper in space["threshold_pairs"]:
                with self.subTest(costs=costs, lower=lower, upper=upper):
                    target = upper + (1.0 - upper) * 0.5
                    ratio = target / (1.0 - target)
                    built = detector(lower, upper, costs=costs, ratio=ratio)
                    result = built.detect_subclaim(Sub(10), FixedScorer(90.0))
                    self.assertEqual(result.documents_used, 1)
                    self.assertGreaterEqual(result.p_factual, upper)
                    self.assertEqual(
                        result.prediction, 1, "a high stop must classify factual"
                    )

    def test_the_old_contradictory_configuration_cannot_be_built(self):
        # Before this change, lower=0.30 stopped at P=0.28 and then predicted
        # FACTUAL. The configuration is now unconstructable.
        with self.assertRaises(ValueError):
            detector(0.30, 0.80, costs=PRIMARY)
        # And the arithmetic that made it contradictory is unchanged:
        self.assertEqual(cost_based_prediction(0.28, 28, 96), 1)


class TestSearchSpace(unittest.TestCase):
    def test_the_primary_effective_lower_grid(self):
        space = cost_consistent_thresholds(*PRIMARY)
        self.assertEqual(space["effective_lower_grid"], [0.05, 0.10, 0.15, 0.20])
        self.assertEqual(space["excluded_lower_grid"], [0.25, 0.30, 0.35, 0.40])

    def test_the_primary_effective_upper_grid_is_unchanged(self):
        space = cost_consistent_thresholds(*PRIMARY)
        self.assertEqual(space["effective_upper_grid"], list(CANDIDATE_UPPER_GRID))
        self.assertEqual(space["excluded_upper_grid"], [])

    def test_the_contradictory_lower_values_cannot_enter_tuning(self):
        space = cost_consistent_thresholds(*PRIMARY)
        entering = {pair[0] for pair in space["threshold_pairs"]}
        for value in (0.25, 0.30, 0.35, 0.40):
            with self.subTest(value=value):
                self.assertNotIn(value, entering)

    def test_the_gate_primary_configuration_keeps_035_and_drops_040(self):
        space = cost_consistent_thresholds(*GATE_PRIMARY)
        self.assertAlmostEqual(space["cost_decision_threshold"], GATE_THRESHOLD)
        self.assertIn(0.35, space["effective_lower_grid"])
        self.assertNotIn(0.40, space["effective_lower_grid"])
        self.assertEqual(space["excluded_lower_grid"], [0.40])

    def test_the_grid_is_derived_not_hardcoded(self):
        # A cost pair whose threshold sits between grid points must produce a
        # search space nobody wrote down in advance.
        space = cost_consistent_thresholds(1.0, 1.0)  # t = 0.5
        self.assertEqual(space["cost_decision_threshold"], 0.5)
        self.assertEqual(space["effective_lower_grid"], list(CANDIDATE_LOWER_GRID))
        self.assertEqual(space["effective_upper_grid"], [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        strict = cost_consistent_thresholds(9.0, 1.0)  # t = 0.9
        self.assertEqual(strict["effective_upper_grid"], [0.95])
        self.assertEqual(strict["effective_lower_grid"], list(CANDIDATE_LOWER_GRID))

    def test_every_generated_pair_satisfies_the_invariant(self):
        for c_miss, c_false_alarm in (
            PRIMARY, GATE_PRIMARY, (1, 1), (9, 1), (1, 9), (3, 7), (50, 50),
        ):
            space = cost_consistent_thresholds(c_miss, c_false_alarm)
            t = space["cost_decision_threshold"]
            for lower, upper in space["threshold_pairs"]:
                with self.subTest(costs=(c_miss, c_false_alarm), pair=(lower, upper)):
                    self.assertTrue(0.0 < lower < upper < 1.0)
                    self.assertLessEqual(lower, t)
                    self.assertLess(t, upper)
                    self.assertTrue(thresholds_are_cost_consistent(lower, upper, t))

    def test_no_valid_pair_is_silently_generated_outside_the_invariant(self):
        # The complement direction: every candidate pair that the filter drops
        # must genuinely violate the invariant, so nothing valid is lost.
        for c_miss, c_false_alarm in (PRIMARY, GATE_PRIMARY, (1, 9), (9, 1)):
            space = cost_consistent_thresholds(c_miss, c_false_alarm)
            t = space["cost_decision_threshold"]
            kept = {tuple(pair) for pair in space["threshold_pairs"]}
            for lower in CANDIDATE_LOWER_GRID:
                for upper in CANDIDATE_UPPER_GRID:
                    if not 0.0 < lower < upper < 1.0:
                        continue
                    consistent = thresholds_are_cost_consistent(lower, upper, t)
                    with self.subTest(costs=(c_miss, c_false_alarm), pair=(lower, upper)):
                        self.assertEqual((lower, upper) in kept, consistent)

    def test_the_record_is_auditable(self):
        space = cost_consistent_thresholds(*PRIMARY)
        for key in (
            "c_miss", "c_false_alarm", "cost_decision_threshold",
            "candidate_lower_grid", "candidate_upper_grid",
            "effective_lower_grid", "effective_upper_grid",
            "excluded_lower_grid", "excluded_upper_grid",
            "threshold_pairs", "threshold_pairs_evaluated",
            "candidate_pairs_before_filtering", "rule",
        ):
            self.assertIn(key, space)
        self.assertEqual(space["candidate_lower_grid"], list(CANDIDATE_LOWER_GRID))
        self.assertEqual(space["candidate_pairs_before_filtering"], 64)
        self.assertEqual(space["threshold_pairs_evaluated"], 32)
        self.assertEqual(
            space["threshold_pairs_evaluated"], len(space["threshold_pairs"])
        )

    def test_the_record_is_json_serializable(self):
        import json

        json.dumps(cost_consistent_thresholds(*PRIMARY))


class TestTunerWiring(unittest.TestCase):
    """Asserted on main.py's AST; running the tuner would need torch."""

    @classmethod
    def setUpClass(cls):
        import ast
        from pathlib import Path

        cls.ast = ast
        cls.source = (
            Path(__file__).resolve().parents[1] / "main.py"
        ).read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.tuner = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "tune_ddre_thresholds"
        )

    def test_the_tuner_derives_its_grids_from_the_configured_costs(self):
        calls = [
            node
            for node in self.ast.walk(self.tuner)
            if isinstance(node, self.ast.Call)
            and isinstance(node.func, self.ast.Name)
            and node.func.id == "cost_consistent_thresholds"
        ]
        self.assertEqual(len(calls), 1)
        passed = {a.id for a in calls[0].args if isinstance(a, self.ast.Name)}
        self.assertIn("c_miss", passed)
        self.assertIn("c_false_alarm", passed)

    def test_the_tuner_searches_the_effective_grids_not_the_candidate_ones(self):
        # Using candidate_* here would reintroduce D-01: the tuner would score
        # contradictory pairs (and, today, crash on the constructor guard).
        subscripts = {
            node.slice.value
            for node in self.ast.walk(self.tuner)
            if isinstance(node, self.ast.Subscript)
            and isinstance(node.value, self.ast.Name)
            and node.value.id == "search_space"
            and isinstance(node.slice, self.ast.Constant)
        }
        self.assertIn("effective_lower_grid", subscripts)
        self.assertIn("effective_upper_grid", subscripts)
        self.assertNotIn("candidate_lower_grid", subscripts)
        self.assertNotIn("candidate_upper_grid", subscripts)

    def test_the_tuner_does_not_build_its_own_grid(self):
        body = self.ast.get_source_segment(self.source, self.tuner)
        self.assertNotIn("np.arange(0.05", body)
        self.assertNotIn("np.arange(0.60", body)

    def test_no_hardcoded_decision_threshold_anywhere_in_the_pipeline(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for path in (root / "main.py", root / "src" / "ddre_core.py"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("0.225806", text)
                self.assertNotIn("0.368421", text)

    def test_the_search_space_reaches_the_experiment_summary(self):
        self.assertIn('"threshold_search_space": threshold_search_space', self.source)
        self.assertIn(
            "selected, threshold_table, selection_rule, threshold_search_space",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
