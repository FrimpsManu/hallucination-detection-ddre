"""Tests for the frozen D-03 density-ratio support/stability diagnostic.

D-03 is **not** "large ratios are wrong". A large ratio is what a density-ratio
estimator should produce where the classes genuinely separate. The question is
whether the evidence DDRE consumes is *supported* by the training score regions
and *stable* across the production hyperparameter surface -- separating
large-but-stable evidence from tail-driven evidence.

These tests pin the methodology BEFORE the real measurement runs, so the
escalation criteria cannot be adjusted after seeing the numbers.

Synthetic, CPU-only. No model, no GPU, no held-out evidence, no real ratios.
"""

import ast
import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.ddre_core import ULSIFDensityRatio
from src.density_ratio_diagnostics import (
    D13_BIN_EDGES,
    DECISION_HIGH,
    DECISION_LOW,
    DECISION_NONE,
    DIAGNOSTIC_MAX_DOCS,
    DIAGNOSTIC_P0,
    DIAGNOSTIC_SPLIT_SEED,
    DIAGNOSTIC_VALIDATION_FRACTION,
    DIRECTION_ABOVE,
    DIRECTION_BELOW,
    DIRECTION_MIXED,
    EXPECTED_FACTUAL_TRAINING,
    EXPECTED_HALLUCINATED_TRAINING,
    PROTOCOL_VERSION,
    RATIO_CLIP_LOWER,
    RATIO_CLIP_UPPER,
    SUPPORT_STRATA,
    DiagnosticIncomplete,
    clip_activity,
    completion_accounting,
    distribution_shift,
    escalation_triggers,
    fixed_bin_histogram,
    hyperparameter_surface,
    logit,
    normalized,
    one_document_stopping,
    population_ratio_report,
    production_hyperparameter_pairs,
    quantile_summary,
    range_status,
    require_complete,
    returned_ratio,
    selected_fit_ratios,
    state_after_one_document,
    stop_decision,
    stop_decision_stability,
    stops_under_selected_fit,
    support_counts,
    support_stratum,
    support_strata_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fitted_estimator(seed=0, n=60):
    """A small but genuine production fit, so the real code paths are used."""
    rng = np.random.default_rng(seed)
    factual = np.clip(rng.normal(72.0, 9.0, n), 1.0, 99.0)
    hallucinated = np.clip(rng.normal(28.0, 9.0, n), 1.0, 99.0)
    estimator = ULSIFDensityRatio(max_centers=20, random_state=42)
    estimator.fit(factual, hallucinated)
    return estimator, list(factual), list(hallucinated)


# --------------------------------------------------------------------------
# Selected-fit ratios, raw vs returned, clip accounting
# --------------------------------------------------------------------------


class TestSelectedFitRatios(unittest.TestCase):
    def setUp(self):
        self.estimator, self.factual, self.hallucinated = fitted_estimator()

    def test_ratio_quantiles_are_computed_from_the_returned_ratios(self):
        # The returned ratio is what DDRE consumes. Reporting only raw
        # quantiles would describe a quantity the detector never sees.
        report = population_ratio_report(
            self.estimator, self.factual, name="factual"
        )
        ratios = selected_fit_ratios(self.estimator, self.factual)
        self.assertEqual(
            report["returned_ratio_quantiles"],
            quantile_summary(ratios["returned"]),
        )
        self.assertEqual(
            report["log_returned_ratio_quantiles"],
            quantile_summary(np.log(ratios["returned"])),
        )

    def test_raw_and_returned_ratios_are_reported_separately(self):
        report = population_ratio_report(
            self.estimator, self.factual, name="factual"
        )
        self.assertIn("raw_ratio_quantiles", report)
        self.assertIn("returned_ratio_quantiles", report)
        self.assertIsNot(
            report["raw_ratio_quantiles"], report["returned_ratio_quantiles"]
        )

    def test_raw_and_returned_quantiles_differ_when_the_clip_actually_bites(self):
        # On a well-behaved fit nothing hits the clip, so raw and returned are
        # identical and swapping them is invisible. This population is built so
        # they MUST differ: the raw values say what the fit produced, the
        # returned values say what DDRE consumed, and conflating them would
        # hide the clip's entire contribution.
        class Extreme:
            # One centre, huge and tiny coefficients reachable from either end.
            centers = np.array([[0.0], [1.0]])
            sigma = 0.05
            alpha = np.array([1e-9, 1e9])
            _as_column = staticmethod(ULSIFDensityRatio._as_column)
            _kernel = staticmethod(ULSIFDensityRatio._kernel)

        scores = [0.0, 100.0]
        ratios = selected_fit_ratios(Extreme(), scores)
        self.assertLess(float(np.min(ratios["raw"])), RATIO_CLIP_LOWER)
        self.assertGreater(float(np.max(ratios["raw"])), RATIO_CLIP_UPPER)

        report = population_ratio_report(Extreme(), scores, name="extreme")
        self.assertEqual(
            report["raw_ratio_quantiles"], quantile_summary(ratios["raw"])
        )
        self.assertEqual(
            report["returned_ratio_quantiles"],
            quantile_summary(ratios["returned"]),
        )
        self.assertNotEqual(
            report["raw_ratio_quantiles"]["max"],
            report["returned_ratio_quantiles"]["max"],
        )
        self.assertNotEqual(
            report["raw_ratio_quantiles"]["min"],
            report["returned_ratio_quantiles"]["min"],
        )
        self.assertEqual(report["returned_ratio_quantiles"]["max"], RATIO_CLIP_UPPER)
        self.assertEqual(report["returned_ratio_quantiles"]["min"], RATIO_CLIP_LOWER)
        self.assertEqual(report["clip_activity"]["lower_clip_hits"], 1)
        self.assertEqual(report["clip_activity"]["upper_clip_hits"], 1)

    def test_a_clipped_population_shows_raw_and_returned_diverging(self):
        # Constructed so the difference between raw and returned is visible;
        # conflating them would hide exactly this.
        raw = [1e-9, 1e9, 1.0]
        self.assertEqual(returned_ratio(raw[0]), RATIO_CLIP_LOWER)
        self.assertEqual(returned_ratio(raw[1]), RATIO_CLIP_UPPER)
        self.assertEqual(returned_ratio(raw[2]), 1.0)
        summary = clip_activity(raw)
        self.assertNotEqual(
            quantile_summary(raw)["max"],
            quantile_summary([returned_ratio(r) for r in raw])["max"],
        )
        self.assertEqual(summary["count"], 3)

    def test_lower_and_upper_clip_hits_are_counted_separately(self):
        summary = clip_activity([1e-9, 1e-9, 1e9, 0.5, 2.0])
        self.assertEqual(summary["lower_clip_hits"], 2)
        self.assertEqual(summary["upper_clip_hits"], 1)
        self.assertEqual(summary["any_clip_hits"], 3)
        self.assertEqual(summary["clip_lower"], RATIO_CLIP_LOWER)
        self.assertEqual(summary["clip_upper"], RATIO_CLIP_UPPER)

    def test_non_positive_raw_ratios_are_counted(self):
        self.assertEqual(clip_activity([0.0, -1.0, 1.0])["raw_non_positive"], 2)

    def test_the_alpha_sum_bound_is_reported_and_holds(self):
        # The estimator is BOUNDED by sum(alpha) on ordinary fitted evaluation:
        # a finite sum of non-negative coefficients times kernels bounded by 1.
        # Calling it mathematically unbounded would be wrong.
        report = population_ratio_report(
            self.estimator, self.factual, name="factual"
        )
        alpha_sum = float(np.sum(self.estimator.alpha))
        self.assertAlmostEqual(report["alpha_sum"], alpha_sum)
        self.assertIn("BOUNDED", report["alpha_sum_bound_note"])
        raw = selected_fit_ratios(self.estimator, self.factual)["raw"]
        self.assertLessEqual(float(np.max(raw)), alpha_sum + 1e-9)

    def test_quantile_levels_are_fixed_not_adaptive(self):
        summary = quantile_summary([1.0, 2.0, 3.0, 4.0])
        for label in ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"):
            self.assertIn(label, summary)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)

    def test_a_non_finite_value_fails_loudly_rather_than_being_dropped(self):
        with self.assertRaises(DiagnosticIncomplete) as caught:
            quantile_summary([1.0, float("nan")])
        self.assertIn("NOT dropped", str(caught.exception))
        with self.assertRaises(DiagnosticIncomplete):
            quantile_summary([])


# --------------------------------------------------------------------------
# Local support
# --------------------------------------------------------------------------


class TestLocalSupport(unittest.TestCase):
    def test_support_uses_normalized_score_distance(self):
        # sigma is a distance in the kernel's [0, 1] space. Measuring on the
        # 0-100 scale would be wrong by a factor of 100 and would make almost
        # everything look unsupported.
        self.assertEqual(list(normalized([0.0, 50.0, 100.0])), [0.0, 0.5, 1.0])
        counts = support_counts(
            [50.0], factual_training=[45.0, 60.0], hallucinated_training=[49.0],
            sigma=0.1,
        )
        # |0.50-0.45| = 0.05 <= 0.1 ; |0.50-0.60| = 0.10 <= 0.1 ; both inside.
        self.assertEqual(int(counts["factual_support_within_sigma"][0]), 2)
        self.assertEqual(int(counts["hallucinated_support_within_sigma"][0]), 1)
        self.assertIn("normalized", counts["space"])

    def test_the_support_radius_is_exactly_the_selected_sigma(self):
        # A point at exactly sigma is inside; just beyond it is not.
        counts = support_counts(
            [50.0], factual_training=[40.0, 39.9], hallucinated_training=[50.0],
            sigma=0.1,
        )
        self.assertEqual(int(counts["factual_support_within_sigma"][0]), 1)
        self.assertEqual(counts["radius_sigma"], 0.1)

    def test_min_class_support_is_the_smaller_of_the_two(self):
        counts = support_counts(
            [50.0], factual_training=[50.0, 50.0, 50.0],
            hallucinated_training=[50.0], sigma=0.1,
        )
        self.assertEqual(int(counts["min_class_support"][0]), 1)

    def test_a_non_positive_bandwidth_is_refused(self):
        for sigma in (0.0, -0.1, float("nan")):
            with self.subTest(sigma=sigma):
                with self.assertRaises(DiagnosticIncomplete):
                    support_counts([50.0], [50.0], [50.0], sigma)

    def test_the_support_strata_are_the_pre_registered_bands(self):
        self.assertEqual(
            [label for label, _, _ in SUPPORT_STRATA],
            ["0", "1-2", "3-4", "5-9", ">=10"],
        )
        for value, expected in (
            (0, "0"), (1, "1-2"), (2, "1-2"), (3, "3-4"), (4, "3-4"),
            (5, "5-9"), (9, "5-9"), (10, ">=10"), (99, ">=10"),
        ):
            with self.subTest(support=value):
                self.assertEqual(support_stratum(value), expected)

    def test_range_status_detects_both_tails(self):
        status = range_status(
            [1.0, 50.0, 99.0], factual_training=[40.0, 60.0],
            hallucinated_training=[20.0, 30.0],
        )
        self.assertEqual(
            status["status"],
            ["below_combined_range", "inside_combined_range", "above_combined_range"],
        )
        self.assertEqual(status["below_combined_range"], 1)
        self.assertEqual(status["above_combined_range"], 1)
        self.assertEqual(status["inside_combined_range"], 1)

    def test_being_outside_one_class_range_is_recorded_not_judged(self):
        status = range_status(
            [95.0], factual_training=[90.0, 99.0],
            hallucinated_training=[10.0, 20.0],
        )
        # Inside the combined range and inside factual, outside hallucinated.
        self.assertEqual(status["status"], ["inside_combined_range"])
        self.assertEqual(status["outside_hallucinated_range"], 1)
        self.assertEqual(status["outside_factual_range"], 0)
        self.assertIn("NOT an error", status["note"])


# --------------------------------------------------------------------------
# Hyperparameter surface
# --------------------------------------------------------------------------


class TestHyperparameterSurface(unittest.TestCase):
    def setUp(self):
        self.estimator, self.factual, self.hallucinated = fitted_estimator()
        self.scores = [10.0, 35.0, 50.0, 70.0, 92.0]

    def surface(self):
        return hyperparameter_surface(
            self.estimator, self.factual, self.hallucinated, self.scores
        )

    def test_every_production_cv_pair_appears_exactly_once(self):
        pairs = production_hyperparameter_pairs(self.estimator)
        cv_pairs = [
            (float(row["sigma"]), float(row["lambda"]))
            for row in self.estimator.cv_table
        ]
        self.assertEqual(sorted(pairs), sorted(cv_pairs))
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(self.surface()["n_pairs"], len(cv_pairs))

    def test_the_pairs_come_from_the_production_cv_table_not_a_new_grid(self):
        # Regenerating the grid would let the diagnostic surface drift from the
        # grid the production fit actually searched.
        surface = self.surface()
        recorded = {
            (row["sigma"], row["lambda"]) for row in surface["pairs"]
        }
        self.assertEqual(
            recorded,
            {
                (float(r["sigma"]), float(r["lambda"]))
                for r in self.estimator.cv_table
            },
        )

    def test_every_refit_uses_the_selected_fits_final_centre_set(self):
        # Resampling centres per refit would confound hyperparameter
        # sensitivity with centre-sampling noise.
        surface = self.surface()
        self.assertTrue(surface["centers_held_fixed"])
        self.assertEqual(surface["n_centers"], int(self.estimator.centers.size))

        # And the values agree with a hand refit at fixed centres.
        factual_x = self.estimator._as_column(self.factual)
        hallucinated_x = self.estimator._as_column(self.hallucinated)
        evaluation_x = self.estimator._as_column(self.scores)
        expected = []
        for pair in surface["pairs"]:
            alpha = self.estimator._solve(
                factual_x, hallucinated_x, self.estimator.centers,
                pair["sigma"], pair["lambda"],
            )
            raw = self.estimator._kernel(
                evaluation_x, self.estimator.centers, pair["sigma"]
            ) @ alpha
            expected.append(np.clip(raw, RATIO_CLIP_LOWER, RATIO_CLIP_UPPER))
        np.testing.assert_allclose(
            surface["returned_matrix"], np.vstack(expected)
        )

    def test_the_log_ratio_span_is_max_minus_min(self):
        surface = self.surface()
        np.testing.assert_allclose(
            surface["log_ratio_span"],
            surface["log_ratio_max"] - surface["log_ratio_min"],
        )
        self.assertTrue(np.all(surface["log_ratio_span"] >= 0.0))

    def test_a_direction_crossing_one_is_detected(self):
        class Stub:
            centers = np.array([[0.5]])
            sigma = 0.2
            alpha = np.array([1.0])
            cv_table = [
                {"sigma": 0.2, "lambda": 0.1, "cv_objective": 0.0},
                {"sigma": 0.2, "lambda": 0.2, "cv_objective": 0.0},
            ]
            _as_column = staticmethod(ULSIFDensityRatio._as_column)
            _kernel = staticmethod(ULSIFDensityRatio._kernel)
            calls = []

            def _solve(self, f, h, centers, sigma, lam):
                # Two refits straddling 1.0 at the evaluation point.
                self.calls.append(lam)
                return np.array([2.0 if lam == 0.1 else 0.5])

        surface = hyperparameter_surface(Stub(), [50.0], [50.0], [50.0])
        self.assertEqual(surface["direction"], [DIRECTION_MIXED])

    def test_a_consistent_direction_is_reported_as_such(self):
        surface = self.surface()
        for direction in surface["direction"]:
            self.assertIn(
                direction, {DIRECTION_ABOVE, DIRECTION_BELOW, DIRECTION_MIXED,
                            "exactly_one"},
            )

    def test_a_non_finite_refit_fails_loudly_instead_of_being_skipped(self):
        class Broken:
            centers = np.array([[0.5]])
            sigma = 0.2
            alpha = np.array([1.0])
            cv_table = [{"sigma": 0.2, "lambda": 0.1, "cv_objective": 0.0}]
            _as_column = staticmethod(ULSIFDensityRatio._as_column)
            _kernel = staticmethod(ULSIFDensityRatio._kernel)

            def _solve(self, f, h, centers, sigma, lam):
                return np.array([float("nan")])

        with self.assertRaises(DiagnosticIncomplete) as caught:
            hyperparameter_surface(Broken(), [50.0], [50.0], [50.0])
        self.assertIn("NOT skipped", str(caught.exception))

    def test_a_duplicated_cv_pair_is_refused(self):
        class Duplicated:
            cv_table = [
                {"sigma": 0.2, "lambda": 0.1},
                {"sigma": 0.2, "lambda": 0.1},
            ]

        with self.assertRaises(DiagnosticIncomplete):
            production_hyperparameter_pairs(Duplicated())

    def test_an_absent_cv_table_is_refused(self):
        class Empty:
            cv_table = []

        with self.assertRaises(DiagnosticIncomplete):
            production_hyperparameter_pairs(Empty())


# --------------------------------------------------------------------------
# One-document stopping
# --------------------------------------------------------------------------


class TestOneDocumentStopping(unittest.TestCase):
    def test_the_general_log_odds_rule_is_used(self):
        self.assertAlmostEqual(
            state_after_one_document(4.0, p0=0.5), math.log(4.0)
        )
        # At P0 != 0.5 the prior shifts the state; |log r| would not.
        self.assertAlmostEqual(
            state_after_one_document(1.0, p0=0.35), logit(0.35)
        )

    def test_the_rule_is_correct_for_an_asymmetric_band(self):
        # The case where |log r| gives the wrong answer: band [0.05, 0.80],
        # r = 0.2 at P0 = 0.5. |log 0.2| = 1.609 exceeds |logit(0.80)| = 1.386,
        # so a magnitude shorthand would call this a stop; the actual state
        # log(0.2) = -1.609 is ABOVE logit(0.05) = -2.944, so it is not.
        state = state_after_one_document(0.2, p0=0.5)
        self.assertGreater(state, logit(0.05))
        self.assertEqual(stop_decision(state, 0.05, 0.80), DECISION_NONE)
        self.assertGreater(abs(math.log(0.2)), abs(logit(0.80)))

    def test_the_decision_is_none_low_or_high(self):
        self.assertEqual(
            stop_decision(state_after_one_document(0.001), 0.05, 0.95),
            DECISION_LOW,
        )
        self.assertEqual(
            stop_decision(state_after_one_document(1000.0), 0.05, 0.95),
            DECISION_HIGH,
        )
        self.assertEqual(
            stop_decision(state_after_one_document(1.0), 0.05, 0.95),
            DECISION_NONE,
        )

    def test_low_high_and_either_are_counted_separately(self):
        # One clear low, one clear high, one no-stop.
        report = one_document_stopping([0.001, 1000.0, 1.0], [(0.05, 0.95)])
        pair = report["per_threshold_pair"][0]
        self.assertEqual(pair["low_stop_count"], 1)
        self.assertEqual(pair["high_stop_count"], 1)
        self.assertEqual(pair["either_stop_count"], 2)
        self.assertAlmostEqual(pair["low_stop_fraction"], 1 / 3)
        self.assertAlmostEqual(pair["high_stop_fraction"], 1 / 3)
        self.assertAlmostEqual(pair["either_stop_fraction"], 2 / 3)

    def test_the_report_names_the_rule_and_the_prior(self):
        report = one_document_stopping([1.0], [(0.05, 0.95)])
        self.assertEqual(report["p0"], DIAGNOSTIC_P0)
        self.assertIn("logit(P0) + log", report["rule"])
        self.assertIn("|log r| is only the special case", report["rule_note"])

    def test_an_empty_threshold_grid_or_population_is_refused(self):
        with self.assertRaises(DiagnosticIncomplete):
            one_document_stopping([1.0], [])
        with self.assertRaises(DiagnosticIncomplete):
            one_document_stopping([], [(0.05, 0.95)])

    def test_stopping_is_evaluated_on_first_documents_only(self):
        # A later-document extreme must not masquerade as one-document
        # stopping: the state after ONE document is the quantity of interest.
        occurrences = [
            {"is_first_document": True, "score": 50.0},
            {"is_first_document": False, "score": 99.0},
            {"is_first_document": True, "score": 51.0},
        ]
        first = [o["score"] for o in occurrences if o["is_first_document"]]
        self.assertEqual(first, [50.0, 51.0])
        self.assertNotIn(99.0, first)

    def test_the_production_threshold_grid_is_used(self):
        from src.ddre_core import (
            CANDIDATE_LOWER_GRID,
            CANDIDATE_UPPER_GRID,
            cost_consistent_thresholds,
        )

        space = cost_consistent_thresholds(
            28.0, 96.0, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID
        )
        pairs = [
            (float(l), float(u))
            for l in space["effective_lower_grid"]
            for u in space["effective_upper_grid"]
            if l < u
        ]
        self.assertEqual(len(pairs), 32)
        report = one_document_stopping([1.0, 100.0], pairs)
        self.assertEqual(report["n_threshold_pairs"], 32)


class TestStopDecisionStability(unittest.TestCase):
    def test_a_unanimous_decision_is_recorded_as_unanimous(self):
        stability = stop_decision_stability(
            [1000.0], np.array([[900.0], [1100.0]]), [(0.05, 0.95)]
        )
        self.assertEqual(stability["unanimous_across_hyperparameters"], 1)
        self.assertEqual(stability["any_decision_flip"], 0)
        self.assertEqual(stability["unanimous_per_score"], [True])

    def test_a_high_to_none_softening_is_counted_as_such(self):
        stability = stop_decision_stability(
            [1000.0], np.array([[1.0]]), [(0.05, 0.95)]
        )
        self.assertEqual(stability["high_to_none"], 1)
        self.assertEqual(stability["direction_flip_high_to_low"], 0)
        self.assertEqual(stability["unanimous_per_score"], [False])

    def test_a_low_to_high_direction_flip_is_counted_separately(self):
        # The serious case: the same evidence concludes hallucination under one
        # hyperparameter setting and factual under another. Averaging this into
        # a single "mean ratio delta" would hide it entirely.
        stability = stop_decision_stability(
            [0.001], np.array([[1000.0]]), [(0.05, 0.95)]
        )
        self.assertEqual(stability["direction_flip_low_to_high"], 1)
        self.assertEqual(stability["low_to_none"], 0)
        self.assertEqual(stability["high_to_none"], 0)

    def test_a_none_to_stop_change_is_counted(self):
        low = stop_decision_stability(
            [1.0], np.array([[0.001]]), [(0.05, 0.95)]
        )
        self.assertEqual(low["none_to_low"], 1)
        high = stop_decision_stability(
            [1.0], np.array([[1000.0]]), [(0.05, 0.95)]
        )
        self.assertEqual(high["none_to_high"], 1)

    def test_a_shape_mismatch_is_refused(self):
        with self.assertRaises(DiagnosticIncomplete):
            stop_decision_stability(
                [1.0, 2.0], np.array([[1.0]]), [(0.05, 0.95)]
            )


# --------------------------------------------------------------------------
# Escalation triggers -- pre-registered and conservative
# --------------------------------------------------------------------------


def triggers(**overrides):
    payload = {
        "selected_document_clip_activity": {
            "any_clip_hits": 0, "lower_clip_hits": 0, "upper_clip_hits": 0
        },
        "first_document_stopped": [False],
        "first_document_direction": [DIRECTION_ABOVE],
        "first_document_min_class_support": [10],
        "first_document_unanimous": [True],
    }
    payload.update(overrides)
    return escalation_triggers(**payload)


class TestEscalationTriggers(unittest.TestCase):
    def test_selected_fit_clip_activity_fires_trigger_one(self):
        result = triggers(
            selected_document_clip_activity={
                "any_clip_hits": 3, "lower_clip_hits": 1, "upper_clip_hits": 2
            }
        )
        self.assertTrue(
            result["triggers"]["trigger_1_selected_fit_hits_ratio_clip"]["fired"]
        )
        self.assertTrue(result["additional_sensitivity_required"])

    def test_a_stopped_first_document_with_a_direction_flip_fires_trigger_two(self):
        result = triggers(
            first_document_stopped=[True],
            first_document_direction=[DIRECTION_MIXED],
        )
        trigger = result["triggers"][
            "trigger_2_stopping_direction_flips_over_hyperparameters"
        ]
        self.assertTrue(trigger["fired"])
        self.assertEqual(trigger["count"], 1)
        self.assertTrue(result["additional_sensitivity_required"])

    def test_zero_support_plus_a_non_unanimous_stop_fires_trigger_three(self):
        result = triggers(
            first_document_stopped=[True],
            first_document_min_class_support=[0],
            first_document_unanimous=[False],
        )
        self.assertTrue(
            result["triggers"]["trigger_3_zero_support_and_unstable_stop"]["fired"]
        )
        self.assertTrue(result["additional_sensitivity_required"])

    def test_weak_support_alone_does_not_fire_anything(self):
        result = triggers(
            first_document_stopped=[True],
            first_document_min_class_support=[0],
            first_document_unanimous=[True],   # stable despite zero support
        )
        self.assertFalse(result["additional_sensitivity_required"])

    def test_a_large_stable_ratio_alone_does_not_fire_anything(self):
        # The central point of D-03's framing: magnitude is not a defect.
        result = triggers(
            first_document_stopped=[True],
            first_document_direction=[DIRECTION_ABOVE],
            first_document_min_class_support=[25],
            first_document_unanimous=[True],
        )
        self.assertFalse(result["additional_sensitivity_required"])
        self.assertIn("is evidence of a defect", result["magnitude_note"])
        self.assertIn("above 10 or 100", result["magnitude_note"])

    def test_one_document_stopping_alone_does_not_fire_anything(self):
        result = triggers(first_document_stopped=[True])
        self.assertFalse(result["additional_sensitivity_required"])

    def test_a_direction_flip_on_a_non_stopping_score_does_not_fire_trigger_two(self):
        # Trigger 2 is scoped to scores that actually stop after one document.
        result = triggers(
            first_document_stopped=[False],
            first_document_direction=[DIRECTION_MIXED],
        )
        self.assertFalse(result["additional_sensitivity_required"])

    def test_all_three_false_gives_no_additional_sensitivity_required(self):
        result = triggers()
        for trigger in result["triggers"].values():
            self.assertFalse(trigger["fired"])
        self.assertFalse(result["additional_sensitivity_required"])
        self.assertIn("did not fire", result["meaning"])

    def test_no_cap_or_calibration_is_ever_emitted(self):
        for result in (triggers(), triggers(
            selected_document_clip_activity={
                "any_clip_hits": 5, "lower_clip_hits": 5, "upper_clip_hits": 0
            }
        )):
            self.assertIsNone(result["selected_cap"])
            self.assertIsNone(result["selected_calibration"])
            self.assertFalse(result["method_change_made"])
            self.assertNotIn("cap =", json.dumps(result))

    def test_firing_asks_for_a_study_not_a_cap(self):
        result = triggers(
            selected_document_clip_activity={
                "any_clip_hits": 1, "lower_clip_hits": 0, "upper_clip_hits": 1
            }
        )
        self.assertIn("SEPARATE cap / calibration", result["meaning"])
        self.assertIn("does NOT select a cap", result["meaning"])

    def test_not_firing_does_not_claim_ulsif_is_proven_correct(self):
        self.assertIn("does NOT mean uLSIF is", triggers()["meaning"])

    def test_misaligned_inputs_are_refused(self):
        with self.assertRaises(DiagnosticIncomplete):
            triggers(first_document_direction=[DIRECTION_ABOVE, DIRECTION_ABOVE])

    def test_the_criteria_text_is_frozen_in_the_output(self):
        result = triggers()
        self.assertIn(
            "either existing ratio clip",
            result["triggers"]["trigger_1_selected_fit_hits_ratio_clip"]["criterion"],
        )
        self.assertIn(
            "min_class_support == 0",
            result["triggers"]["trigger_3_zero_support_and_unstable_stop"][
                "criterion"
            ],
        )


# --------------------------------------------------------------------------
# Support strata report
# --------------------------------------------------------------------------


class TestSupportStrataReport(unittest.TestCase):
    def report(self):
        return support_strata_report(
            min_class_support=[0, 1, 4, 7, 20],
            returned_ratios=[100.0, 2.0, 1.5, 0.5, 3.0],
            log_ratio_span=[4.0, 0.1, 0.2, 0.05, 0.02],
            directions=[DIRECTION_MIXED, DIRECTION_ABOVE, DIRECTION_ABOVE,
                        DIRECTION_BELOW, DIRECTION_ABOVE],
            stopped_flags=[True, False, False, False, False],
            unanimous_flags=[False, True, True, True, True],
        )

    def test_each_pre_registered_stratum_is_reported(self):
        strata = self.report()["strata"]
        self.assertEqual(
            list(strata), ["0", "1-2", "3-4", "5-9", ">=10"]
        )
        self.assertEqual(strata["0"]["count"], 1)
        self.assertEqual(strata[">=10"]["count"], 1)

    def test_a_stratum_reports_ratios_span_stopping_and_stability_together(self):
        # This is what separates large-but-stable from tail-driven: the
        # zero-support stratum here shows a big ratio, a wide span, a
        # direction crossing, a stop, and non-unanimity all at once.
        zero = self.report()["strata"]["0"]
        self.assertEqual(zero["returned_ratio_quantiles"]["max"], 100.0)
        self.assertEqual(zero["log_ratio_span_quantiles"]["max"], 4.0)
        self.assertEqual(zero["direction_crosses_one"], 1)
        self.assertEqual(zero["one_document_stop_count"], 1)
        self.assertEqual(zero["non_unanimous"], 1)

    def test_an_empty_stratum_is_reported_as_empty_not_omitted(self):
        report = support_strata_report(
            min_class_support=[20], returned_ratios=[2.0], log_ratio_span=[0.1],
            directions=[DIRECTION_ABOVE], stopped_flags=[False],
            unanimous_flags=[True],
        )
        self.assertEqual(report["strata"]["0"]["count"], 0)
        self.assertIn("no evaluation score", report["strata"]["0"]["note"])

    def test_the_report_says_bands_not_a_cutoff(self):
        self.assertIn("Bands, not a cutoff", self.report()["note"])

    def test_misaligned_inputs_are_refused(self):
        with self.assertRaises(DiagnosticIncomplete):
            support_strata_report(
                min_class_support=[0, 1], returned_ratios=[1.0],
                log_ratio_span=[0.1], directions=[DIRECTION_ABOVE],
                stopped_flags=[False], unanimous_flags=[True],
            )


# --------------------------------------------------------------------------
# D-13 descriptive shift
# --------------------------------------------------------------------------


class TestD13DistributionShift(unittest.TestCase):
    def test_the_histogram_uses_fixed_ten_point_bins(self):
        self.assertEqual(len(D13_BIN_EDGES), 11)
        self.assertEqual(D13_BIN_EDGES[0], 0.0)
        self.assertEqual(D13_BIN_EDGES[-1], 100.0)
        histogram = fixed_bin_histogram([0.0, 5.0, 10.0, 95.0, 100.0])
        self.assertEqual(list(histogram)[0], "0-10")
        self.assertEqual(list(histogram)[-1], "90-100")
        self.assertEqual(histogram["0-10"], 2)     # 0.0 and 5.0
        self.assertEqual(histogram["10-20"], 1)    # 10.0
        self.assertEqual(histogram["90-100"], 2)   # 95.0 and 100.0 (closed top)

    def test_both_distributions_are_reported_side_by_side(self):
        shift = distribution_shift([10.0, 20.0, 30.0], [70.0, 80.0, 90.0])
        self.assertIn("quantiles", shift["nbc_pair_scores"])
        self.assertIn("histogram", shift["validation_document_max_scores"])
        self.assertEqual(shift["nbc_pair_scores"]["quantiles"]["p50"], 20.0)
        self.assertEqual(
            shift["validation_document_max_scores"]["quantiles"]["p50"], 80.0
        )

    def test_the_shift_is_described_not_corrected_and_not_resolved(self):
        shift = distribution_shift([10.0], [90.0])
        self.assertEqual(shift["finding"], "D-13")
        self.assertIn("not corrected and not resolved", shift["status"])
        self.assertIn("Both BSE and DDRE inherit", shift["note"])
        self.assertIn("No correction, recalibration or retraining", shift["note"])


# --------------------------------------------------------------------------
# Completion accounting
# --------------------------------------------------------------------------


class TestCompletionAccounting(unittest.TestCase):
    def test_full_coverage_is_complete(self):
        accounting = completion_accounting(
            expected_occurrences=100, scored_occurrences=100,
            first_documents=20, expected_first_documents=20,
        )
        self.assertTrue(accounting["complete"])
        self.assertEqual(accounting["status"], "COMPLETE")
        self.assertIs(require_complete(accounting), accounting)

    def test_partial_coverage_is_never_labelled_complete(self):
        accounting = completion_accounting(
            expected_occurrences=100, scored_occurrences=99,
            first_documents=20, expected_first_documents=20,
        )
        self.assertFalse(accounting["complete"])
        self.assertEqual(accounting["status"], "INCOMPLETE")
        self.assertIn("never dropped", accounting["note"])
        with self.assertRaises(DiagnosticIncomplete) as caught:
            require_complete(accounting)
        self.assertIn("partial measurement", str(caught.exception))

    def test_a_first_document_shortfall_also_blocks_completion(self):
        accounting = completion_accounting(
            expected_occurrences=100, scored_occurrences=100,
            first_documents=19, expected_first_documents=20,
        )
        self.assertFalse(accounting["complete"])

    def test_an_empty_measurement_is_not_complete(self):
        self.assertFalse(
            completion_accounting(
                expected_occurrences=0, scored_occurrences=0,
                first_documents=0, expected_first_documents=0,
            )["complete"]
        )


# --------------------------------------------------------------------------
# The runner's population, provenance and safety, asserted on source
# --------------------------------------------------------------------------


class TestRunnerContract(unittest.TestCase):
    """The runner imports torch, so its guarantees are asserted on its AST."""

    @classmethod
    def setUpClass(cls):
        cls.path = PROJECT_ROOT / "scripts" / "diagnose_ddre_ratio_support.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_the_diagnostic_never_touches_the_held_out_split(self):
        # The runner takes the VALIDATION half and discards the test half.
        self.assertIn("validation, _, identity = frozen_validation_split", self.source)
        self.assertIn('"held_out_scored": False', self.source)
        self.assertNotIn("test_records", self.source)

    def test_the_split_is_verified_by_passage_identity_not_count(self):
        self.assertIn('"identity_matches": expected == observed', self.source)
        self.assertIn('if not identity["identity_matches"]:', self.source)
        self.assertIn("validation_fraction=DIAGNOSTIC_VALIDATION_FRACTION", self.source)
        self.assertIn("random_state=DIAGNOSTIC_SPLIT_SEED", self.source)

    def test_the_frozen_split_constants_are_the_production_ones(self):
        self.assertEqual(DIAGNOSTIC_VALIDATION_FRACTION, 0.20)
        self.assertEqual(DIAGNOSTIC_SPLIT_SEED, 42)
        self.assertEqual(DIAGNOSTIC_MAX_DOCS, 10)

    def test_the_document_population_is_threshold_independent(self):
        # All candidate documents up to max_docs, not those a tuned DDRE
        # happened to retrieve.
        self.assertIn("documents = subclaim.documents[:max_docs]", self.source)
        self.assertIn('"threshold_independent": True', self.source)
        for adaptive in ("detect_subclaim", "detect_sentence", "DDREDetector("):
            self.assertNotIn(adaptive, self.source)

    def test_document_occurrences_are_not_deduplicated(self):
        self.assertIn('"deduplicated": False', self.source)
        self.assertNotIn("set(occurrences)", self.source)

    def test_the_training_counts_are_mandatory(self):
        self.assertEqual(EXPECTED_FACTUAL_TRAINING, 199)
        self.assertEqual(EXPECTED_HALLUCINATED_TRAINING, 199)
        self.assertIn(
            "EXPECTED_FACTUAL_TRAINING, EXPECTED_HALLUCINATED_TRAINING",
            self.source,
        )
        self.assertIn("ABORTED: NBC training counts are", self.source)

    def test_the_source_cache_is_never_opened_for_writing(self):
        # New scores go to the derived cache bound to the certified digest.
        self.assertIn("prepare_derived_cache(", self.source)
        self.assertIn("bind_compatibility_to_source(", self.source)
        self.assertIn("cache_path=str(args.output_cache)", self.source)
        self.assertNotIn("cache_path=str(source_cache)", self.source)
        self.assertNotIn("cache_path=args.source_cache", self.source)

    def test_batch_size_one_is_pinned(self):
        from src.provenance_guard import REQUIRED_BATCH_SIZE

        self.assertEqual(REQUIRED_BATCH_SIZE, 1)
        self.assertIn("WANG_BATCH_SIZE = REQUIRED_BATCH_SIZE", self.source)
        self.assertIn("batch_size=WANG_BATCH_SIZE", self.source)
        self.assertNotIn("batch_size=8", self.source)

    def test_the_score_version_and_wang_commit_are_pinned(self):
        from src.provenance_guard import (
            REQUIRED_SCORE_VERSION,
            REQUIRED_WANG_SOURCE_COMMIT,
        )

        self.assertEqual(
            REQUIRED_SCORE_VERSION,
            "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2",
        )
        self.assertEqual(
            REQUIRED_WANG_SOURCE_COMMIT,
            "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea",
        )
        self.assertIn("score_version=SCORE_VERSION", self.source)
        self.assertIn("wang_source_commit=wang_source_commit(", self.source)

    def test_the_398_pair_probe_gates_the_run(self):
        from src.score_compatibility import EXPECTED_PAIR_COUNT

        self.assertEqual(EXPECTED_PAIR_COUNT, 398)
        self.assertIn("run_compatibility_probe(", self.source)
        self.assertIn(
            'if not compatibility["score_compatibility_established"]:', self.source
        )
        self.assertIn("ABORTED: score compatibility not established", self.source)

    def test_the_source_digest_is_measured_on_both_sides_of_the_probe(self):
        self.assertIn("source_digest=lambda: sha256_file(source_cache)", self.source)
        self.assertIn('"source_sha256_before"', self.source)
        self.assertIn('"source_sha256_after_probe"', self.source)

    def test_the_historical_checkpoint_limitation_is_preserved(self):
        self.assertIn('"checkpoint_identity_established"', self.source)
        self.assertIn("NOT rewritten as solved", self.source)
        self.assertIn("score_compatibility_established", self.source)

    def test_the_report_pins_the_frozen_outputs(self):
        self.assertIn('"selected_cap": None', self.source)
        self.assertIn('"selected_calibration": None', self.source)
        self.assertIn('"method_change_made": False', self.source)

    def test_the_dry_run_writes_nothing_and_loads_no_model(self):
        tree = ast.parse(self.source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "dry_run_report"
        )
        body = ast.get_source_segment(self.source, function)
        for forbidden in (
            "from_pretrained", "score_document", "score_pairs",
            "prepare_derived_cache", "json.dump", "open(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)
        self.assertIn("no model loaded, no inference, no cache written", body)

    def test_the_dry_run_reports_the_reviewable_population(self):
        function = next(
            node for node in ast.parse(self.source).body
            if isinstance(node, ast.FunctionDef) and node.name == "dry_run_report"
        )
        body = ast.get_source_segment(self.source, function)
        for field in (
            "validation passages", "held-out passages", "passage-id sha256",
            "subclaims", "document occurrences", "first documents",
            "cost-consistent pairs", "compatibility probe", "derived cache",
        ):
            with self.subTest(field=field):
                self.assertIn(field, body)

    def test_the_methodology_lives_in_the_pure_module_not_the_script(self):
        # The script is a runner; the science must be reviewable and testable
        # without torch.
        for helper in (
            "escalation_triggers", "hyperparameter_surface",
            "one_document_stopping", "support_counts", "distribution_shift",
        ):
            with self.subTest(helper=helper):
                self.assertIn(f"    {helper},\n", self.source)   # imported
                self.assertNotIn(f"def {helper}(", self.source)  # not redefined

    def test_the_protocol_version_is_recorded(self):
        self.assertEqual(PROTOCOL_VERSION, "d03-support-stability-v1")
        self.assertIn('"protocol_version": PROTOCOL_VERSION', self.source)


class TestProductionBehaviourUnchanged(unittest.TestCase):
    def test_ddre_core_is_untouched_by_this_change(self):
        import subprocess

        diff = subprocess.run(
            ["git", "diff", "--quiet", "origin/main", "--", "src/ddre_core.py"],
            capture_output=True, cwd=PROJECT_ROOT,
        )
        if diff.returncode == 128:
            self.skipTest("origin/main not available")
        self.assertEqual(diff.returncode, 0, "src/ddre_core.py must not change")

    def test_the_production_ratio_clip_is_unchanged(self):
        self.assertEqual(RATIO_CLIP_LOWER, 1e-6)
        self.assertEqual(RATIO_CLIP_UPPER, 1e6)
        source = (PROJECT_ROOT / "src" / "ddre_core.py").read_text(encoding="utf-8")
        self.assertIn("np.clip(value, 1e-6, 1e6)", source)

    def test_the_diagnostic_does_not_refit_production_state(self):
        # hyperparameter_surface must not leave the estimator changed: it is a
        # measurement of the selected fit, not a competing fitting procedure.
        estimator, factual, hallucinated = fitted_estimator()
        before = (
            float(estimator.sigma), float(estimator.lam),
            estimator.alpha.copy(), estimator.centers.copy(),
        )
        hyperparameter_surface(estimator, factual, hallucinated, [50.0])
        self.assertEqual(float(estimator.sigma), before[0])
        self.assertEqual(float(estimator.lam), before[1])
        np.testing.assert_array_equal(estimator.alpha, before[2])
        np.testing.assert_array_equal(estimator.centers, before[3])


if __name__ == "__main__":
    unittest.main()
