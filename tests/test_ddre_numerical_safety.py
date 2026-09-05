"""Tests for fail-closed DDRE numerical safety (audit findings D-04 and D-06).

The principle under test: **an invalid numerical state must never be converted
into evidence.** A NaN is not a weak signal, an infinite score is not a perfect
one, and an identically-zero fitted ratio is not overwhelming proof of
hallucination -- each is a broken computation, and the code must say so.

What is deliberately NOT tested here, because it is deliberately not enforced:
no fit is rejected for the SCALE or SPREAD of its ratios. Large ratios, small
positive ratios, a narrow range and heavy class overlap all remain acceptable.
Those are D-03 questions and no threshold for them is introduced.

numpy + stdlib only, so CI's numpy-only install runs the whole file.
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.ddre_core import (
    DDRENumericalError,
    DDREDetector,
    DegenerateULSIFFit,
    NonFiniteScoreError,
    ULSIFDensityRatio,
)

C_MISS, C_FALSE_ALARM = 28.0, 96.0
NON_FINITE = (float("nan"), float("inf"), float("-inf"))


def fitted(seed=7, n=60):
    rng = np.random.default_rng(seed)
    return ULSIFDensityRatio(max_centers=25, random_state=seed).fit(
        np.clip(rng.normal(75, 8, n), 0, 100),
        np.clip(rng.normal(25, 8, n), 0, 100),
        folds=4,
    )


class ConstantRatio:
    """A ratio estimator that returns whatever it is told, including nonsense."""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def ratio(self, score):
        self.calls.append(score)
        return self.value


class FixedScorer:
    def __init__(self, score, spans=1):
        self.score = score
        self.spans = spans
        self.calls = 0

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        self.calls += 1
        return self.score, self.spans


class SequenceScorer:
    """Serves a list of scores in order, so 'stopped early' is observable."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        score = self.scores[self.calls]
        self.calls += 1
        return score, 1


class Doc:
    page_content = "word " * 10
    url = "https://example.invalid/doc"


class Sub:
    def __init__(self, n_documents, text="a subclaim"):
        self.text = text
        self.documents = [Doc() for _ in range(n_documents)]


def detector(estimator, **overrides):
    kwargs = {
        "lower_threshold": 0.20,
        "upper_threshold": 0.80,
        "p0": 0.5,
        "c_miss": C_MISS,
        "c_false_alarm": C_FALSE_ALARM,
        "max_docs": 10,
    }
    kwargs.update(overrides)
    return DDREDetector(estimator, **kwargs)


# --------------------------------------------------------------------------
# A. uLSIF training inputs
# --------------------------------------------------------------------------


class TestTrainingInputValidation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.factual = np.clip(rng.normal(75, 8, 40), 0, 100)
        self.hallucinated = np.clip(rng.normal(25, 8, 40), 0, 100)

    def fit(self, factual=None, hallucinated=None):
        return ULSIFDensityRatio(max_centers=20, random_state=3).fit(
            self.factual if factual is None else factual,
            self.hallucinated if hallucinated is None else hallucinated,
            folds=4,
        )

    def test_ordinary_finite_data_still_fits(self):
        estimator = self.fit()
        self.assertIsNotNone(estimator.alpha)
        self.assertGreater(estimator.ratio(80.0), 0.0)

    def test_a_nan_factual_score_is_rejected(self):
        bad = self.factual.copy()
        bad[5] = float("nan")
        with self.assertRaises(NonFiniteScoreError) as caught:
            self.fit(factual=bad)
        message = str(caught.exception)
        self.assertIn("factual_scores", message)
        self.assertIn("index 5", message)

    def test_a_positive_infinite_factual_score_is_rejected(self):
        bad = self.factual.copy()
        bad[0] = float("inf")
        with self.assertRaises(NonFiniteScoreError):
            self.fit(factual=bad)

    def test_a_negative_infinite_hallucinated_score_is_rejected(self):
        bad = self.hallucinated.copy()
        bad[9] = float("-inf")
        with self.assertRaises(NonFiniteScoreError) as caught:
            self.fit(hallucinated=bad)
        message = str(caught.exception)
        self.assertIn("hallucinated_scores", message)
        self.assertIn("index 9", message)

    def test_the_error_names_the_count_and_stays_bounded(self):
        bad = self.factual.copy()
        bad[:20] = float("nan")
        with self.assertRaises(NonFiniteScoreError) as caught:
            self.fit(factual=bad)
        message = str(caught.exception)
        self.assertIn("20 non-finite value(s)", message)
        self.assertIn("and 15 more", message)  # only 5 offenders are listed
        self.assertLess(len(message), 500, "must not dump the array")

    def test_finite_normalization_behaviour_is_unchanged(self):
        # score / 100 then clip to [0, 1]: untouched by this PR.
        estimator = ULSIFDensityRatio()
        np.testing.assert_allclose(
            estimator._as_column([0.0, 50.0, 100.0]).ravel(), [0.0, 0.5, 1.0]
        )
        np.testing.assert_allclose(
            estimator._as_column([-10.0, 250.0]).ravel(), [0.0, 1.0]
        )


# --------------------------------------------------------------------------
# B. Hyperparameter selection safety
# --------------------------------------------------------------------------


class TestModelSelectionSafety(unittest.TestCase):
    def test_the_selected_objective_is_finite_in_a_normal_fit(self):
        estimator = fitted()
        objectives = [row["cv_objective"] for row in estimator.cv_table]
        self.assertTrue(all(math.isfinite(v) for v in objectives))
        self.assertTrue(math.isfinite(estimator.fit_diagnostics["cv_objective"]))
        self.assertEqual(estimator.fit_diagnostics["cv_objective"], min(objectives))

    def test_a_non_finite_cv_objective_is_rejected_not_skipped(self):
        # Silently dropping a candidate would change the effective
        # model-selection search space, which is a scientific behaviour change.
        class PoisonedObjective(ULSIFDensityRatio):
            def _objective(self, *args, **kwargs):
                return float("nan")

        rng = np.random.default_rng(5)
        with self.assertRaises(DegenerateULSIFFit) as caught:
            PoisonedObjective(max_centers=15, random_state=5).fit(
                np.clip(rng.normal(75, 8, 30), 0, 100),
                np.clip(rng.normal(25, 8, 30), 0, 100),
                folds=3,
            )
        message = str(caught.exception)
        self.assertIn("non-finite objective", message)
        self.assertIn("Refusing to select", message)

    def test_a_nan_objective_cannot_win_selection_by_losing_comparisons(self):
        # NaN < x is False and x < NaN is False, so an unchecked NaN could
        # survive as `best` purely through comparison behaviour.
        self.assertFalse(float("nan") < 1.0)
        self.assertFalse(1.0 < float("nan"))

    def test_the_candidate_grids_are_unchanged(self):
        estimator = fitted()
        self.assertEqual(
            sorted({row["lambda"] for row in estimator.cv_table}),
            [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        )
        self.assertTrue(
            all(math.isfinite(row["sigma"]) and row["sigma"] > 0
                for row in estimator.cv_table)
        )


# --------------------------------------------------------------------------
# C. Final-fit sanity check (D-04)
# --------------------------------------------------------------------------


class TestFinalFitSanityCheck(unittest.TestCase):
    def setUp(self):
        self.estimator = fitted()
        self.factual_x = self.estimator._as_column([70.0, 80.0, 90.0])
        self.hallucinated_x = self.estimator._as_column([10.0, 20.0, 30.0])

    def validate(self):
        return self.estimator._validate_final_fit(
            self.factual_x, self.hallucinated_x, cv_objective=-0.5
        )

    def test_a_valid_finite_nonzero_fit_passes(self):
        record = self.validate()
        self.assertTrue(record["sanity_check_passed"])
        self.assertGreater(record["n_positive_alpha"], 0)

    def test_an_all_zero_alpha_is_rejected(self):
        self.estimator.alpha = np.zeros_like(self.estimator.alpha)
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("no coefficient is strictly positive", str(caught.exception))
        self.assertIn("1e-6", str(caught.exception))

    def test_a_nan_alpha_is_rejected_as_a_coefficient_problem(self):
        # Asserting WHICH check fires matters: a non-finite alpha also makes the
        # fitted ratios non-finite, so a later check would catch it too and the
        # coefficient check could be removed unnoticed. The diagnosis would then
        # point at the ratios rather than at the parameters that produced them.
        self.estimator.alpha = self.estimator.alpha.copy()
        self.estimator.alpha[0] = float("nan")
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("coefficients (alpha)", str(caught.exception))

    def test_an_infinite_alpha_is_rejected_as_a_coefficient_problem(self):
        self.estimator.alpha = self.estimator.alpha.copy()
        self.estimator.alpha[1] = float("inf")
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("coefficients (alpha)", str(caught.exception))

    def test_finite_coefficients_that_overflow_the_ratio_are_rejected(self):
        # Every coefficient is finite, so the alpha check passes; their sum
        # overflows to infinity, so only the fitted-ratio check can catch it.
        self.estimator.alpha = np.full_like(self.estimator.alpha, 1e308)
        self.assertTrue(np.all(np.isfinite(self.estimator.alpha)))
        with np.errstate(over="ignore"):  # the overflow is the point of the test
            with self.assertRaises(DegenerateULSIFFit) as caught:
                self.validate()
        self.assertIn("fitted ratios on the training support", str(caught.exception))

    def test_an_empty_alpha_is_rejected(self):
        self.estimator.alpha = np.zeros(0)
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("no coefficients", str(caught.exception))

    def test_a_negative_alpha_is_rejected(self):
        self.estimator.alpha = self.estimator.alpha.copy()
        self.estimator.alpha[0] = -1.0
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("non-negative", str(caught.exception))

    def test_non_finite_centers_are_rejected(self):
        self.estimator.centers = self.estimator.centers.copy()
        self.estimator.centers[0, 0] = float("nan")
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("kernel centers", str(caught.exception))

    def test_empty_centers_are_rejected(self):
        self.estimator.centers = np.zeros((0, 1))
        with self.assertRaises(DegenerateULSIFFit):
            self.validate()

    def test_an_invalid_sigma_is_rejected(self):
        for sigma in (0.0, -1.0, float("nan"), float("inf"), None):
            with self.subTest(sigma=sigma):
                self.setUp()
                self.estimator.sigma = sigma
                with self.assertRaises(DegenerateULSIFFit):
                    self.validate()

    def test_an_invalid_lambda_is_rejected(self):
        for lam in (-1.0, float("nan"), float("inf"), None):
            with self.subTest(lam=lam):
                self.setUp()
                self.estimator.lam = lam
                with self.assertRaises(DegenerateULSIFFit):
                    self.validate()

    def test_identically_zero_fitted_ratios_are_rejected(self):
        # Positive alpha but centres so far from the data that every kernel
        # value underflows: the ratio is zero everywhere it is observed.
        self.estimator.centers = np.array([[1e6]])
        self.estimator.alpha = np.array([1.0])
        self.estimator.sigma = 1e-6
        with self.assertRaises(DegenerateULSIFFit) as caught:
            self.validate()
        self.assertIn("identically zero", str(caught.exception))

    def test_a_failed_fit_leaves_no_usable_estimator(self):
        class DegenerateSolve(ULSIFDensityRatio):
            def _solve(self, factual_x, hallucinated_x, centers, sigma, lam):
                alpha = super()._solve(factual_x, hallucinated_x, centers, sigma, lam)
                # Degenerate only on the final fit, so cross-validation still runs.
                if len(centers) == min(self.max_centers, len(factual_x)) and \
                        len(factual_x) == self._n_final:
                    return np.zeros_like(alpha)
                return alpha

        rng = np.random.default_rng(11)
        factual = np.clip(rng.normal(75, 8, 30), 0, 100)
        hallucinated = np.clip(rng.normal(25, 8, 30), 0, 100)
        estimator = DegenerateSolve(max_centers=15, random_state=11)
        estimator._n_final = len(factual)
        with self.assertRaises(DegenerateULSIFFit):
            estimator.fit(factual, hallucinated, folds=3)
        self.assertIsNone(estimator.alpha)
        self.assertIsNone(estimator.centers)
        self.assertIsNone(estimator.fit_diagnostics)
        with self.assertRaises(RuntimeError):
            estimator.ratio(50.0)

    def test_no_ratio_scale_or_separation_threshold_was_introduced(self):
        # The claim boundary: validity is about numerical invalidity, NOT about
        # how large, small, spread out or well separated the ratios are. A fit
        # whose ratios are all tiny-but-positive, or all nearly equal, or built
        # from heavily overlapping classes, must still pass.
        rng = np.random.default_rng(23)
        overlapping = ULSIFDensityRatio(max_centers=20, random_state=23).fit(
            np.clip(rng.normal(50, 20, 60), 0, 100),
            np.clip(rng.normal(50, 20, 60), 0, 100),
            folds=4,
        )
        self.assertTrue(overlapping.fit_diagnostics["sanity_check_passed"])

        tiny = fitted()
        tiny.alpha = tiny.alpha * 1e-12
        record = tiny._validate_final_fit(
            self.factual_x, self.hallucinated_x, cv_objective=-0.5
        )
        self.assertTrue(record["sanity_check_passed"])
        self.assertLess(record["raw_ratio_max_on_all_train"], 1e-6)

        flat = fitted()
        flat.alpha = np.full_like(flat.alpha, 0.01)
        self.assertTrue(
            flat._validate_final_fit(
                self.factual_x, self.hallucinated_x, cv_objective=-0.5
            )["sanity_check_passed"]
        )


# --------------------------------------------------------------------------
# D. Fit diagnostics
# --------------------------------------------------------------------------


class TestFitDiagnostics(unittest.TestCase):
    def test_a_successful_fit_exposes_the_required_record(self):
        record = fitted().fit_diagnostics
        for key in (
            "n_factual", "n_hallucinated", "sigma", "lambda", "cv_objective",
            "n_centers", "n_alpha", "n_positive_alpha", "alpha_sum",
            "alpha_min", "alpha_max",
            "raw_ratio_min_on_factual_train", "raw_ratio_max_on_factual_train",
            "raw_ratio_min_on_hallucinated_train",
            "raw_ratio_max_on_hallucinated_train",
            "raw_ratio_min_on_all_train", "raw_ratio_max_on_all_train",
            "sanity_check_passed",
        ):
            self.assertIn(key, record)
        self.assertTrue(record["sanity_check_passed"])

    def test_the_record_is_json_serializable(self):
        json.dumps(fitted().fit_diagnostics)

    def test_the_record_is_read_only(self):
        estimator = fitted()
        estimator.fit_diagnostics["sigma"] = 999.0
        self.assertNotEqual(estimator.fit_diagnostics["sigma"], 999.0)

    def test_it_is_none_before_a_successful_fit(self):
        self.assertIsNone(ULSIFDensityRatio().fit_diagnostics)

    def test_the_counts_match_the_fitted_state(self):
        estimator = fitted(n=50)
        record = estimator.fit_diagnostics
        self.assertEqual(record["n_factual"], 50)
        self.assertEqual(record["n_hallucinated"], 50)
        self.assertEqual(record["n_alpha"], np.size(estimator.alpha))
        self.assertEqual(record["n_centers"], np.size(estimator.centers))
        self.assertEqual(
            record["n_positive_alpha"], int(np.count_nonzero(estimator.alpha > 0))
        )
        self.assertAlmostEqual(record["alpha_sum"], float(np.sum(estimator.alpha)))

    def test_the_record_states_it_does_not_tune_the_model(self):
        self.assertIn("do not tune or", fitted().fit_diagnostics["note"])

    def test_the_experiment_summary_carries_the_diagnostics(self):
        # AST only; running main needs torch.
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "main.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn(
            '"ulsif_fit_diagnostics": ratio_estimator.fit_diagnostics', source
        )


# --------------------------------------------------------------------------
# Failed / repeated fit state management
# --------------------------------------------------------------------------


class TestFailedFitInvalidatesPreviousState(unittest.TestCase):
    """A fit attempt must leave the object representing THAT attempt, or nothing.

    Otherwise a second fit that fails early leaves the previous successful model
    in place, and ratio() goes on serving evidence from a fit the caller
    believes was replaced -- while fit_diagnostics describes that older fit,
    making provenance ambiguous exactly when something has gone wrong.
    """

    def setUp(self):
        rng = np.random.default_rng(4)
        self.factual = np.clip(rng.normal(75, 8, 40), 0, 100)
        self.hallucinated = np.clip(rng.normal(25, 8, 40), 0, 100)
        self.estimator = ULSIFDensityRatio(max_centers=20, random_state=4)
        self.estimator.fit(self.factual, self.hallucinated, folds=4)
        # A usable model really is in place before each scenario.
        self.assertIsNotNone(self.estimator.alpha)
        self.assertTrue(self.estimator.fit_diagnostics["sanity_check_passed"])
        self.assertGreater(self.estimator.ratio(80.0), 0.0)
        self.first_sigma = self.estimator.sigma

    def assert_no_usable_fit(self):
        self.assertIsNone(self.estimator.alpha)
        self.assertIsNone(self.estimator.centers)
        self.assertIsNone(self.estimator.sigma)
        self.assertIsNone(self.estimator.lam)
        self.assertIsNone(self.estimator.fit_diagnostics)
        with self.assertRaises(RuntimeError) as caught:
            self.estimator.ratio(50.0)
        self.assertIn("must be fit before use", str(caught.exception))

    def test_a_failed_input_validation_invalidates_the_previous_fit(self):
        bad = self.factual.copy()
        bad[0] = float("nan")
        with self.assertRaises(NonFiniteScoreError):
            self.estimator.fit(bad, self.hallucinated, folds=4)
        self.assert_no_usable_fit()

    def test_a_failed_cv_objective_invalidates_the_previous_fit(self):
        original_objective = ULSIFDensityRatio._objective
        try:
            ULSIFDensityRatio._objective = lambda *a, **k: float("nan")
            with self.assertRaises(DegenerateULSIFFit):
                self.estimator.fit(self.factual, self.hallucinated, folds=4)
        finally:
            ULSIFDensityRatio._objective = original_objective
        self.assert_no_usable_fit()

    def test_a_failed_final_sanity_check_invalidates_sigma_and_lambda_too(self):
        # The earlier version of this path cleared centers/alpha/diagnostics but
        # left sigma and lambda behind, describing a model that no longer exists.
        # Cross-validation runs normally here; only the final check fails, so
        # sigma and lambda ARE set by the time it does.
        original_validate = ULSIFDensityRatio._validate_final_fit
        seen = {}

        def failing_validate(instance, *args, **kwargs):
            seen["sigma"] = instance.sigma
            seen["lam"] = instance.lam
            raise DegenerateULSIFFit("synthetic final-fit failure")

        try:
            ULSIFDensityRatio._validate_final_fit = failing_validate
            with self.assertRaises(DegenerateULSIFFit):
                self.estimator.fit(self.factual, self.hallucinated, folds=4)
        finally:
            ULSIFDensityRatio._validate_final_fit = original_validate

        self.assertIsNotNone(seen["sigma"], "sigma was set before the failure")
        self.assertIsNotNone(seen["lam"], "lambda was set before the failure")
        self.assert_no_usable_fit()

    def test_a_third_valid_fit_recovers_normally(self):
        bad = self.factual.copy()
        bad[3] = float("inf")
        with self.assertRaises(NonFiniteScoreError):
            self.estimator.fit(bad, self.hallucinated, folds=4)
        self.assert_no_usable_fit()

        self.estimator.fit(self.factual, self.hallucinated, folds=4)
        self.assertIsNotNone(self.estimator.alpha)
        self.assertTrue(self.estimator.fit_diagnostics["sanity_check_passed"])
        self.assertEqual(self.estimator.fit_diagnostics["n_factual"], 40)
        self.assertEqual(self.estimator.sigma, self.first_sigma)
        self.assertGreater(self.estimator.ratio(80.0), self.estimator.ratio(20.0))

    def test_the_diagnostics_describe_the_current_fit_not_an_earlier_one(self):
        smaller_factual = self.factual[:20]
        smaller_hallucinated = self.hallucinated[:20]
        self.estimator.fit(smaller_factual, smaller_hallucinated, folds=4)
        self.assertEqual(self.estimator.fit_diagnostics["n_factual"], 20)
        self.assertEqual(self.estimator.fit_diagnostics["n_hallucinated"], 20)

    def test_the_cv_table_belongs_only_to_the_current_attempt(self):
        self.assertGreater(len(self.estimator.cv_table), 0)
        bad = self.hallucinated.copy()
        bad[1] = float("-inf")
        with self.assertRaises(NonFiniteScoreError):
            self.estimator.fit(self.factual, bad, folds=4)
        self.assertEqual(
            self.estimator.cv_table, [],
            "a stale CV table would masquerade as provenance for the failed attempt",
        )

    def test_the_state_is_cleared_before_input_validation_runs(self):
        # The clear must precede validation, or an input that fails immediately
        # would leave the old model untouched.
        with self.assertRaises(NonFiniteScoreError):
            self.estimator.fit([float("nan")] * 10, self.hallucinated, folds=4)
        self.assert_no_usable_fit()

    def test_clearing_is_idempotent_and_safe_on_a_fresh_estimator(self):
        fresh = ULSIFDensityRatio()
        fresh._clear_fit_state()
        fresh._clear_fit_state()
        self.assertIsNone(fresh.alpha)
        self.assertEqual(fresh.cv_table, [])


# --------------------------------------------------------------------------
# E. ratio() input and output validation
# --------------------------------------------------------------------------


class TestRatioValidation(unittest.TestCase):
    def test_a_non_finite_score_is_rejected_before_normalization(self):
        estimator = fitted()
        for bad in NON_FINITE:
            with self.subTest(score=bad):
                with self.assertRaises(NonFiniteScoreError):
                    estimator.ratio(bad)

    def test_infinity_is_not_silently_clipped_to_a_perfect_score(self):
        estimator = fitted()
        with self.assertRaises(NonFiniteScoreError):
            estimator.ratio(float("inf"))
        # The old behaviour was to return exactly this.
        self.assertGreater(estimator.ratio(100.0), 0.0)

    def test_a_non_finite_raw_fitted_ratio_is_rejected_not_clipped(self):
        estimator = fitted()
        estimator.alpha = estimator.alpha.copy()
        estimator.alpha[0] = float("inf")
        with self.assertRaises(NonFiniteScoreError) as caught:
            estimator.ratio(80.0)
        self.assertIn("Refusing to clip", str(caught.exception))

    def test_the_existing_clip_bounds_are_unchanged(self):
        estimator = fitted()
        estimator.alpha = estimator.alpha * 1e30
        self.assertEqual(estimator.ratio(80.0), 1e6)
        estimator = fitted()
        estimator.alpha = estimator.alpha * 1e-30
        self.assertEqual(estimator.ratio(80.0), 1e-6)

    def test_a_normal_finite_score_is_unaffected(self):
        estimator = fitted()
        for score in (0.0, 25.0, 50.0, 75.0, 100.0):
            with self.subTest(score=score):
                value = estimator.ratio(score)
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 1e-6)
                self.assertLessEqual(value, 1e6)

    def test_the_orientation_is_unchanged(self):
        estimator = fitted()
        self.assertGreater(estimator.ratio(80.0), estimator.ratio(20.0))

    def test_an_unfit_estimator_still_reports_that_first(self):
        with self.assertRaises(RuntimeError):
            ULSIFDensityRatio().ratio(50.0)


# --------------------------------------------------------------------------
# F/G. Detector boundary (D-06) and accounting
# --------------------------------------------------------------------------


class TestDetectorScoreValidation(unittest.TestCase):
    def test_a_non_finite_document_score_raises_on_the_first_bad_document(self):
        for bad in NON_FINITE:
            with self.subTest(score=bad):
                estimator = ConstantRatio(2.0)
                scorer = FixedScorer(bad)
                with self.assertRaises(NonFiniteScoreError) as caught:
                    detector(estimator).detect_subclaim(Sub(10), scorer)
                message = str(caught.exception)
                self.assertIn("document-score numerical failure", message)
                self.assertIn(repr(bad), message)
                self.assertIn("a subclaim", message)
                self.assertEqual(scorer.calls, 1, "retrieval must stop immediately")
                self.assertEqual(
                    estimator.calls, [], "ratio() must not be invoked"
                )

    def test_later_documents_are_never_scored(self):
        scorer = SequenceScorer([50.0, float("nan"), 60.0, 70.0])
        estimator = ConstantRatio(1.0)
        with self.assertRaises(NonFiniteScoreError):
            detector(estimator).detect_subclaim(Sub(4), scorer)
        self.assertEqual(scorer.calls, 2)
        self.assertEqual(estimator.calls, [50.0], "only the valid score was used")

    def test_the_error_locates_the_document_without_dumping_page_text(self):
        with self.assertRaises(NonFiniteScoreError) as caught:
            detector(ConstantRatio(1.0)).detect_subclaim(
                Sub(3), FixedScorer(float("nan"))
            )
        message = str(caught.exception)
        self.assertIn("document 1 of 3", message)
        self.assertIn("https://example.invalid/doc", message)
        self.assertNotIn(Doc.page_content, message)

    def test_no_detection_result_is_produced_for_a_failed_score(self):
        with self.assertRaises(NonFiniteScoreError):
            detector(ConstantRatio(1.0)).detect_sentence(
                type("Rec", (), {"subclaims": [Sub(3)]})(),
                FixedScorer(float("nan")),
            )


class TestDetectorRatioValidation(unittest.TestCase):
    def bad_ratio(self, value):
        estimator = ConstantRatio(value)
        scorer = FixedScorer(50.0)
        with self.assertRaises(NonFiniteScoreError) as caught:
            detector(estimator).detect_subclaim(Sub(10), scorer)
        self.assertIn("density-ratio numerical failure", str(caught.exception))
        self.assertEqual(scorer.calls, 1)
        return str(caught.exception)

    def test_a_zero_ratio_raises_before_log(self):
        # log(0) is -inf; the guard must fire first.
        self.assertIn("0.0", self.bad_ratio(0.0))

    def test_a_negative_ratio_raises(self):
        self.bad_ratio(-1.0)

    def test_a_nan_ratio_raises(self):
        self.bad_ratio(float("nan"))

    def test_an_infinite_ratio_raises(self):
        self.bad_ratio(float("inf"))
        self.bad_ratio(float("-inf"))

    def test_a_non_finite_posterior_is_rejected_before_the_stopping_rule(self):
        # Defence in depth. With the score and ratio guards in place this state
        # is not reachable through them, so the accumulator itself is faulted:
        # if the log-odds update ever produced a non-finite posterior, the
        # stopping comparisons would all be False and the whole retrieval budget
        # would be spent on a meaningless value.
        class BrokenAccumulator(DDREDetector):
            @staticmethod
            def _sigmoid(x):
                return float("nan")

        scorer = FixedScorer(50.0)
        built = BrokenAccumulator(
            ConstantRatio(2.0), lower_threshold=0.20, upper_threshold=0.80,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        with self.assertRaises(NonFiniteScoreError) as caught:
            built.detect_subclaim(Sub(10), scorer)
        self.assertIn("posterior numerical failure", str(caught.exception))
        self.assertEqual(scorer.calls, 1, "must not spend the retrieval budget")

    def test_the_guard_is_not_repaired_with_an_epsilon(self):
        message = self.bad_ratio(0.0)
        self.assertIn("Not repaired with an epsilon", message)

    def test_a_normal_positive_finite_ratio_behaves_exactly_as_before(self):
        # log O_n = logit(0.5) + n*log(r), unchanged by the guards.
        estimator = ConstantRatio(2.0)
        result = detector(
            estimator, lower_threshold=1e-3, upper_threshold=1 - 1e-3, max_docs=4
        ).detect_subclaim(Sub(4), FixedScorer(50.0))
        self.assertEqual(result.documents_used, 4)
        self.assertAlmostEqual(
            result.p_factual, 1.0 / (1.0 + math.exp(-4 * math.log(2.0)))
        )

    def test_a_ratio_of_one_still_leaves_the_posterior_at_the_prior(self):
        result = detector(ConstantRatio(1.0), max_docs=5).detect_subclaim(
            Sub(5), FixedScorer(50.0)
        )
        self.assertAlmostEqual(result.p_factual, 0.5)
        self.assertEqual(result.documents_used, 5)


class TestNoNaNPosteriorIsReachable(unittest.TestCase):
    def test_no_detection_result_with_a_nan_posterior_can_be_produced(self):
        for score, ratio_value in (
            (float("nan"), 2.0), (float("inf"), 2.0), (float("-inf"), 2.0),
            (50.0, float("nan")), (50.0, float("inf")), (50.0, 0.0), (50.0, -1.0),
        ):
            with self.subTest(score=score, ratio=ratio_value):
                with self.assertRaises(DDRENumericalError):
                    detector(ConstantRatio(ratio_value)).detect_subclaim(
                        Sub(10), FixedScorer(score)
                    )

    def test_valid_inputs_always_give_a_finite_posterior(self):
        for ratio_value in (1e-6, 0.5, 1.0, 2.0, 1e6):
            with self.subTest(ratio=ratio_value):
                result = detector(ConstantRatio(ratio_value)).detect_subclaim(
                    Sub(10), FixedScorer(50.0)
                )
                self.assertTrue(math.isfinite(result.p_factual))
                self.assertGreaterEqual(result.p_factual, 0.0)
                self.assertLessEqual(result.p_factual, 1.0)

    def test_the_exceptions_are_value_errors(self):
        # Existing callers that catch ValueError keep working.
        self.assertTrue(issubclass(DDRENumericalError, ValueError))
        self.assertTrue(issubclass(NonFiniteScoreError, DDRENumericalError))
        self.assertTrue(issubclass(DegenerateULSIFFit, DDRENumericalError))


# --------------------------------------------------------------------------
# Regression preservation
# --------------------------------------------------------------------------


class TestProtectedBehaviourUnchanged(unittest.TestCase):
    def test_bse_decision_logic_is_untouched_by_this_change(self):
        # Originally a whole-file byte diff against origin/main. PR #12 added
        # BSEDetector.detect_sentence_with_trace and made detect_sentence
        # delegate to it (audit finding D-10), which is instrumentation, not a
        # decision change -- so the guard is TIGHTENED rather than dropped:
        # every other top-level and method definition in the file must still be
        # byte-identical to origin/main, and only those two names may differ.
        import ast
        import subprocess

        show = subprocess.run(
            ["git", "show", "origin/main:src/baseline_core.py"],
            capture_output=True, text=True,
        )
        if show.returncode != 0:
            self.skipTest("origin/main not available")

        def definitions(source):
            tree = ast.parse(source)
            found = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[node.name] = ast.get_source_segment(source, node)
            return found

        baseline = definitions(show.stdout)
        current = definitions(
            (Path(__file__).resolve().parents[1] / "src" / "baseline_core.py")
            .read_text(encoding="utf-8")
        )
        permitted_to_differ = {"detect_sentence", "detect_sentence_with_trace"}

        self.assertEqual(
            set(baseline) - set(current), set(),
            "no BSE definition may be removed",
        )
        self.assertEqual(
            set(current) - set(baseline), {"detect_sentence_with_trace"},
            "only the trace accessor may be added",
        )
        for name, source in baseline.items():
            if name in permitted_to_differ:
                continue
            with self.subTest(definition=name):
                self.assertEqual(
                    current[name], source,
                    f"src/baseline_core.py::{name} must not change",
                )
        # The decision rules by name, so the list above cannot quietly shrink.
        for decision in (
            "should_continue", "detect_subclaim", "bayes_update",
            "cost_based_prediction", "_official_expected_next_posterior",
        ):
            self.assertIn(decision, baseline)
            self.assertEqual(current[decision], baseline[decision])

    def bse(self):
        from src.baseline_core import BSEDetector

        return BSEDetector(
            [1, 78, 19, 13, 6, 5, 28, 57, 1, 1],
            [1, 141, 38, 15, 4, 1, 3, 4, 1, 1],
            mode="official", p0=0.5, c_miss=28, c_false_alarm=96,
            c_retrieve=1, max_docs=10,
        )

    def test_bse_still_scores_valid_documents_as_published(self):
        result = self.bse().detect_subclaim(Sub(3), FixedScorer(50.0))
        self.assertGreater(result.documents_used, 0)
        self.assertTrue(math.isfinite(result.p_factual))

    def test_bse_numerical_behaviour_on_a_bad_score_is_unchanged(self):
        # Worth recording: D-06 was DDRE-specific. BSE already fails on a
        # non-finite score, because its bucket discretizer calls int() on it --
        # ValueError for NaN, OverflowError for an infinity. That is incidental
        # rather than a designed guard, and its message names neither the
        # subclaim nor the document, but it does mean BSE never silently
        # converted NaN into evidence. This PR does not touch that path.
        with self.assertRaises(ValueError):
            self.bse().detect_subclaim(Sub(3), FixedScorer(float("nan")))
        with self.assertRaises(OverflowError):
            self.bse().detect_subclaim(Sub(3), FixedScorer(float("inf")))

    def test_the_d01_cost_consistency_guard_still_applies(self):
        from src.ddre_core import cost_consistent_thresholds

        space = cost_consistent_thresholds(28, 96)
        self.assertEqual(space["effective_lower_grid"], [0.05, 0.10, 0.15, 0.20])
        with self.assertRaises(ValueError):
            detector(ConstantRatio(1.0), lower_threshold=0.30)

    def test_the_ulsif_orientation_and_clip_are_unchanged(self):
        estimator = fitted()
        self.assertGreater(estimator.ratio(80.0), estimator.ratio(20.0))
        source = (
            Path(__file__).resolve().parents[1] / "src" / "ddre_core.py"
        ).read_text(encoding="utf-8")
        self.assertIn("np.clip(value, 1e-6, 1e6)", source)

    def test_no_model_or_gpu_dependency_was_added(self):
        source = (
            Path(__file__).resolve().parents[1] / "src" / "ddre_core.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("torch", "transformers", "cuda"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
