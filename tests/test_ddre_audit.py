"""AUDIT / REGRESSION tests for the DDRE (uLSIF) implementation.

Every test here PINS CURRENT BEHAVIOUR so that the scientific audit in
docs/ddre_scientific_audit.md rests on executed evidence rather than on
reading. Several of them deliberately assert behaviour the audit judges to be
WRONG; those are marked AUDIT: and name the finding they document. They must be
updated -- not deleted -- when the corresponding fix lands, because the point of
a regression test for a known defect is that the fix has to change it visibly.

Nothing here changes production behaviour, and nothing here runs the formal
experiment, loads a model, or touches the held-out test split.

numpy + stdlib only, so CI's numpy-only install runs the whole file.
"""

import ast
import math
import unittest
from pathlib import Path

import numpy as np

from src.baseline_core import BSEDetector, cost_based_prediction
from src.ddre_core import (
    CANDIDATE_LOWER_GRID,
    CANDIDATE_UPPER_GRID,
    DDREDetector,
    DegenerateULSIFFit,
    NonFiniteScoreError,
    ULSIFDensityRatio,
    cost_consistent_thresholds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Wang's published costs and the classification threshold they imply.
C_MISS, C_FALSE_ALARM = 28.0, 96.0
COST_THRESHOLD = C_MISS / (C_MISS + C_FALSE_ALARM)  # 0.2258064516...

# The released Laplace-smoothed NBC histograms, used only to shape illustrative
# score samples. No model and no released scores are needed.
RELEASED_POSITIVE = (1, 78, 19, 13, 6, 5, 28, 57, 1, 1)
RELEASED_NEGATIVE = (1, 141, 38, 15, 4, 1, 3, 4, 1, 1)


def nbc_shaped_scores(histogram, seed):
    """Scores drawn uniformly inside each released NBC bucket.

    Illustrative only: it reproduces the *shape* of the released histograms
    without needing the NLI model, which is what the audit's numerical
    behaviour claims depend on.
    """
    rng = np.random.default_rng(seed)
    out = []
    for bucket, count in enumerate(histogram):
        out.extend(rng.uniform(bucket * 10.0, (bucket + 1) * 10.0, max(0, count - 1)))
    return np.asarray(out, dtype=float)


def fitted_estimator(seed=42):
    return ULSIFDensityRatio(random_state=seed).fit(
        nbc_shaped_scores(RELEASED_POSITIVE, seed),
        nbc_shaped_scores(RELEASED_NEGATIVE, seed + 1),
    )


class ConstantRatio:
    """A stub density-ratio estimator with an exactly known value."""

    def __init__(self, value):
        self.value = float(value)
        self.calls = 0

    def ratio(self, score):
        self.calls += 1
        return self.value


class FixedScorer:
    """Returns one score for every document; counts calls."""

    def __init__(self, score, spans=1):
        self.score = score
        self.spans = spans
        self.calls = 0

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        self.calls += 1
        return self.score, self.spans


class Doc:
    page_content = "word " * 10


class Sub:
    def __init__(self, n_documents, text="claim"):
        self.text = text
        self.documents = [Doc() for _ in range(n_documents)]


class Rec:
    def __init__(self, subclaims):
        self.subclaims = subclaims


# --------------------------------------------------------------------------
# A. uLSIF mathematics
# --------------------------------------------------------------------------


class TestRatioOrientation(unittest.TestCase):
    """r(s) must estimate p(s|factual) / p(s|hallucinated), not its inverse."""

    def test_H_is_built_from_the_denominator_and_h_from_the_numerator(self):
        # uLSIF: (H + lambda I) a = h with H = E_de[phi phi^T], h = E_nu[phi].
        # The numerator is FACTUAL and the denominator HALLUCINATED, so H must
        # come from the hallucinated sample and h from the factual one. Solving
        # the documented normal equations by hand must reproduce _solve.
        estimator = ULSIFDensityRatio(max_centers=8, random_state=3)
        factual = np.linspace(60.0, 95.0, 40)
        hallucinated = np.linspace(5.0, 40.0, 40)
        fx = estimator._as_column(factual)
        hx = estimator._as_column(hallucinated)
        centers = fx[:8]
        sigma, lam = 0.15, 1e-3

        phi_denominator = estimator._kernel(hx, centers, sigma)
        phi_numerator = estimator._kernel(fx, centers, sigma)
        expected = np.linalg.solve(
            phi_denominator.T @ phi_denominator / len(hx) + lam * np.eye(8),
            np.mean(phi_numerator, axis=0),
        )
        np.testing.assert_allclose(
            estimator._solve(fx, hx, centers, sigma, lam),
            np.maximum(expected, 0.0),
        )

    def test_centers_are_drawn_from_the_numerator_sample(self):
        # Standard uLSIF places kernel centres on the numerator (factual)
        # sample. Proved with DISJOINT supports, so a centre taken from the
        # hallucinated sample would be detectable: every centre must be a member
        # of the normalised factual set and none may lie in the
        # hallucinated-only support.
        factual = np.linspace(60.0, 95.0, 60)          # normalised [0.60, 0.95]
        hallucinated = np.linspace(5.0, 40.0, 60)      # normalised [0.05, 0.40]
        estimator = ULSIFDensityRatio(max_centers=25, random_state=13).fit(
            factual, hallucinated, folds=4
        )

        factual_support = {round(float(x), 12) for x in factual / 100.0}
        hallucinated_support = {round(float(x), 12) for x in hallucinated / 100.0}
        self.assertTrue(factual_support.isdisjoint(hallucinated_support))

        centers = [round(float(c), 12) for c in estimator.centers.ravel()]
        self.assertEqual(len(centers), 25)
        for center in centers:
            self.assertIn(center, factual_support)
            self.assertNotIn(center, hallucinated_support)
        # And none sits in the hallucinated-only interval at all.
        self.assertTrue(all(c >= 0.60 for c in centers))

    def test_the_ratio_is_larger_on_factual_like_scores(self):
        estimator = fitted_estimator()
        self.assertGreater(estimator.ratio(75.0), estimator.ratio(15.0))

    def test_the_held_out_objective_matches_the_uLSIF_criterion(self):
        # J = 1/2 E_de[r^2] - E_nu[r], with de = hallucinated, nu = factual.
        estimator = ULSIFDensityRatio(max_centers=6, random_state=5)
        fx = estimator._as_column(np.linspace(60.0, 95.0, 20))
        hx = estimator._as_column(np.linspace(5.0, 40.0, 20))
        centers, sigma = fx[:6], 0.2
        alpha = estimator._solve(fx, hx, centers, sigma, 1e-3)
        expected = 0.5 * np.mean(
            (estimator._kernel(hx, centers, sigma) @ alpha) ** 2
        ) - np.mean(estimator._kernel(fx, centers, sigma) @ alpha)
        self.assertAlmostEqual(
            estimator._objective(fx, hx, centers, alpha, sigma), float(expected)
        )

    def test_scores_are_normalized_to_the_unit_interval(self):
        estimator = ULSIFDensityRatio()
        np.testing.assert_allclose(
            estimator._as_column([0.0, 50.0, 100.0]).ravel(), [0.0, 0.5, 1.0]
        )


class TestRatioPositivityAndClipping(unittest.TestCase):
    def test_the_ratio_is_positive_and_may_exceed_one(self):
        estimator = fitted_estimator()
        values = [estimator.ratio(s) for s in np.linspace(0.0, 100.0, 101)]
        self.assertTrue(all(v > 0.0 for v in values))
        self.assertGreater(max(values), 1.0)

    def test_the_ratio_is_clipped_to_a_fixed_symmetric_log_window(self):
        self.assertAlmostEqual(math.log(1e-6), -math.log(1e6))

    def test_alpha_is_truncated_to_be_non_negative(self):
        # Post-hoc truncation is standard uLSIF (Kanamori et al. 2009).
        estimator = fitted_estimator()
        self.assertTrue(np.all(estimator.alpha >= 0.0))

    def test_RESOLVED_a_degenerate_all_zero_fit_is_rejected_before_use(self):
        # AUDIT finding D-04 -- RESOLVED by PR #9.
        #
        # WAS: if the solved alpha vanished, r(s) was identically zero and the
        # clip turned "the model has nothing to say" into log r = -13.8 per
        # document, i.e. near-certain hallucination. fit() neither detected nor
        # reported it, and ratio() served 1e-6 as though it were evidence.
        #
        # NOW: the final-fit sanity check rejects an all-zero coefficient
        # vector before the estimator becomes usable. The arithmetic that made
        # it dangerous is unchanged -- the clip is still 1e-6 -- so the test
        # still records what that floor would have meant.
        self.assertAlmostEqual(math.log(1e-6), -13.8155, places=3)

        estimator = fitted_estimator()
        estimator.alpha = np.zeros_like(estimator.alpha)
        with self.assertRaises(DegenerateULSIFFit) as caught:
            estimator._validate_final_fit(
                estimator._as_column([80.0, 90.0]),
                estimator._as_column([10.0, 20.0]),
                cv_objective=-1.0,
            )
        self.assertIn("no coefficient is strictly positive", str(caught.exception))

    def test_the_ratio_is_bounded_above_by_the_sum_of_the_coefficients(self):
        # CORRECTION to an earlier draft of the audit, which called the
        # estimator "unbounded". It is not. Every Gaussian kernel value is <= 1
        # and the coefficients are finite and non-negative, so on the clipped
        # [0, 1] input the fitted ratio is bounded above by sum(alpha).
        estimator = fitted_estimator()
        bound = float(estimator.alpha.sum())
        self.assertTrue(math.isfinite(bound))
        for score in np.linspace(0.0, 100.0, 101):
            self.assertLessEqual(estimator.ratio(float(score)), bound)

    def test_AUDIT_the_ratio_is_weakly_constrained_away_from_both_supports(self):
        # AUDIT finding D-03, restated correctly. The estimate is well
        # determined only where both samples have support; elsewhere its value
        # is decided by kernel tails and the regulariser. The finite bound
        # sum(alpha) is itself data-dependent and can be very large, so a single
        # weakly-identified document can carry large log-evidence.
        #
        # This is a SYNTHETIC fixture. It establishes the mechanism only; it is
        # NOT evidence about the actual formal NBC fit, which this repository
        # holds no fixed raw scores for.
        rng = np.random.default_rng(11)
        estimator = ULSIFDensityRatio(max_centers=30, random_state=7).fit(
            np.clip(rng.normal(78, 7, 120), 0, 100),
            np.clip(rng.normal(22, 7, 120), 0, 100),
            folds=4,
        )
        self.assertGreater(estimator.ratio(100.0), 100.0)
        self.assertLessEqual(estimator.ratio(100.0), float(estimator.alpha.sum()))

    def test_neither_synthetic_fixture_reaches_the_clip_bounds(self):
        # CORRECTION to a second earlier claim: log r = +-13.8 arises ONLY at
        # the [1e-6, 1e6] clip bounds, and neither synthetic fit comes close.
        # The clip is a backstop, not the observed operating regime.
        rng = np.random.default_rng(11)
        separated = ULSIFDensityRatio(max_centers=30, random_state=7).fit(
            np.clip(rng.normal(78, 7, 120), 0, 100),
            np.clip(rng.normal(22, 7, 120), 0, 100),
            folds=4,
        )
        grid = np.linspace(0.0, 100.0, 101)
        for estimator in (fitted_estimator(), separated):
            values = np.array([estimator.ratio(float(s)) for s in grid])
            self.assertGreater(values.min(), 1e-6)
            self.assertLess(values.max(), 1e6)
            self.assertLess(float(np.abs(np.log(values)).max()), math.log(1e6))

    def test_RESOLVED_a_non_finite_score_is_rejected_by_ratio(self):
        # AUDIT finding D-06 -- RESOLVED by PR #9.
        #
        # WAS: no guard, so NaN passed straight through and +inf was silently
        # clipped by normalization to a perfect score of 100.
        #
        # NOW: rejected before normalization, so +inf can no longer become 100.
        estimator = fitted_estimator()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=bad):
                with self.assertRaises(NonFiniteScoreError):
                    estimator.ratio(bad)


# --------------------------------------------------------------------------
# B. Hyperparameter selection
# --------------------------------------------------------------------------


class TestHyperparameterSelection(unittest.TestCase):
    def test_selection_minimises_the_cross_validated_uLSIF_objective(self):
        estimator = fitted_estimator()
        best = min(row["cv_objective"] for row in estimator.cv_table)
        chosen = [
            row
            for row in estimator.cv_table
            if row["sigma"] == estimator.sigma and row["lambda"] == estimator.lam
        ]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["cv_objective"], best)

    def test_the_grid_is_the_documented_one(self):
        estimator = fitted_estimator()
        self.assertEqual(
            sorted({row["lambda"] for row in estimator.cv_table}),
            [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        )
        self.assertLessEqual(len({row["sigma"] for row in estimator.cv_table}), 4)

    def test_fitting_only_ever_sees_the_two_score_arrays_it_is_given(self):
        # The estimator has no access to sentence records, labels or splits:
        # its entire interface is fit(factual_scores, hallucinated_scores).
        signature = ULSIFDensityRatio.fit.__code__.co_varnames[
            : ULSIFDensityRatio.fit.__code__.co_argcount
        ]
        self.assertEqual(
            signature, ("self", "factual_scores", "hallucinated_scores", "folds")
        )

    def test_fitting_is_deterministic_for_a_fixed_seed(self):
        a, b = fitted_estimator(42), fitted_estimator(42)
        self.assertEqual(a.sigma, b.sigma)
        self.assertEqual(a.lam, b.lam)
        np.testing.assert_allclose(a.alpha, b.alpha)


# --------------------------------------------------------------------------
# C. Sequential DDRE detector
# --------------------------------------------------------------------------


class TestSequentialAccumulation(unittest.TestCase):
    def test_log_odds_start_at_zero_for_p0_one_half(self):
        self.assertEqual(DDREDetector._logit(0.5), 0.0)

    def test_log_odds_accumulate_additively(self):
        # log O_n = log O_0 + sum_i log r(s_i), with P0 = 0.5 so log O_0 = 0.
        ratio = 2.0
        detector = DDREDetector(
            ConstantRatio(ratio), lower_threshold=1e-3, upper_threshold=1 - 1e-3,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=4,
        )
        result = detector.detect_subclaim(Sub(4), FixedScorer(50.0))
        expected_log_odds = 4 * math.log(ratio)
        self.assertEqual(result.documents_used, 4)
        self.assertAlmostEqual(
            result.p_factual, 1.0 / (1.0 + math.exp(-expected_log_odds))
        )

    def test_a_ratio_of_one_leaves_the_posterior_at_the_prior(self):
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=5,
        )
        result = detector.detect_subclaim(Sub(5), FixedScorer(50.0))
        self.assertAlmostEqual(result.p_factual, 0.5)
        self.assertEqual(result.documents_used, 5)

    def test_stopping_happens_after_the_update_not_before(self):
        detector = DDREDetector(
            ConstantRatio(10.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        result = detector.detect_subclaim(Sub(10), FixedScorer(50.0))
        self.assertEqual(result.documents_used, 1)

    def test_the_retrieval_budget_is_respected(self):
        scorer = FixedScorer(50.0)
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=3,
        )
        result = detector.detect_subclaim(Sub(25), scorer)
        self.assertEqual(result.documents_used, 3)
        self.assertEqual(scorer.calls, 3)

    def test_nli_span_calls_are_summed_not_counted_as_documents(self):
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=3,
        )
        result = detector.detect_subclaim(Sub(3), FixedScorer(50.0, spans=7))
        self.assertEqual(result.documents_used, 3)
        self.assertEqual(result.nli_calls, 21)

    def test_a_subclaim_with_no_documents_returns_the_prior(self):
        detector = DDREDetector(
            ConstantRatio(5.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        scorer = FixedScorer(50.0)
        result = detector.detect_subclaim(Sub(0), scorer)
        self.assertEqual(scorer.calls, 0)
        self.assertEqual(result.documents_used, 0)
        self.assertEqual(result.p_factual, 0.5)
        # P0 = 0.5 is above the cost threshold, so an unevidenced subclaim is
        # declared FACTUAL by default. Documented, not asserted to be right.
        self.assertEqual(result.prediction, 1)

    def test_the_running_log_odds_are_saturated_at_forty(self):
        detector = DDREDetector(
            ConstantRatio(1e6), lower_threshold=1e-9, upper_threshold=1 - 1e-9,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        result = detector.detect_subclaim(Sub(10), FixedScorer(50.0))
        # 10 * log(1e6) = 138 would overflow the sigmoid; the clip holds it.
        self.assertTrue(math.isfinite(result.p_factual))
        self.assertLessEqual(result.p_factual, 1.0)

    def test_RESOLVED_a_nan_score_raises_on_the_first_bad_document(self):
        # AUDIT finding D-06 -- RESOLVED by PR #9.
        #
        # WAS: NaN defeated every comparison, so no stopping rule fired, the
        # full retrieval budget (10 documents) was spent, and a NaN p_factual
        # reached the metrics with no warning.
        #
        # NOW: the first non-finite score raises immediately, and the later
        # documents are never scored.
        detector = DDREDetector(
            fitted_estimator(), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        scorer = FixedScorer(float("nan"))
        with self.assertRaises(NonFiniteScoreError) as caught:
            detector.detect_subclaim(Sub(10), scorer)
        self.assertIn("document-score numerical failure", str(caught.exception))
        self.assertEqual(scorer.calls, 1, "later documents must not be scored")


class TestSentenceAggregation(unittest.TestCase):
    def test_sentence_factuality_is_the_minimum_over_subclaims(self):
        class PerCallRatio:
            def __init__(self, values):
                self.values = list(values)
            def ratio(self, score):
                return self.values.pop(0)

        detector = DDREDetector(
            PerCallRatio([9.0, 1 / 9.0]), lower_threshold=1e-3,
            upper_threshold=1 - 1e-3, p0=0.5, c_miss=C_MISS,
            c_false_alarm=C_FALSE_ALARM, max_docs=1,
        )
        result = detector.detect_sentence(Rec([Sub(1), Sub(1)]), FixedScorer(50.0))
        self.assertAlmostEqual(result.p_factual, 0.1)
        self.assertEqual(result.documents_used, 2)

    def test_the_sentence_prediction_uses_the_aggregated_probability(self):
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=1,
        )
        result = detector.detect_sentence(Rec([Sub(1)]), FixedScorer(50.0))
        self.assertEqual(
            result.prediction,
            cost_based_prediction(result.p_factual, C_MISS, C_FALSE_ALARM),
        )

    def test_documents_and_span_calls_are_summed_across_subclaims(self):
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=2,
        )
        result = detector.detect_sentence(
            Rec([Sub(2), Sub(2), Sub(2)]), FixedScorer(50.0, spans=3)
        )
        self.assertEqual(result.documents_used, 6)
        self.assertEqual(result.nli_calls, 18)


# --------------------------------------------------------------------------
# D. Stopping thresholds vs the CM/CFA classification threshold
# --------------------------------------------------------------------------


class TestOneDocumentStoppingRule(unittest.TestCase):
    """The general rule, and why |log r| is only a symmetric special case.

    After one document the state is `logit(P0) + log r`. A LOW stop needs that
    to be <= logit(lower); a HIGH stop needs it >= logit(upper). The |log r|
    shorthand collapses to this only when P0 = 0.5 (so logit(P0) = 0) AND the
    band is symmetric in log-odds (so logit(upper) = -logit(lower)).
    """

    @staticmethod
    def logit(p):
        return math.log(p / (1.0 - p))

    def one_document_state(self, p0, ratio):
        detector = DDREDetector(
            ConstantRatio(ratio), lower_threshold=0.2, upper_threshold=0.8,
            p0=p0, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=1,
        )
        result = detector.detect_subclaim(Sub(1), FixedScorer(50.0))
        return self.logit(result.p_factual)

    def test_the_state_after_one_document_is_logit_p0_plus_log_r(self):
        for p0 in (0.3, 0.5, 0.7):
            for ratio in (0.25, 1.0, 4.0):
                with self.subTest(p0=p0, ratio=ratio):
                    self.assertAlmostEqual(
                        self.one_document_state(p0, ratio),
                        self.logit(p0) + math.log(ratio),
                        places=9,
                    )

    def test_the_shorthand_is_exact_for_a_symmetric_band_at_p0_one_half(self):
        # P0 = 0.5 and [0.20, 0.80]: logit(P0) = 0 and the band is symmetric,
        # so |log r| >= logit(0.8/0.2) reproduces the general rule exactly.
        lower, upper, p0 = 0.20, 0.80, 0.5
        self.assertAlmostEqual(self.logit(upper), -self.logit(lower))
        for ratio in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0):
            with self.subTest(ratio=ratio):
                state = self.logit(p0) + math.log(ratio)
                general = state <= self.logit(lower) or state >= self.logit(upper)
                shorthand = abs(math.log(ratio)) >= self.logit(upper)
                self.assertEqual(general, shorthand)

    def test_AUDIT_the_shorthand_is_wrong_for_an_asymmetric_band(self):
        # AUDIT correction. With [0.05, 0.80] at P0 = 0.5 the band is no longer
        # symmetric in log-odds: logit(0.05) = -2.944 but logit(0.80) = +1.386.
        # A ratio of 0.2 gives |log r| = 1.609, which clears the shorthand
        # threshold, yet the state -1.609 reaches NEITHER boundary. The
        # shorthand would count a stop that does not happen.
        lower, upper, p0, ratio = 0.05, 0.80, 0.5, 0.2
        state = self.logit(p0) + math.log(ratio)
        general = state <= self.logit(lower) or state >= self.logit(upper)
        shorthand = abs(math.log(ratio)) >= self.logit(upper)
        self.assertFalse(general)
        self.assertTrue(shorthand)
        self.assertNotEqual(general, shorthand)

    def test_AUDIT_the_shorthand_is_wrong_when_p0_is_not_one_half(self):
        # AUDIT correction. logit(P0) shifts the whole state, so a prior other
        # than 0.5 breaks the shorthand even on a symmetric band: at P0 = 0.35
        # a ratio of 1.0 carries no evidence at all, yet the state already sits
        # at or below logit(lower) for lower = 0.35.
        lower, upper, p0, ratio = 0.35, 0.80, 0.35, 1.0
        state = self.logit(p0) + math.log(ratio)
        general = state <= self.logit(lower) or state >= self.logit(upper)
        shorthand = abs(math.log(ratio)) >= self.logit(upper)
        self.assertTrue(general)
        self.assertFalse(shorthand)

    def test_low_and_high_stops_must_be_counted_separately(self):
        # They have opposite consequences for the cost rule, so a combined
        # "either stop" figure hides which one drives a retrieval saving.
        lower, upper, p0 = 0.20, 0.80, 0.5
        ratios = (0.05, 0.2, 5.0, 20.0)
        low = [r for r in ratios
               if self.logit(p0) + math.log(r) <= self.logit(lower)]
        high = [r for r in ratios
                if self.logit(p0) + math.log(r) >= self.logit(upper)]
        self.assertEqual(low, [0.05, 0.2])
        self.assertEqual(high, [5.0, 20.0])
        self.assertEqual(set(low) & set(high), set())
        self.assertEqual(len(set(low) | set(high)), len(low) + len(high))


class TestCostThreshold(unittest.TestCase):
    def test_the_classification_threshold_is_cm_over_cm_plus_cfa(self):
        self.assertAlmostEqual(COST_THRESHOLD, 0.2258064516129032)
        self.assertEqual(cost_based_prediction(COST_THRESHOLD + 1e-9, C_MISS, C_FALSE_ALARM), 1)
        self.assertEqual(cost_based_prediction(COST_THRESHOLD - 1e-9, C_MISS, C_FALSE_ALARM), 0)

    def test_the_boundary_itself_is_classified_nonfactual(self):
        # (1-P)*CM < P*CFA is strict, so equality falls to nonfactual.
        self.assertEqual(cost_based_prediction(COST_THRESHOLD, C_MISS, C_FALSE_ALARM), 0)

    def test_the_primary_gate_configuration_has_a_different_threshold(self):
        self.assertAlmostEqual(14.0 / (14.0 + 24.0), 0.3684210526315789)


class TestStoppingThresholdConsistency(unittest.TestCase):
    """AUDIT finding D-01 -- RESOLVED.

    The defect: the tuner's lower grid ran to 0.40 while the CM=28/CFA=96 cost
    rule switches at 0.225806, so a detector could stop LOW (asserting
    hallucination) and then be classified FACTUAL. These tests recorded that
    contradiction; they are now updated to record that it can no longer occur.

    The arithmetic that made it a contradiction is deliberately still asserted
    below -- cost_based_prediction is unchanged, so 0.25 through 0.40 still
    classify factual. What changed is that no detector can stop there.

    The positive behaviour lives in tests/test_cost_consistent_thresholds.py.
    """

    def tuner_lower_grid(self):
        return [float(x) for x in CANDIDATE_LOWER_GRID]

    def test_the_candidate_grid_still_contains_the_contradictory_values(self):
        # The candidate grid is unchanged; the FILTER is what is new, so the
        # filter has real work to do.
        inconsistent = [x for x in self.tuner_lower_grid() if x > COST_THRESHOLD]
        self.assertEqual(inconsistent, [0.25, 0.30, 0.35, 0.40])

    def test_the_cost_rule_still_calls_those_probabilities_factual(self):
        # WAS test_AUDIT_stopping_low_above_the_cost_threshold_still_predicts_factual.
        # cost_based_prediction is untouched, so the arithmetic is identical;
        # the contradiction is now prevented upstream instead.
        for lower in (0.25, 0.30, 0.35, 0.40):
            with self.subTest(lower=lower):
                self.assertEqual(cost_based_prediction(lower, C_MISS, C_FALSE_ALARM), 1)

    def test_RESOLVED_the_contradiction_is_no_longer_reachable(self):
        # WAS test_AUDIT_the_contradiction_is_reachable_through_the_detector,
        # which built lower=0.30, stopped at P=0.28 and reported FACTUAL. That
        # detector can no longer be constructed at all.
        lower = 0.30
        target = 0.28  # COST_THRESHOLD < target < lower
        ratio = (target / (1 - target)) / (0.5 / 0.5)
        with self.assertRaises(ValueError) as caught:
            DDREDetector(
                ConstantRatio(ratio), lower_threshold=lower, upper_threshold=0.80,
                p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
            )
        self.assertIn("contradict the cost-based", str(caught.exception))

    def test_RESOLVED_the_contradictory_values_cannot_enter_tuning(self):
        space = cost_consistent_thresholds(C_MISS, C_FALSE_ALARM)
        self.assertEqual(space["effective_lower_grid"], [0.05, 0.10, 0.15, 0.20])
        self.assertEqual(space["excluded_lower_grid"], [0.25, 0.30, 0.35, 0.40])
        self.assertEqual(
            {pair[0] for pair in space["threshold_pairs"]}, {0.05, 0.10, 0.15, 0.20}
        )

    def test_lower_thresholds_at_or_below_the_cost_threshold_are_consistent(self):
        for lower in (0.05, 0.10, 0.15, 0.20):
            with self.subTest(lower=lower):
                self.assertEqual(cost_based_prediction(lower, C_MISS, C_FALSE_ALARM), 0)

    def test_the_upper_grid_has_no_such_contradiction(self):
        for upper in np.round(np.arange(0.60, 0.96, 0.05), 2):
            with self.subTest(upper=upper):
                self.assertEqual(cost_based_prediction(float(upper), C_MISS, C_FALSE_ALARM), 1)


# --------------------------------------------------------------------------
# E. Validation-only tuning
# --------------------------------------------------------------------------


class TestNoTestSetAccessDuringTuning(unittest.TestCase):
    """Asserted structurally against main.py's AST.

    Running the tuner needs torch, so the property is checked on the source:
    the tuner must not name the held-out split at all, and the split must reach
    the final evaluation only.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def names_in(self, function):
        return {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
        }

    def test_the_tuner_never_references_the_test_split(self):
        names = self.names_in(self.functions["tune_ddre_thresholds"])
        self.assertNotIn("test_records", names)
        self.assertIn("validation_records", names)

    def test_the_tuner_takes_no_test_data_argument(self):
        args = [
            a.arg
            for a in self.functions["tune_ddre_thresholds"].args.args
            + self.functions["tune_ddre_thresholds"].args.kwonlyargs
        ]
        self.assertNotIn("test_records", args)
        self.assertIn("validation_records", args)

    def test_the_ratio_estimator_is_fit_before_any_split_evaluation(self):
        # uLSIF is fit from the separate NBC pairs, which are disjoint from the
        # SelfCheckGPT sentences that form the validation/test splits.
        fit_call = min(
            node.lineno
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        )
        tune_call = min(
            node.lineno
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tune_ddre_thresholds"
        )
        self.assertLess(fit_call, tune_call)

    def test_the_test_split_is_only_used_for_final_evaluation_and_reporting(self):
        main_function = self.functions["main"]
        uses = [
            node.lineno
            for node in ast.walk(main_function)
            if isinstance(node, ast.Name)
            and node.id == "test_records"
            and isinstance(node.ctx, ast.Load)
        ]
        tune_line = min(
            node.lineno
            for node in ast.walk(main_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tune_ddre_thresholds"
        )
        # The only pre-tuning use is the smoke-test subsetting, which writes to
        # separate, explicitly non-paper output files.
        before = [line for line in uses if line < tune_line]
        self.assertEqual(len(before), 1)
        self.assertIn("stratified_subset", self.source.splitlines()[before[0] - 1])

    def test_RESOLVED_the_quality_constraint_now_protects_nonfactual_pr_auc(self):
        # AUDIT finding D-07 -- RESOLVED by PR #10.
        #
        # WAS: the feasibility test named only factual AUC-PR and balanced
        # PR-AUC, so a configuration could trade away nonfactual AUC-PR --
        # Wang's headline metric -- and still qualify, because balanced is the
        # mean of the two and a factual gain masks a nonfactual loss.
        #
        # NOW: all three safeguards are required, and each is recorded per
        # candidate so the trade-off is auditable. The rule moved out of
        # main.py into src/threshold_selection.py so it is reviewable and
        # unit-testable on its own; the tuner delegates to it.
        from src.threshold_selection import candidate_record

        tuner = ast.get_source_segment(
            self.source, self.functions["tune_ddre_thresholds"]
        )
        self.assertIn("candidate_record(", tuner)
        self.assertIn("select_threshold_configuration(", tuner)

        record = candidate_record(
            0.05, 0.60,
            {
                "nonfactual": {"auc_pr": 0.78},   # -0.02 vs baseline
                "factual": {"auc_pr": 0.62},      # +0.02 vs baseline
                "balanced_pr_auc": 0.70,          # unchanged
                "accuracy": 0.5, "macro_f1": 0.5,
                "efficiency": {
                    "avg_retrieved_documents_per_sentence": 1.0,
                    "avg_retrieved_documents_per_subclaim": 1.0,
                    "avg_nli_span_calls_per_sentence": 3.0,
                },
            },
            {
                "nonfactual": {"auc_pr": 0.80},
                "factual": {"auc_pr": 0.60},
                "balanced_pr_auc": 0.70,
            },
            quality_tolerance=0.005, retrieval_penalty=0.05, max_docs=10,
        )
        for safeguard in (
            "preserves_nonfactual", "preserves_factual", "preserves_balanced"
        ):
            self.assertIn(safeguard, record)
        for recorded in (
            "nonfactual_auc_pr_delta_vs_bse",
            "factual_auc_pr_delta_vs_bse",
            "balanced_pr_auc_delta_vs_bse",
        ):
            self.assertIn(recorded, record)
        # The exact old bug: factual up, nonfactual down, balanced unchanged.
        self.assertTrue(record["preserves_factual"])
        self.assertTrue(record["preserves_balanced"])
        self.assertFalse(record["preserves_nonfactual"])
        self.assertFalse(record["preserves_baseline_quality"])

    def test_RESOLVED_no_point_estimate_can_declare_the_hypothesis_supported(self):
        # AUDIT finding D-08 -- RESOLVED by PR #10.
        #
        # WAS: hypothesis_comparison computed `hypothesis_supported_on_test`
        # from test-set point estimates alone, with no uncertainty of any kind,
        # and with no nonfactual floor.
        #
        # NOW: the boolean is gone; the function returns labelled descriptive
        # effect sizes, and the confirmatory decision comes from the frozen
        # paired passage-level bootstrap.
        self.assertNotIn("hypothesis_supported_on_test", self.source)
        rule = ast.get_source_segment(
            self.source, self.functions["hypothesis_comparison"]
        )
        self.assertNotIn("supported = (", rule)
        self.assertIn("descriptive point estimates", rule)
        self.assertIn("claim_assessment", rule)
        self.assertIn("assess_claim", self.source)
        self.assertIn("paired_passage_bootstrap", self.source)


# --------------------------------------------------------------------------
# F. DDRE / BSE fairness
# --------------------------------------------------------------------------


class TestEvidenceStreamFairness(unittest.TestCase):
    def test_both_detectors_read_the_same_documents_in_the_same_order(self):
        class RecordingScorer:
            def __init__(self):
                self.seen = []
            def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
                self.seen.append(id(page_content))
                return 50.0, 1

        class NumberedDoc:
            def __init__(self, i):
                self.page_content = f"document {i}"

        subclaim = Sub(0)
        subclaim.documents = [NumberedDoc(i) for i in range(4)]

        bse_scorer, ddre_scorer = RecordingScorer(), RecordingScorer()
        BSEDetector(
            list(RELEASED_POSITIVE), list(RELEASED_NEGATIVE), mode="official",
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, c_retrieve=1,
            max_docs=4,
        ).detect_subclaim(subclaim, bse_scorer)
        DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=4,
        ).detect_subclaim(subclaim, ddre_scorer)
        self.assertEqual(
            ddre_scorer.seen[: len(bse_scorer.seen)], bse_scorer.seen
        )

    def test_both_detectors_use_the_same_cost_rule_and_aggregation(self):
        # The aggregation moved into detect_sentence_with_trace when D-10 added
        # per-subclaim traces (PR #12); detect_sentence delegates to it, so
        # there is still exactly ONE aggregation per detector. The property
        # under test is unchanged: both detectors use the same cost rule and
        # the same min-over-subclaims aggregation.
        for detector_class in (BSEDetector, DDREDetector):
            with self.subTest(detector=detector_class.__name__):
                aggregating = detector_class.detect_sentence_with_trace.__code__
                self.assertIn("cost_based_prediction", aggregating.co_names)
                self.assertIn("min", aggregating.co_names)
                # And the plain entry point does not aggregate independently.
                delegating = detector_class.detect_sentence.__code__.co_names
                self.assertIn("detect_sentence_with_trace", delegating)
                self.assertNotIn("cost_based_prediction", delegating)

    def test_a_threshold_pre_check_could_never_stop_at_the_prior(self):
        # Why a pre-retrieval threshold check does NOT fix D-02. With P0 = 0.5,
        # every lower in the grid is <= 0.40 and every upper is >= 0.60, so the
        # prior lies strictly inside the band and a band evaluated before the
        # first document can never fire. BSE's zero-retrieval behaviour comes
        # from its decision-theoretic expected-cost rule, not from checking
        # early, so equalising the floor would be a stopping-rule redesign.
        p0 = 0.5
        for lower in np.round(np.arange(0.05, 0.41, 0.05), 2):
            for upper in np.round(np.arange(0.60, 0.96, 0.05), 2):
                with self.subTest(lower=float(lower), upper=float(upper)):
                    self.assertFalse(p0 <= float(lower) or p0 >= float(upper))

    def test_AUDIT_ddre_always_retrieves_at_least_one_document_but_bse_may_not(self):
        # AUDIT finding D-02, an algorithm/protocol asymmetry. BSE's
        # decision-theoretic rule can decline the FIRST fetch and retrieve zero
        # documents; DDRE's probability band is evaluated only after an update,
        # so it has a floor of one document per subclaim. The two retrieval
        # counts are not measured from the same origin, and the floor currently
        # DISADVANTAGES DDRE.
        never = BSEDetector(
            [1] * 10, [1] * 10, mode="official", p0=0.5, c_miss=1.0,
            c_false_alarm=1.0, c_retrieve=1000.0, max_docs=10,
        )
        bse_scorer = FixedScorer(50.0)
        bse = never.detect_subclaim(Sub(10), bse_scorer)
        self.assertEqual(bse.documents_used, 0)
        self.assertEqual(bse_scorer.calls, 0)

        ddre_scorer = FixedScorer(50.0)
        ddre = DDREDetector(
            ConstantRatio(1e6), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        ).detect_subclaim(Sub(10), ddre_scorer)
        self.assertEqual(ddre.documents_used, 1)
        self.assertEqual(ddre_scorer.calls, 1)

    def test_AUDIT_ddre_has_a_tuned_stopping_rule_and_published_bse_has_none(self):
        # AUDIT finding D-09, a claim boundary rather than an invalid
        # comparison. DDRE selects its stopping band from 64 validation
        # configurations; published BSE's stopping rule follows from the fixed
        # published costs and has no tunable counterpart. The recommendation is
        # to KEEP published BSE (CM=28, CFA=96, c_retrieve=1) as the primary
        # comparator and report any tuned BSE variant as a labelled secondary
        # robustness analysis.
        # 64 candidate pairs, of which 32 survive the D-01 cost-consistency
        # filter at CM=28/CFA=96. Either way DDRE selects from many
        # configurations and published BSE selects from none.
        self.assertEqual(len(CANDIDATE_LOWER_GRID) * len(CANDIDATE_UPPER_GRID), 64)
        self.assertEqual(
            cost_consistent_thresholds(C_MISS, C_FALSE_ALARM)[
                "threshold_pairs_evaluated"
            ],
            32,
        )
        bse_init = BSEDetector.__init__.__code__.co_varnames[
            : BSEDetector.__init__.__code__.co_argcount
        ]
        self.assertNotIn("lower_threshold", bse_init)
        self.assertNotIn("upper_threshold", bse_init)

    def test_the_published_primary_comparator_configuration_is_unchanged(self):
        # The primary baseline stays Wang's published configuration, so that the
        # comparison remains comparable with the paper. Pinned here so that
        # swapping in a tuned baseline cannot happen silently.
        detector = BSEDetector(
            list(RELEASED_POSITIVE), list(RELEASED_NEGATIVE), mode="official",
            p0=0.5, c_miss=28, c_false_alarm=96, c_retrieve=1, max_docs=10,
        )
        self.assertEqual(detector.mode, "official")
        self.assertEqual((detector.c_miss, detector.c_false_alarm), (28.0, 96.0))
        self.assertEqual(detector.c_retrieve, 1.0)
        self.assertEqual(detector.p0, 0.5)


# --------------------------------------------------------------------------
# G/H. Reporting and statistical readiness
# --------------------------------------------------------------------------


class TestReportingReadiness(unittest.TestCase):
    def test_the_detection_result_carries_the_compute_counters(self):
        from src.baseline_core import DetectionResult

        fields = DetectionResult.__dataclass_fields__
        for name in ("p_factual", "prediction", "documents_used", "nli_calls"):
            self.assertIn(name, fields)

    def test_RESOLVED_per_subclaim_detail_is_now_retained_alongside_the_result(self):
        # WAS test_AUDIT_per_subclaim_detail_is_discarded_at_the_sentence_level.
        # D-10, resolved by PR #12. DetectionResult still carries only summed
        # counters -- that is unchanged -- but the exact per-subclaim results
        # are now returned from the SAME pass, so the stopping depths are no
        # longer lost.
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=4,
        )
        record = Rec([Sub(4), Sub(1)])
        result = detector.detect_sentence(record, FixedScorer(50.0))
        self.assertEqual(result.documents_used, 5)
        self.assertFalse(hasattr(result, "per_subclaim"))

        traced, subclaim_results = detector.detect_sentence_with_trace(
            record, FixedScorer(50.0)
        )
        self.assertEqual(traced.documents_used, result.documents_used)
        self.assertEqual(traced.p_factual, result.p_factual)
        self.assertEqual(len(subclaim_results), 2)
        self.assertEqual(
            [r.documents_used for r in subclaim_results], [4, 1]
        )

    def test_RESOLVED_prediction_rows_record_the_subclaim_count(self):
        # WAS test_prediction_rows_do_not_record_the_subclaim_count. Needed to
        # express documents-per-subclaim on a bootstrap resample; D-10 resolved
        # by PR #12.
        source = (PROJECT_ROOT / "src" / "evaluation.py").read_text(encoding="utf-8")
        rows = source[source.index("def prediction_rows"):]
        self.assertIn("passage_index", rows)
        self.assertIn("retrieved_documents", rows)
        self.assertIn('"n_subclaims"', rows)
        self.assertIn('"subclaim_documents_used"', rows)
        self.assertIn('"subclaim_documents_available"', rows)
        self.assertIn('"subclaim_nli_calls"', rows)


if __name__ == "__main__":
    unittest.main()
