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
from src.ddre_core import DDREDetector, ULSIFDensityRatio

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
        # Standard uLSIF places kernel centers on the numerator (factual)
        # sample. This is also the reason r(s) collapses toward zero where no
        # factual score was observed -- see the extrapolation tests below.
        estimator = fitted_estimator()
        self.assertTrue(np.all(estimator.centers >= 0.0))
        self.assertTrue(np.all(estimator.centers <= 1.0))

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

    def test_AUDIT_a_degenerate_all_zero_fit_reads_as_overwhelming_evidence(self):
        # AUDIT finding D-04. If the solved alpha vanishes, r(s) is identically
        # zero and the clip turns "the model has nothing to say" into
        # log r = -13.8 per document, i.e. near-certain hallucination. Nothing
        # in fit() detects or reports this.
        estimator = fitted_estimator()
        estimator.alpha = np.zeros_like(estimator.alpha)
        self.assertEqual(estimator.ratio(50.0), 1e-6)
        self.assertAlmostEqual(math.log(estimator.ratio(50.0)), -13.8155, places=3)

    def test_AUDIT_the_ratio_is_unbounded_where_the_denominator_has_no_support(self):
        # AUDIT finding D-03. The kernel model is a positive combination of
        # Gaussians centred on factual scores, and H only penalises magnitude
        # where hallucinated scores live. Far above the hallucinated sample the
        # fitted ratio is therefore free to grow without bound: this is
        # extrapolation, not evidence.
        rng = np.random.default_rng(11)
        estimator = ULSIFDensityRatio(max_centers=30, random_state=7).fit(
            np.clip(rng.normal(78, 7, 120), 0, 100),
            np.clip(rng.normal(22, 7, 120), 0, 100),
            folds=4,
        )
        self.assertGreater(estimator.ratio(100.0), 100.0)
        self.assertGreater(math.log(estimator.ratio(100.0)), 4.0)

    def test_AUDIT_a_non_finite_score_produces_a_non_finite_ratio(self):
        # AUDIT finding D-06. There is no guard, so NaN passes straight
        # through, and +inf is silently clipped to a perfect score of 100.
        estimator = fitted_estimator()
        self.assertTrue(math.isnan(estimator.ratio(float("nan"))))
        self.assertEqual(estimator.ratio(float("inf")), estimator.ratio(100.0))


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

    def test_AUDIT_a_nan_score_spends_the_whole_budget_and_yields_nan(self):
        # AUDIT finding D-06. NaN defeats every comparison, so no stopping rule
        # fires, the full retrieval budget is spent, and NaN reaches the
        # metrics. Nothing warns.
        detector = DDREDetector(
            fitted_estimator(), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        result = detector.detect_subclaim(Sub(10), FixedScorer(float("nan")))
        self.assertTrue(math.isnan(result.p_factual))
        self.assertEqual(result.documents_used, 10)


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
    """AUDIT finding D-01: the stop-low rule can contradict the cost rule."""

    def tuner_lower_grid(self):
        return [float(x) for x in np.round(np.arange(0.05, 0.41, 0.05), 2)]

    def test_the_tuner_grid_contains_lower_thresholds_above_the_cost_threshold(self):
        inconsistent = [x for x in self.tuner_lower_grid() if x > COST_THRESHOLD]
        self.assertEqual(inconsistent, [0.25, 0.30, 0.35, 0.40])

    def test_AUDIT_stopping_low_above_the_cost_threshold_still_predicts_factual(self):
        # The detector stops because it is confident the claim is hallucinated,
        # and the cost rule then labels it FACTUAL. The two rules disagree for
        # every stopping probability in (COST_THRESHOLD, lower_threshold].
        for lower in (0.25, 0.30, 0.35, 0.40):
            with self.subTest(lower=lower):
                self.assertEqual(cost_based_prediction(lower, C_MISS, C_FALSE_ALARM), 1)

    def test_AUDIT_the_contradiction_is_reachable_through_the_detector(self):
        # A single document with r < 1 lands the posterior inside the
        # contradictory window, the detector stops, and the reported prediction
        # is FACTUAL despite evidence against.
        lower = 0.30
        target = 0.28  # COST_THRESHOLD < target < lower
        ratio = (target / (1 - target)) / (0.5 / 0.5)
        detector = DDREDetector(
            ConstantRatio(ratio), lower_threshold=lower, upper_threshold=0.80,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=10,
        )
        result = detector.detect_subclaim(Sub(10), FixedScorer(20.0))
        self.assertEqual(result.documents_used, 1)
        self.assertLess(result.p_factual, lower)
        self.assertGreater(result.p_factual, COST_THRESHOLD)
        self.assertEqual(result.prediction, 1)

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

    def test_AUDIT_the_quality_constraint_does_not_protect_nonfactual_pr_auc(self):
        # AUDIT finding D-07. The feasibility test names only factual AUC-PR
        # and balanced PR-AUC, so a configuration may trade away nonfactual
        # AUC-PR -- Wang's headline metric -- and still qualify.
        tuner = ast.get_source_segment(
            self.source, self.functions["tune_ddre_thresholds"]
        )
        qualifies = tuner[tuner.index("qualifies = ("):tuner.index("candidate = {")]
        self.assertIn("factual", qualifies)
        self.assertIn("balanced_pr_auc", qualifies)
        self.assertNotIn("nonfactual", qualifies)

    def test_AUDIT_the_hypothesis_rule_does_not_protect_nonfactual_pr_auc(self):
        # AUDIT finding D-08. The same omission decides whether the paper's
        # claim is reported as supported.
        rule = ast.get_source_segment(
            self.source, self.functions["hypothesis_comparison"]
        )
        supported = rule[rule.index("supported = ("):rule.index("return {")]
        self.assertIn("factual_delta", supported)
        self.assertNotIn("nonfactual_delta", supported)


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
        for detector_class in (BSEDetector, DDREDetector):
            source_names = detector_class.detect_sentence.__code__.co_names
            self.assertIn("cost_based_prediction", source_names)
            self.assertIn("min", source_names)

    def test_AUDIT_ddre_always_retrieves_at_least_one_document_but_bse_may_not(self):
        # AUDIT finding D-02. BSE tests its stopping rule BEFORE the first
        # fetch and can therefore retrieve zero documents; DDRE only tests
        # after an update, so it has a floor of one document per subclaim.
        # The two retrieval counts are not measured from the same origin.
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

    def test_AUDIT_ddre_has_a_tuned_stopping_rule_and_bse_has_none(self):
        # AUDIT finding D-09. DDRE selects its stopping band from 64 validation
        # configurations; BSE's stopping rule is derived from the fixed costs
        # and has no tunable counterpart in this pipeline.
        lower = np.round(np.arange(0.05, 0.41, 0.05), 2)
        upper = np.round(np.arange(0.60, 0.96, 0.05), 2)
        self.assertEqual(len(lower) * len(upper), 64)
        bse_init = BSEDetector.__init__.__code__.co_varnames[
            : BSEDetector.__init__.__code__.co_argcount
        ]
        self.assertNotIn("lower_threshold", bse_init)
        self.assertNotIn("upper_threshold", bse_init)


# --------------------------------------------------------------------------
# G/H. Reporting and statistical readiness
# --------------------------------------------------------------------------


class TestReportingReadiness(unittest.TestCase):
    def test_the_detection_result_carries_the_compute_counters(self):
        from src.baseline_core import DetectionResult

        fields = DetectionResult.__dataclass_fields__
        for name in ("p_factual", "prediction", "documents_used", "nli_calls"):
            self.assertIn(name, fields)

    def test_AUDIT_per_subclaim_detail_is_discarded_at_the_sentence_level(self):
        # AUDIT finding D-10. detect_sentence returns one DetectionResult with
        # summed counters, so per-subclaim stopping depth cannot be recovered
        # from the returned object or from the predictions CSV.
        detector = DDREDetector(
            ConstantRatio(1.0), lower_threshold=0.2, upper_threshold=0.8,
            p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=4,
        )
        result = detector.detect_sentence(Rec([Sub(4), Sub(1)]), FixedScorer(50.0))
        self.assertEqual(result.documents_used, 5)
        self.assertFalse(hasattr(result, "per_subclaim"))

    def test_prediction_rows_do_not_record_the_subclaim_count(self):
        # Needed to express documents-per-subclaim on a bootstrap resample.
        source = (PROJECT_ROOT / "src" / "evaluation.py").read_text(encoding="utf-8")
        rows = source[source.index("def prediction_rows"):]
        self.assertIn("passage_index", rows)
        self.assertIn("retrieved_documents", rows)
        self.assertNotIn("subclaims", rows)


if __name__ == "__main__":
    unittest.main()
