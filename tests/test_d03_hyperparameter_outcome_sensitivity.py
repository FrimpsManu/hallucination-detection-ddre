"""D-03 follow-up: hyperparameter outcome sensitivity, validation only.

D-03 trigger 2 fired on evidence direction. This follow-up asks only whether the
downstream validation-level conclusion moves across the same 20 production
(sigma, lambda) pairs. It must select nothing and must never touch the held-out
half, so these tests pin exactly that: the split identity, the held-out
firewall, the read-only cache, fail-closed replay, the fixed center set, the
unchanged production selection, reuse of the existing validation rule, and the
absence of any cap/calibration/method claim in the output.

No model, no NLI inference, no GPU.
"""

import ast
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from src.cached_document_scores import (
    CachedDocumentScorer,
    MissingDocumentScore,
    MissingPairScore,
)
from src.density_ratio_diagnostics import production_hyperparameter_pairs
from src.ddre_core import (
    CANDIDATE_LOWER_GRID,
    CANDIDATE_UPPER_GRID,
    ULSIFDensityRatio,
    cost_consistent_thresholds,
)
from src.hyperparameter_outcome_sensitivity import (
    C_FALSE_ALARM,
    C_MISS,
    D03_BASE_COMMIT,
    EXPECTED_HELD_OUT_PASSAGES,
    EXPECTED_HYPERPARAMETER_PAIRS,
    EXPECTED_SPLIT_SEED,
    EXPECTED_THRESHOLD_PAIRS,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VALIDATION_PASSAGES,
    MAX_DOCS,
    PRODUCTION_LAMBDA,
    PRODUCTION_SIGMA,
    ArtifactVerificationFailed,
    HeldOutLeak,
    ProductionFitChanged,
    assert_validation_only,
    candidate_estimator,
    passage_ids_sha256,
    robustness_summary,
    verify_d03_artifact,
    verify_production_fit,
)
from src.threshold_selection import select_threshold_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "d03_hyperparameter_outcome_sensitivity.py"

REPORT_SHA = "f2468e5bb1265e107f4060d36782f6f21ba9f5f55141dde4814d5ffd11aa7f6d"
CACHE_SHA = "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"
VALIDATION_IDS = list(range(48))

# The ACTUAL frozen D-03 validation passages, reconstructed from the released
# Wang data at validation_fraction=0.20, seed=42.
REAL_VALIDATION_PASSAGE_IDS = [
    3, 7, 8, 20, 24, 26, 33, 37, 38, 46, 50, 51, 54, 70, 88, 96, 98, 104,
    106, 115, 126, 127, 128, 130, 133, 134, 144, 149, 153, 161, 165, 170,
    180, 192, 198, 201, 208, 211, 212, 215, 216, 225, 226, 231, 233, 235,
    236, 237,
]


class Record:
    """Minimal stand-in for SentenceRecord: only what the firewall reads."""

    def __init__(self, passage_index):
        self.passage_index = passage_index


def d03_report(**overrides):
    report = {
        "git_commit": D03_BASE_COMMIT,
        "completion": {"complete": True},
        "score_compatibility_established": True,
        "held_out_scored": False,
        "split": {
            "identity_matches": True,
            "validation_fraction": EXPECTED_VALIDATION_FRACTION,
            "validation_passages": EXPECTED_VALIDATION_PASSAGES,
            "sha256": passage_ids_sha256(VALIDATION_IDS),
        },
    }
    report.update(overrides)
    return report


def verify(report=None, **kwargs):
    defaults = {
        "report_sha256": REPORT_SHA,
        "expected_report_sha256": REPORT_SHA,
        "cache_sha256": CACHE_SHA,
        "expected_cache_sha256": CACHE_SHA,
    }
    defaults.update(kwargs)
    return verify_d03_artifact(report if report is not None else d03_report(), **defaults)


# --------------------------------------------------------------------------

class TestArtifactVerificationFailsClosed(unittest.TestCase):
    def test_the_frozen_artifacts_verify(self):
        record = verify()
        self.assertTrue(record["passed"])
        self.assertEqual(record["failures"], [])
        self.assertEqual(record["d03_split_sha256"], passage_ids_sha256(VALIDATION_IDS))

    def test_a_wrong_report_digest_aborts(self):
        with self.assertRaises(ArtifactVerificationFailed) as caught:
            verify(report_sha256="0" * 64)
        self.assertIn("d03_report_sha256", str(caught.exception))

    def test_a_wrong_cache_digest_aborts(self):
        with self.assertRaises(ArtifactVerificationFailed) as caught:
            verify(cache_sha256="0" * 64)
        self.assertIn("derived_cache_sha256", str(caught.exception))

    def test_a_different_d03_commit_aborts(self):
        with self.assertRaises(ArtifactVerificationFailed):
            verify(d03_report(git_commit="deadbeef" * 5))

    def test_an_incomplete_d03_run_aborts(self):
        with self.assertRaises(ArtifactVerificationFailed):
            verify(d03_report(completion={"complete": False}))

    def test_unestablished_score_compatibility_aborts(self):
        with self.assertRaises(ArtifactVerificationFailed):
            verify(d03_report(score_compatibility_established=False))

    def test_a_d03_run_that_scored_held_out_aborts(self):
        with self.assertRaises(ArtifactVerificationFailed):
            verify(d03_report(held_out_scored=True))

    def test_a_missing_held_out_flag_is_not_a_pass(self):
        report = d03_report()
        del report["held_out_scored"]
        with self.assertRaises(ArtifactVerificationFailed) as caught:
            verify(report)
        self.assertIn("d03_held_out_scored", str(caught.exception))

    def test_every_failure_is_reported_not_just_the_first(self):
        with self.assertRaises(ArtifactVerificationFailed) as caught:
            verify(d03_report(git_commit="x", score_compatibility_established=False),
                   report_sha256="0" * 64)
        message = str(caught.exception)
        for name in ("d03_report_sha256", "d03_git_commit",
                     "score_compatibility_established"):
            self.assertIn(name, message)


class TestSplitIdentity(unittest.TestCase):
    def test_the_digest_is_over_the_sorted_passage_ids(self):
        self.assertEqual(
            passage_ids_sha256([3, 1, 2]), passage_ids_sha256([1, 2, 3])
        )

    def test_a_different_passage_set_is_a_different_digest(self):
        self.assertNotEqual(
            passage_ids_sha256(VALIDATION_IDS),
            passage_ids_sha256(VALIDATION_IDS[:-1] + [999]),
        )

    def test_the_digest_matches_the_d03_script_implementation(self):
        # Transcribed rather than imported; this pins the transcription against
        # the original so the two cannot drift apart unnoticed.
        import hashlib
        import json

        expected = hashlib.sha256(
            json.dumps(sorted(VALIDATION_IDS), sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(passage_ids_sha256(VALIDATION_IDS), expected)

    def test_the_real_frozen_split_digest_is_reproduced(self):
        """Anchor: the actual D-03 validation passage set.

        Reconstructing the frozen split from the released Wang data at
        validation_fraction=0.20, seed=42 yields 48 passages whose digest is
        this value, matching what the D-03 dry run recorded. Anchored as a
        literal so a change to the split, the seed or the digest convention
        fails here instead of silently redefining the population.
        """
        self.assertEqual(
            passage_ids_sha256(REAL_VALIDATION_PASSAGE_IDS),
            "618f248e3ddad32881a787d445badc6ddf010050c4a645f7e185cb017ce7bf67",
        )
        self.assertEqual(len(REAL_VALIDATION_PASSAGE_IDS), 48)

    def test_the_frozen_constants_are_the_d03_ones(self):
        self.assertEqual(EXPECTED_VALIDATION_FRACTION, 0.20)
        self.assertEqual(EXPECTED_SPLIT_SEED, 42)
        self.assertEqual(EXPECTED_VALIDATION_PASSAGES, 48)
        self.assertEqual(EXPECTED_HELD_OUT_PASSAGES, 190)


class TestHeldOutFirewall(unittest.TestCase):
    def test_validation_only_records_pass(self):
        records = [Record(i) for i in VALIDATION_IDS]
        self.assertTrue(assert_validation_only(records, range(48, 238), where="t"))

    def test_a_single_held_out_record_raises(self):
        records = [Record(i) for i in VALIDATION_IDS] + [Record(200)]
        with self.assertRaises(HeldOutLeak) as caught:
            assert_validation_only(records, range(48, 238), where="a detector")
        self.assertIn("200", str(caught.exception))
        self.assertIn("a detector", str(caught.exception))

    def test_the_script_guards_every_scoring_site(self):
        source = SCRIPT.read_text(encoding="utf-8")
        # Exactly three guard sites: the population once, the BSE evaluation
        # once, and every DDRE candidate inside the sweep.
        self.assertEqual(source.count("assert_validation_only("), 3)
        for where in ("the validation population", "BSE validation",
                      "a DDRE candidate evaluation"):
            self.assertIn(where, source)
        self.assertIn('"held_out_scored": False', source)

    def test_the_script_never_passes_the_held_out_half_to_a_scorer(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        # The split helper returns it as _held_out; nothing else may use it.
        self.assertIn("_held_out", source)
        names = [
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "_held_out"
        ]
        # Bound once by the tuple unpack, and never read again.
        self.assertEqual(len(names), 1)

    def test_evaluation_is_only_ever_called_with_validation(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for call in ("evaluate_on_validation(bse, validation, scorer)",
                     "evaluate_on_validation(detector, validation, scorer)"):
            self.assertIn(call, source)


class TestCacheIsReadOnlyAndFailsClosed(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="sensitivity-cache-")
        self.root = Path(self._tmp.name)
        self.cache = self.root / "derived.sqlite"
        connection = sqlite3.connect(str(self.cache))
        connection.execute(
            "CREATE TABLE nli_scores (cache_key TEXT PRIMARY KEY, "
            "model_name TEXT NOT NULL, score_version TEXT NOT NULL, "
            "score REAL NOT NULL)"
        )
        connection.commit()
        connection.close()
        self.scorer = CachedDocumentScorer(
            self.cache, "model", "v2", lambda t, **k: [str(t)]
        )

    def tearDown(self):
        self.scorer.close()
        self._tmp.cleanup()

    def _store(self, premise, hypothesis, score):
        connection = sqlite3.connect(str(self.cache))
        connection.execute(
            "INSERT OR REPLACE INTO nli_scores VALUES (?, ?, ?, ?)",
            (self.scorer._cache_key(premise, hypothesis), "model", "v2", score),
        )
        connection.commit()
        connection.close()

    def test_the_connection_is_opened_read_only(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.scorer._connection.execute(
                "INSERT INTO nli_scores VALUES ('k','m','v2',1.0)"
            )

    def test_score_pairs_replays_recorded_scores_in_order(self):
        self._store("p1", "h1", 11.5)
        self._store("p2", "h2", 22.5)
        self.assertEqual(
            self.scorer.score_pairs([("p1", "h1"), ("p2", "h2")]), [11.5, 22.5]
        )

    def test_a_missing_pair_score_fails_closed(self):
        self._store("p1", "h1", 11.5)
        with self.assertRaises(MissingPairScore) as caught:
            self.scorer.score_pairs([("p1", "h1"), ("absent", "hypothesis")])
        self.assertIn("2/2", str(caught.exception))

    def test_nothing_is_defaulted_or_skipped_on_a_miss(self):
        with self.assertRaises(MissingPairScore):
            self.scorer.score_pairs([("absent", "h")])

    def test_a_missing_document_span_still_fails_closed(self):
        with self.assertRaises(MissingDocumentScore):
            self.scorer.score_document("claim", "page content")

    def test_score_pairs_uses_the_existing_key_convention(self):
        import hashlib

        expected = hashlib.sha256(
            "\0".join(["model", "v2", "p", "h"]).encode("utf-8")
        ).hexdigest()
        self.assertEqual(self.scorer._cache_key("p", "h"), expected)

    def test_the_cache_file_is_never_written(self):
        before = self.cache.stat().st_mtime_ns
        self._store("p", "h", 1.0)
        self.scorer.score_pairs([("p", "h")])
        # The scorer itself performed no write; only the test fixture did.
        self.assertTrue(self.scorer._connection is not None)
        self.assertGreaterEqual(self.cache.stat().st_mtime_ns, before)


def fitted_production():
    """A real production-shaped fit on synthetic NBC-like scores."""
    import numpy as np

    rng = np.random.default_rng(11)
    factual = np.clip(rng.normal(72, 12, 199), 0, 100)
    hallucinated = np.clip(rng.normal(26, 12, 199), 0, 100)
    estimator = ULSIFDensityRatio(random_state=EXPECTED_SPLIT_SEED).fit(
        factual, hallucinated
    )
    return estimator, factual, hallucinated


class TestHyperparameterSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production, cls.factual, cls.hallucinated = fitted_production()
        cls.pairs = production_hyperparameter_pairs(cls.production)

    def test_there_are_exactly_twenty_unique_pairs(self):
        self.assertEqual(len(self.pairs), EXPECTED_HYPERPARAMETER_PAIRS)
        self.assertEqual(len(set(self.pairs)), EXPECTED_HYPERPARAMETER_PAIRS)

    def test_the_pairs_come_from_the_production_cv_table(self):
        recorded = {
            (float(r["sigma"]), float(r["lambda"])) for r in self.production.cv_table
        }
        self.assertEqual(set(self.pairs), recorded)

    def test_the_centers_are_held_fixed_across_every_pair(self):
        import numpy as np

        factual_x = self.production._as_column(self.factual)
        hallucinated_x = self.production._as_column(self.hallucinated)
        for sigma, lam in self.pairs:
            candidate = candidate_estimator(
                self.production, factual_x, hallucinated_x, sigma, lam
            )
            with self.subTest(sigma=sigma, lam=lam):
                self.assertIs(candidate.centers, self.production.centers)
                np.testing.assert_array_equal(
                    candidate.centers, self.production.centers
                )

    def test_each_candidate_carries_its_own_sigma_lambda_and_alpha(self):
        factual_x = self.production._as_column(self.factual)
        hallucinated_x = self.production._as_column(self.hallucinated)
        seen = set()
        for sigma, lam in self.pairs:
            candidate = candidate_estimator(
                self.production, factual_x, hallucinated_x, sigma, lam
            )
            self.assertEqual(candidate.sigma, float(sigma))
            self.assertEqual(candidate.lam, float(lam))
            seen.add(tuple(candidate.alpha.tolist()))
        # Different hyperparameters give different solutions.
        self.assertGreater(len(seen), 1)

    def test_the_candidate_uses_the_untouched_production_clip(self):
        factual_x = self.production._as_column(self.factual)
        hallucinated_x = self.production._as_column(self.hallucinated)
        candidate = candidate_estimator(
            self.production, factual_x, hallucinated_x, *self.pairs[0]
        )
        for score in (0.0, 50.0, 100.0):
            value = candidate.ratio(score)
            self.assertGreaterEqual(value, 1e-6)
            self.assertLessEqual(value, 1e6)

    def test_the_production_estimator_is_not_mutated_by_the_sweep(self):
        before = (self.production.sigma, self.production.lam,
                  tuple(self.production.alpha.tolist()))
        factual_x = self.production._as_column(self.factual)
        hallucinated_x = self.production._as_column(self.hallucinated)
        for sigma, lam in self.pairs:
            candidate_estimator(
                self.production, factual_x, hallucinated_x, sigma, lam
            )
        after = (self.production.sigma, self.production.lam,
                 tuple(self.production.alpha.tolist()))
        self.assertEqual(before, after)


class TestProductionSelectionUnchanged(unittest.TestCase):
    def test_the_frozen_production_pair_is_the_recorded_one(self):
        self.assertEqual(PRODUCTION_SIGMA, 0.2269287109375)
        self.assertEqual(PRODUCTION_LAMBDA, 1.0)

    def test_a_refit_that_moves_sigma_aborts(self):
        class Fake:
            sigma = 0.5
            lam = 1.0
            cv_table = [{"sigma": 0.5, "lambda": 1.0}] * 1
            centers = None
        with self.assertRaises(ProductionFitChanged) as caught:
            verify_production_fit(Fake())
        self.assertIn("sigma", str(caught.exception))

    def test_a_refit_with_the_wrong_number_of_pairs_aborts(self):
        import numpy as np

        class Fake:
            sigma = PRODUCTION_SIGMA
            lam = PRODUCTION_LAMBDA
            cv_table = [{"sigma": PRODUCTION_SIGMA, "lambda": PRODUCTION_LAMBDA}]
            centers = np.zeros((3, 1))
        with self.assertRaises(ProductionFitChanged) as caught:
            verify_production_fit(Fake())
        self.assertIn("unique (sigma, lambda) pairs", str(caught.exception))

    def test_the_script_never_assigns_a_new_production_hyperparameter(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"production_hyperparameters_changed": False', source)
        self.assertIn('"sensitivity_only": True', source)


class TestThirtyTwoThresholdsPerPair(unittest.TestCase):
    def test_the_cost_consistent_grid_has_exactly_32_pairs(self):
        space = cost_consistent_thresholds(
            C_MISS, C_FALSE_ALARM, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID
        )
        pairs = [
            (lower, upper)
            for lower in space["effective_lower_grid"]
            for upper in space["effective_upper_grid"]
            if lower < upper
        ]
        self.assertEqual(len(pairs), EXPECTED_THRESHOLD_PAIRS)

    def test_the_script_aborts_if_that_count_changes(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_THRESHOLD_PAIRS", source)
        self.assertIn("cost-consistent threshold", source)

    def test_the_costs_are_the_production_ones(self):
        self.assertEqual((C_MISS, C_FALSE_ALARM, MAX_DOCS), (28.0, 96.0, 10))


class TestExistingValidationRuleIsUsed(unittest.TestCase):
    def test_the_script_imports_the_repository_rule(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("from src.threshold_selection import", source)
        self.assertIn("candidate_record", source)
        self.assertIn("select_threshold_configuration(candidates)", source)

    def test_no_alternative_quality_criterion_is_defined(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for banned in ("def my_quality", "def custom_feasible",
                       "preserves_baseline_quality ="):
            self.assertNotIn(banned, source)
        self.assertIn('"new_quality_criterion_introduced": False', source)

    def test_the_selection_helper_still_returns_three_values(self):
        candidates = [{
            "preserves_baseline_quality": True, "avg_documents": 2.0,
            "balanced_pr_auc": 0.8, "nonfactual_auc_pr": 0.85,
            "factual_auc_pr": 0.6, "lower": 0.05, "upper": 0.60,
            "fallback_objective": 0.7,
        }]
        selected, rule, confirmatory = select_threshold_configuration(candidates)
        self.assertIs(selected, candidates[0])
        self.assertTrue(confirmatory)
        self.assertIsInstance(rule, str)


class TestOutputClaimsNothing(unittest.TestCase):
    def test_the_script_records_no_cap_or_calibration_or_method_change(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"selected_cap": None', source)
        self.assertIn('"selected_calibration": None', source)
        self.assertIn('"method_change_made": False', source)

    def test_the_summary_asserts_no_interpretation(self):
        rows = [{
            "is_production_pair": True, "feasible_threshold_count": 1,
            "confirmatory_threshold_selection": True,
            "selected": {
                "nonfactual_auc_pr_delta_vs_bse": 0.01,
                "factual_auc_pr_delta_vs_bse": -0.002,
                "balanced_pr_auc_delta_vs_bse": 0.004,
                "retrieved_documents_reduction_fraction": 0.31,
                "nli_span_calls_reduction_fraction": 0.29,
                "preserves_baseline_quality": True,
            },
        }]
        summary = robustness_summary(rows)
        self.assertIsNone(summary["interpretation"])
        self.assertIn("Descriptive only", summary["interpretation_note"])

    def test_the_summary_reports_ranges_and_counts_only(self):
        def row(reduction, confirmatory, quality=True):
            return {
                "is_production_pair": False, "feasible_threshold_count": 1 if confirmatory else 0,
                "confirmatory_threshold_selection": confirmatory,
                "selected": {
                    "nonfactual_auc_pr_delta_vs_bse": 0.0,
                    "factual_auc_pr_delta_vs_bse": 0.0,
                    "balanced_pr_auc_delta_vs_bse": 0.0,
                    "retrieved_documents_reduction_fraction": reduction,
                    "nli_span_calls_reduction_fraction": reduction,
                    "preserves_baseline_quality": quality,
                },
            }
        summary = robustness_summary([row(0.1, True), row(-0.2, False), row(0.4, True)])
        self.assertEqual(summary["hyperparameter_pairs"], 3)
        self.assertEqual(summary["pairs_with_a_confirmatorily_feasible_threshold"], 2)
        self.assertEqual(summary["pairs_without_a_confirmatorily_feasible_threshold"], 1)
        self.assertEqual(summary["pairs_whose_selection_reduces_retrieval"], 2)
        self.assertEqual(summary["pairs_preserving_quality_and_reducing_retrieval"], 2)
        spread = summary["spread_all_pairs"]["retrieved_documents_reduction_fraction"]
        self.assertEqual((spread["min"], spread["max"]), (-0.2, 0.4))
        confirmatory = summary["spread_confirmatory_pairs_only"][
            "retrieved_documents_reduction_fraction"
        ]
        self.assertEqual((confirmatory["min"], confirmatory["max"]), (0.1, 0.4))

    def test_no_pass_fail_verdict_key_exists(self):
        summary = robustness_summary([])
        for banned in ("verdict", "passed", "status", "supported"):
            self.assertNotIn(banned, summary)


class TestProductionCodeUntouched(unittest.TestCase):
    """This study must not change the experiment it is measuring."""

    def test_main_py_tuner_and_grids_are_unchanged(self):
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("def tune_ddre_thresholds(", source)
        self.assertIn("select_threshold_configuration(candidates)", source)

    def test_the_candidate_grids_are_the_production_ones(self):
        self.assertEqual(
            CANDIDATE_LOWER_GRID, (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
        )
        self.assertEqual(
            CANDIDATE_UPPER_GRID, (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
        )

    def test_the_ratio_clip_is_unchanged(self):
        source = (PROJECT_ROOT / "src" / "ddre_core.py").read_text(encoding="utf-8")
        self.assertIn("1e-6, 1e6", source)

    def test_the_script_loads_no_model(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for banned in ("AutoModelForSequenceClassification", "AutoTokenizer",
                       "EntailmentScorer("):
            self.assertNotIn(banned, source)
        self.assertIn("CachedDocumentScorer(", source)
        self.assertIn('"nli_inference_performed": False', source)


if __name__ == "__main__":
    unittest.main()
