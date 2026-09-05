"""Tests for the frozen confirmatory statistical protocol (D-07 and D-08).

Two properties matter most.

**D-07:** validation selection must protect nonfactual PR-AUC explicitly.
Balanced PR-AUC is the mean of the two class metrics, so a factual gain can
mask a nonfactual loss and still clear a balanced-only floor. That is the exact
bug, and it is tested directly.

**D-08:** the scientific claim must come from the pre-registered paired
passage-level bootstrap, never from test-set point estimates. `NOT_CONFIRMATORY`
and `NOT_SUPPORTED` are different statements and are never collapsed.

Synthetic data only -- no model, no GPU, no held-out inference.
"""

import ast
import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.evaluation import wang_pr_auc
from src.threshold_selection import (
    FALLBACK_SELECTION_RULE,
    FEASIBLE_SELECTION_RULE,
    SAFEGUARD_NOTE,
    candidate_record,
    select_threshold_configuration,
)
from src.paired_bootstrap import (
    CLAIM_NOT_CONFIRMATORY,
    CLAIM_NOT_SUPPORTED,
    CLAIM_SUPPORTED,
    CONFIRMATORY_BOOTSTRAP_RESAMPLES,
    CONFIRMATORY_BOOTSTRAP_SEED,
    CONFIRMATORY_BOOTSTRAP_UNIT,
    CONFIRMATORY_CI_LEVEL,
    CONFIRMATORY_PR_AUC_MARGIN,
    PRIMARY_EFFICIENCY_ENDPOINT,
    SECONDARY_EFFICIENCY_ENDPOINT,
    BootstrapUnavailable,
    PairedInputMismatch,
    assess_claim,
    confirmatory_protocol,
    paired_passage_bootstrap,
    passage_blocks,
    percentile_interval,
    replicate_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Record:
    def __init__(self, passage_index, sentence_index, label):
        self.passage_index = passage_index
        self.sentence_index = sentence_index
        self.label = label
        self.subclaims = []


class Result:
    def __init__(self, p_factual, documents_used=1.0, nli_calls=1.0):
        self.p_factual = float(p_factual)
        self.prediction = 1 if p_factual > 28.0 / 124.0 else 0
        self.documents_used = documents_used
        self.nli_calls = nli_calls


def synthetic_sample(n_passages=12, per_passage=4, seed=0):
    """Passages of several sentences each, both classes present."""
    rng = np.random.default_rng(seed)
    records, scores = [], []
    for passage in range(n_passages):
        for sentence in range(per_passage):
            label = int((passage + sentence) % 2)
            records.append(Record(passage, sentence, label))
            scores.append(
                float(np.clip(rng.normal(0.7 if label else 0.3, 0.15), 0.01, 0.99))
            )
    return records, scores


def results_from(scores, documents=1.0, nli=1.0):
    return [Result(s, documents, nli) for s in scores]


# --------------------------------------------------------------------------
# D-07: validation selection protects nonfactual PR-AUC
# --------------------------------------------------------------------------


def scored_candidates(table, baseline, quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN):
    """Build candidate records for a small synthetic grid.

    Uses the production helper, so the safeguards under test are the ones the
    experiment actually applies. No torch, no model, no validation data.
    """
    return [
        candidate_record(
            lower, upper,
            {
                "nonfactual": {"auc_pr": nonfactual},
                "factual": {"auc_pr": factual},
                "balanced_pr_auc": 0.5 * (nonfactual + factual),
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "efficiency": {
                    "avg_retrieved_documents_per_sentence": documents,
                    "avg_nli_span_calls_per_sentence": documents * 3.0,
                },
            },
            baseline,
            quality_tolerance=quality_tolerance,
            retrieval_penalty=0.05,
            max_docs=10,
        )
        for (lower, upper), (nonfactual, factual, documents) in sorted(table.items())
    ]


BASELINE = {
    "nonfactual": {"auc_pr": 0.80},
    "factual": {"auc_pr": 0.60},
    "balanced_pr_auc": 0.70,
}


class TestValidationProtectsNonfactual(unittest.TestCase):
    def run_selection(self, table, quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN):
        candidates = scored_candidates(table, BASELINE, quality_tolerance)
        selected, rule, confirmatory = select_threshold_configuration(candidates)
        return selected, candidates, rule, confirmatory

    def candidate(self, candidates, lower, upper):
        for entry in candidates:
            if entry["lower"] == lower and entry["upper"] == upper:
                return entry
        raise AssertionError(f"({lower}, {upper}) was not scored")

    def test_a_factual_gain_cannot_mask_a_nonfactual_loss(self):
        # THE D-07 BUG, directly. factual +0.02, nonfactual -0.02: balanced is
        # unchanged, so a balanced-and-factual-only floor would accept it.
        table = {(0.05, 0.60): (0.78, 0.62, 1.0)}
        selected, candidates, rule, confirmatory = self.run_selection(table)
        entry = self.candidate(candidates, 0.05, 0.60)
        self.assertAlmostEqual(entry["nonfactual_auc_pr_delta_vs_bse"], -0.02)
        self.assertAlmostEqual(entry["factual_auc_pr_delta_vs_bse"], +0.02)
        self.assertAlmostEqual(entry["balanced_pr_auc_delta_vs_bse"], 0.0, places=12)
        self.assertTrue(entry["preserves_factual"])
        self.assertTrue(entry["preserves_balanced"])
        self.assertFalse(entry["preserves_nonfactual"])
        self.assertFalse(entry["preserves_baseline_quality"])

    def test_a_nonfactual_loss_of_exactly_the_tolerance_is_permitted(self):
        table = {(0.05, 0.60): (0.80 - CONFIRMATORY_PR_AUC_MARGIN, 0.60, 1.0)}
        _, candidates, _, _ = self.run_selection(table)
        entry = self.candidate(candidates, 0.05, 0.60)
        self.assertTrue(entry["preserves_nonfactual"])

    def test_a_nonfactual_loss_beyond_the_tolerance_is_rejected(self):
        worse = 0.80 - CONFIRMATORY_PR_AUC_MARGIN
        table = {(0.05, 0.60): (math.nextafter(worse, 0.0), 0.60, 1.0)}
        _, candidates, _, _ = self.run_selection(table)
        self.assertFalse(self.candidate(candidates, 0.05, 0.60)["preserves_nonfactual"])

    def test_the_factual_and_balanced_safeguards_still_work(self):
        table = {
            (0.05, 0.60): (0.90, 0.50, 1.0),   # factual -0.10, balanced +0.00
            (0.10, 0.60): (0.70, 0.50, 1.0),   # both down
        }
        _, candidates, _, _ = self.run_selection(table)
        first = self.candidate(candidates, 0.05, 0.60)
        self.assertTrue(first["preserves_nonfactual"])
        self.assertFalse(first["preserves_factual"])
        self.assertFalse(first["preserves_baseline_quality"])
        second = self.candidate(candidates, 0.10, 0.60)
        self.assertFalse(second["preserves_balanced"])

    def test_the_balanced_safeguard_is_implied_but_still_recorded(self):
        # Honest accounting: given the same tolerance t, the balanced floor is
        # implied by the other two -- if n >= bn - t and f >= bf - t then
        # (n+f)/2 >= (bn+bf)/2 - t. So removing the balanced clause from the
        # conjunction cannot change any feasibility verdict, and no behavioural
        # test can detect it. It is kept for EXPLICIT AUDITABILITY, and both
        # facts are pinned here: the implication, and the recording.
        rng = np.random.default_rng(0)
        tolerance = CONFIRMATORY_PR_AUC_MARGIN
        for _ in range(2000):
            bn, bf = rng.uniform(0, 1, 2)
            n = bn - tolerance + rng.uniform(0, 0.5)
            f = bf - tolerance + rng.uniform(0, 0.5)
            baseline = {
                "nonfactual": {"auc_pr": float(bn)},
                "factual": {"auc_pr": float(bf)},
                "balanced_pr_auc": float(0.5 * (bn + bf)),
            }
            record = scored_candidates(
                {(0.05, 0.60): (float(n), float(f), 1.0)}, baseline, tolerance
            )[0]
            self.assertTrue(record["preserves_nonfactual"])
            self.assertTrue(record["preserves_factual"])
            self.assertTrue(
                record["preserves_balanced"],
                "balanced must be implied by the two class floors",
            )

        # ...and it is still evaluated and recorded by name, not inferred.
        source = (
            PROJECT_ROOT / "src" / "threshold_selection.py"
        ).read_text(encoding="utf-8")
        conjunction = source[
            source.index('"preserves_baseline_quality"') : source.index(
                '"fallback_objective"'
            )
        ]
        for safeguard in (
            "preserves_nonfactual", "preserves_factual", "preserves_balanced"
        ):
            self.assertIn(safeguard, conjunction)

    def test_feasible_candidates_still_prioritise_fewest_documents(self):
        table = {
            (0.05, 0.60): (0.85, 0.65, 5.0),   # better quality, more documents
            (0.10, 0.60): (0.80, 0.60, 2.0),   # exactly baseline, fewest
        }
        selected, _, rule, confirmatory = self.run_selection(table)
        self.assertEqual((selected["lower"], selected["upper"]), (0.10, 0.60))
        self.assertEqual(selected["avg_documents"], 2.0)
        self.assertIn("minimum retrieval cost", rule)
        self.assertIn("nonfactual", rule)
        self.assertEqual(rule, FEASIBLE_SELECTION_RULE)
        self.assertTrue(confirmatory)

    def test_exact_document_ties_break_on_quality_deterministically(self):
        table = {
            (0.05, 0.60): (0.80, 0.60, 3.0),
            (0.10, 0.60): (0.86, 0.62, 3.0),   # same cost, better quality
            (0.15, 0.60): (0.82, 0.61, 3.0),
        }
        selected, _, _, _ = self.run_selection(table)
        self.assertEqual((selected["lower"], selected["upper"]), (0.10, 0.60))

    def test_a_fallback_selection_is_marked_non_confirmatory(self):
        # Every candidate is far below the baseline, so nothing is feasible.
        table = {(0.05, 0.60): (0.50, 0.50, 1.0), (0.10, 0.60): (0.52, 0.51, 2.0)}
        selected, candidates, rule, confirmatory = self.run_selection(table)
        self.assertFalse(any(c["preserves_baseline_quality"] for c in candidates))
        self.assertFalse(confirmatory)
        self.assertEqual(rule, FALLBACK_SELECTION_RULE)
        self.assertIn("NOT ELIGIBLE", rule)

    def test_the_safeguards_are_recorded_on_every_candidate(self):
        _, candidates, _, _ = self.run_selection({(0.05, 0.60): (0.5, 0.5, 9.0)})
        self.assertGreater(len(candidates), 0)
        for entry in candidates:
            for key in (
                "preserves_nonfactual", "preserves_factual", "preserves_balanced",
                "preserves_baseline_quality",
                "nonfactual_auc_pr_delta_vs_bse",
                "factual_auc_pr_delta_vs_bse",
                "balanced_pr_auc_delta_vs_bse",
            ):
                self.assertIn(key, entry)
        self.assertIn("nonfactual", SAFEGUARD_NOTE)


# --------------------------------------------------------------------------
# Frozen protocol constants
# --------------------------------------------------------------------------


class TestFrozenProtocol(unittest.TestCase):
    def test_the_constants_are_the_pre_registered_ones(self):
        self.assertEqual(CONFIRMATORY_PR_AUC_MARGIN, 0.005)
        self.assertEqual(CONFIRMATORY_BOOTSTRAP_RESAMPLES, 10_000)
        self.assertEqual(CONFIRMATORY_BOOTSTRAP_SEED, 42)
        self.assertEqual(CONFIRMATORY_CI_LEVEL, 0.95)
        self.assertEqual(CONFIRMATORY_BOOTSTRAP_UNIT, "passage")

    def test_the_protocol_record_is_json_serializable_and_complete(self):
        protocol = confirmatory_protocol()
        json.dumps(protocol)
        for key in (
            "bootstrap_unit", "n_resamples", "seed", "ci_method", "ci_level",
            "pr_auc_noninferiority_margin", "sign_conventions",
            "primary_performance_endpoints", "primary_efficiency_endpoint",
            "secondary_efficiency_endpoint", "primary_claim_rule",
            "multiplicity", "p_values", "invalid_replicate_policy",
            "validation_fallback_policy",
        ):
            self.assertIn(key, protocol)
        self.assertTrue(protocol["frozen_before_held_out_evaluation"])
        self.assertEqual(protocol["ci_percentiles"], [2.5, 97.5])
        self.assertEqual(protocol["p_values"], "none produced")

    def test_the_claim_rule_is_documented_as_conjunctive(self):
        protocol = confirmatory_protocol()
        self.assertIn("CONJUNCTIVE", protocol["primary_claim_rule"])
        self.assertIn("EVERY primary gate", protocol["multiplicity"])

    def test_the_sign_conventions_are_stated(self):
        conventions = confirmatory_protocol()["sign_conventions"]
        self.assertEqual(conventions["performance"], "DDRE - BSE; positive favours DDRE")
        self.assertEqual(conventions["efficiency"], "BSE - DDRE; positive favours DDRE")


# --------------------------------------------------------------------------
# Bootstrap mechanics
# --------------------------------------------------------------------------


class TestBootstrapMechanics(unittest.TestCase):
    def setUp(self):
        self.records, self.scores = synthetic_sample()
        self.results = results_from(self.scores)

    def bootstrap(self, ddre=None, baseline=None, **kwargs):
        kwargs.setdefault("n_resamples", 200)
        return paired_passage_bootstrap(
            self.records,
            self.results if ddre is None else ddre,
            self.results if baseline is None else baseline,
            **kwargs,
        )

    def test_it_is_deterministic_for_a_fixed_seed(self):
        first = self.bootstrap(seed=42)
        second = self.bootstrap(seed=42)
        self.assertEqual(first["endpoints"], second["endpoints"])

    def test_a_different_seed_gives_a_different_draw(self):
        blocks = passage_blocks(self.records)
        a, _ = replicate_indices(blocks, np.random.default_rng(42))
        b, _ = replicate_indices(blocks, np.random.default_rng(43))
        self.assertFalse(np.array_equal(a, b))

    def test_it_resamples_passage_blocks_not_individual_sentences(self):
        blocks = passage_blocks(self.records)
        self.assertEqual(len(blocks), 12)
        for _, members in blocks:
            self.assertEqual(len(members), 4)
        indices, draw = replicate_indices(blocks, np.random.default_rng(42))
        # Every sampled index belongs to a WHOLE block that was drawn.
        self.assertEqual(len(indices), len(self.records))
        rebuilt = []
        for position in draw:
            rebuilt.extend(blocks[int(position)][1])
        self.assertEqual(list(indices), rebuilt)

    def test_a_passage_drawn_twice_contributes_its_block_twice(self):
        blocks = passage_blocks(self.records)

        class TwiceRng:
            def integers(self, low, high, size):
                # Draw passage 0 twice, then distinct passages.
                return np.array([0, 0] + list(range(1, size - 1)))

        indices, draw = replicate_indices(blocks, TwiceRng())
        block_zero = list(blocks[0][1])
        self.assertEqual(list(indices[: 2 * len(block_zero)]), block_zero + block_zero)
        self.assertEqual(len(indices), len(self.records))

    def test_both_methods_receive_the_same_passage_draw(self):
        # If the draws differed, an identical pair of methods could show a
        # non-zero delta. That they never do is the evidence.
        report = self.bootstrap(n_resamples=300, seed=42)
        for name in report["endpoints"]:
            with self.subTest(endpoint=name):
                self.assertEqual(report["endpoints"][name]["ci_lower"], 0.0)
                self.assertEqual(report["endpoints"][name]["ci_upper"], 0.0)

    def test_the_percentile_interval_uses_exactly_2_5_and_97_5(self):
        values = np.arange(10_000, dtype=float)
        lower, upper, lower_pct, upper_pct = percentile_interval(values, 0.95)
        self.assertEqual((lower_pct, upper_pct), (2.5, 97.5))
        self.assertAlmostEqual(lower, float(np.percentile(values, 2.5)))
        self.assertAlmostEqual(upper, float(np.percentile(values, 97.5)))

    def test_every_endpoint_records_its_provenance(self):
        report = self.bootstrap(seed=42)
        for name, endpoint in report["endpoints"].items():
            for key in (
                "observed", "bootstrap_mean", "ci_lower", "ci_upper", "ci_level",
                "ci_method", "n_resamples", "seed", "bootstrap_unit",
                "sign_convention",
            ):
                self.assertIn(key, endpoint, name)
            self.assertEqual(endpoint["bootstrap_unit"], "passage")
        self.assertEqual(report["unique_passages"], 12)

    def test_mismatched_inputs_are_rejected(self):
        with self.assertRaises(PairedInputMismatch):
            paired_passage_bootstrap(self.records, self.results[:-1], self.results)
        with self.assertRaises(PairedInputMismatch):
            paired_passage_bootstrap([], [], [])

    def test_a_single_class_sample_is_rejected_up_front(self):
        records = [Record(0, i, 1) for i in range(4)]
        results = results_from([0.5] * 4)
        with self.assertRaises(PairedInputMismatch):
            paired_passage_bootstrap(records, results, results)

    def test_an_invalid_replicate_fails_loudly_rather_than_being_dropped(self):
        # One passage holds every nonfactual sentence, so a draw that misses it
        # leaves a one-class replicate and PR-AUC is undefined.
        records = (
            [Record(0, i, 0) for i in range(3)]
            + [Record(p, i, 1) for p in range(1, 6) for i in range(3)]
        )
        results = results_from([0.5] * len(records))
        with self.assertRaises(BootstrapUnavailable) as caught:
            paired_passage_bootstrap(records, results, results, n_resamples=500, seed=1)
        message = str(caught.exception)
        self.assertIn("NOT dropped", message)
        self.assertIn("NOT redrawn", message)
        self.assertIn("NO substitute metric", message)


class TestSignConventionsAndValues(unittest.TestCase):
    def setUp(self):
        self.records, self.scores = synthetic_sample()

    def test_performance_delta_is_ddre_minus_bse(self):
        # DDRE scores separate the classes; BSE is uninformative. The delta must
        # be POSITIVE, i.e. DDRE - BSE.
        good = results_from(self.scores)
        flat = results_from([0.5] * len(self.records))
        report = paired_passage_bootstrap(
            self.records, good, flat, n_resamples=200, seed=42
        )
        self.assertGreater(report["endpoints"]["factual_auc_pr_delta"]["observed"], 0.0)
        reversed_report = paired_passage_bootstrap(
            self.records, flat, good, n_resamples=200, seed=42
        )
        self.assertLess(
            reversed_report["endpoints"]["factual_auc_pr_delta"]["observed"], 0.0
        )

    def test_efficiency_saving_is_bse_minus_ddre(self):
        cheap = results_from(self.scores, documents=2.0, nli=6.0)
        costly = results_from(self.scores, documents=3.0, nli=12.0)
        report = paired_passage_bootstrap(
            self.records, cheap, costly, n_resamples=200, seed=42
        )
        documents = report["endpoints"][PRIMARY_EFFICIENCY_ENDPOINT]
        self.assertAlmostEqual(documents["observed"], 1.0)
        self.assertAlmostEqual(documents["ci_lower"], 1.0)
        self.assertAlmostEqual(documents["ci_upper"], 1.0)
        self.assertIn("BSE - DDRE", documents["sign_convention"])
        self.assertAlmostEqual(
            report["endpoints"][SECONDARY_EFFICIENCY_ENDPOINT]["observed"], 6.0
        )

    def test_the_bootstrap_uses_the_repository_wang_pr_auc(self):
        labels = np.asarray([r.label for r in self.records])
        p_factual = np.asarray(self.scores)
        flat = results_from([0.5] * len(self.records))
        report = paired_passage_bootstrap(
            self.records, results_from(self.scores), flat, n_resamples=10, seed=42
        )
        expected_factual = wang_pr_auc(labels, p_factual) - wang_pr_auc(
            labels, np.full_like(p_factual, 0.5)
        )
        self.assertAlmostEqual(
            report["endpoints"]["factual_auc_pr_delta"]["observed"], expected_factual
        )
        expected_nonfactual = wang_pr_auc(1 - labels, 1.0 - p_factual) - wang_pr_auc(
            1 - labels, np.full_like(p_factual, 0.5)
        )
        self.assertAlmostEqual(
            report["endpoints"]["nonfactual_auc_pr_delta"]["observed"],
            expected_nonfactual,
        )
        self.assertIn("wang_pr_auc", report["pr_auc_definition"])


# --------------------------------------------------------------------------
# D-08: the claim assessment
# --------------------------------------------------------------------------


def bootstrap_with(**lower_bounds):
    """A minimal bootstrap-shaped record with the given CI lower bounds."""
    defaults = {
        "nonfactual_auc_pr_delta": 0.0,
        "factual_auc_pr_delta": 0.0,
        "balanced_pr_auc_delta": 0.0,
        PRIMARY_EFFICIENCY_ENDPOINT: 1.0,
        SECONDARY_EFFICIENCY_ENDPOINT: 1.0,
    }
    defaults.update(lower_bounds)
    return {
        "endpoints": {
            name: {"ci_lower": value, "ci_upper": value + 0.1, "observed": value}
            for name, value in defaults.items()
        }
    }


def claim(**overrides):
    kwargs = {
        "validation_selection_confirmatory": True,
        "quality_tolerance": CONFIRMATORY_PR_AUC_MARGIN,
        "smoke_test": False,
    }
    bootstrap = overrides.pop("bootstrap", bootstrap_with())
    kwargs.update(overrides)
    return assess_claim(bootstrap, **kwargs)


class TestClaimAssessment(unittest.TestCase):
    def test_a_clean_run_with_all_gates_passing_is_supported(self):
        assessment = claim()
        self.assertTrue(assessment["confirmatory_eligible"])
        self.assertTrue(assessment["performance_noninferiority"]["all_pass"])
        self.assertTrue(assessment["retrieval_efficiency_superiority_pass"])
        self.assertTrue(assessment["primary_claim_supported"])
        self.assertEqual(assessment["claim_status"], CLAIM_SUPPORTED)

    def test_zero_retrieval_saving_is_not_superiority(self):
        assessment = claim(bootstrap=bootstrap_with(**{PRIMARY_EFFICIENCY_ENDPOINT: 0.0}))
        self.assertFalse(assessment["retrieval_efficiency_superiority_pass"])
        self.assertFalse(assessment["primary_claim_supported"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def test_a_nonfactual_loss_beyond_the_margin_blocks_the_claim(self):
        beyond = math.nextafter(-CONFIRMATORY_PR_AUC_MARGIN, -1.0)
        assessment = claim(bootstrap=bootstrap_with(nonfactual_auc_pr_delta=beyond))
        self.assertFalse(assessment["performance_noninferiority"]["nonfactual_pass"])
        self.assertFalse(assessment["primary_claim_supported"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def test_exactly_the_margin_is_non_inferior(self):
        assessment = claim(
            bootstrap=bootstrap_with(nonfactual_auc_pr_delta=-CONFIRMATORY_PR_AUC_MARGIN)
        )
        self.assertTrue(assessment["performance_noninferiority"]["nonfactual_pass"])

    def test_a_factual_gain_cannot_compensate_for_a_nonfactual_failure(self):
        # The conjunctive rule in one test: a large factual improvement and a
        # large retrieval saving do not rescue a failed nonfactual endpoint.
        assessment = claim(
            bootstrap=bootstrap_with(
                nonfactual_auc_pr_delta=-0.05,
                factual_auc_pr_delta=+0.20,
                balanced_pr_auc_delta=+0.05,
                **{PRIMARY_EFFICIENCY_ENDPOINT: 5.0},
            )
        )
        self.assertTrue(assessment["performance_noninferiority"]["factual_pass"])
        self.assertTrue(assessment["retrieval_efficiency_superiority_pass"])
        self.assertFalse(assessment["performance_noninferiority"]["nonfactual_pass"])
        self.assertFalse(assessment["primary_claim_supported"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def test_a_validation_fallback_is_never_confirmatory(self):
        # Even with spectacular held-out effects.
        assessment = claim(
            validation_selection_confirmatory=False,
            bootstrap=bootstrap_with(
                nonfactual_auc_pr_delta=+0.30,
                factual_auc_pr_delta=+0.30,
                balanced_pr_auc_delta=+0.30,
                **{PRIMARY_EFFICIENCY_ENDPOINT: 8.0},
            ),
        )
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertFalse(assessment["primary_claim_supported"])
        self.assertIn("fallback", " ".join(assessment["confirmatory_disqualifiers"]))

    def test_a_different_quality_tolerance_is_not_confirmatory(self):
        assessment = claim(quality_tolerance=0.02)
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertFalse(assessment["validation_tolerance_matches_frozen_margin"])

    def test_a_smoke_run_is_not_confirmatory(self):
        assessment = claim(smoke_test=True)
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertIn("smoke", " ".join(assessment["confirmatory_disqualifiers"]))

    def test_an_unavailable_bootstrap_is_not_confirmatory(self):
        assessment = claim(bootstrap=None, bootstrap_error="BootstrapUnavailable: x")
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertFalse(assessment["bootstrap_available"])

    def test_not_confirmatory_is_never_reported_as_a_negative_result(self):
        assessment = claim(validation_selection_confirmatory=False)
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertNotEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)
        self.assertIn("NOT a negative result", assessment["interpretation"])

    def test_the_nli_endpoint_is_secondary(self):
        supported = claim(bootstrap=bootstrap_with(**{SECONDARY_EFFICIENCY_ENDPOINT: 0.0}))
        self.assertFalse(supported["nli_efficiency_superiority_pass"])
        self.assertTrue(supported["primary_claim_supported"])
        self.assertEqual(supported["claim_status"], CLAIM_SUPPORTED)
        self.assertIn("NOT a statistically supported", supported["interpretation"])

        both = claim()
        self.assertTrue(both["nli_efficiency_superiority_pass"])
        self.assertIn("AND lower NLI-evaluation cost", both["interpretation"])

    def test_every_gate_is_recorded_separately(self):
        assessment = claim()
        json.dumps(assessment)
        for key in (
            "confirmatory_eligible", "validation_selection_confirmatory",
            "performance_noninferiority", "retrieval_efficiency_superiority_pass",
            "nli_efficiency_superiority_pass", "primary_claim_supported",
            "claim_status", "claim_rule",
        ):
            self.assertIn(key, assessment)
        for key in ("nonfactual_pass", "factual_pass", "balanced_pass", "all_pass"):
            self.assertIn(key, assessment["performance_noninferiority"])


# --------------------------------------------------------------------------
# End-to-end: bootstrap feeding the claim
# --------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.records, self.scores = synthetic_sample()

    def test_identical_methods_give_zero_deltas_and_no_supported_claim(self):
        results = results_from(self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(
            self.records, results, results, n_resamples=200, seed=42
        )
        for name in report["endpoints"]:
            self.assertEqual(report["endpoints"][name]["observed"], 0.0)
            self.assertEqual(report["endpoints"][name]["ci_lower"], 0.0)
        assessment = assess_claim(
            report, validation_selection_confirmatory=True,
            quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN,
        )
        self.assertTrue(assessment["performance_noninferiority"]["all_pass"])
        self.assertFalse(assessment["retrieval_efficiency_superiority_pass"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def test_identical_quality_with_one_fewer_document_supports_the_claim(self):
        ddre = results_from(self.scores, documents=2.0, nli=6.0)
        baseline = results_from(self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(
            self.records, ddre, baseline, n_resamples=200, seed=42
        )
        for name in (
            "nonfactual_auc_pr_delta", "factual_auc_pr_delta", "balanced_pr_auc_delta"
        ):
            self.assertEqual(report["endpoints"][name]["ci_lower"], 0.0)
            self.assertEqual(report["endpoints"][name]["ci_upper"], 0.0)
        documents = report["endpoints"][PRIMARY_EFFICIENCY_ENDPOINT]
        self.assertEqual((documents["ci_lower"], documents["ci_upper"]), (1.0, 1.0))
        assessment = assess_claim(
            report, validation_selection_confirmatory=True,
            quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN,
        )
        self.assertEqual(assessment["claim_status"], CLAIM_SUPPORTED)

    def test_fewer_documents_but_worse_nonfactual_does_not_support_the_claim(self):
        degraded = [
            Result(0.5 if r.label == 0 else s, 2.0, 6.0)
            for r, s in zip(self.records, self.scores)
        ]
        baseline = results_from(self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(
            self.records, degraded, baseline, n_resamples=200, seed=42
        )
        self.assertLess(
            report["endpoints"]["nonfactual_auc_pr_delta"]["ci_lower"],
            -CONFIRMATORY_PR_AUC_MARGIN,
        )
        assessment = assess_claim(
            report, validation_selection_confirmatory=True,
            quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN,
        )
        self.assertTrue(assessment["retrieval_efficiency_superiority_pass"])
        self.assertFalse(assessment["performance_noninferiority"]["nonfactual_pass"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def test_the_function_defaults_are_the_frozen_constants(self):
        # The pre-registration only binds the analysis if the defaults ARE it.
        import inspect

        defaults = inspect.signature(paired_passage_bootstrap).parameters
        self.assertEqual(
            defaults["n_resamples"].default, CONFIRMATORY_BOOTSTRAP_RESAMPLES
        )
        self.assertEqual(defaults["seed"].default, CONFIRMATORY_BOOTSTRAP_SEED)
        self.assertEqual(defaults["ci_level"].default, CONFIRMATORY_CI_LEVEL)

    def test_the_full_frozen_settings_run_end_to_end(self):
        # A genuine 10,000-resample run at seed 42, exercising the whole
        # resampling loop. The Wang PR-AUC is replaced by a cheap stub ONLY to
        # keep the suite fast -- its identity with the normal evaluation path is
        # covered by test_the_bootstrap_uses_the_repository_wang_pr_auc, which
        # uses the real helper.
        def cheap_pr_auc(y_binary, score):
            return float(np.mean(np.asarray(score, dtype=float)))

        ddre = results_from(self.scores, documents=2.0, nli=6.0)
        baseline = results_from(self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(
            self.records, ddre, baseline, pr_auc=cheap_pr_auc
        )
        self.assertEqual(report["n_resamples"], CONFIRMATORY_BOOTSTRAP_RESAMPLES)
        self.assertEqual(report["seed"], CONFIRMATORY_BOOTSTRAP_SEED)
        self.assertEqual(report["ci_level"], CONFIRMATORY_CI_LEVEL)
        self.assertEqual(report["ci_method"], "percentile")
        documents = report["endpoints"][PRIMARY_EFFICIENCY_ENDPOINT]
        self.assertEqual((documents["ci_lower"], documents["ci_upper"]), (1.0, 1.0))
        self.assertEqual(
            report["endpoints"]["factual_auc_pr_delta"]["ci_lower"],
            report["endpoints"]["factual_auc_pr_delta"]["ci_upper"],
        )
        json.dumps(report)


class TestSummaryWiring(unittest.TestCase):
    """Asserted on main.py's AST; running it needs torch and the model."""

    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_the_summary_carries_the_protocol_bootstrap_and_claim(self):
        for field in (
            '"confirmatory_statistical_protocol": confirmatory_protocol()',
            '"confirmatory_bootstrap": bootstrap',
            '"claim_assessment": claim_assessment',
            '"descriptive_point_estimates": comparison',
        ):
            self.assertIn(field, self.source)

    def test_the_bootstrap_is_computed_from_the_paired_test_results(self):
        self.assertIn(
            "paired_passage_bootstrap(\n                test_records, ddre_results, "
            "bse_official_results\n            )",
            self.source,
        )

    def test_the_claim_reads_the_validation_selection_and_tolerance(self):
        self.assertIn("validation_selection_confirmatory=selected[", self.source)
        self.assertIn("quality_tolerance=args.quality_tolerance", self.source)
        self.assertIn("smoke_test=args.smoke_test", self.source)

    def test_bootstrap_failure_is_caught_and_reported_not_swallowed(self):
        self.assertIn("except (BootstrapUnavailable, PairedInputMismatch)", self.source)
        self.assertIn("bootstrap_error=bootstrap_error", self.source)

    def test_the_old_point_estimate_boolean_is_gone(self):
        self.assertNotIn("hypothesis_supported_on_test", self.source)


if __name__ == "__main__":
    unittest.main()
