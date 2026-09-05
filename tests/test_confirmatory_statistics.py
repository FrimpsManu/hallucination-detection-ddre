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
import contextlib
import io
import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.evaluation import EvaluatedSentence, wang_pr_auc
from src.threshold_selection import (
    FALLBACK_SELECTION_RULE,
    FEASIBLE_SELECTION_RULE,
    SAFEGUARD_NOTE,
    candidate_record,
    select_threshold_configuration,
)
from src.paired_bootstrap import (
    CONFIRMATORY_CI_METHOD,
    CONFIRMATORY_C_FALSE_ALARM,
    CONFIRMATORY_C_MISS,
    CONFIRMATORY_C_RETRIEVE,
    CONFIRMATORY_MAX_DOCS,
    CONFIRMATORY_P0,
    CONFIRMATORY_SPLIT_SEED,
    CONFIRMATORY_VALIDATION_FRACTION,
    EFFICIENCY_SIGN_CONVENTION,
    FROZEN_RUN_CONFIGURATION,
    PERFORMANCE_ENDPOINTS,
    PERFORMANCE_SIGN_CONVENTION,
    BOOTSTRAP_ENDPOINTS,
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
    split_identity,
    validate_bootstrap_provenance,
    validate_paired_inputs,
    validate_run_configuration,
    validate_split_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Result:
    def __init__(self, p_factual, documents_used=1.0, nli_calls=1.0):
        self.p_factual = float(p_factual)
        self.prediction = 1 if p_factual > 28.0 / 124.0 else 0
        self.documents_used = documents_used
        self.nli_calls = nli_calls


class Record:
    def __init__(self, passage_index, sentence_index, label):
        self.passage_index = passage_index
        self.sentence_index = sentence_index
        self.label = label
        self.subclaims = []


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


def observations_from(records, scores, documents=1.0, nli=1.0):
    """Identity-bearing observations, as evaluate_detector_with_identity builds."""
    return [
        EvaluatedSentence(
            passage_index=record.passage_index,
            sentence_index=record.sentence_index,
            gold_label=int(record.label),
            result=Result(score, documents, nli),
        )
        for record, score in zip(records, scores)
    ]


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
        self.assertEqual(CONFIRMATORY_CI_METHOD, "percentile")

    def test_the_frozen_run_configuration_is_the_published_comparator(self):
        # Wang's published cost pair and this repository's primary settings.
        # If these drift, the protocol document describes a different run from
        # the one the gate enforces.
        self.assertEqual(CONFIRMATORY_C_MISS, 28.0)
        self.assertEqual(CONFIRMATORY_C_FALSE_ALARM, 96.0)
        self.assertEqual(CONFIRMATORY_C_RETRIEVE, 1.0)
        self.assertEqual(CONFIRMATORY_P0, 0.5)
        self.assertEqual(CONFIRMATORY_MAX_DOCS, 10)
        self.assertEqual(CONFIRMATORY_VALIDATION_FRACTION, 0.20)
        self.assertEqual(CONFIRMATORY_SPLIT_SEED, 42)
        self.assertEqual(
            FROZEN_RUN_CONFIGURATION,
            {
                "c_miss": 28.0,
                "c_false_alarm": 96.0,
                "c_retrieve": 1.0,
                "p0": 0.5,
                "max_docs": 10,
                "validation_fraction": 0.20,
                "split_seed": 42,
            },
        )
        protocol = confirmatory_protocol()
        self.assertEqual(
            protocol["frozen_run_configuration"], dict(FROZEN_RUN_CONFIGURATION)
        )
        self.assertIn("CM=28, CFA=96", protocol["primary_comparator"])
        self.assertIn("frozen_split_requirement", protocol)
        self.assertIn("bootstrap_provenance_requirement", protocol)

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
        self.observations = observations_from(self.records, self.scores)

    def bootstrap(self, ddre=None, baseline=None, **kwargs):
        kwargs.setdefault("n_resamples", 200)
        return paired_passage_bootstrap(
            self.observations if ddre is None else ddre,
            self.observations if baseline is None else baseline,
            **kwargs,
        )

    def test_it_is_deterministic_for_a_fixed_seed(self):
        first = self.bootstrap(seed=42)
        second = self.bootstrap(seed=42)
        self.assertEqual(first["endpoints"], second["endpoints"])

    def test_a_different_seed_gives_a_different_draw(self):
        blocks = passage_blocks(self.observations)
        a, _ = replicate_indices(blocks, np.random.default_rng(42))
        b, _ = replicate_indices(blocks, np.random.default_rng(43))
        self.assertFalse(np.array_equal(a, b))

    def test_it_resamples_passage_blocks_not_individual_sentences(self):
        blocks = passage_blocks(self.observations)
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
        blocks = passage_blocks(self.observations)

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
            paired_passage_bootstrap(self.observations[:-1], self.observations)
        with self.assertRaises(PairedInputMismatch):
            paired_passage_bootstrap([], [])

    def test_a_single_class_sample_is_rejected_up_front(self):
        records = [Record(0, i, 1) for i in range(4)]
        observations = observations_from(records, [0.5] * 4)
        with self.assertRaises(PairedInputMismatch):
            paired_passage_bootstrap(observations, observations)

    def test_an_invalid_replicate_fails_loudly_rather_than_being_dropped(self):
        # One passage holds every nonfactual sentence, so a draw that misses it
        # leaves a one-class replicate and PR-AUC is undefined.
        records = (
            [Record(0, i, 0) for i in range(3)]
            + [Record(p, i, 1) for p in range(1, 6) for i in range(3)]
        )
        observations = observations_from(records, [0.5] * len(records))
        with self.assertRaises(BootstrapUnavailable) as caught:
            paired_passage_bootstrap(
                observations, observations, n_resamples=500, seed=1
            )
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
        good = observations_from(self.records, self.scores)
        flat = observations_from(self.records, [0.5] * len(self.records))
        report = paired_passage_bootstrap(good, flat, n_resamples=200, seed=42)
        self.assertGreater(report["endpoints"]["factual_auc_pr_delta"]["observed"], 0.0)
        reversed_report = paired_passage_bootstrap(flat, good, n_resamples=200, seed=42)
        self.assertLess(
            reversed_report["endpoints"]["factual_auc_pr_delta"]["observed"], 0.0
        )

    def test_efficiency_saving_is_bse_minus_ddre(self):
        cheap = observations_from(self.records, self.scores, documents=2.0, nli=6.0)
        costly = observations_from(self.records, self.scores, documents=3.0, nli=12.0)
        report = paired_passage_bootstrap(cheap, costly, n_resamples=200, seed=42)
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
        flat = observations_from(self.records, [0.5] * len(self.records))
        report = paired_passage_bootstrap(
            observations_from(self.records, self.scores), flat, n_resamples=10, seed=42
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


# The claim-gate tests are about the GATES, not about resampling, so they use a
# synthetic bootstrap record. That record must nevertheless satisfy the frozen
# provenance in full: if it did not, every one of these tests would be answered
# by the provenance gate and the claim logic underneath would go untested.

FROZEN_VALIDATION_PASSAGE_IDS = [3, 7, 11, 19]
FROZEN_TEST_PASSAGE_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20]


def endpoint_record(name, value, **overrides):
    record = {
        "observed": value,
        "bootstrap_mean": value,
        "ci_lower": value,
        "ci_upper": value + 0.1,
        "ci_level": CONFIRMATORY_CI_LEVEL,
        "ci_method": CONFIRMATORY_CI_METHOD,
        "ci_lower_percentile": 2.5,
        "ci_upper_percentile": 97.5,
        "n_resamples": CONFIRMATORY_BOOTSTRAP_RESAMPLES,
        "seed": CONFIRMATORY_BOOTSTRAP_SEED,
        "bootstrap_unit": CONFIRMATORY_BOOTSTRAP_UNIT,
        "sign_convention": (
            PERFORMANCE_SIGN_CONVENTION
            if name in PERFORMANCE_ENDPOINTS
            else EFFICIENCY_SIGN_CONVENTION
        ),
    }
    record.update(overrides)
    return record


def bootstrap_with(**lower_bounds):
    """A bootstrap-shaped record carrying the EXACT frozen provenance."""
    defaults = {
        "nonfactual_auc_pr_delta": 0.0,
        "factual_auc_pr_delta": 0.0,
        "balanced_pr_auc_delta": 0.0,
        PRIMARY_EFFICIENCY_ENDPOINT: 1.0,
        SECONDARY_EFFICIENCY_ENDPOINT: 1.0,
    }
    defaults.update(lower_bounds)
    return {
        "bootstrap_unit": CONFIRMATORY_BOOTSTRAP_UNIT,
        "n_resamples": CONFIRMATORY_BOOTSTRAP_RESAMPLES,
        "seed": CONFIRMATORY_BOOTSTRAP_SEED,
        "ci_level": CONFIRMATORY_CI_LEVEL,
        "ci_method": CONFIRMATORY_CI_METHOD,
        "endpoints": {
            name: endpoint_record(name, value) for name, value in defaults.items()
        },
    }


def frozen_split_metadata(**overrides):
    metadata = {
        "random_state": CONFIRMATORY_SPLIT_SEED,
        "validation_fraction": CONFIRMATORY_VALIDATION_FRACTION,
        "validation_passage_ids": list(FROZEN_VALIDATION_PASSAGE_IDS),
        "test_passage_ids": list(FROZEN_TEST_PASSAGE_IDS),
    }
    metadata.update(overrides)
    return metadata


def claim(**overrides):
    kwargs = {
        "validation_selection_confirmatory": True,
        "quality_tolerance": CONFIRMATORY_PR_AUC_MARGIN,
        "smoke_test": False,
        "run_configuration": dict(FROZEN_RUN_CONFIGURATION),
        "split_metadata": frozen_split_metadata(),
        "expected_validation_passage_ids": list(FROZEN_VALIDATION_PASSAGE_IDS),
        "expected_test_passage_ids": list(FROZEN_TEST_PASSAGE_IDS),
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
# The pre-registration is only binding if deviations are DETECTED
# --------------------------------------------------------------------------


class TestBootstrapProvenanceGate(unittest.TestCase):
    """A bootstrap that is not the frozen one can never be confirmatory.

    Without this gate the pre-registration is only a comment: a 200-resample,
    seed-7, 80%-interval bootstrap could be handed to ``assess_claim`` and
    receive SUPPORTED whenever its bounds happened to clear the thresholds.
    """

    def assert_rejected(self, bootstrap, needle):
        mismatches = validate_bootstrap_provenance(bootstrap)
        self.assertTrue(mismatches, "the deviation was not detected at all")
        self.assertIn(needle, " ".join(mismatches))
        assessment = claim(bootstrap=bootstrap)
        self.assertFalse(assessment["bootstrap_provenance_matches_frozen_protocol"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertFalse(assessment["primary_claim_supported"])
        # Never a negative result: the run could not address the claim.
        self.assertNotEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def header(self, **overrides):
        bootstrap = bootstrap_with()
        bootstrap.update(overrides)
        for endpoint in bootstrap["endpoints"].values():
            for field in overrides:
                if field in endpoint:
                    endpoint[field] = overrides[field]
        return bootstrap

    def test_the_exact_frozen_provenance_is_eligible(self):
        bootstrap = bootstrap_with()
        self.assertEqual(validate_bootstrap_provenance(bootstrap), [])
        assessment = claim(bootstrap=bootstrap)
        self.assertTrue(assessment["bootstrap_provenance_matches_frozen_protocol"])
        self.assertTrue(assessment["confirmatory_eligible"])
        self.assertEqual(assessment["claim_status"], CLAIM_SUPPORTED)

    def test_a_two_hundred_resample_bootstrap_is_rejected(self):
        self.assert_rejected(self.header(n_resamples=200), "n_resamples")

    def test_a_different_seed_is_rejected(self):
        self.assert_rejected(self.header(seed=43), "seed")

    def test_a_ninety_percent_interval_is_rejected(self):
        self.assert_rejected(self.header(ci_level=0.90), "ci_level")

    def test_a_non_percentile_interval_method_is_rejected(self):
        self.assert_rejected(self.header(ci_method="bca"), "ci_method")

    def test_a_sentence_level_bootstrap_is_rejected(self):
        # The whole point of the cluster bootstrap: resampling sentences would
        # understate the variance and narrow every interval.
        self.assert_rejected(self.header(bootstrap_unit="sentence"), "bootstrap_unit")

    def test_an_endpoint_disagreeing_with_its_own_header_is_rejected(self):
        bootstrap = bootstrap_with()
        bootstrap["endpoints"]["factual_auc_pr_delta"]["n_resamples"] = 500
        self.assert_rejected(bootstrap, "but the bootstrap header records")

    def test_a_wrong_sign_convention_is_rejected(self):
        bootstrap = bootstrap_with()
        bootstrap["endpoints"][PRIMARY_EFFICIENCY_ENDPOINT]["sign_convention"] = (
            PERFORMANCE_SIGN_CONVENTION
        )
        self.assert_rejected(bootstrap, "sign convention")

    def test_a_missing_endpoint_is_rejected(self):
        bootstrap = bootstrap_with()
        del bootstrap["endpoints"][SECONDARY_EFFICIENCY_ENDPOINT]
        self.assert_rejected(bootstrap, "missing required endpoint")

    def test_an_endpoint_without_an_interval_is_rejected(self):
        bootstrap = bootstrap_with()
        bootstrap["endpoints"]["balanced_pr_auc_delta"]["ci_lower"] = None
        self.assert_rejected(bootstrap, "has no confidence interval")

    def test_a_missing_endpoint_fails_its_gate_rather_than_raising(self):
        bootstrap = bootstrap_with()
        del bootstrap["endpoints"][PRIMARY_EFFICIENCY_ENDPOINT]
        assessment = claim(bootstrap=bootstrap)
        self.assertFalse(assessment["retrieval_efficiency_superiority_pass"])
        self.assertIsNone(assessment["retrieval_efficiency_ci_lower"])

    def test_a_real_bootstrap_run_at_the_frozen_settings_passes_provenance(self):
        # The production function must actually emit what the gate demands;
        # a gate no real run can satisfy would be worse than no gate.
        def cheap_pr_auc(y_binary, score):
            return float(np.mean(np.asarray(score, dtype=float)))

        records, scores = synthetic_sample(n_passages=6, per_passage=3)
        ddre = observations_from(records, scores, documents=2.0, nli=6.0)
        baseline = observations_from(records, scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(ddre, baseline, pr_auc=cheap_pr_auc)
        self.assertEqual(validate_bootstrap_provenance(report), [])


class TestRunConfigurationGate(unittest.TestCase):
    """The run must BE the pre-registered run, not merely resemble it."""

    def assert_rejected(self, configuration, needle):
        mismatches = validate_run_configuration(configuration)
        self.assertTrue(mismatches, "the deviation was not detected at all")
        self.assertIn(needle, " ".join(mismatches))
        assessment = claim(run_configuration=configuration)
        self.assertFalse(assessment["run_configuration_matches_frozen"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)

    def altered(self, **overrides):
        configuration = dict(FROZEN_RUN_CONFIGURATION)
        configuration.update(overrides)
        return configuration

    def test_the_frozen_configuration_is_eligible(self):
        self.assertEqual(validate_run_configuration(dict(FROZEN_RUN_CONFIGURATION)), [])
        assessment = claim()
        self.assertTrue(assessment["run_configuration_matches_frozen"])
        self.assertEqual(assessment["claim_status"], CLAIM_SUPPORTED)

    def test_the_secondary_cost_pair_is_not_the_frozen_comparator(self):
        # CM=14 / CFA=24 is a legitimate secondary analysis in this repository.
        # It is simply not the comparator this protocol pre-registered.
        self.assert_rejected(self.altered(c_miss=14.0, c_false_alarm=24.0), "c_miss")

    def test_a_different_retrieval_cost_is_rejected(self):
        self.assert_rejected(self.altered(c_retrieve=2.0), "c_retrieve")

    def test_a_different_prior_is_rejected(self):
        self.assert_rejected(self.altered(p0=0.4), "p0")

    def test_a_different_document_budget_is_rejected(self):
        self.assert_rejected(self.altered(max_docs=5), "max_docs")

    def test_a_different_validation_fraction_is_rejected(self):
        self.assert_rejected(
            self.altered(validation_fraction=0.30), "validation_fraction"
        )

    def test_a_different_split_seed_is_rejected(self):
        self.assert_rejected(self.altered(split_seed=7), "split_seed")

    def test_an_unrecorded_field_is_rejected(self):
        configuration = dict(FROZEN_RUN_CONFIGURATION)
        del configuration["c_false_alarm"]
        self.assert_rejected(configuration, "does not record")

    def test_an_absent_configuration_is_rejected(self):
        self.assertTrue(validate_run_configuration(None))
        assessment = claim(run_configuration=None)
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertIn(
            "run configuration was not recorded",
            " ".join(assessment["confirmatory_disqualifiers"]),
        )

    def test_the_frozen_configuration_is_reported_alongside_the_actual_one(self):
        assessment = claim(run_configuration=self.altered(c_miss=14.0))
        self.assertEqual(assessment["frozen_run_configuration"]["c_miss"], 28.0)
        self.assertEqual(assessment["run_configuration"]["c_miss"], 14.0)


class TestFrozenSplitGate(unittest.TestCase):
    """The held-out set is checked by IDENTITY, not by size."""

    def test_the_frozen_split_is_eligible(self):
        mismatches, identity = validate_split_identity(
            frozen_split_metadata(),
            FROZEN_VALIDATION_PASSAGE_IDS,
            FROZEN_TEST_PASSAGE_IDS,
        )
        self.assertEqual(mismatches, [])
        self.assertTrue(identity["matches_frozen_split"])
        self.assertEqual(identity["sha256"], identity["expected_sha256"])
        assessment = claim()
        self.assertTrue(assessment["split_matches_frozen"])
        self.assertEqual(assessment["claim_status"], CLAIM_SUPPORTED)

    def test_the_same_number_of_different_passages_is_rejected(self):
        # This is the case a count check cannot see: 16 held-out passages, but
        # not THESE 16. A different set of passages is a different experiment.
        swapped = list(FROZEN_TEST_PASSAGE_IDS)
        swapped[0] = 3  # a validation passage, moved into the held-out set
        metadata = frozen_split_metadata(test_passage_ids=swapped)
        mismatches, identity = validate_split_identity(
            metadata, FROZEN_VALIDATION_PASSAGE_IDS, FROZEN_TEST_PASSAGE_IDS
        )
        self.assertEqual(
            len(metadata["test_passage_ids"]), len(FROZEN_TEST_PASSAGE_IDS)
        )
        self.assertTrue(mismatches)
        self.assertIn("not the frozen split", " ".join(mismatches))
        self.assertFalse(identity["matches_frozen_split"])
        assessment = claim(split_metadata=metadata)
        self.assertFalse(assessment["split_matches_frozen"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)

    def test_different_validation_passages_are_rejected(self):
        metadata = frozen_split_metadata(validation_passage_ids=[3, 7, 11, 20])
        assessment = claim(split_metadata=metadata)
        self.assertIn(
            "validation passage IDs are not the frozen split",
            " ".join(assessment["split_mismatches"]),
        )
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)

    def test_a_different_validation_fraction_is_rejected(self):
        metadata = frozen_split_metadata(validation_fraction=0.30)
        assessment = claim(split_metadata=metadata)
        self.assertIn("validation_fraction", " ".join(assessment["split_mismatches"]))
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)

    def test_a_different_split_seed_is_rejected(self):
        metadata = frozen_split_metadata(random_state=7)
        assessment = claim(split_metadata=metadata)
        self.assertIn("random_state", " ".join(assessment["split_mismatches"]))
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)

    def test_absent_split_metadata_is_rejected(self):
        assessment = claim(split_metadata=None)
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertIn(
            "split metadata was not recorded", " ".join(assessment["split_mismatches"])
        )

    def test_an_unverified_split_is_rejected_rather_than_assumed_correct(self):
        # No expected IDs supplied means the split was never checked. Fail
        # closed: an unchecked split is not a passing split.
        assessment = claim(expected_validation_passage_ids=None)
        self.assertFalse(assessment["split_matches_frozen"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertIn(
            "no expected passage IDs", " ".join(assessment["split_mismatches"])
        )

    def test_the_split_identity_is_a_stable_order_independent_fingerprint(self):
        forward = split_identity([1, 2, 3], [4, 5])
        shuffled = split_identity([3, 1, 2], [5, 4])
        self.assertEqual(forward["sha256"], shuffled["sha256"])
        self.assertNotEqual(forward["sha256"], split_identity([1, 2, 4], [3, 5])["sha256"])
        self.assertEqual(forward["validation_passages"], 3)
        self.assertEqual(forward["test_passages"], 2)


class TestPairedIdentityIsVerified(unittest.TestCase):
    """Equal lengths are not evidence of pairing."""

    def setUp(self):
        self.records, self.scores = synthetic_sample()
        self.ddre = observations_from(self.records, self.scores)

    def test_a_permuted_baseline_is_rejected_before_any_resampling(self):
        # The failure this catches is silent otherwise: DDRE on sentence i
        # differenced against BSE on some other sentence, with every length
        # check, passage count and sentence count still agreeing.
        permuted = list(self.ddre)
        permuted[0], permuted[-1] = permuted[-1], permuted[0]
        self.assertEqual(len(permuted), len(self.ddre))

        calls = []

        def counting_pr_auc(y_binary, score):
            calls.append(1)
            return 0.5

        with self.assertRaises(PairedInputMismatch) as caught:
            paired_passage_bootstrap(
                self.ddre, permuted, n_resamples=10, seed=42,
                pr_auc=counting_pr_auc,
            )
        message = str(caught.exception)
        self.assertIn("position 0", message)
        self.assertIn("not paired", message)
        self.assertEqual(calls, [], "the bootstrap started before validating pairing")

    def test_a_within_passage_permutation_is_rejected(self):
        # Passage membership and counts are unchanged, so only the per-position
        # identity comparison can see this.
        permuted = list(self.ddre)
        permuted[0], permuted[1] = permuted[1], permuted[0]
        with self.assertRaises(PairedInputMismatch):
            validate_paired_inputs(self.ddre, permuted)

    def test_a_relabelled_sentence_is_rejected(self):
        # Same passage and sentence, different gold label: the two methods are
        # being scored against different ground truth.
        relabelled = list(self.ddre)
        first = relabelled[0]
        relabelled[0] = EvaluatedSentence(
            passage_index=first.passage_index,
            sentence_index=first.sentence_index,
            gold_label=1 - first.gold_label,
            result=first.result,
        )
        with self.assertRaises(PairedInputMismatch) as caught:
            validate_paired_inputs(self.ddre, relabelled)
        self.assertIn("label", str(caught.exception))

    def test_a_duplicated_sentence_is_rejected(self):
        duplicated = list(self.ddre)
        duplicated[1] = duplicated[0]
        with self.assertRaises(PairedInputMismatch) as caught:
            validate_paired_inputs(duplicated, duplicated)
        self.assertIn("double-count", str(caught.exception))

    def test_the_identical_sequence_is_accepted_and_its_shape_reported(self):
        shape = validate_paired_inputs(self.ddre, list(self.ddre))
        self.assertEqual(shape["sentences"], len(self.records))
        self.assertEqual(shape["passages"], 12)
        self.assertEqual(
            shape["factual_sentences"] + shape["nonfactual_sentences"],
            shape["sentences"],
        )

    def test_the_identity_is_attached_inside_the_evaluation_loop(self):
        # Not zipped on afterwards: the observation is built from the same
        # record that produced the result, in the same iteration.
        from src.evaluation import evaluate_detector_with_identity

        class Detector:
            def detect_sentence(self, record, scorer, use_cache=True):
                assert use_cache is False
                return Result(0.9 if record.label else 0.1, 2.0, 6.0)

        records = [Record(p, i, (p + i) % 2) for p in range(3) for i in range(2)]
        with contextlib.redirect_stderr(io.StringIO()):  # quiet the progress bar
            _, results, observations = evaluate_detector_with_identity(
                Detector(), records, scorer=None, description="unit", use_cache=False
            )
        self.assertEqual(len(observations), len(records))
        for record, result, observation in zip(records, results, observations):
            self.assertIs(observation.result, result)
            self.assertEqual(
                observation.identity,
                (record.passage_index, record.sentence_index, int(record.label)),
            )

    def test_the_identity_is_built_from_what_was_scored(self):
        observation = self.ddre[0]
        self.assertEqual(
            observation.identity,
            (
                observation.passage_index,
                observation.sentence_index,
                observation.gold_label,
            ),
        )
        with self.assertRaises(Exception):
            observation.passage_index = 99  # frozen: identities cannot drift


# --------------------------------------------------------------------------
# End-to-end: bootstrap feeding the claim
# --------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.records, self.scores = synthetic_sample()

    def claim_from(self, report):
        """Judge a real bootstrap's bounds under the frozen provenance.

        The bootstraps below use 200 resamples to keep the suite fast, which
        makes them exploratory by construction: they can never be confirmatory,
        and that is asserted directly in
        ``test_an_exploratory_bootstrap_can_never_be_confirmatory``. To exercise
        the claim GATES on genuine bootstrap output, the observed lower bounds
        are carried unchanged into a record that does carry the frozen
        provenance. Only the provenance is substituted; no number is.
        """
        bounds = {
            name: report["endpoints"][name]["ci_lower"] for name in BOOTSTRAP_ENDPOINTS
        }
        return claim(bootstrap=bootstrap_with(**bounds))

    def test_an_exploratory_bootstrap_can_never_be_confirmatory(self):
        # 200 resamples at seed 7 with an 80% interval is a perfectly legal
        # exploratory analysis. It is not the pre-registered one, so however
        # good its bounds are it cannot produce a confirmatory claim.
        ddre = observations_from(self.records, self.scores, documents=2.0, nli=6.0)
        baseline = observations_from(self.records, self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(
            ddre, baseline, n_resamples=200, seed=7, ci_level=0.80
        )
        assessment = claim(bootstrap=report)
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_CONFIRMATORY)
        self.assertFalse(
            assessment["bootstrap_provenance_matches_frozen_protocol"]
        )
        joined = " ".join(assessment["bootstrap_provenance_mismatches"])
        self.assertIn("n_resamples", joined)
        self.assertIn("seed", joined)
        self.assertIn("ci_level", joined)

    def test_identical_methods_give_zero_deltas_and_no_supported_claim(self):
        results = observations_from(self.records, self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(results, results, n_resamples=200, seed=42)
        for name in report["endpoints"]:
            self.assertEqual(report["endpoints"][name]["observed"], 0.0)
            self.assertEqual(report["endpoints"][name]["ci_lower"], 0.0)
        assessment = self.claim_from(report)
        self.assertTrue(assessment["performance_noninferiority"]["all_pass"])
        self.assertFalse(assessment["retrieval_efficiency_superiority_pass"])
        self.assertEqual(assessment["claim_status"], CLAIM_NOT_SUPPORTED)

    def test_identical_quality_with_one_fewer_document_supports_the_claim(self):
        ddre = observations_from(self.records, self.scores, documents=2.0, nli=6.0)
        baseline = observations_from(self.records, self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(ddre, baseline, n_resamples=200, seed=42)
        for name in (
            "nonfactual_auc_pr_delta", "factual_auc_pr_delta", "balanced_pr_auc_delta"
        ):
            self.assertEqual(report["endpoints"][name]["ci_lower"], 0.0)
            self.assertEqual(report["endpoints"][name]["ci_upper"], 0.0)
        documents = report["endpoints"][PRIMARY_EFFICIENCY_ENDPOINT]
        self.assertEqual((documents["ci_lower"], documents["ci_upper"]), (1.0, 1.0))
        assessment = self.claim_from(report)
        self.assertEqual(assessment["claim_status"], CLAIM_SUPPORTED)

    def test_fewer_documents_but_worse_nonfactual_does_not_support_the_claim(self):
        degraded = observations_from(
            self.records,
            [0.5 if r.label == 0 else s for r, s in zip(self.records, self.scores)],
            documents=2.0, nli=6.0,
        )
        baseline = observations_from(self.records, self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(degraded, baseline, n_resamples=200, seed=42)
        self.assertLess(
            report["endpoints"]["nonfactual_auc_pr_delta"]["ci_lower"],
            -CONFIRMATORY_PR_AUC_MARGIN,
        )
        assessment = self.claim_from(report)
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

        ddre = observations_from(self.records, self.scores, documents=2.0, nli=6.0)
        baseline = observations_from(self.records, self.scores, documents=3.0, nli=9.0)
        report = paired_passage_bootstrap(ddre, baseline, pr_auc=cheap_pr_auc)
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

    def test_the_bootstrap_is_computed_from_identity_bearing_observations(self):
        # Identities are attached inside the evaluation loop, not reconstructed
        # afterwards from a result list whose order nobody checked.
        self.assertIn(
            "paired_passage_bootstrap(\n                ddre_observations, "
            "bse_official_observations\n            )",
            self.source,
        )
        self.assertIn("ddre_metrics, ddre_results, ddre_observations", self.source)
        self.assertIn(
            "bse_official_metrics, bse_official_results, bse_official_observations",
            self.source,
        )
        self.assertIn("evaluate_detector_with_identity", self.source)

    def test_the_claim_receives_the_actual_run_configuration(self):
        # Not the frozen constants echoed back: the values the run used.
        for field in (
            '"c_miss": args.c_miss',
            '"c_false_alarm": args.c_false_alarm',
            '"c_retrieve": args.c_retrieve',
            '"p0": args.p0',
            '"max_docs": args.max_docs',
            '"validation_fraction": args.validation_fraction',
            '"split_seed": RANDOM_STATE',
        ):
            self.assertIn(field, self.source)

    def test_the_frozen_split_is_re_derived_and_compared_by_identity(self):
        self.assertIn("validation_fraction=CONFIRMATORY_VALIDATION_FRACTION", self.source)
        self.assertIn("random_state=CONFIRMATORY_SPLIT_SEED", self.source)
        self.assertIn(
            'expected_validation_passage_ids=frozen_split["validation_passage_ids"]',
            self.source,
        )
        self.assertIn(
            'expected_test_passage_ids=frozen_split["test_passage_ids"]', self.source
        )
        self.assertIn("split_metadata=split_metadata", self.source)
        self.assertIn(
            '"confirmatory_split_identity": claim_assessment["split_identity"]',
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
