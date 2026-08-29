"""Tests for the Gate 1 scoring-path diagnostic helpers.

The diagnostic is an instrument, so its own arithmetic has to be trustworthy
before any measurement it produces means anything. These tests pin:

* the duplicated discretizers against the production ones in
  ``src.baseline_core``, so the copies cannot silently drift;
* deterministic sampling, so a recorded sample is genuinely reproducible;
* the delta statistics;
* the materiality bands and the verdict mapping, including the case that
  matters most -- that a bucket disagreement is MATERIAL even when every raw
  delta is tiny;
* the token_type_ids assessment, which must not conclude anything from
  ``type_vocab_size`` alone.

Standard library only, like the module under test.
"""

import unittest

from src.baseline_core import discretize_document_score, discretize_nbc_score
from src.scoring_diagnostics import (
    EQUIVALENCE_MAX_ABS_DELTA,
    MATERIAL_MAX_ABS_DELTA,
    STATUS_EQUIVALENT,
    STATUS_MATERIAL,
    STATUS_MINOR,
    arm_histograms,
    bucket_disagreements,
    compare_arms,
    classify_comparison,
    content_digest,
    delta_stats,
    document_address_key,
    forward_call_accounting,
    document_bucket,
    enumerate_documents,
    laplace_histogram,
    nbc_bucket,
    one_decimal_disagreements,
    overall_verdict,
    percentile,
    round_one_decimal,
    sample_documents,
    selection_rank,
    token_id_agreement,
    token_type_id_assessment,
)
from src.wang_data import EvidenceDocument, SentenceRecord, Subclaim


def make_records(n_sentences=6, n_subclaims=2, n_documents=4):
    records = []
    for sentence in range(n_sentences):
        subclaims = [
            Subclaim(
                text=f"claim-{sentence}-{s}",
                documents=[
                    EvidenceDocument(url=f"u{sentence}{s}{d}", page_content=f"doc {sentence} {s} {d}")
                    for d in range(n_documents)
                ],
            )
            for s in range(n_subclaims)
        ]
        records.append(
            SentenceRecord(
                passage_index=sentence // 2,
                sentence_index=sentence % 2,
                sentence=f"sentence-{sentence}",
                label=sentence % 2,
                raw_label="accurate",
                subclaims=subclaims,
            )
        )
    return records


class TestDiscretizersMatchProduction(unittest.TestCase):
    """The diagnostic re-derives Wang's discretizers; they must not drift."""

    def test_nbc_bucket_matches_production(self):
        for i in range(0, 1001):
            score = i / 10.0
            self.assertEqual(nbc_bucket(score), discretize_nbc_score(score))

    def test_document_bucket_matches_production(self):
        for i in range(0, 1001):
            score = i / 10.0
            self.assertEqual(document_bucket(score), discretize_document_score(score))

    def test_the_two_discretizers_still_differ(self):
        self.assertEqual(nbc_bucket(10.0), 1)
        self.assertEqual(document_bucket(10.0), 0)

    def test_round_one_decimal_is_wangs_rounding(self):
        self.assertEqual(round_one_decimal(12.34), 12.3)
        self.assertEqual(round_one_decimal(12.36), 12.4)

    def test_laplace_histogram_smooths_every_bin(self):
        histogram = laplace_histogram([5.0, 5.0, 95.0])
        self.assertEqual(len(histogram), 10)
        self.assertEqual(histogram[0], 3)
        self.assertEqual(histogram[9], 2)
        self.assertEqual(sum(histogram), 3 + 10)


class TestSampling(unittest.TestCase):
    def test_enumeration_is_complete_and_ordered(self):
        records = make_records()
        addresses = enumerate_documents(records)
        self.assertEqual(len(addresses), 6 * 2 * 4)
        keys = [
            (a["passage_index"], a["sentence_index"], a["subclaim_index"], a["document_index"])
            for a in addresses
        ]
        self.assertEqual(len(set(keys)), len(keys))

    def test_sampling_is_deterministic_for_a_fixed_seed(self):
        records = make_records()
        first = sample_documents(records, 10, seed=7)
        second = sample_documents(records, 10, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)

    def test_different_seeds_give_different_samples(self):
        records = make_records()
        self.assertNotEqual(
            sample_documents(records, 10, seed=7),
            sample_documents(records, 10, seed=8),
        )

    def test_sample_is_returned_in_corpus_order(self):
        records = make_records()
        sample = sample_documents(records, 12, seed=3)
        keys = [
            (s["passage_index"], s["sentence_index"], s["subclaim_index"], s["document_index"])
            for s in sample
        ]
        self.assertEqual(keys, sorted(keys))

    def test_selection_rank_is_platform_stable(self):
        # Pinned literal: the sample must be identical on any interpreter, so
        # this value is part of the contract, not an implementation detail.
        address = {
            "passage_index": 3,
            "sentence_index": 1,
            "subclaim_index": 0,
            "document_index": 7,
        }
        self.assertEqual(selection_rank(address, 20231215)[:16], "b077dc97f347619b")

    def test_selection_rank_ignores_fields_outside_the_address(self):
        base = {
            "passage_index": 1,
            "sentence_index": 2,
            "subclaim_index": 3,
            "document_index": 4,
        }
        noisy = dict(base, url="https://example.invalid", page_content="anything")
        self.assertEqual(selection_rank(base, 5), selection_rank(noisy, 5))

    def test_document_address_key_is_the_full_address(self):
        records = make_records()
        addresses = enumerate_documents(records)
        keys = [document_address_key(a) for a in addresses]
        self.assertEqual(len(set(keys)), len(keys))

    def test_growing_the_corpus_does_not_move_existing_ranks(self):
        small = enumerate_documents(make_records(n_sentences=4))
        large = enumerate_documents(make_records(n_sentences=6))
        by_key = {document_address_key(a): selection_rank(a, 11) for a in large}
        for address in small:
            self.assertEqual(
                selection_rank(address, 11), by_key[document_address_key(address)]
            )

    def test_oversized_request_returns_everything(self):
        records = make_records()
        self.assertEqual(len(sample_documents(records, 10_000, seed=1)), 6 * 2 * 4)

    def test_content_digest_is_stable_and_order_sensitive(self):
        self.assertEqual(content_digest("a", "b"), content_digest("a", "b"))
        self.assertNotEqual(content_digest("a", "b"), content_digest("b", "a"))


class TestDeltaStats(unittest.TestCase):
    def test_identical_vectors_have_zero_delta(self):
        stats = delta_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(stats["signed_mean"], 0.0)
        self.assertEqual(stats["max_absolute"], 0.0)
        self.assertEqual(stats["count_above_threshold"], 0)

    def test_signed_mean_keeps_direction_while_abs_mean_does_not(self):
        stats = delta_stats([2.0, 0.0], [0.0, 2.0])
        self.assertEqual(stats["signed_mean"], 0.0)
        self.assertEqual(stats["absolute_mean"], 2.0)

    def test_threshold_count(self):
        stats = delta_stats([0.0, 0.0, 1.0], [0.0, 1e-9, 0.0])
        self.assertEqual(stats["count_above_threshold"], 1)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            delta_stats([1.0], [1.0, 2.0])

    def test_empty_input_is_reported_not_raised(self):
        stats = delta_stats([], [])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["max_absolute"])

    def test_percentile_is_nearest_rank(self):
        values = [float(i) for i in range(101)]
        self.assertEqual(percentile(values, 0.0), 0.0)
        self.assertEqual(percentile(values, 0.5), 50.0)
        self.assertEqual(percentile(values, 1.0), 100.0)
        self.assertIsNone(percentile([], 0.5))


class TestDisagreementCounters(unittest.TestCase):
    def test_one_decimal_disagreement_detects_a_rounding_flip(self):
        result = one_decimal_disagreements([12.34, 50.0], [12.36, 50.0])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["examples"][0]["left_rounded"], 12.3)
        self.assertEqual(result["examples"][0]["right_rounded"], 12.4)

    def test_sub_rounding_noise_is_not_a_disagreement(self):
        result = one_decimal_disagreements([12.30001], [12.30002])
        self.assertEqual(result["count"], 0)

    def test_bucket_disagreement_uses_the_supplied_discretizer(self):
        # 9.94 -> 9.9 and 10.04 -> 10.0 straddle the NBC bin edge but not the
        # document bin edge, because the two discretizers are offset by 0.1.
        self.assertEqual(bucket_disagreements([9.94], [10.04], nbc_bucket)["count"], 1)
        self.assertEqual(bucket_disagreements([9.94], [10.04], document_bucket)["count"], 0)

    def test_token_id_agreement_reports_first_divergence(self):
        result = token_id_agreement([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 6]])
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["exact_matches"], 1)
        self.assertEqual(result["mismatches"][0]["first_divergence"], 1)

    def test_token_id_agreement_handles_length_differences(self):
        result = token_id_agreement([[1, 2]], [[1, 2, 3]])
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["mismatches"][0]["first_divergence"], 2)

    def test_token_id_length_mismatch_between_arms_is_an_error(self):
        with self.assertRaises(ValueError):
            token_id_agreement([[1]], [[1], [2]])


def build_blocks(tokens_mismatch=0, max_delta=0.0, one_decimal=0, buckets=0):
    """Minimal comparison-shaped inputs for the classifier."""
    return (
        {"mismatch_count": tokens_mismatch, "pairs": 10, "exact_match_rate": 1.0},
        {"max_absolute": max_delta, "count": 10},
        {"count": one_decimal},
        [{"count": buckets}, {"count": 0}],
    )


class TestClassification(unittest.TestCase):
    def test_clean_comparison_is_equivalent(self):
        self.assertEqual(
            classify_comparison(*build_blocks())["status"], STATUS_EQUIVALENT
        )

    def test_token_mismatch_is_always_material(self):
        result = classify_comparison(*build_blocks(tokens_mismatch=1))
        self.assertEqual(result["status"], STATUS_MATERIAL)
        self.assertIn("input_ids", result["reasons"][0])

    def test_bucket_disagreement_is_material_even_with_tiny_deltas(self):
        # The decisive case: the BSE update consumes only the bucket, so a
        # bucket flip matters however small the underlying score change was.
        result = classify_comparison(*build_blocks(max_delta=1e-9, buckets=1))
        self.assertEqual(result["status"], STATUS_MATERIAL)

    def test_large_delta_alone_is_material(self):
        result = classify_comparison(
            *build_blocks(max_delta=MATERIAL_MAX_ABS_DELTA * 2)
        )
        self.assertEqual(result["status"], STATUS_MATERIAL)

    def test_rounding_flip_without_bucket_change_is_minor(self):
        result = classify_comparison(*build_blocks(max_delta=1e-4, one_decimal=3))
        self.assertEqual(result["status"], STATUS_MINOR)

    def test_delta_above_equivalence_bound_without_rounding_change_is_minor(self):
        result = classify_comparison(
            *build_blocks(max_delta=EQUIVALENCE_MAX_ABS_DELTA * 2)
        )
        self.assertEqual(result["status"], STATUS_MINOR)

    def test_equivalence_bound_is_inclusive(self):
        result = classify_comparison(*build_blocks(max_delta=EQUIVALENCE_MAX_ABS_DELTA))
        self.assertEqual(result["status"], STATUS_EQUIVALENT)


def comparison(status, token_mismatches=0):
    return {
        "classification": {"status": status, "reasons": []},
        "token_ids": {"mismatch_count": token_mismatches},
    }


class TestOverallVerdict(unittest.TestCase):
    def test_all_equivalent_exonerates_the_scorer(self):
        verdict = overall_verdict(
            comparison(STATUS_EQUIVALENT),
            comparison(STATUS_EQUIVALENT),
            comparison(STATUS_EQUIVALENT),
        )
        self.assertEqual(verdict["headline"], "SCORER_EXONERATED")

    def test_minor_differences_still_exonerate(self):
        verdict = overall_verdict(
            comparison(STATUS_MINOR), comparison(STATUS_MINOR), comparison(STATUS_MINOR)
        )
        self.assertEqual(verdict["headline"], "SCORER_EXONERATED")

    def test_argument_path_material_is_reported(self):
        verdict = overall_verdict(
            comparison(STATUS_MATERIAL),
            comparison(STATUS_EQUIVALENT),
            comparison(STATUS_MATERIAL),
        )
        self.assertEqual(verdict["headline"], "ARGUMENT_PATH_IMPLICATED")

    def test_batching_material_is_reported(self):
        verdict = overall_verdict(
            comparison(STATUS_EQUIVALENT),
            comparison(STATUS_MATERIAL),
            comparison(STATUS_MATERIAL),
        )
        self.assertEqual(verdict["headline"], "BATCHING_IMPLICATED")

    def test_both_material_is_reported(self):
        verdict = overall_verdict(
            comparison(STATUS_MATERIAL),
            comparison(STATUS_MATERIAL),
            comparison(STATUS_MATERIAL),
        )
        self.assertEqual(verdict["headline"], "ARGUMENT_PATH_AND_BATCHING_IMPLICATED")

    def test_tokenization_mismatch_outranks_everything(self):
        verdict = overall_verdict(
            comparison(STATUS_MATERIAL, token_mismatches=4),
            comparison(STATUS_MATERIAL),
            comparison(STATUS_MATERIAL),
        )
        self.assertEqual(verdict["headline"], "TOKENIZATION_DIFFERS")


class TestTokenTypeIdAssessment(unittest.TestCase):
    """type_vocab_size alone must never decide this."""

    def test_not_emitted_is_a_no_op(self):
        result = token_type_id_assessment(2, False, None)
        self.assertFalse(result["can_differ_from_wang"])

    def test_all_zero_values_are_a_no_op_even_with_a_type_embedding(self):
        result = token_type_id_assessment(2, True, [0])
        self.assertFalse(result["can_differ_from_wang"])

    def test_non_zero_values_with_zero_type_vocab_size_are_a_no_op(self):
        result = token_type_id_assessment(0, True, [0, 1])
        self.assertFalse(result["can_differ_from_wang"])
        self.assertIn("no token-type embedding", result["note"])

    def test_non_zero_values_with_a_type_embedding_are_a_candidate_cause(self):
        result = token_type_id_assessment(2, True, [0, 1])
        self.assertTrue(result["can_differ_from_wang"])
        self.assertEqual(result["unique_token_type_ids"], [0, 1])

    def test_non_zero_type_vocab_size_alone_does_not_implicate_anything(self):
        self.assertFalse(token_type_id_assessment(2, False, None)["can_differ_from_wang"])
        self.assertFalse(token_type_id_assessment(2, True, [0])["can_differ_from_wang"])

    def test_unknown_type_vocab_size_is_carried_through(self):
        result = token_type_id_assessment(None, True, [0, 1])
        self.assertIsNone(result["type_vocab_size"])
        self.assertTrue(result["can_differ_from_wang"])


# --------------------------------------------------------------------------
# Decision-level materiality.
#
# Wang's runtime loop (released main.py:237-250) scores every span of a
# document, keeps the MAXIMUM entailment score, and discretizes only that
# maximum before the Bayesian update. An individual span is never discretized
# into the update. These tests pin that distinction, because getting it wrong
# in either direction produces a wrong verdict: counting span buckets would
# report differences the baseline cannot see, and not counting document-max
# buckets would miss the ones it can.
# --------------------------------------------------------------------------

# Two scores 2e-4 apart that straddle a one-decimal rounding boundary, which in
# turn straddles a document-bucket edge: round(10.0499, 1) == 10.0 -> bucket 0,
# round(10.0501, 1) == 10.1 -> bucket 1. The raw delta stays far below both
# materiality bounds, so any MATERIAL verdict below comes from the bucket alone.
BUCKET_EDGE_LOW = 10.0499
BUCKET_EDGE_HIGH = 10.0501


def make_arm(scores, n_tokens=5):
    """An arm-shaped dict with identical tokenization across arms."""
    return {
        "scores": list(scores),
        "token_ids": [list(range(n_tokens)) for _ in scores],
    }


def two_span_fixture(left_spans, right_spans):
    """One NBC pair plus one two-span document, scored identically elsewhere.

    Layout of the flat pair list: index 0 is a positive NBC pair, index 1 a
    negative one, indices 2 and 3 are the document's two spans.
    """
    units = [
        {"kind": "nbc", "polarity": "positive", "index": 0, "pair_index": 0},
        {"kind": "nbc", "polarity": "negative", "index": 0, "pair_index": 1},
    ]
    document_units = [
        {
            "kind": "document",
            "passage_index": 4,
            "sentence_index": 2,
            "subclaim_index": 1,
            "document_index": 3,
            "span_count": 2,
            "pair_indices": [2, 3],
        }
    ]
    left = make_arm([50.0, 50.0] + list(left_spans))
    right = make_arm([50.0, 50.0] + list(right_spans))
    return left, right, units, document_units


class TestDecisionLevelMateriality(unittest.TestCase):
    def test_case_a_non_max_span_crossing_a_bucket_is_not_material(self):
        # The document maximum is 55.0 in both arms and sits at span 0. Span 1
        # crosses a bucket edge, but BSE never sees span 1's bucket.
        left, right, units, documents = two_span_fixture(
            [55.0, BUCKET_EDGE_LOW], [55.0, BUCKET_EDGE_HIGH]
        )
        block = compare_arms("case A", left, right, units, documents)

        self.assertEqual(block["spans"]["bucket_disagreements"]["count"], 1)
        self.assertEqual(
            block["documents"]["max_score_bucket_disagreements"]["count"], 0
        )
        self.assertEqual(block["nbc"]["bucket_disagreements"]["count"], 0)
        self.assertNotEqual(block["classification"]["status"], STATUS_MATERIAL)
        for reason in block["classification"]["reasons"]:
            self.assertNotIn("bucket disagreement", reason)

    def test_case_b_document_max_crossing_a_bucket_is_material(self):
        # The maximum itself moves across the bucket edge, so the posterior
        # update differs and the arms are not interchangeable.
        left, right, units, documents = two_span_fixture(
            [BUCKET_EDGE_LOW, 5.0], [BUCKET_EDGE_HIGH, 5.0]
        )
        block = compare_arms("case B", left, right, units, documents)

        self.assertEqual(
            block["documents"]["max_score_bucket_disagreements"]["count"], 1
        )
        self.assertEqual(block["classification"]["status"], STATUS_MATERIAL)
        self.assertTrue(
            any("bucket disagreement" in r for r in block["classification"]["reasons"])
        )

    def test_case_a_and_case_b_differ_only_in_which_span_moved(self):
        # Same two scores, same delta, same span-bucket disagreement count.
        # The only difference is whether the moving span is the maximum.
        case_a = compare_arms(
            "a", *two_span_fixture([55.0, BUCKET_EDGE_LOW], [55.0, BUCKET_EDGE_HIGH])
        )
        case_b = compare_arms(
            "b", *two_span_fixture([BUCKET_EDGE_LOW, 5.0], [BUCKET_EDGE_HIGH, 5.0])
        )
        self.assertEqual(
            case_a["spans"]["bucket_disagreements"]["count"],
            case_b["spans"]["bucket_disagreements"]["count"],
        )
        self.assertEqual(
            case_a["score_deltas_all_pairs"]["max_absolute"],
            case_b["score_deltas_all_pairs"]["max_absolute"],
        )
        self.assertNotEqual(
            case_a["classification"]["status"], case_b["classification"]["status"]
        )

    def test_nbc_bucket_disagreement_is_still_material(self):
        units = [
            {"kind": "nbc", "polarity": "positive", "index": 0, "pair_index": 0},
            {"kind": "nbc", "polarity": "negative", "index": 0, "pair_index": 1},
        ]
        left = make_arm([9.9499, 50.0])
        right = make_arm([9.9501, 50.0])
        block = compare_arms("nbc", left, right, units, [])
        self.assertEqual(block["nbc"]["bucket_disagreements"]["count"], 1)
        self.assertEqual(block["classification"]["status"], STATUS_MATERIAL)

    def test_identical_arms_are_equivalent(self):
        left, right, units, documents = two_span_fixture([55.0, 10.0], [55.0, 10.0])
        block = compare_arms("identical", left, right, units, documents)
        self.assertEqual(block["classification"]["status"], STATUS_EQUIVALENT)
        self.assertEqual(block["token_ids"]["mismatch_count"], 0)

    def test_span_block_is_labelled_as_diagnostic_only(self):
        left, right, units, documents = two_span_fixture([55.0, 10.0], [55.0, 10.0])
        block = compare_arms("labels", left, right, units, documents)
        self.assertFalse(block["spans"]["decision_level"])
        self.assertTrue(block["nbc"]["decision_level"])
        self.assertTrue(block["documents"]["decision_level"])

    def test_argmax_move_without_a_max_change_is_not_material(self):
        # The two spans tie in the left arm (argmax falls on span 0) and the
        # right arm nudges span 0 down by 1e-5, moving the argmax to span 1.
        # The maximum score itself is unchanged, so nothing downstream moves.
        left, right, units, documents = two_span_fixture([55.0, 55.0], [54.99999, 55.0])
        block = compare_arms("argmax", left, right, units, documents)
        self.assertEqual(block["documents"]["argmax_span_disagreements"]["count"], 1)
        self.assertEqual(
            block["documents"]["max_score_bucket_disagreements"]["count"], 0
        )
        self.assertNotEqual(block["classification"]["status"], STATUS_MATERIAL)

    def test_document_examples_name_the_document(self):
        left, right, units, documents = two_span_fixture(
            [BUCKET_EDGE_LOW, 5.0], [BUCKET_EDGE_HIGH, 5.0]
        )
        block = compare_arms("addresses", left, right, units, documents)
        example = block["documents"]["max_score_bucket_disagreements"]["examples"][0]
        self.assertEqual(example["passage_index"], 4)
        self.assertEqual(example["sentence_index"], 2)
        self.assertEqual(example["subclaim_index"], 1)
        self.assertEqual(example["document_index"], 3)

    def test_documents_without_spans_are_skipped_not_counted(self):
        units = [{"kind": "nbc", "polarity": "positive", "index": 0, "pair_index": 0}]
        empty = [
            {
                "kind": "document",
                "passage_index": 0,
                "sentence_index": 0,
                "subclaim_index": 0,
                "document_index": 0,
                "span_count": 0,
                "pair_indices": [],
            }
        ]
        block = compare_arms("empty", make_arm([1.0]), make_arm([1.0]), units, empty)
        self.assertEqual(block["documents"]["documents"], 0)
        self.assertEqual(block["classification"]["status"], STATUS_EQUIVALENT)


class TestArmHistograms(unittest.TestCase):
    def test_histograms_split_by_polarity(self):
        units = [
            {"polarity": "positive", "pair_index": 0},
            {"polarity": "positive", "pair_index": 1},
            {"polarity": "negative", "pair_index": 2},
        ]
        histograms = arm_histograms([5.0, 5.0, 95.0], units)
        self.assertEqual(histograms["positive"][0], 3)
        self.assertEqual(histograms["negative"][9], 2)


class TestForwardCallAccounting(unittest.TestCase):
    """Pair evaluations and model forward calls are not the same number."""

    def test_pair_evaluations_count_every_arm(self):
        accounting = forward_call_accounting(589, 8)
        self.assertEqual(accounting["pair_evaluations"], 3 * 589)

    def test_batched_arm_issues_fewer_calls(self):
        calls = forward_call_accounting(589, 8)["forward_calls"]
        self.assertEqual(calls["A1"], 589)
        self.assertEqual(calls["A3"], 589)
        self.assertEqual(calls["A2"], 74)  # ceil(589 / 8)
        self.assertEqual(calls["total"], 1252)

    def test_forward_calls_are_fewer_than_pair_evaluations(self):
        accounting = forward_call_accounting(589, 8)
        self.assertLess(
            accounting["forward_calls"]["total"], accounting["pair_evaluations"]
        )

    def test_batch_size_one_makes_the_two_counts_agree(self):
        accounting = forward_call_accounting(100, 1)
        self.assertEqual(
            accounting["forward_calls"]["total"], accounting["pair_evaluations"]
        )

    def test_partial_final_batch_is_rounded_up(self):
        self.assertEqual(forward_call_accounting(9, 8)["forward_calls"]["A2"], 2)
        self.assertEqual(forward_call_accounting(8, 8)["forward_calls"]["A2"], 1)

    def test_zero_pairs_issues_no_calls(self):
        accounting = forward_call_accounting(0, 8)
        self.assertEqual(accounting["forward_calls"]["total"], 0)
        self.assertEqual(accounting["pair_evaluations"], 0)

    def test_invalid_batch_size_is_clamped_not_divided_by_zero(self):
        self.assertEqual(forward_call_accounting(10, 0)["forward_calls"]["A2"], 10)



if __name__ == "__main__":
    unittest.main()
