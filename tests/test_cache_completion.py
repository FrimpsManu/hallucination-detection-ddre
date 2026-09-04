"""Tests for the targeted NBC cache-completion tool.

Synthetic values and fake scorers only. No torch, no model and no released data,
so CI runs the whole file with numpy alone.

Two properties matter most. The counting mixin must observe without altering
behaviour, since it wraps the production scorer and a distorted score would
poison the shared v2 cache. And the accounting must cross-check itself: the
number of evaluations performed and the number of rows the cache gained measure
the same quantity two ways, and a disagreement means a write did not land.
"""

import unittest

from src.cache_completion import (
    INCOMPLETE_PRIMARY_PLACEMENTS,
    PRIMARY_CONFIGURATION,
    WANG_BATCH_SIZE,
    SpanCountingMixin,
    completion_report,
    parse_placement,
    placement_label,
    verification_summary,
)


class FakeScorer:
    """Stands in for EntailmentScorer: serves a cache, computes on a miss."""

    def __init__(self, cached=None, spans_per_document=3):
        self.cached = dict(cached or {})
        self.spans_per_document = spans_per_document
        self.inferred = []

    def _infer_batch(self, pairs):
        self.inferred.extend(pairs)
        return [42.0 for _ in pairs]

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        spans = [f"{page_content}#{i}" for i in range(self.spans_per_document)]
        scores = []
        for span in spans:
            if span in self.cached:
                scores.append(self.cached[span])
            else:
                scores.extend(self._infer_batch([(span, claim)]))
                self.cached[span] = 42.0
        return max(scores), len(spans)


def counting_scorer(**kwargs):
    cls = type("CountingFake", (SpanCountingMixin, FakeScorer), {})
    scorer = cls(**kwargs)
    scorer.reset_counts()
    return scorer


class TestPlacementConstants(unittest.TestCase):
    def test_the_three_reported_placements_are_recorded(self):
        self.assertEqual(INCOMPLETE_PRIMARY_PLACEMENTS, ((0, 4), (8, 4), (9, 4)))

    def test_only_the_primary_configuration_is_completed(self):
        self.assertEqual(PRIMARY_CONFIGURATION, "CM_14_CFA_24")

    def test_batch_size_matches_released_wang_semantics(self):
        self.assertEqual(WANG_BATCH_SIZE, 1)

    def test_placement_label_is_stable(self):
        self.assertEqual(placement_label(0, 4), "pos_bin=0,neg_bin=4")

    def test_placement_parsing(self):
        self.assertEqual(parse_placement("0,4"), (0, 4))
        self.assertEqual(parse_placement(" 8 , 4 "), (8, 4))

    def test_malformed_placements_are_rejected(self):
        for bad in ("4", "1,2,3", "", "a,b"):
            with self.assertRaises(ValueError):
                parse_placement(bad)

    def test_out_of_range_bins_are_rejected(self):
        for bad in ("10,4", "0,10", "-1,4"):
            with self.assertRaises(ValueError):
                parse_placement(bad)


class TestSpanCounting(unittest.TestCase):
    def test_a_fully_cached_document_needs_no_evaluation(self):
        cached = {f"doc#{i}": 5.0 for i in range(3)}
        scorer = counting_scorer(cached=cached)
        scorer.score_document("claim", "doc")
        counts = scorer._counts()
        self.assertEqual(counts["spans_requested"], 3)
        self.assertEqual(counts["spans_evaluated_now"], 0)
        self.assertEqual(counts["spans_served_from_cache"], 3)
        self.assertEqual(counts["documents_scored"], 1)

    def test_only_the_missing_spans_are_evaluated(self):
        scorer = counting_scorer(cached={"doc#0": 5.0, "doc#2": 7.0})
        scorer.score_document("claim", "doc")
        counts = scorer._counts()
        self.assertEqual(counts["spans_requested"], 3)
        self.assertEqual(counts["spans_evaluated_now"], 1)
        self.assertEqual(counts["spans_served_from_cache"], 2)

    def test_a_span_is_not_re_evaluated_once_written(self):
        # The second document reuses what the first one wrote.
        scorer = counting_scorer(cached={})
        scorer.score_document("claim", "doc")
        first = scorer._counts()["spans_evaluated_now"]
        scorer.score_document("claim", "doc")
        self.assertEqual(first, 3)
        self.assertEqual(scorer._counts()["spans_evaluated_now"], 3)

    def test_counting_does_not_change_the_returned_score(self):
        # Deliberately many decimals: this mixin wraps the production scorer
        # that writes into the shared v2 cache, so any rounding or reshaping it
        # introduced would be persisted. Whole numbers would hide that.
        cached = {"doc#0": 1.0, "doc#1": 19.9462890625, "doc#2": 3.0}
        plain = FakeScorer(cached=dict(cached))
        counted = counting_scorer(cached=dict(cached))
        plain_result = plain.score_document("claim", "doc")
        counted_result = counted.score_document("claim", "doc")
        self.assertEqual(plain_result, counted_result)
        self.assertEqual(counted_result[0], 19.9462890625)
        self.assertNotEqual(counted_result[0], round(counted_result[0], 1))

    def test_counting_preserves_continuous_scores_on_a_cache_miss(self):
        # A computed score must reach the cache untouched as well.
        counted = counting_scorer(cached={})
        counted._infer_batch = lambda pairs: [19.9462890625 for _ in pairs]
        score, _ = counted.score_document("claim", "doc")
        self.assertEqual(score, 19.9462890625)

    def test_infer_batch_calls_equal_spans_at_batch_size_one(self):
        scorer = counting_scorer(cached={})
        scorer.score_document("claim", "doc")
        counts = scorer._counts()
        self.assertEqual(counts["infer_batch_calls"], counts["spans_evaluated_now"])

    def test_reset_counts_clears_everything(self):
        scorer = counting_scorer(cached={})
        scorer.score_document("claim", "doc")
        scorer.reset_counts()
        self.assertEqual(
            scorer._counts(),
            {
                "documents_scored": 0,
                "spans_requested": 0,
                "spans_evaluated_now": 0,
                "spans_served_from_cache": 0,
                "infer_batch_calls": 0,
            },
        )

    def test_counts_accumulate_across_documents(self):
        scorer = counting_scorer(cached={})
        scorer.score_document("claim", "a")
        scorer.score_document("claim", "b")
        counts = scorer._counts()
        self.assertEqual(counts["documents_scored"], 2)
        self.assertEqual(counts["spans_requested"], 6)
        self.assertEqual(counts["spans_evaluated_now"], 6)


def placement_entry(label, evaluated, requested):
    return {
        "placement": label,
        "spans_evaluated_now": evaluated,
        "spans_requested": requested,
        "spans_served_from_cache": requested - evaluated,
        "documents_scored": 1,
        "infer_batch_calls": evaluated,
    }


def verified(label, complete=True, reason=None):
    return {"placement": label, "complete": complete, "reason": reason}


class TestCompletionReport(unittest.TestCase):
    def report(self, per_placement, rows_before, rows_after, verification):
        return completion_report(
            INCOMPLETE_PRIMARY_PLACEMENTS,
            rows_before,
            rows_after,
            per_placement,
            verification,
        )

    def test_totals_are_summed_across_placements(self):
        report = self.report(
            [placement_entry("a", 10, 100), placement_entry("b", 5, 80)],
            1000,
            1015,
            [verified("a"), verified("b")],
        )
        self.assertEqual(report["previously_missing_span_scores"], 15)
        self.assertEqual(report["new_nli_evaluations_performed"], 15)
        self.assertEqual(report["span_requests_total"], 180)
        self.assertEqual(report["spans_served_from_existing_cache"], 165)

    def test_cache_growth_is_reported_before_and_after(self):
        report = self.report(
            [placement_entry("a", 15, 100)], 1000, 1015, [verified("a")]
        )
        self.assertEqual(report["cache_rows_before"], 1000)
        self.assertEqual(report["cache_rows_after"], 1015)
        self.assertEqual(report["cache_rows_added"], 15)

    def test_accounting_is_cross_checked(self):
        report = self.report(
            [placement_entry("a", 15, 100)], 1000, 1015, [verified("a")]
        )
        self.assertTrue(report["accounting_consistent"])
        self.assertIn("agree unless", report["accounting_note"])

    def test_a_write_that_did_not_land_is_flagged(self):
        # 15 evaluations but the cache only grew by 12: a write failed or a key
        # collided, and the expanded cache must not be trusted.
        report = self.report(
            [placement_entry("a", 15, 100)], 1000, 1012, [verified("a")]
        )
        self.assertFalse(report["accounting_consistent"])
        self.assertIn("MISMATCH", report["accounting_note"])
        self.assertIn("do not treat the expanded cache as sound", report["accounting_note"])

    def test_completion_requires_every_placement_to_verify(self):
        report = self.report(
            [placement_entry("a", 1, 10)],
            1,
            2,
            [verified("a"), verified("b", complete=False, reason="miss")],
        )
        self.assertFalse(report["all_requested_placements_complete"])

    def test_all_verified_reports_complete(self):
        report = self.report(
            [placement_entry("a", 1, 10)], 1, 2, [verified("a"), verified("b")]
        )
        self.assertTrue(report["all_requested_placements_complete"])

    def test_empty_verification_is_not_treated_as_complete(self):
        report = self.report([placement_entry("a", 1, 10)], 1, 2, [])
        self.assertFalse(report["all_requested_placements_complete"])

    def test_only_the_primary_configuration_is_recorded(self):
        report = self.report([placement_entry("a", 1, 10)], 1, 2, [verified("a")])
        self.assertEqual(report["configuration_completed"], "CM_14_CFA_24")

    def test_the_report_disclaims_reinterpretation(self):
        report = self.report([placement_entry("a", 1, 10)], 1, 2, [verified("a")])
        self.assertIn("Cache completion only", report["scope_note"])
        self.assertIn("not reinterpreted", report["scope_note"])
        self.assertIn("unchanged", report["scope_note"])

    def test_zero_missing_spans_is_a_valid_no_op(self):
        report = self.report(
            [placement_entry("a", 0, 100)], 5000, 5000, [verified("a")]
        )
        self.assertEqual(report["previously_missing_span_scores"], 0)
        self.assertTrue(report["accounting_consistent"])
        self.assertTrue(report["all_requested_placements_complete"])


class TestVerificationSummary(unittest.TestCase):
    def test_complete_placements_render_cleanly(self):
        lines = verification_summary([verified("pos_bin=0,neg_bin=4")])
        self.assertIn("COMPLETE", lines[0])
        self.assertNotIn("STILL INCOMPLETE", lines[0])

    def test_incomplete_placements_show_the_reason(self):
        lines = verification_summary(
            [verified("pos_bin=8,neg_bin=4", complete=False, reason="span 3/7 missing")]
        )
        self.assertIn("STILL INCOMPLETE", lines[0])
        self.assertIn("span 3/7 missing", lines[0])

    def test_one_line_per_placement(self):
        lines = verification_summary([verified("a"), verified("b"), verified("c")])
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
