"""Tests for audit findings D-10 and D-02.

**D-10:** per-subclaim retrieval depths were collapsed into one sentence total
and discarded. They are now returned from the *same* detector pass that
produces the sentence result, checked against it, carried on the identity-
bearing observation, and exported to the predictions CSV.

**D-02:** BSE and DDRE stop under different protocols. BSE's decision-theoretic
``should_continue`` can decline the first retrieval; DDRE's stopping band is
evaluated only after an evidence update, so every non-empty evaluated subclaim
has a one-document structural floor. This is REPORTED, not corrected --
equalising it would be a new stopping algorithm, not instrumentation.

Both are accounting. No detector mathematics changes, and the frozen
confirmatory endpoints are untouched.

CPU-only, synthetic. No model, no GPU, no held-out inference.
"""

import ast
import json
import math
import unittest
from pathlib import Path

from src.baseline_core import BSEDetector
from src.ddre_core import DDREDetector
from src.evaluation import (
    ASYMMETRY_NOTE,
    EvaluatedSentence,
    SubclaimTrace,
    SubclaimTraceMismatch,
    prediction_rows,
    retrieval_protocol_asymmetry,
    summarize_subclaim_efficiency,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Wang's released NBC histograms, as used elsewhere in the suite.
RELEASED_POSITIVE = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
RELEASED_NEGATIVE = [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]
C_MISS, C_FALSE_ALARM = 28, 96


class Doc:
    def __init__(self, index=0):
        self.page_content = f"document {index}"


class Sub:
    def __init__(self, n_documents, text="claim"):
        self.text = text
        self.documents = [Doc(i) for i in range(n_documents)]


class Rec:
    def __init__(self, subclaims, passage_index=0, sentence_index=0, label=1):
        self.subclaims = subclaims
        self.passage_index = passage_index
        self.sentence_index = sentence_index
        self.label = label


class RecordingScorer:
    """Returns a fixed score and records the exact call sequence."""

    def __init__(self, score=50.0, spans=1):
        self.score = score
        self.spans = spans
        self.seen = []

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        self.seen.append((claim, page_content))
        return self.score, self.spans


class ConstantRatio:
    def __init__(self, value):
        self.value = float(value)

    def ratio(self, score):
        return self.value


def bse(mode="official", max_docs=4):
    return BSEDetector(
        list(RELEASED_POSITIVE), list(RELEASED_NEGATIVE), mode=mode,
        p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, c_retrieve=1,
        max_docs=max_docs,
    )


def ddre(max_docs=4, lower=0.2, upper=0.8, ratio=1.0):
    return DDREDetector(
        ConstantRatio(ratio), lower_threshold=lower, upper_threshold=upper,
        p0=0.5, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=max_docs,
    )


def trace(index=0, available=10, used=1, nli_calls=2, p_factual=0.5, prediction=1):
    return SubclaimTrace(
        subclaim_index=index,
        documents_available=available,
        documents_used=used,
        nli_calls=nli_calls,
        p_factual=p_factual,
        prediction=prediction,
    )


def observation(traces, passage_index=0, sentence_index=0, label=1, result=None):
    """An EvaluatedSentence whose sentence result accounts for its traces."""
    traces = tuple(traces)

    class Result:
        documents_used = sum(t.documents_used for t in traces)
        nli_calls = sum(t.nli_calls for t in traces)
        p_factual = min((t.p_factual for t in traces), default=0.5)
        prediction = 1

    return EvaluatedSentence(
        passage_index=passage_index,
        sentence_index=sentence_index,
        gold_label=label,
        result=Result() if result is None else result,
        n_subclaims=len(traces),
        subclaim_traces=traces,
    )


# --------------------------------------------------------------------------
# M. Detector trace equivalence: the trace IS the computation
# --------------------------------------------------------------------------


class TestDetectorTraceEquivalence(unittest.TestCase):
    """detect_sentence and detect_sentence_with_trace must not diverge."""

    def assert_same_sentence_result(self, detector_factory, record):
        untraced = detector_factory().detect_sentence(record, RecordingScorer())
        traced, _ = detector_factory().detect_sentence_with_trace(
            record, RecordingScorer()
        )
        for field in ("p_factual", "prediction", "documents_used", "nli_calls"):
            with self.subTest(field=field):
                self.assertEqual(getattr(traced, field), getattr(untraced, field))

    def assert_same_scorer_calls(self, detector_factory, record):
        untraced_scorer = RecordingScorer()
        detector_factory().detect_sentence(record, untraced_scorer)
        traced_scorer = RecordingScorer()
        detector_factory().detect_sentence_with_trace(record, traced_scorer)
        # Same calls, same order, same count.
        self.assertEqual(traced_scorer.seen, untraced_scorer.seen)

    def test_bse_official_results_agree(self):
        self.assert_same_sentence_result(
            lambda: bse("official"), Rec([Sub(4), Sub(2), Sub(0)])
        )

    def test_bse_equation8_results_agree(self):
        self.assert_same_sentence_result(
            lambda: bse("eq8"), Rec([Sub(4), Sub(2), Sub(0)])
        )

    def test_ddre_results_agree(self):
        self.assert_same_sentence_result(
            lambda: ddre(), Rec([Sub(4), Sub(2), Sub(0)])
        )

    def test_bse_official_makes_the_same_scorer_calls_in_the_same_order(self):
        self.assert_same_scorer_calls(
            lambda: bse("official"), Rec([Sub(3), Sub(2)])
        )

    def test_bse_equation8_makes_the_same_scorer_calls_in_the_same_order(self):
        self.assert_same_scorer_calls(lambda: bse("eq8"), Rec([Sub(3), Sub(2)]))

    def test_ddre_makes_the_same_scorer_calls_in_the_same_order(self):
        self.assert_same_scorer_calls(lambda: ddre(), Rec([Sub(3), Sub(2)]))

    def test_the_trace_is_in_exact_subclaim_order(self):
        # Distinct depths per subclaim, so a reordering would be visible.
        record = Rec([Sub(4), Sub(1), Sub(2)])
        for detector_factory in (lambda: bse("official"), lambda: ddre()):
            with self.subTest(detector=detector_factory().__class__.__name__):
                _, subclaim_results = detector_factory().detect_sentence_with_trace(
                    record, RecordingScorer()
                )
                self.assertEqual(len(subclaim_results), 3)
                for subclaim, result in zip(record.subclaims, subclaim_results):
                    self.assertLessEqual(
                        result.documents_used, len(subclaim.documents)
                    )

    def test_the_trace_is_an_immutable_tuple(self):
        _, subclaim_results = ddre().detect_sentence_with_trace(
            Rec([Sub(2)]), RecordingScorer()
        )
        self.assertIsInstance(subclaim_results, tuple)

    def test_detect_sentence_delegates_rather_than_re_aggregating(self):
        # One aggregation implementation per detector. Two copies could drift,
        # and a traced run could then disagree with an untraced one.
        for detector_class in (BSEDetector, DDREDetector):
            with self.subTest(detector=detector_class.__name__):
                names = detector_class.detect_sentence.__code__.co_names
                self.assertIn("detect_sentence_with_trace", names)
                self.assertNotIn("detect_subclaim", names)
                self.assertNotIn("min", names)


class TestZeroRetrievalIsRepresentable(unittest.TestCase):
    """BSE may decline the first fetch; DDRE currently may not."""

    def test_a_bse_subclaim_can_use_zero_documents_while_documents_are_available(self):
        # A prior far enough from 0.5 that the expected-cost rule declines the
        # first retrieval outright.
        detector = BSEDetector(
            list(RELEASED_POSITIVE), list(RELEASED_NEGATIVE), mode="official",
            p0=0.999, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM,
            c_retrieve=1000.0, max_docs=4,
        )
        scorer = RecordingScorer()
        _, subclaim_results = detector.detect_sentence_with_trace(
            Rec([Sub(4)]), scorer
        )
        self.assertEqual(subclaim_results[0].documents_used, 0)
        # And a zero-document subclaim makes zero scorer calls.
        self.assertEqual(scorer.seen, [])
        self.assertEqual(subclaim_results[0].nli_calls, 0)

    def test_a_non_empty_ddre_subclaim_always_uses_at_least_one_document(self):
        # The structural floor whose size D-02 reports. Not a bug: the band is
        # evaluated only after an evidence update.
        for documents in (1, 2, 5):
            with self.subTest(documents=documents):
                _, subclaim_results = ddre(max_docs=4).detect_sentence_with_trace(
                    Rec([Sub(documents)]), RecordingScorer()
                )
                self.assertGreaterEqual(subclaim_results[0].documents_used, 1)

    def test_stopping_after_the_first_document_is_recorded_as_depth_one(self):
        # Depth 1, not 0: the document WAS fetched and scored before the band
        # was tested. Recording it as 0 would understate DDRE's retrieval cost.
        stops_immediately = ddre(max_docs=4, lower=0.2, upper=0.8, ratio=100.0)
        _, subclaim_results = stops_immediately.detect_sentence_with_trace(
            Rec([Sub(4)]), RecordingScorer()
        )
        self.assertEqual(subclaim_results[0].documents_used, 1)
        self.assertGreaterEqual(subclaim_results[0].p_factual, 0.8)


# --------------------------------------------------------------------------
# N. Trace integrity and the D-10 export
# --------------------------------------------------------------------------


class TestTraceIntegrity(unittest.TestCase):
    def evaluate(self, detector, records):
        from src.evaluation import evaluate_detector_with_traces
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            return evaluate_detector_with_traces(
                detector, records, RecordingScorer(),
                description="unit", use_cache=False,
            )

    def records(self):
        return [
            Rec([Sub(4), Sub(2)], passage_index=0, sentence_index=0, label=1),
            Rec([Sub(0)], passage_index=0, sentence_index=1, label=0),
            Rec([Sub(3)], passage_index=1, sentence_index=0, label=1),
        ]

    def test_n_subclaims_equals_the_record_subclaim_count(self):
        records = self.records()
        _, _, observations = self.evaluate(ddre(), records)
        for record, obs in zip(records, observations):
            self.assertEqual(obs.n_subclaims, len(record.subclaims))
            self.assertEqual(len(obs.subclaim_traces), len(record.subclaims))

    def test_sentence_totals_equal_the_trace_totals(self):
        _, results, observations = self.evaluate(ddre(), self.records())
        for result, obs in zip(results, observations):
            self.assertEqual(
                result.documents_used,
                sum(t.documents_used for t in obs.subclaim_traces),
            )
            self.assertEqual(
                result.nli_calls, sum(t.nli_calls for t in obs.subclaim_traces)
            )
            self.assertEqual(
                result.p_factual,
                min(t.p_factual for t in obs.subclaim_traces),
            )

    def test_the_check_uses_min_not_mean_over_differing_subclaim_posteriors(self):
        # Every other fixture here happens to give each subclaim the SAME
        # posterior, which makes min and mean indistinguishable and lets a
        # mean-aggregating check pass unnoticed. This one forces them apart.
        detector = ddre(max_docs=4, lower=0.2, upper=0.8, ratio=3.0)
        record = Rec([Sub(1), Sub(4)])
        _, results, observations = self.evaluate(detector, [record])
        posteriors = [t.p_factual for t in observations[0].subclaim_traces]
        self.assertNotAlmostEqual(
            min(posteriors), sum(posteriors) / len(posteriors),
            msg="the fixture must make min and mean differ to be useful",
        )
        self.assertEqual(results[0].p_factual, min(posteriors))
        self.assertNotAlmostEqual(
            results[0].p_factual, sum(posteriors) / len(posteriors)
        )

    def test_a_mean_aggregating_sentence_result_fails_the_check(self):
        # The aggregation semantics are part of what the trace must account
        # for, not just the counters.
        class MeanAggregatingDetector:
            max_docs = 4

            def detect_sentence_with_trace(self, record, scorer, *, use_cache=True):
                real = ddre(max_docs=4, lower=0.2, upper=0.8, ratio=3.0)
                result, subclaim_results = real.detect_sentence_with_trace(
                    record, scorer, use_cache=use_cache
                )
                average = sum(r.p_factual for r in subclaim_results) / len(
                    subclaim_results
                )

                class Averaged:
                    p_factual = average
                    prediction = result.prediction
                    documents_used = result.documents_used
                    nli_calls = result.nli_calls

                return Averaged(), subclaim_results

        with self.assertRaises(SubclaimTraceMismatch) as caught:
            self.evaluate(MeanAggregatingDetector(), [Rec([Sub(1), Sub(4)])])
        self.assertIn("minimum subclaim p_factual", str(caught.exception))

    def test_documents_available_is_capped_by_max_docs(self):
        # Not len(subclaim.documents): a budget of 2 means only 2 were ever
        # available to this protocol, and calling it 6 would make a full-budget
        # run look like early stopping.
        _, _, observations = self.evaluate(ddre(max_docs=2), [Rec([Sub(6)])])
        self.assertEqual(observations[0].subclaim_traces[0].documents_available, 2)

    def test_documents_available_is_the_document_count_when_it_is_smaller(self):
        _, _, observations = self.evaluate(ddre(max_docs=10), [Rec([Sub(3)])])
        self.assertEqual(observations[0].subclaim_traces[0].documents_available, 3)

    def test_a_trace_length_mismatch_fails_loudly(self):
        class ShortTraceDetector:
            max_docs = 4

            def detect_sentence_with_trace(self, record, scorer, *, use_cache=True):
                real = ddre(max_docs=4)
                result, subclaim_results = real.detect_sentence_with_trace(
                    record, scorer, use_cache=use_cache
                )
                return result, subclaim_results[:-1]   # drop one

        with self.assertRaises(SubclaimTraceMismatch) as caught:
            self.evaluate(ShortTraceDetector(), [Rec([Sub(2), Sub(2)])])
        self.assertIn("cannot be aligned", str(caught.exception))

    def test_a_trace_that_does_not_account_for_the_result_fails_loudly(self):
        class InconsistentDetector:
            max_docs = 4

            def detect_sentence_with_trace(self, record, scorer, *, use_cache=True):
                real = ddre(max_docs=4)
                result, subclaim_results = real.detect_sentence_with_trace(
                    record, scorer, use_cache=use_cache
                )

                class Inflated:
                    p_factual = result.p_factual
                    prediction = result.prediction
                    documents_used = result.documents_used + 5
                    nli_calls = result.nli_calls

                return Inflated(), subclaim_results

        with self.assertRaises(SubclaimTraceMismatch) as caught:
            self.evaluate(InconsistentDetector(), [Rec([Sub(2)])])
        self.assertIn("retrieval depths sum to", str(caught.exception))

    def test_an_observation_without_traces_is_not_silently_treated_as_empty(self):
        # None, not (): a trace-less observation must never look like a
        # sentence that genuinely had no subclaims.
        bare = EvaluatedSentence(0, 0, 1, object())
        self.assertIsNone(bare.subclaim_traces)
        self.assertIsNone(bare.n_subclaims)
        self.assertFalse(bare.has_traces)
        with self.assertRaises(SubclaimTraceMismatch) as caught:
            summarize_subclaim_efficiency([bare])
        self.assertIn("did not record them", str(caught.exception))

    def test_the_pairing_identity_is_unchanged_by_the_new_fields(self):
        # The confirmatory bootstrap pairs on this tuple. Instrumentation must
        # not alter it, or PR #10's pairing check changes meaning.
        traced = observation([trace()], passage_index=3, sentence_index=7, label=0)
        bare = EvaluatedSentence(3, 7, 0, traced.result)
        self.assertEqual(traced.identity, (3, 7, 0))
        self.assertEqual(traced.identity, bare.identity)


class TestPredictionExport(unittest.TestCase):
    """D-10: the CSV must let a reader reconstruct subclaim efficiency."""

    def rows(self):
        records = [Rec([Sub(4), Sub(3), Sub(2)], passage_index=1, sentence_index=2)]
        traces = (
            trace(0, available=4, used=0, nli_calls=0),
            trace(1, available=3, used=3, nli_calls=6),
            trace(2, available=2, used=1, nli_calls=2),
        )
        observations = [observation(traces, passage_index=1, sentence_index=2)]
        results = [observations[0].result]
        return prediction_rows("bse_official", records, results, observations)

    def test_the_row_records_the_subclaim_count(self):
        self.assertEqual(self.rows()[0]["n_subclaims"], 3)

    def test_the_list_columns_are_valid_json(self):
        row = self.rows()[0]
        for column in (
            "subclaim_documents_used",
            "subclaim_documents_available",
            "subclaim_nli_calls",
        ):
            with self.subTest(column=column):
                self.assertIsInstance(row[column], str)
                self.assertIsInstance(json.loads(row[column]), list)

    def test_each_list_length_equals_the_subclaim_count(self):
        row = self.rows()[0]
        for column in (
            "subclaim_documents_used",
            "subclaim_documents_available",
            "subclaim_nli_calls",
        ):
            with self.subTest(column=column):
                self.assertEqual(len(json.loads(row[column])), row["n_subclaims"])

    def test_the_exact_depth_vector_survives_export(self):
        row = self.rows()[0]
        self.assertEqual(json.loads(row["subclaim_documents_used"]), [0, 3, 1])
        self.assertEqual(json.loads(row["subclaim_documents_available"]), [4, 3, 2])
        self.assertEqual(json.loads(row["subclaim_nli_calls"]), [0, 6, 2])

    def test_the_row_counts_declined_first_retrievals(self):
        self.assertEqual(self.rows()[0]["zero_retrieval_nonempty_subclaims"], 1)

    def test_the_existing_columns_are_all_still_present(self):
        row = self.rows()[0]
        for column in (
            "method", "passage_index", "sentence_index", "gold_label",
            "p_factual", "prediction", "retrieved_documents", "nli_span_calls",
        ):
            self.assertIn(column, row)

    def test_rows_without_observations_still_work_for_diagnostic_scripts(self):
        records = [Rec([Sub(2)])]
        results = [observation((trace(0, used=2),)).result]
        rows = prediction_rows("bse_official", records, results)
        self.assertIn("retrieved_documents", rows[0])
        self.assertNotIn("n_subclaims", rows[0])

    def test_misaligned_observations_fail_loudly(self):
        records = [Rec([Sub(2)], passage_index=0, sentence_index=0)]
        results = [observation((trace(0, used=2),)).result]
        wrong = [observation((trace(0, used=2),), passage_index=9, sentence_index=9)]
        with self.assertRaises(SubclaimTraceMismatch):
            prediction_rows("bse_official", records, results, wrong)


# --------------------------------------------------------------------------
# G. Subclaim efficiency, from actual traces
# --------------------------------------------------------------------------


class TestSubclaimEfficiencySummary(unittest.TestCase):
    def summary(self, depths, available=None):
        available = available if available is not None else [10] * len(depths)
        traces = [
            trace(i, available=a, used=d, nli_calls=2 * d)
            for i, (d, a) in enumerate(zip(depths, available))
        ]
        return summarize_subclaim_efficiency([observation(traces)])

    def test_the_counts_come_from_individual_subclaims(self):
        summary = self.summary([0, 2, 0, 4])
        self.assertEqual(summary["total_subclaims"], 4)
        self.assertEqual(summary["nonempty_subclaims"], 4)
        self.assertEqual(summary["empty_subclaims"], 0)
        self.assertEqual(summary["total_documents_used"], 6)

    def test_empty_subclaims_are_counted_separately(self):
        summary = self.summary([0, 3], available=[0, 10])
        self.assertEqual(summary["empty_subclaims"], 1)
        self.assertEqual(summary["nonempty_subclaims"], 1)
        # The empty one is a zero-retrieval subclaim but NOT a declined fetch.
        self.assertEqual(summary["zero_retrieval_subclaims"], 1)
        self.assertEqual(summary["zero_retrieval_nonempty_subclaims"], 0)

    def test_the_histogram_includes_depth_zero(self):
        summary = self.summary([0, 0, 1, 3, 3, 3])
        self.assertEqual(
            summary["retrieval_depth_histogram"],
            {"0": 2, "1": 1, "3": 3},
        )
        self.assertEqual(list(summary["retrieval_depth_histogram"]), ["0", "1", "3"])

    def test_the_averages_use_subclaim_values_not_sentence_totals(self):
        # Two sentences: one with subclaim depths [0, 4], one with [2].
        # Sentence totals are 4 and 2; subclaim depths are 0, 4, 2.
        first = observation([trace(0, used=0), trace(1, used=4)])
        second = observation([trace(0, used=2)], sentence_index=1)
        summary = summarize_subclaim_efficiency([first, second])
        self.assertEqual(summary["total_subclaims"], 3)
        self.assertAlmostEqual(summary["avg_documents_per_subclaim"], 2.0)
        self.assertEqual(summary["p50_documents_per_subclaim"], 2.0)
        self.assertEqual(summary["max_documents_per_subclaim_observed"], 4)
        # Dividing sentence totals by sentence count would give 3.0, and would
        # invent a distribution that never existed.
        self.assertNotAlmostEqual(summary["avg_documents_per_subclaim"], 3.0)

    def test_one_retrieval_subclaims_are_counted(self):
        summary = self.summary([1, 1, 2, 0])
        self.assertEqual(summary["one_retrieval_nonempty_subclaims"], 2)

    def test_nli_calls_are_summarised_per_subclaim(self):
        summary = self.summary([0, 2, 4])
        self.assertEqual(summary["total_nli_calls"], 12)
        self.assertAlmostEqual(summary["avg_nli_calls_per_subclaim"], 4.0)

    def test_the_summary_states_where_its_numbers_come_from(self):
        self.assertIn("per-subclaim traces", self.summary([1])["source"])

    def test_all_fields_are_json_serialisable(self):
        json.dumps(self.summary([0, 1, 2, 3]))


# --------------------------------------------------------------------------
# H/I. D-02 protocol asymmetry
# --------------------------------------------------------------------------


def observations_with(depths, available=None):
    available = available if available is not None else [10] * len(depths)
    traces = [
        trace(i, available=a, used=d, nli_calls=2 * d)
        for i, (d, a) in enumerate(zip(depths, available))
    ]
    return [observation(traces)]


class TestProtocolAsymmetryReport(unittest.TestCase):
    def report(self, bse_depths, ddre_depths, bse_available=None,
               ddre_available=None, max_docs=10):
        return retrieval_protocol_asymmetry(
            observations_with(bse_depths, bse_available),
            observations_with(ddre_depths, ddre_available),
            max_docs=max_docs,
        )

    def test_bse_zero_retrieval_frequency_is_reported(self):
        report = self.report([0, 2, 0, 4], [1, 1, 3, 2])
        side = report["bse_official"]
        self.assertEqual(side["nonempty_subclaims"], 4)
        self.assertEqual(side["zero_retrieval_nonempty_subclaims"], 2)
        self.assertEqual(side["zero_retrieval_nonempty_fraction"], 0.5)
        self.assertTrue(side["may_decline_first_retrieval"])

    def test_the_ddre_floor_is_the_number_of_non_empty_subclaims(self):
        report = self.report([0, 2, 0, 4], [1, 1, 3, 2])
        self.assertEqual(report["ddre_first_retrieval_floor_documents"], 4)
        self.assertFalse(report["ddre_ulsif"]["may_decline_first_retrieval"])

    def test_documents_above_the_floor_are_reported(self):
        report = self.report([0, 2, 0, 4], [1, 1, 3, 2])
        self.assertEqual(report["ddre_ulsif"]["observed_total_documents"], 7)
        self.assertEqual(report["ddre_documents_above_first_retrieval_floor"], 3)

    def test_an_empty_subclaim_does_not_contribute_to_the_floor(self):
        # documents_available == 0: the protocol never had a first document to
        # take, so it imposes no floor.
        report = self.report(
            [1, 1], [1, 0], ddre_available=[10, 0],
        )
        self.assertEqual(report["ddre_first_retrieval_floor_documents"], 1)
        self.assertEqual(report["ddre_documents_above_first_retrieval_floor"], 0)

    def test_the_floor_is_defined_from_the_protocol_not_from_observed_depths(self):
        # Same observed total, different numbers of non-empty subclaims: the
        # floor tracks the subclaims, not the documents.
        deep = self.report([1], [4], ddre_available=[10])
        wide = self.report([1], [1, 1, 1, 1], ddre_available=[10, 10, 10, 10])
        self.assertEqual(deep["ddre_ulsif"]["observed_total_documents"], 4)
        self.assertEqual(wide["ddre_ulsif"]["observed_total_documents"], 4)
        self.assertEqual(deep["ddre_first_retrieval_floor_documents"], 1)
        self.assertEqual(wide["ddre_first_retrieval_floor_documents"], 4)
        self.assertIn(
            "documents_available > 0",
            deep["ddre_first_retrieval_floor_definition"],
        )

    def test_a_ddre_subclaim_that_declined_a_first_retrieval_fails_loudly(self):
        # The floor is a claim about the CURRENT protocol. If this happens the
        # implementation no longer matches it, and the report would be false.
        with self.assertRaises(SubclaimTraceMismatch) as caught:
            self.report([1], [0], ddre_available=[10])
        message = str(caught.exception)
        self.assertIn("documents available but", message)
        self.assertIn("no longer matches the protocol", message)

    def test_the_same_pattern_is_accepted_and_counted_for_bse(self):
        # Exactly the case that fails for DDRE is the one being MEASURED for
        # BSE: its pre-first-fetch stop.
        report = self.report([0], [1], bse_available=[10])
        self.assertEqual(
            report["bse_official"]["zero_retrieval_nonempty_subclaims"], 1
        )
        self.assertEqual(report["bse_official"]["zero_retrieval_nonempty_fraction"], 1.0)

    def test_documents_above_the_floor_can_never_be_negative(self):
        report = self.report([1], [1, 1, 1])
        self.assertGreaterEqual(
            report["ddre_documents_above_first_retrieval_floor"], 0
        )

    def test_the_report_says_no_floor_adjustment_enters_the_claim(self):
        report = self.report([0, 1], [1, 1])
        self.assertIn("No floor adjustment is applied", report["note"])
        self.assertIn(
            "DESCRIPTIVE only", report["confirmatory_endpoint_unchanged"]
        )
        self.assertIn("per SENTENCE", report["confirmatory_endpoint_unchanged"])

    def test_the_report_calls_it_a_protocol_asymmetry_not_a_bug(self):
        report = self.report([0, 1], [1, 1])
        self.assertEqual(report["finding"], "D-02")
        self.assertIn("protocol asymmetry", report["status"])
        self.assertIn("no algorithmic change made", report["status"])
        for wrong_framing in ("bug", "unfair implementation", "fixed", "fairer"):
            self.assertNotIn(wrong_framing, report["note"].lower())
            self.assertNotIn(wrong_framing, report["status"].lower())

    def test_the_note_states_both_protocols_explicitly(self):
        self.assertIn("should_continue", ASYMMETRY_NOTE)
        self.assertIn("only\nafter an evidence update".replace("\n", " "),
                      " ".join(ASYMMETRY_NOTE.split()))
        self.assertIn("disadvantages DDRE", ASYMMETRY_NOTE)

    def test_the_report_is_json_serialisable(self):
        json.dumps(self.report([0, 2, 0, 4], [1, 1, 3, 2]))


# --------------------------------------------------------------------------
# K. The confirmatory protocol is untouched
# --------------------------------------------------------------------------


class TestConfirmatoryProtocolUnchanged(unittest.TestCase):
    def test_the_frozen_endpoints_are_unchanged(self):
        from src.paired_bootstrap import (
            BOOTSTRAP_ENDPOINTS,
            CONFIRMATORY_BOOTSTRAP_RESAMPLES,
            CONFIRMATORY_BOOTSTRAP_SEED,
            CONFIRMATORY_BOOTSTRAP_UNIT,
            CONFIRMATORY_CI_LEVEL,
            CONFIRMATORY_CI_METHOD,
            CONFIRMATORY_PR_AUC_MARGIN,
            PRIMARY_EFFICIENCY_ENDPOINT,
            SECONDARY_EFFICIENCY_ENDPOINT,
        )

        self.assertEqual(
            BOOTSTRAP_ENDPOINTS,
            (
                "nonfactual_auc_pr_delta",
                "factual_auc_pr_delta",
                "balanced_pr_auc_delta",
                "retrieved_documents_per_sentence_savings",
                "nli_span_calls_per_sentence_savings",
            ),
        )
        self.assertEqual(
            PRIMARY_EFFICIENCY_ENDPOINT, "retrieved_documents_per_sentence_savings"
        )
        self.assertEqual(
            SECONDARY_EFFICIENCY_ENDPOINT, "nli_span_calls_per_sentence_savings"
        )
        self.assertEqual(CONFIRMATORY_BOOTSTRAP_RESAMPLES, 10_000)
        self.assertEqual(CONFIRMATORY_BOOTSTRAP_SEED, 42)
        self.assertEqual(CONFIRMATORY_CI_LEVEL, 0.95)
        self.assertEqual(CONFIRMATORY_CI_METHOD, "percentile")
        self.assertEqual(CONFIRMATORY_BOOTSTRAP_UNIT, "passage")
        self.assertEqual(CONFIRMATORY_PR_AUC_MARGIN, 0.005)

    def test_the_bootstrap_still_reads_sentence_level_counters(self):
        # Not documents/subclaim: replacing the confirmatory retrieval endpoint
        # would change the frozen estimand under cover of an accounting PR.
        source = (PROJECT_ROOT / "src" / "paired_bootstrap.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("o.result.documents_used", source)
        self.assertIn("o.result.nli_calls", source)
        self.assertNotIn("subclaim_traces", source)
        self.assertNotIn("n_subclaims", source)

    def test_no_floor_adjustment_reaches_the_bootstrap_or_the_claim(self):
        # The D-02 retrieval floor must stay in the descriptive report. (The
        # word "floor" also appears in threshold_selection's D-07 prose about a
        # balanced-only quality floor, so the check names the D-02 concepts
        # rather than the bare word.)
        for module in ("paired_bootstrap.py", "threshold_selection.py"):
            source = (PROJECT_ROOT / "src" / module).read_text(encoding="utf-8")
            with self.subTest(module=module):
                for concept in (
                    "retrieval_floor",
                    "first_retrieval",
                    "above_first_retrieval",
                    "documents_available",
                    "protocol_asymmetry",
                ):
                    self.assertNotIn(concept, source)

    def test_traces_do_not_change_the_paired_bootstrap_result(self):
        from src.paired_bootstrap import paired_passage_bootstrap

        def cheap_pr_auc(y_binary, score):
            import numpy as np

            return float(np.mean(np.asarray(score, dtype=float)))

        class Result:
            def __init__(self, p, docs, nli):
                self.p_factual, self.documents_used, self.nli_calls = p, docs, nli

        def build(with_traces):
            observations = []
            for passage in range(4):
                for sentence in range(3):
                    label = (passage + sentence) % 2
                    result = Result(0.4 + 0.1 * label, 2.0 + label, 6.0)
                    observations.append(
                        EvaluatedSentence(
                            passage_index=passage,
                            sentence_index=sentence,
                            gold_label=label,
                            result=result,
                            n_subclaims=2 if with_traces else None,
                            subclaim_traces=(trace(0), trace(1))
                            if with_traces else None,
                        )
                    )
            return observations

        without = paired_passage_bootstrap(
            build(False), build(False), n_resamples=50, seed=42,
            pr_auc=cheap_pr_auc,
        )
        with_traces = paired_passage_bootstrap(
            build(True), build(True), n_resamples=50, seed=42, pr_auc=cheap_pr_auc,
        )
        self.assertEqual(without["endpoints"], with_traces["endpoints"])


# --------------------------------------------------------------------------
# P. No double inference in the formal path
# --------------------------------------------------------------------------


class TestFormalPathEvaluatesEachMethodOnce(unittest.TestCase):
    """Asserted on main.py's AST; running it needs torch and the model."""

    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def held_out_evaluator_calls(self):
        """Every evaluator call in main.py that scores ``test_records``."""
        calls = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name not in {
                "evaluate_detector",
                "evaluate_detector_with_identity",
                "evaluate_detector_with_traces",
            }:
                continue
            arguments = [getattr(a, "id", None) for a in node.args]
            if "test_records" in arguments:
                calls.append((name, arguments[0] if arguments else None))
        return calls

    def test_each_held_out_method_is_evaluated_exactly_once(self):
        calls = self.held_out_evaluator_calls()
        detectors = [detector for _, detector in calls]
        self.assertEqual(
            sorted(detectors), ["bse_eq8", "bse_official", "ddre"],
            f"expected one held-out pass per method, got {calls}",
        )
        self.assertEqual(len(detectors), len(set(detectors)))

    def test_every_held_out_pass_is_the_trace_bearing_evaluator(self):
        # A trace obtained from a second pass would be a trace of a different
        # computation from the one that produced the metrics.
        for evaluator, detector in self.held_out_evaluator_calls():
            with self.subTest(detector=detector):
                self.assertEqual(evaluator, "evaluate_detector_with_traces")

    def test_the_reporting_helpers_read_those_same_observations(self):
        self.assertIn(
            "summarize_subclaim_efficiency(bse_official_observations)", self.source
        )
        self.assertIn(
            "summarize_subclaim_efficiency(bse_eq8_observations)", self.source
        )
        self.assertIn("summarize_subclaim_efficiency(ddre_observations)", self.source)
        self.assertIn("retrieval_protocol_asymmetry(\n", self.source)
        self.assertIn("bse_official_observations,\n            ddre_observations,",
                      self.source)

    def test_the_predictions_csv_receives_the_trace_bearing_observations(self):
        for method in (
            "bse_official_observations",
            "bse_eq8_observations",
            "ddre_observations",
        ):
            self.assertIn(method + ",\n        ))", self.source)

    def test_the_summary_carries_both_new_sections(self):
        self.assertIn('"subclaim_efficiency": subclaim_efficiency', self.source)
        self.assertIn(
            '"retrieval_protocol_asymmetry": protocol_asymmetry', self.source
        )

    def test_the_console_reports_the_asymmetry(self):
        self.assertIn("Retrieval protocol asymmetry (D-02, descriptive)", self.source)
        self.assertIn("BSE zero-retrieval non-empty subclaims", self.source)
        self.assertIn("DDRE current first-retrieval floor", self.source)
        self.assertIn("DDRE documents above floor", self.source)


if __name__ == "__main__":
    unittest.main()
