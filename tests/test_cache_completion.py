"""Tests for the targeted NBC cache-completion tool.

Synthetic values and fake scorers only. No torch, no model and no released data,
so CI runs the whole file with numpy alone.

Two properties matter most. The counting mixin must observe without altering
behaviour, since it wraps the production scorer and a distorted score would
poison the shared v2 cache. And the accounting must cross-check itself: the
number of evaluations performed and the number of rows the cache gained measure
the same quantity two ways, and a disagreement means a write did not land.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.cache_completion import (
    INCOMPLETE_PRIMARY_PLACEMENTS,
    PRIMARY_CONFIGURATION,
    WANG_BATCH_SIZE,
    CompatibilitySourceMismatch,
    SpanCountingMixin,
    UnfaithfulDerivedCache,
    UnsafeCacheTarget,
    assert_compatibility_source_bound,
    assert_copy_faithful,
    bind_compatibility_to_source,
    build_counting_scorer,
    cache_row_count,
    discard_invalid_derived_cache,
    completion_report,
    parse_placement,
    placement_label,
    prepare_derived_cache,
    sha256_file,
    verification_summary,
    verify_source_unchanged,
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


# --------------------------------------------------------------------------
# Derived-cache handling.
#
# The formal v2 cache is an immutable completed Gate 1 artifact. Writing into it
# would make the recorded sensitivity result unreproducible, so the formal path
# always copies first and extends only the copy.
# --------------------------------------------------------------------------

def build_cache(path, rows=5):
    connection = sqlite3.connect(str(path))
    connection.execute(
        "CREATE TABLE nli_scores ("
        "cache_key TEXT PRIMARY KEY, model_name TEXT NOT NULL, "
        "score_version TEXT NOT NULL, score REAL NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO nli_scores VALUES (?, ?, ?, ?)",
        [(f"key{i}", "m", "v2", float(i)) for i in range(rows)],
    )
    connection.commit()
    connection.close()


def append_row(path, key="extra"):
    connection = sqlite3.connect(str(path))
    connection.execute(
        "INSERT INTO nli_scores VALUES (?, ?, ?, ?)", (key, "m", "v2", 1.0)
    )
    connection.commit()
    connection.close()


SOURCE_DIGEST_A = "a" * 64
SOURCE_DIGEST_B = "b" * 64
BOUND = {
    "compatibility_source_sha256": SOURCE_DIGEST_A,
    "copy_source_sha256": SOURCE_DIGEST_A,
    "bound": True,
    "message": "",
}


class DerivedCacheTestCase(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix="derived-cache-")
        self.root = Path(self._tempdir.name)
        self.source = self.root / "formal_v2.sqlite"
        self.destination = self.root / "formal_v2_nbc_complete.sqlite"
        build_cache(self.source, rows=7)

    def tearDown(self):
        self._tempdir.cleanup()


class TestPrepareDerivedCache(DerivedCacheTestCase):
    def test_the_destination_is_a_faithful_copy(self):
        identity = prepare_derived_cache(self.source, self.destination)
        self.assertTrue(identity["copied"])
        self.assertTrue(identity["copy_faithful"])
        self.assertEqual(
            identity["destination_sha256_after_copy"],
            identity["source_sha256_before"],
        )
        self.assertEqual(identity["destination_rows_after_copy"], 7)
        self.assertEqual(sha256_file(self.destination), sha256_file(self.source))

    def test_the_source_is_unchanged_by_writes_to_the_copy(self):
        identity = prepare_derived_cache(self.source, self.destination)
        append_row(self.destination)
        check = verify_source_unchanged(self.source, identity["source_sha256_before"])
        self.assertTrue(check["unchanged"])
        self.assertEqual(cache_row_count(self.source), 7)
        self.assertEqual(cache_row_count(self.destination), 8)

    def test_source_equal_to_destination_is_refused(self):
        with self.assertRaises(UnsafeCacheTarget) as caught:
            prepare_derived_cache(self.source, self.source)
        self.assertIn("immutable artifact", str(caught.exception))

    def test_source_equal_to_destination_by_relative_path_is_refused(self):
        alias = Path(str(self.source))
        with self.assertRaises(UnsafeCacheTarget):
            prepare_derived_cache(self.source, alias)

    def test_the_in_place_override_is_opt_in_only(self):
        identity = prepare_derived_cache(
            self.source, self.source, allow_in_place=True
        )
        self.assertTrue(identity["in_place"])
        self.assertFalse(identity["copied"])
        self.assertIn("UNSAFE", identity["warning"])

    def test_an_existing_destination_is_not_silently_overwritten(self):
        build_cache(self.destination, rows=2)
        with self.assertRaises(UnsafeCacheTarget) as caught:
            prepare_derived_cache(self.source, self.destination)
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(cache_row_count(self.destination), 2)

    def test_overwrite_is_honoured_when_requested(self):
        build_cache(self.destination, rows=2)
        identity = prepare_derived_cache(
            self.source, self.destination, overwrite=True
        )
        self.assertTrue(identity["copy_faithful"])
        self.assertEqual(cache_row_count(self.destination), 7)

    def test_a_missing_source_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            prepare_derived_cache(self.root / "nope.sqlite", self.destination)

    def test_nested_destination_directories_are_created(self):
        nested = self.root / "a" / "b" / "derived.sqlite"
        identity = prepare_derived_cache(self.source, nested)
        self.assertTrue(nested.exists())
        self.assertTrue(identity["copy_faithful"])

    def test_a_modified_source_is_detected(self):
        identity = prepare_derived_cache(self.source, self.destination)
        append_row(self.source, key="tamper")
        check = verify_source_unchanged(self.source, identity["source_sha256_before"])
        self.assertFalse(check["unchanged"])
        self.assertIn("SOURCE CACHE CHANGED", check["message"])


class TestFaithfulCopyGate(DerivedCacheTestCase):
    """A derived cache that did not copy correctly is never extended.

    The check lives in ``build_counting_scorer``, which is the only place a
    scorer capable of a forward pass or a cache write is created. So the
    "zero inference, zero rows" property is structural: on an unfaithful copy
    there is nothing that could perform either.
    """

    def unfaithful(self):
        identity = prepare_derived_cache(self.source, self.destination)
        identity["copy_faithful"] = False
        identity["copy_faithful_note"] = "digest differs immediately after copy"
        return identity

    def test_an_unfaithful_copy_performs_zero_inference(self):
        # No scorer is constructed, so nothing exists that could score a span.
        # torch is never imported: the check runs before that import.
        with self.assertRaises(UnfaithfulDerivedCache) as caught:
            build_counting_scorer(
                None, None, "model", str(self.destination),
                cache_identity=self.unfaithful(), source_binding=BOUND,
            )
        self.assertIn("zero inference", str(caught.exception))

    def test_an_unfaithful_copy_writes_zero_new_rows(self):
        rows_before = cache_row_count(self.destination) if self.destination.exists() else None
        identity = self.unfaithful()
        rows_after_copy = cache_row_count(self.destination)
        with self.assertRaises(UnfaithfulDerivedCache):
            build_counting_scorer(
                None, None, "model", str(self.destination),
                cache_identity=identity, source_binding=BOUND,
            )
        self.assertIsNone(rows_before)
        self.assertEqual(cache_row_count(self.destination), rows_after_copy)
        self.assertEqual(cache_row_count(self.destination), 7)
        self.assertEqual(cache_row_count(self.source), 7)

    def test_a_missing_cache_identity_is_refused(self):
        # Fail closed: no identity means the copy was never verified at all.
        for identity in (None, {}):
            with self.subTest(identity=identity):
                with self.assertRaises(UnfaithfulDerivedCache):
                    assert_copy_faithful(identity)

    def test_a_faithful_copy_passes_the_gate(self):
        identity = prepare_derived_cache(self.source, self.destination)
        self.assertTrue(identity["copy_faithful"])
        assert_copy_faithful(identity)  # does not raise

    def test_copy_faithful_is_computed_from_both_digest_and_row_count(self):
        # A truncated or partially-written copy can differ in either, so both
        # are required. Each is faulted independently.
        import src.cache_completion as module

        real_sha, real_rows = module.sha256_file, module.cache_row_count
        try:
            module.sha256_file = lambda path, *a, **k: (
                "digest-differs" if Path(path) == self.destination else real_sha(path)
            )
            identity = prepare_derived_cache(
                self.source, self.destination, overwrite=True
            )
            self.assertFalse(identity["copy_faithful"])
            self.assertIn("NOT FAITHFUL", identity["copy_faithful_note"])
        finally:
            module.sha256_file = real_sha

        try:
            module.cache_row_count = lambda path: (
                999 if Path(path) == self.destination else real_rows(path)
            )
            identity = prepare_derived_cache(
                self.source, self.destination, overwrite=True
            )
            self.assertFalse(identity["copy_faithful"])
        finally:
            module.cache_row_count = real_rows

        with self.assertRaises(UnfaithfulDerivedCache):
            assert_copy_faithful(identity)

    def test_run_sound_is_false_when_the_copy_is_not_faithful(self):
        identity = self.unfaithful()
        source_check = {
            "source_cache": str(self.source),
            "expected_sha256": identity["source_sha256_before"],
            "observed_sha256": identity["source_sha256_before"],
            "unchanged": True,
            "message": "",
        }
        report = completion_report(
            INCOMPLETE_PRIMARY_PLACEMENTS, 7, 9,
            [placement_entry("a", 2, 50)], [verified("a")],
            cache_identity=identity, source_check=source_check,
            guard={"passed": True}, compatibility=PASSING_COMPATIBILITY,
            source_binding=BOUND,
        )
        self.assertFalse(report["derived_cache_copy_faithful"])
        self.assertFalse(report["run_sound"])
        self.assertTrue(report["all_requested_placements_complete"])

    def test_run_sound_requires_a_source_integrity_check_at_all(self):
        # Fail closed: nobody confirmed the formal artifact survived the run.
        identity = prepare_derived_cache(self.source, self.destination)
        report = completion_report(
            INCOMPLETE_PRIMARY_PLACEMENTS, 7, 9,
            [placement_entry("a", 2, 50)], [verified("a")],
            cache_identity=identity, source_check=None,
            guard={"passed": True}, compatibility=PASSING_COMPATIBILITY,
            source_binding=BOUND,
        )
        self.assertFalse(report["source_cache_unchanged"])
        self.assertFalse(report["run_sound"])

    def test_run_sound_requires_a_cache_identity_at_all(self):
        report = completion_report(
            INCOMPLETE_PRIMARY_PLACEMENTS, 7, 9,
            [placement_entry("a", 2, 50)], [verified("a")],
        )
        self.assertFalse(report["derived_cache_copy_faithful"])
        self.assertFalse(report["run_sound"])


class TestScriptOrdering(unittest.TestCase):
    """The gates must sit ahead of the expensive, irreversible steps.

    Asserted structurally against the script's AST rather than by running it,
    because the ordering is the property under review and running it needs a
    real checkpoint. A gate that fires after the model has been fetched, or
    after the scorer exists, is not the gate the review asked for.
    """

    @classmethod
    def setUpClass(cls):
        import ast

        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "complete_nbc_cache.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls.ast = ast
        cls.main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

    def line_of_import(self, module):
        return min(
            node.lineno
            for node in self.ast.walk(self.main)
            if isinstance(node, self.ast.ImportFrom) and node.module == module
        )

    def line_of_call(self, name):
        return min(
            node.lineno
            for node in self.ast.walk(self.main)
            if isinstance(node, self.ast.Call)
            and isinstance(node.func, self.ast.Name)
            and node.func.id == name
        )

    def test_the_static_guard_aborts_before_from_pretrained(self):
        # The whole point of the static half: a missing or wrong revision must
        # abort before anything is fetched from Hugging Face.
        static_abort = self.line_of_call("_write_aborted")
        self.assertLess(static_abort, self.line_of_import("transformers"))

    def test_the_static_guard_runs_before_the_derived_cache_is_prepared(self):
        self.assertLess(
            self.line_of_call("check_static_preconditions"),
            self.line_of_call("prepare_derived_cache"),
        )

    def test_the_runtime_guard_runs_before_the_derived_cache_is_prepared(self):
        self.assertLess(
            self.line_of_call("check_runtime_preconditions"),
            self.line_of_call("prepare_derived_cache"),
        )

    def test_the_compatibility_probe_runs_before_the_derived_cache_is_prepared(self):
        # A compatibility failure must leave nothing behind, so the probe has
        # to come before the copy, not after it.
        self.assertLess(
            self.line_of_call("_run_compatibility_probe"),
            self.line_of_call("prepare_derived_cache"),
        )

    def test_the_compatibility_abort_runs_before_the_derived_cache_is_prepared(self):
        self.assertLess(
            self.line_of_call("_write_compatibility_abort"),
            self.line_of_call("prepare_derived_cache"),
        )

    def test_the_compatibility_probe_runs_after_the_runtime_guard(self):
        # It needs the revision-pinned, provenance-checked model.
        self.assertLess(
            self.line_of_call("check_runtime_preconditions"),
            self.line_of_call("_run_compatibility_probe"),
        )

    def test_the_compatibility_probe_runs_before_any_completion_scorer(self):
        self.assertLess(
            self.line_of_call("_run_compatibility_probe"),
            self.line_of_call("build_counting_scorer"),
        )

    def test_the_source_binding_runs_before_the_scorer_is_constructed(self):
        self.assertLess(
            self.line_of_call("bind_compatibility_to_source"),
            self.line_of_call("build_counting_scorer"),
        )

    def test_the_source_binding_runs_after_the_copy(self):
        # It compares the digest prepare_derived_cache recorded, so it cannot
        # run before that digest exists.
        self.assertLess(
            self.line_of_call("prepare_derived_cache"),
            self.line_of_call("bind_compatibility_to_source"),
        )

    def test_the_binding_abort_runs_before_the_scorer_is_constructed(self):
        self.assertLess(
            self.line_of_call("_write_binding_abort"),
            self.line_of_call("build_counting_scorer"),
        )

    def test_an_invalidated_derived_cache_is_discarded_before_scoring(self):
        self.assertLess(
            self.line_of_call("discard_invalid_derived_cache"),
            self.line_of_call("build_counting_scorer"),
        )

    def test_the_completion_loop_runs_after_every_gate(self):
        completion = self.line_of_call("complete_placement")
        for gate in ("check_static_preconditions", "check_runtime_preconditions",
                     "_run_compatibility_probe", "prepare_derived_cache",
                     "bind_compatibility_to_source", "build_counting_scorer"):
            with self.subTest(gate=gate):
                self.assertLess(self.line_of_call(gate), completion)

    def test_the_copy_gate_runs_before_the_scorer_is_constructed(self):
        self.assertLess(
            self.line_of_call("_write_copy_abort"),
            self.line_of_call("build_counting_scorer"),
        )

    def test_the_reference_bundle_is_merged_before_any_check(self):
        self.assertLess(
            self.line_of_call("merge_reference"),
            self.line_of_call("check_static_preconditions"),
        )


PASSING_COMPATIBILITY = {
    "pairs_probed": 398,
    "exact_raw_matches": 398,
    "score_compatibility_established": True,
    "source_sha256_after_probe": SOURCE_DIGEST_A,
}


class TestCompatibilitySourceBinding(DerivedCacheTestCase):
    """The copied source must be the artifact compatibility was established on.

    The probe certifies one file, identified by the digest it read after its
    last forward pass. ``prepare_derived_cache`` re-hashes the source when it
    copies. If the formal cache changed in between, the derived cache is a
    faithful copy of a *different* artifact than the verdict describes.
    """

    def compatibility(self, digest=SOURCE_DIGEST_A):
        return dict(PASSING_COMPATIBILITY, source_sha256_after_probe=digest)

    def identity(self, digest=SOURCE_DIGEST_A):
        return {
            "copied": True,
            "destination_cache": str(self.destination),
            "source_sha256_before": digest,
            "copy_faithful": True,
        }

    def test_the_clean_case_binds(self):
        binding = bind_compatibility_to_source(
            self.compatibility(), self.identity()
        )
        self.assertTrue(binding["bound"])
        self.assertEqual(binding["compatibility_source_sha256"], SOURCE_DIGEST_A)
        self.assertEqual(binding["copy_source_sha256"], SOURCE_DIGEST_A)
        assert_compatibility_source_bound(binding)  # does not raise

    def test_a_source_replaced_between_the_probe_and_the_copy_is_not_bound(self):
        # 1. compatibility passes on digest A
        # 2. the source is replaced
        # 3. prepare_derived_cache sees digest B
        # 4. the binding fails
        binding = bind_compatibility_to_source(
            self.compatibility(SOURCE_DIGEST_A), self.identity(SOURCE_DIGEST_B)
        )
        self.assertFalse(binding["bound"])
        self.assertEqual(binding["compatibility_source_sha256"], SOURCE_DIGEST_A)
        self.assertEqual(binding["copy_source_sha256"], SOURCE_DIGEST_B)
        self.assertIn("CHANGED BETWEEN THE COMPATIBILITY PROBE AND THE COPY",
                      binding["message"])

    def test_an_unbound_run_never_constructs_a_completion_scorer(self):
        # 5. the completion scorer is never constructed, so 6. zero completion
        # inference and 7. zero new rows follow structurally: there is nothing
        # that could score or write.
        binding = bind_compatibility_to_source(
            self.compatibility(SOURCE_DIGEST_A), self.identity(SOURCE_DIGEST_B)
        )
        identity = prepare_derived_cache(self.source, self.destination)
        rows_after_copy = cache_row_count(self.destination)
        with self.assertRaises(CompatibilitySourceMismatch) as caught:
            build_counting_scorer(
                None, None, "model", str(self.destination),
                cache_identity=identity, source_binding=binding,
            )
        self.assertIn("zero completion inference", str(caught.exception))
        self.assertEqual(cache_row_count(self.destination), rows_after_copy)
        self.assertEqual(cache_row_count(self.destination), 7)
        self.assertEqual(cache_row_count(self.source), 7)

    def test_a_missing_binding_is_refused(self):
        for binding in (None, {}, {"bound": False}):
            with self.subTest(binding=binding):
                with self.assertRaises(CompatibilitySourceMismatch):
                    assert_compatibility_source_bound(binding)

    def test_a_digest_missing_on_either_side_is_not_bound(self):
        for compatibility, identity in (
            ({}, self.identity()),
            (self.compatibility(), {"source_sha256_before": None}),
            (None, None),
        ):
            with self.subTest(compatibility=compatibility):
                binding = bind_compatibility_to_source(compatibility, identity)
                self.assertFalse(binding["bound"])
                self.assertIn("Unverifiable is treated as failed",
                              binding["message"])

    def test_an_invalidated_derived_cache_is_removed(self):
        # Policy: it carries the name a sensitivity rerun is told to use, and
        # it holds no new scores, so it is deleted rather than left on disk.
        identity = prepare_derived_cache(self.source, self.destination)
        discarded = discard_invalid_derived_cache(identity)
        self.assertTrue(discarded["removed"])
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.source.exists())
        self.assertEqual(cache_row_count(self.source), 7)

    def test_an_in_place_cache_is_never_removed(self):
        # The destination is the source; the source is never destroyed.
        identity = prepare_derived_cache(
            self.source, self.source, allow_in_place=True
        )
        discarded = discard_invalid_derived_cache(identity)
        self.assertFalse(discarded["removed"])
        self.assertTrue(self.source.exists())
        self.assertIn("never destroyed", discarded["reason"])


class TestReportCarriesCacheIdentity(DerivedCacheTestCase):
    def build_report(self, source_unchanged=True, guard_passed=True,
                     compatibility=PASSING_COMPATIBILITY, source_binding=BOUND):
        identity = prepare_derived_cache(
            self.source, self.destination, overwrite=True
        )
        source_check = {
            "source_cache": str(self.source),
            "expected_sha256": identity["source_sha256_before"],
            "observed_sha256": identity["source_sha256_before"]
            if source_unchanged
            else "different",
            "unchanged": source_unchanged,
            "message": "",
        }
        return completion_report(
            INCOMPLETE_PRIMARY_PLACEMENTS,
            7,
            9,
            [placement_entry("a", 2, 50)],
            [verified("a")],
            cache_identity=identity,
            source_check=source_check,
            guard={"passed": guard_passed},
            compatibility=compatibility,
            source_binding=source_binding,
        )

    def test_both_cache_paths_and_hashes_are_recorded(self):
        report = self.build_report()
        identity = report["cache_identity"]
        self.assertEqual(identity["source_cache"], str(self.source))
        self.assertEqual(identity["destination_cache"], str(self.destination))
        self.assertTrue(identity["source_sha256_before"])
        self.assertEqual(identity["source_rows"], 7)

    def test_a_sound_run_requires_the_source_to_be_unchanged(self):
        self.assertTrue(self.build_report()["run_sound"])
        tampered = self.build_report(source_unchanged=False)
        self.assertFalse(tampered["source_cache_unchanged"])
        self.assertFalse(tampered["run_sound"])

    def test_the_scope_note_points_at_the_derived_cache(self):
        report = self.build_report()
        self.assertIn("DERIVED", report["scope_note"])
        self.assertIn("source cache is unmodified", report["scope_note"])

    def test_the_guard_is_carried_in_the_report(self):
        report = self.build_report()
        self.assertEqual(report["provenance_guard"], {"passed": True})

    def test_a_sound_run_requires_the_provenance_guard_to_have_passed(self):
        self.assertFalse(self.build_report(guard_passed=False)["run_sound"])

    def test_a_sound_run_requires_score_compatibility(self):
        failed = dict(PASSING_COMPATIBILITY, score_compatibility_established=False)
        report = self.build_report(compatibility=failed)
        self.assertFalse(report["score_compatibility_established"])
        self.assertFalse(report["run_sound"])

    def test_run_sound_is_false_when_the_source_is_not_bound(self):
        unbound = dict(BOUND, bound=False, copy_source_sha256=SOURCE_DIGEST_B)
        report = self.build_report(source_binding=unbound)
        self.assertFalse(report["compatibility_source_bound"])
        self.assertFalse(report["run_sound"])
        self.assertTrue(report["score_compatibility_established"])

    def test_run_sound_is_false_when_no_binding_was_formed(self):
        report = self.build_report(source_binding=None)
        self.assertFalse(report["compatibility_source_bound"])
        self.assertFalse(report["run_sound"])

    def test_a_bound_run_reports_both_digests(self):
        report = self.build_report()
        self.assertTrue(report["compatibility_source_bound"])
        binding = report["compatibility_source_binding"]
        self.assertEqual(binding["compatibility_source_sha256"], SOURCE_DIGEST_A)
        self.assertEqual(binding["copy_source_sha256"], SOURCE_DIGEST_A)
        self.assertTrue(report["run_sound"])

    def test_run_sound_is_false_when_no_compatibility_probe_was_run(self):
        # Fail closed: an absent probe has not established anything.
        report = self.build_report(compatibility=None)
        self.assertFalse(report["score_compatibility_established"])
        self.assertFalse(report["run_sound"])


if __name__ == "__main__":
    unittest.main()
