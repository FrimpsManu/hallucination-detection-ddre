"""Targeted NLI cache completion for the incomplete NBC sensitivity placements.

The formal NBC sensitivity run against the corrected Wang-fidelity v2 cache
completed 97 of the 100 CM=14/CFA=24 placements. Three could not be evaluated
because the recorded Gate 1 run never consumed the documents those histograms
cause the policy to retrieve, so their span scores were absent from the cache:

    (positive_bin=0, negative_bin=4)
    (positive_bin=8, negative_bin=4)
    (positive_bin=9, negative_bin=4)

Under the sensitivity analysis's own branch logic, zero passes with any
unevaluated placement is ``INCONCLUSIVE_PARTIAL_CACHE_COVERAGE``, not a negative
result. Completing exactly those three placements is what lets the full grid be
evaluated, and a negative conclusion drawn if one is warranted.

This module supports a **cache-completion** step, not a new experiment. It adds
the missing span scores to the existing v2 cache using the ordinary production
scorer and the ordinary production cache mechanism, and reports what it did. It
computes no metric, reaches no verdict, and reinterprets nothing: the sensitivity
analysis is rerun afterwards, unchanged, against the expanded cache.

Why the work is interleaved rather than precomputed
---------------------------------------------------
Retrieval is adaptive. Which document the policy consumes next depends on the
posterior, which depends on the scores of the documents already consumed. There
is no way to enumerate the required spans up front without replaying the
decision loop. So the loop is replayed with the real read-through/write-through
scorer: a span already in the cache is reused, and only a span that is genuinely
missing is evaluated and written back. That is precisely "do not rerun
already-cached spans, do not score unrelated documents".

Only CM=14/CFA=24 is evaluated. CM=28/CFA=96 retrieves different documents from
the same placement, and scoring for it here would compute spans this step was
not asked to compute.

The original formal cache is treated as an immutable completed Gate 1 artifact.
It is never opened for writing. New scores go into a **derived copy**, and the
source's SHA-256 is recorded before the copy and recomputed afterwards to prove
it did not change. The later sensitivity rerun uses the derived cache.

``run_sound`` is the conjunction of every gate that was actually passed: the
provenance guard, the measured 398-pair score compatibility, the verified copy,
the intact source, the self-consistent accounting, and the read-only
completeness replay. Each clause is fail-closed -- a run that cannot show it
passed a gate has not passed it -- because a ``run_sound`` a reader trusts is
what licenses the next unchanged sensitivity run to publish a headline.

That copy is itself verified. ``prepare_derived_cache`` reports
``copy_faithful``: whether the derived file matches the source in both digest
and row count immediately after copying, before anything is written. The caller
refuses to construct a scorer at all unless it is True, and ``run_sound`` is
False without it. Extending an unfaithful copy would put correct new scores on
top of a base that is already wrong.

Standard library only at module scope, so the accounting and reporting logic is
testable without torch or a model.
"""

import hashlib
import shutil
import sqlite3
from pathlib import Path

# The three CM=14/CFA=24 placements the formal v2 run could not evaluate,
# recorded as (positive_bin, negative_bin). Overridable on the command line so
# the tool stays usable if a later run reports a different set, but this is the
# set it exists for.
INCOMPLETE_PRIMARY_PLACEMENTS = ((0, 4), (8, 4), (9, 4))

# Only the primary configuration is completed here, by instruction.
PRIMARY_CONFIGURATION = "CM_14_CFA_24"

# Wang's released inference scores one pair at a time. Batch size 1 keeps this
# step's semantics identical to the released implementation and makes the
# accounting exact: one _infer_batch call is one span evaluation.
WANG_BATCH_SIZE = 1


def sha256_file(path, chunk_size=1024 * 1024):
    """SHA-256 of a file, streamed so a multi-gigabyte cache is affordable."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_row_count(path):
    """Row count of an NLI cache, read-only."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT COUNT(*) FROM nli_scores").fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


class UnsafeCacheTarget(RuntimeError):
    """The requested source/destination pair would modify a formal artifact."""


class UnfaithfulDerivedCache(RuntimeError):
    """The derived cache did not match the source immediately after copying."""


def assert_copy_faithful(cache_identity):
    """Refuse to go any further unless the copy was verified faithful.

    Called from ``build_counting_scorer``, which is the only place a scorer
    capable of inference or of writing a cache row comes into existence. So a
    derived cache that did not match the source cannot be extended: there is
    nothing to extend it with.
    """
    if not cache_identity:
        raise UnfaithfulDerivedCache(
            "No cache identity was supplied, so the derived cache was never "
            "shown to be a faithful copy of the source. Refusing to score."
        )
    if cache_identity.get("copy_faithful") is not True:
        raise UnfaithfulDerivedCache(
            "The derived cache is not a faithful copy of the source: "
            f"{cache_identity.get('copy_faithful_note')} Refusing to construct a "
            "scorer, so zero inference is performed and zero rows are written."
        )


def prepare_derived_cache(
    source, destination, *, allow_in_place=False, overwrite=False
):
    """Copy the formal cache to a derived path, and record both identities.

    The source is an immutable completed Gate 1 artifact. Writing into it would
    make the recorded sensitivity result unreproducible, so the formal path
    always writes to a copy.

    ``allow_in_place`` exists for local debugging only and is never used by the
    documented formal command. Without it, source == destination is refused.
    """
    source_path = Path(source)
    destination_path = Path(destination)

    if not source_path.exists():
        raise FileNotFoundError(f"source cache not found: {source_path}")

    same_file = (
        destination_path.exists()
        and source_path.samefile(destination_path)
    ) or source_path.resolve() == destination_path.resolve()

    if same_file and not allow_in_place:
        raise UnsafeCacheTarget(
            f"source and destination are the same file ({source_path}). The "
            "formal cache is an immutable artifact; write to a derived copy "
            "instead. --unsafe-allow-in-place exists for local debugging and is "
            "not used by the documented formal command."
        )

    source_sha_before = sha256_file(source_path)
    source_rows = cache_row_count(source_path)

    if same_file:
        return {
            "in_place": True,
            "source_cache": str(source_path),
            "destination_cache": str(destination_path),
            "source_sha256_before": source_sha_before,
            "source_rows": source_rows,
            "destination_sha256_after_copy": source_sha_before,
            "destination_rows_after_copy": source_rows,
            "copied": False,
            # No copy was made, so the destination is byte-identical to the
            # source because it *is* the source. Stated explicitly because the
            # caller refuses to score unless this key is True.
            "copy_faithful": True,
            "copy_faithful_note": (
                "No copy was made: --unsafe-allow-in-place is set and the "
                "destination is the source file."
            ),
            "warning": (
                "UNSAFE: writing in place into the source cache. This modifies a "
                "formal artifact and must never be used for a formal result."
            ),
        }

    if destination_path.exists() and not overwrite:
        raise UnsafeCacheTarget(
            f"destination cache already exists: {destination_path}. Refusing to "
            "overwrite an existing derived cache; pass --overwrite-output only "
            "if you intend to discard it."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)

    destination_sha = sha256_file(destination_path)
    destination_rows = cache_row_count(destination_path)

    return {
        "in_place": False,
        "source_cache": str(source_path),
        "destination_cache": str(destination_path),
        "source_sha256_before": source_sha_before,
        "source_rows": source_rows,
        "destination_sha256_after_copy": destination_sha,
        "destination_rows_after_copy": destination_rows,
        "copied": True,
        "copy_faithful": destination_sha == source_sha_before
        and destination_rows == source_rows,
        "copy_faithful_note": (
            "The derived cache is byte-identical to the source immediately "
            "after the copy."
            if destination_sha == source_sha_before and destination_rows == source_rows
            else (
                "THE COPY IS NOT FAITHFUL. The derived cache differs from the "
                "source in digest or row count immediately after copying, before "
                "anything was written. It must not be extended: new rows would "
                "sit on top of an already-wrong base. Investigate the copy "
                "(disk space, a concurrent writer, a truncated write) before "
                "retrying."
            )
        ),
        "warning": None,
    }


def verify_source_unchanged(source, expected_sha256):
    """Recompute the source digest and confirm the formal artifact is intact."""
    observed = sha256_file(source)
    return {
        "source_cache": str(source),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "unchanged": observed == expected_sha256,
        "message": (
            "The source cache is byte-identical to before the run."
            if observed == expected_sha256
            else (
                "SOURCE CACHE CHANGED. A formal Gate 1 artifact was modified. "
                "Do not use either cache until this is understood."
            )
        ),
    }


def placement_label(positive_bin, negative_bin):
    return f"pos_bin={positive_bin},neg_bin={negative_bin}"


def parse_placement(text):
    """Parse a ``"0,4"`` style placement argument."""
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"placement must be 'positive_bin,negative_bin', got {text!r}")
    positive, negative = (int(part) for part in parts)
    for value in (positive, negative):
        if not 0 <= value < 10:
            raise ValueError(f"bin index out of range in {text!r}")
    return positive, negative


class SpanCountingMixin:
    """Counts span traffic through a scorer without changing its behaviour.

    Mixed in ahead of ``EntailmentScorer`` so every call still runs the real
    production code; this only observes. ``src/utils.py`` is not modified.

    ``_infer_batch`` is the exact point where a score is computed rather than
    read from the cache, so at batch size 1 its call count is the number of
    spans that were genuinely missing.
    """

    def reset_counts(self):
        self.documents_scored = 0
        self.spans_requested = 0
        self.new_evaluations = 0
        self.infer_batch_calls = 0

    def _counts(self):
        return {
            "documents_scored": self.documents_scored,
            "spans_requested": self.spans_requested,
            "spans_evaluated_now": self.new_evaluations,
            "spans_served_from_cache": self.spans_requested - self.new_evaluations,
            "infer_batch_calls": self.infer_batch_calls,
        }

    def _infer_batch(self, pairs):
        self.infer_batch_calls += 1
        self.new_evaluations += len(pairs)
        return super()._infer_batch(pairs)

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        score, spans = super().score_document(
            claim, page_content, use_cache=use_cache, write_cache=write_cache
        )
        self.documents_scored += 1
        self.spans_requested += spans
        return score, spans


def build_counting_scorer(
    tokenizer,
    model,
    model_name,
    cache_path,
    batch_size=WANG_BATCH_SIZE,
    *,
    cache_identity,
):
    """A production ``EntailmentScorer`` that also counts what it did.

    ``cache_identity`` is the ``prepare_derived_cache`` result for the cache
    being written, and is required: the faithfulness of the copy is checked
    here, before torch is even imported, so an unfaithful copy cannot produce a
    scorer and therefore cannot cause a forward pass or a cache write.

    ``EntailmentScorer`` is imported after that check rather than at module
    scope so this module stays importable without torch. The scorer is the
    unmodified production class; the mixin only observes.
    """
    assert_copy_faithful(cache_identity)

    from src.utils import EntailmentScorer

    counting_class = type(
        "CountingEntailmentScorer", (SpanCountingMixin, EntailmentScorer), {}
    )
    scorer = counting_class(
        tokenizer, model, model_name, cache_path=cache_path, batch_size=batch_size
    )
    scorer.reset_counts()
    return scorer


def completion_report(
    placements,
    rows_before,
    rows_after,
    per_placement,
    verification,
    cache_identity=None,
    source_check=None,
    guard=None,
    compatibility=None,
):
    """Assemble the accounting for one cache-completion run.

    ``rows_after - rows_before`` and the summed evaluation count are two
    independent measurements of the same quantity: how many span scores were
    previously missing. They are both reported and cross-checked, because a
    disagreement would mean a write did not land or a key collided, and either
    would quietly corrupt the sensitivity rerun.
    """
    evaluated = sum(entry["spans_evaluated_now"] for entry in per_placement)
    requested = sum(entry["spans_requested"] for entry in per_placement)
    rows_added = rows_after - rows_before

    source_intact = source_check is None or bool(source_check.get("unchanged"))
    # Fail closed: a run whose derived cache was never shown to be a faithful
    # copy of the source cannot be sound, and a report that carries no cache
    # identity at all has not shown it either.
    copy_faithful = bool(cache_identity) and cache_identity.get("copy_faithful") is True
    # Same fail-closed rule for the two pre-write gates: a run that cannot show
    # it passed them has not passed them.
    guard_passed = bool(guard) and guard.get("passed") is True
    compatible = (
        bool(compatibility)
        and compatibility.get("score_compatibility_established") is True
    )

    return {
        "placements_requested": [list(p) for p in placements],
        "configuration_completed": PRIMARY_CONFIGURATION,
        "cache_identity": cache_identity,
        "derived_cache_copy_faithful": copy_faithful,
        "score_compatibility": compatibility,
        "score_compatibility_established": compatible,
        "source_cache_check": source_check,
        "source_cache_unchanged": source_intact,
        "provenance_guard": guard,
        "previously_missing_span_scores": evaluated,
        "new_nli_evaluations_performed": evaluated,
        "span_requests_total": requested,
        "spans_served_from_existing_cache": requested - evaluated,
        "cache_rows_before": rows_before,
        "cache_rows_after": rows_after,
        "cache_rows_added": rows_added,
        "accounting_consistent": rows_added == evaluated,
        "accounting_note": (
            "cache_rows_added and new_nli_evaluations_performed measure the same "
            "quantity two ways. They agree unless a write failed or a cache key "
            "collided."
            if rows_added == evaluated
            else (
                f"MISMATCH: {evaluated} evaluations were performed but the cache "
                f"grew by {rows_added} rows. Investigate before rerunning the "
                "sensitivity analysis; do not treat the expanded cache as sound."
            )
        ),
        "per_placement": per_placement,
        "verification": verification,
        "all_requested_placements_complete": bool(
            verification and all(entry["complete"] for entry in verification)
        ),
        # Every clause is a gate that was actually passed, not an assumption:
        # the provenance guard, the measured 398-pair score compatibility, the
        # verified copy, the intact source, the self-consistent accounting, and
        # the read-only completeness replay.
        "run_sound": bool(
            verification
            and all(entry["complete"] for entry in verification)
            and rows_after - rows_before == evaluated
            and source_intact
            and copy_faithful
            and guard_passed
            and compatible
        ),
        "scope_note": (
            "Cache completion only. No metric was computed, no verdict reached, "
            "and the sensitivity result is not reinterpreted here. Rerun "
            "scripts/diagnose_nbc_sensitivity.py unchanged against the DERIVED "
            "cache; the source cache is unmodified."
        ),
    }


def verification_summary(verification):
    """One line per placement, for the printed report."""
    lines = []
    for entry in verification:
        state = "COMPLETE" if entry["complete"] else "STILL INCOMPLETE"
        detail = "" if entry["complete"] else f"  ({entry.get('reason')})"
        lines.append(f"  {entry['placement']:<28}{state}{detail}")
    return lines
