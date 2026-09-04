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

Standard library only at module scope, so the accounting and reporting logic is
testable without torch or a model.
"""

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
    tokenizer, model, model_name, cache_path, batch_size=WANG_BATCH_SIZE
):
    """A production ``EntailmentScorer`` that also counts what it did.

    ``EntailmentScorer`` is imported here rather than at module scope so this
    module stays importable without torch. The scorer is the unmodified
    production class; the mixin only observes.
    """
    from src.utils import EntailmentScorer

    counting_class = type(
        "CountingEntailmentScorer", (SpanCountingMixin, EntailmentScorer), {}
    )
    scorer = counting_class(
        tokenizer, model, model_name, cache_path=cache_path, batch_size=batch_size
    )
    scorer.reset_counts()
    return scorer


def completion_report(placements, rows_before, rows_after, per_placement, verification):
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

    return {
        "placements_requested": [list(p) for p in placements],
        "configuration_completed": PRIMARY_CONFIGURATION,
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
        "scope_note": (
            "Cache completion only. No metric was computed, no verdict reached, "
            "and the sensitivity result is not reinterpreted here. Rerun "
            "scripts/diagnose_nbc_sensitivity.py unchanged against the expanded "
            "cache."
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
