"""Comparison primitives for the Gate 1 scoring-path diagnostic.

This module answers one question and nothing else: *does this repository's
entailment scorer make the same decisions as Wang et al.'s literal released
scorer, in the same environment?* It is a measuring instrument. It never
changes ``bse_official``, the tolerances, the published reference values, the
splits, or the cost parameters, and nothing in the normal experiment pipeline
imports it.

Deliberately standard-library only. ``src/utils.py`` imports torch at module
scope, so anything that lives beside the scorer drags torch into every test
process. These helpers are the part worth unit-testing, so they are kept
importable without torch, numpy, scipy, or sklearn.

Materiality
-----------
"Materially different" is defined here, before any measurement, so that a
result cannot be re-interpreted after the fact:

``EQUIVALENT``
    No token-id mismatch, no one-decimal disagreement, no bucket disagreement,
    and every raw score within :data:`EQUIVALENCE_MAX_ABS_DELTA`. The two arms
    cannot produce a different Gate 1 number.
``MINOR``
    Raw scores differ by more than the equivalence bound, or a handful of
    one-decimal values disagree, but no discretized bucket disagrees. Nothing
    downstream of the discretizer can see the difference *on this sample*.
``MATERIAL``
    Any token-id mismatch, any bucket disagreement, or any raw delta beyond
    :data:`MATERIAL_MAX_ABS_DELTA`. The arms are not interchangeable.

Bucket disagreement is the sharpest of these, because the BSE update consumes
only the bucket. A score difference that never crosses a bucket edge cannot
change a posterior, a stopping decision, or an evidence count.
"""

import hashlib


# A raw entailment score is on a 0-100 scale. Both arms compute it in float32
# on the same device, so the floor on agreement is float32 rounding, not
# algorithmic. 1e-3 is roughly two orders of magnitude above that floor and
# two orders below the 0.1 one-decimal quantum.
EQUIVALENCE_MAX_ABS_DELTA = 1e-3

# Beyond this a difference is larger than the one-decimal quantum Wang rounds
# to, so it can move a rounded score by a whole step on its own.
MATERIAL_MAX_ABS_DELTA = 1e-2

# The threshold the run is asked to count at, independent of the verdict bands.
REPORTED_DELTA_THRESHOLD = 1e-4

STATUS_EQUIVALENT = "EQUIVALENT"
STATUS_MINOR = "MINOR"
STATUS_MATERIAL = "MATERIAL"


# --------------------------------------------------------------------------
# Wang's discretizers, re-derived here rather than imported.
#
# src/baseline_core.py owns the production discretizers and is off limits to
# this work. Importing them would also make a diagnostic that is supposed to
# audit the pipeline depend on the pipeline. These are transcribed from the
# released code directly:
#   NBC_feature.py:34   int(score / 10)          on a one-decimal score
#   main.py:249         int((score - 0.1) / 10)  on a one-decimal score
# tests/test_scoring_diagnostics.py pins them against src.baseline_core so the
# duplication cannot silently drift.
# --------------------------------------------------------------------------

def round_one_decimal(score):
    """Wang's ``round(float(pred) * 100, 1)`` (utils.py:62), applied to a score."""
    return round(float(score), 1)


def nbc_bucket(score):
    """Released NBC_feature.py:34 discretizer, clamped to a valid index."""
    return max(0, min(int(round_one_decimal(score) / 10.0), 9))


def document_bucket(score):
    """Released main.py:249 discretizer, clamped to a valid index."""
    return max(0, min(int((round_one_decimal(score) - 0.1) / 10.0), 9))


def laplace_histogram(scores):
    """Ten-bin NBC histogram with Wang's +1 smoothing (main.py:161-162)."""
    histogram = [0] * 10
    for score in scores:
        histogram[nbc_bucket(score)] += 1
    return [count + 1 for count in histogram]


# --------------------------------------------------------------------------
# Deterministic sampling
# --------------------------------------------------------------------------

def content_digest(*parts):
    """Stable short digest of text, so a sample can be verified after the fact."""
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def enumerate_documents(records):
    """List every (sentence, subclaim, document) address in a stable order.

    The order is the released data's own order -- passages, then sentences,
    then decomposed subclaims, then retrieved documents -- so the enumeration
    does not depend on dict iteration or filesystem listing.
    """
    addresses = []
    for record in records:
        for subclaim_index, subclaim in enumerate(record.subclaims):
            for document_index, document in enumerate(subclaim.documents):
                addresses.append(
                    {
                        "passage_index": record.passage_index,
                        "sentence_index": record.sentence_index,
                        "subclaim_index": subclaim_index,
                        "document_index": document_index,
                        "subclaim_text": subclaim.text,
                        "url": document.url,
                        "page_content": document.page_content,
                    }
                )
    return addresses


def document_address_key(address):
    """Stable identity of a document within the released corpus."""
    return (
        address["passage_index"],
        address["sentence_index"],
        address["subclaim_index"],
        address["document_index"],
    )


def selection_rank(address, seed):
    """Deterministic per-document sort key derived from a seed and an address.

    Sampling is done by hashing rather than by ``random.sample`` on purpose.
    ``random.sample``'s selection algorithm is a CPython implementation detail
    and is not contracted to be stable across interpreter versions, so a sample
    drawn here and a sample drawn in a Colab runtime could silently differ. A
    SHA-256 rank over ``(seed, address)`` is identical on every platform and
    every Python version, which is what "record the selected sample" has to
    mean if the recording is to be worth anything.

    It also degrades gracefully: adding documents to the corpus would leave the
    ranks of existing documents unchanged.
    """
    payload = f"{seed}:" + ":".join(str(part) for part in document_address_key(address))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_documents(records, n_documents, seed):
    """Draw a reproducible sample of retrieved documents.

    Uniform over every released document rather than biased toward the early
    retrieval positions the policy actually consumes. What is being measured is
    whether two scorers agree on a given (span, subclaim) pair, and that
    agreement has no reason to depend on retrieval position; a uniform draw is
    the one that needs no justification. The realized position distribution is
    recorded in the sample file so the choice stays visible.

    Returned in corpus order, not selection order, so the sample file reads as
    an ordered slice of the dataset.
    """
    addresses = enumerate_documents(records)
    if n_documents >= len(addresses):
        return list(addresses)
    ranked = sorted(addresses, key=lambda address: selection_rank(address, seed))
    chosen = ranked[:n_documents]
    return sorted(chosen, key=document_address_key)


# --------------------------------------------------------------------------
# Delta statistics
# --------------------------------------------------------------------------

def percentile(sorted_values, fraction):
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_values:
        return None
    index = int(round(fraction * (len(sorted_values) - 1)))
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def delta_stats(left, right, threshold=REPORTED_DELTA_THRESHOLD):
    """Summarize ``left - right`` elementwise."""
    if len(left) != len(right):
        raise ValueError("score vectors must be the same length")
    if not left:
        return {
            "count": 0,
            "signed_mean": None,
            "absolute_mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max_absolute": None,
            "threshold": threshold,
            "count_above_threshold": 0,
        }

    signed = [float(a) - float(b) for a, b in zip(left, right)]
    absolute = sorted(abs(value) for value in signed)
    return {
        "count": len(signed),
        "signed_mean": sum(signed) / len(signed),
        "absolute_mean": sum(absolute) / len(absolute),
        "p50": percentile(absolute, 0.50),
        "p95": percentile(absolute, 0.95),
        "p99": percentile(absolute, 0.99),
        "max_absolute": absolute[-1],
        "threshold": threshold,
        "count_above_threshold": sum(1 for value in absolute if value > threshold),
    }


def one_decimal_disagreements(left, right, limit=10):
    """Indices where Wang's one-decimal rounding of the two arms differs."""
    indices = [
        i
        for i, (a, b) in enumerate(zip(left, right))
        if round_one_decimal(a) != round_one_decimal(b)
    ]
    return {
        "count": len(indices),
        "rate": (len(indices) / len(left)) if left else 0.0,
        "examples": [
            {
                "index": i,
                "left": float(left[i]),
                "right": float(right[i]),
                "left_rounded": round_one_decimal(left[i]),
                "right_rounded": round_one_decimal(right[i]),
            }
            for i in indices[:limit]
        ],
    }


def bucket_disagreements(left, right, bucket_fn, limit=10):
    """Indices where the two arms fall in different discretized bins."""
    indices = [
        i for i, (a, b) in enumerate(zip(left, right)) if bucket_fn(a) != bucket_fn(b)
    ]
    return {
        "count": len(indices),
        "rate": (len(indices) / len(left)) if left else 0.0,
        "examples": [
            {
                "index": i,
                "left": float(left[i]),
                "right": float(right[i]),
                "left_bucket": bucket_fn(left[i]),
                "right_bucket": bucket_fn(right[i]),
            }
            for i in indices[:limit]
        ],
    }


def token_id_agreement(left_ids, right_ids, limit=10):
    """Exact-match rate over per-pair token id sequences.

    A mismatch here outranks every numeric comparison: two arms that tokenize
    differently are not scoring the same input, and no downstream delta is
    interpretable until it is resolved.
    """
    if len(left_ids) != len(right_ids):
        raise ValueError("token id lists must be the same length")
    mismatches = []
    for i, (a, b) in enumerate(zip(left_ids, right_ids)):
        if list(a) != list(b):
            mismatches.append(
                {
                    "index": i,
                    "left_length": len(a),
                    "right_length": len(b),
                    "first_divergence": _first_divergence(a, b),
                    "left_ids_head": list(a)[:24],
                    "right_ids_head": list(b)[:24],
                }
            )
    total = len(left_ids)
    return {
        "pairs": total,
        "exact_matches": total - len(mismatches),
        "exact_match_rate": ((total - len(mismatches)) / total) if total else 1.0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:limit],
    }


def _first_divergence(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def classify_comparison(token_agreement, deltas, one_decimal, buckets):
    """Apply the predeclared materiality bands to one arm-vs-arm comparison.

    ``buckets`` is an iterable of bucket-disagreement blocks (NBC and document
    discretizers are both consulted, since the released code uses two).
    """
    reasons = []
    status = STATUS_EQUIVALENT

    if token_agreement is not None and token_agreement["mismatch_count"] > 0:
        reasons.append(
            f"{token_agreement['mismatch_count']} pair(s) tokenize to different input_ids"
        )
        status = STATUS_MATERIAL

    bucket_total = sum(block["count"] for block in buckets)
    if bucket_total > 0:
        reasons.append(f"{bucket_total} discretized bucket disagreement(s)")
        status = STATUS_MATERIAL

    max_absolute = deltas.get("max_absolute")
    if max_absolute is not None and max_absolute > MATERIAL_MAX_ABS_DELTA:
        reasons.append(
            f"max |delta| {max_absolute:.3e} exceeds the material bound "
            f"{MATERIAL_MAX_ABS_DELTA:.0e}"
        )
        status = STATUS_MATERIAL

    if status != STATUS_MATERIAL:
        if one_decimal["count"] > 0:
            reasons.append(
                f"{one_decimal['count']} one-decimal disagreement(s), no bucket change"
            )
            status = STATUS_MINOR
        elif max_absolute is not None and max_absolute > EQUIVALENCE_MAX_ABS_DELTA:
            reasons.append(
                f"max |delta| {max_absolute:.3e} exceeds the equivalence bound "
                f"{EQUIVALENCE_MAX_ABS_DELTA:.0e} but changes no rounded score"
            )
            status = STATUS_MINOR

    if not reasons:
        reasons.append(
            "identical tokenization, no rounded-score change, no bucket change"
        )
    return {"status": status, "reasons": reasons}


def overall_verdict(a1_vs_a3, a3_vs_a2, a1_vs_a2):
    """Translate the three comparisons into the predeclared next investigation.

    The decision rules are fixed in advance:

    * token ids differ anywhere -> tokenizer invocation semantics come first;
    * A1 != A3                  -> the scorer's argument/inference path;
    * A3 != A2                  -> batching and padding numerics;
    * all equivalent            -> the scorer is exonerated, look at the
      environment (dependency versions and checkpoint identity).
    """
    tokenization_broken = any(
        block["token_ids"] is not None and block["token_ids"]["mismatch_count"] > 0
        for block in (a1_vs_a3, a3_vs_a2, a1_vs_a2)
    )
    argument_path = a1_vs_a3["classification"]["status"] == STATUS_MATERIAL
    batching = a3_vs_a2["classification"]["status"] == STATUS_MATERIAL

    if tokenization_broken:
        headline = "TOKENIZATION_DIFFERS"
        conclusion = (
            "Token input_ids differ between arms inside a single environment. "
            "Investigate tokenizer invocation semantics before interpreting any "
            "score delta: the arms are not scoring the same input."
        )
    elif argument_path and batching:
        headline = "ARGUMENT_PATH_AND_BATCHING_IMPLICATED"
        conclusion = (
            "Both the scorer's argument/inference path (A1 vs A3) and its "
            "batching/padding behaviour (A3 vs A2) differ materially."
        )
    elif argument_path:
        headline = "ARGUMENT_PATH_IMPLICATED"
        conclusion = (
            "A1 differs materially from A3 with identical tokenization. The "
            "scorer's argument/inference path is implicated: the candidates are "
            "attention_mask/token_type_ids being passed and the explicit "
            "max_length, not batching."
        )
    elif batching:
        headline = "BATCHING_IMPLICATED"
        conclusion = (
            "A3 differs materially from A2 while A1 and A3 agree. Batching and "
            "padding numerics are implicated."
        )
    else:
        headline = "SCORER_EXONERATED"
        conclusion = (
            "A1 ~ A3 ~ A2 within the predeclared bands. The current scorer "
            "implementation cannot explain the Gate 1 discrepancy; the next "
            "investigation is dependency/environment drift and checkpoint "
            "identity, not this repository's scoring code."
        )
    return {
        "headline": headline,
        "conclusion": conclusion,
        "a1_vs_a3": a1_vs_a3["classification"]["status"],
        "a3_vs_a2": a3_vs_a2["classification"]["status"],
        "a1_vs_a2": a1_vs_a2["classification"]["status"],
    }


def token_type_id_assessment(type_vocab_size, emitted, unique_values):
    """Judge whether passing token_type_ids can change anything.

    Wang's released ``utils.py:57`` calls ``model(inputs["input_ids"])`` and so
    never passes token_type_ids; this repository calls ``model(**inputs)`` and
    passes whatever the tokenizer emits. That difference is only observable if
    the tokenizer actually emits token_type_ids, those ids are not all zero
    (zeros are what the model defaults to when the argument is absent), *and*
    the model instantiates a token-type embedding at all.

    A non-zero ``type_vocab_size`` on its own proves nothing, and neither does
    the mere presence of the key. All three facts are required, so all three
    are reported.
    """
    values = sorted({int(v) for v in (unique_values or [])})
    size = None if type_vocab_size is None else int(type_vocab_size)

    if not emitted:
        return {
            "type_vocab_size": size,
            "token_type_ids_emitted": False,
            "unique_token_type_ids": [],
            "can_differ_from_wang": False,
            "note": (
                "The tokenizer emits no token_type_ids, so model(**inputs) passes "
                "none and is identical to Wang's model(inputs['input_ids'])."
            ),
        }

    if values in ([], [0]):
        return {
            "type_vocab_size": size,
            "token_type_ids_emitted": True,
            "unique_token_type_ids": values,
            "can_differ_from_wang": False,
            "note": (
                "token_type_ids are emitted but are all zero, which is exactly "
                "what the model assumes when the argument is omitted. Passing "
                "them is a no-op relative to Wang."
            ),
        }

    if size == 0:
        return {
            "type_vocab_size": size,
            "token_type_ids_emitted": True,
            "unique_token_type_ids": values,
            "can_differ_from_wang": False,
            "note": (
                f"token_type_ids carry non-zero values {values}, but "
                "type_vocab_size is 0, so no token-type embedding is "
                "instantiated and the argument is ignored. A1 vs A3 is the "
                "empirical confirmation of this."
            ),
        }

    return {
        "type_vocab_size": size,
        "token_type_ids_emitted": True,
        "unique_token_type_ids": values,
        "can_differ_from_wang": True,
        "note": (
            f"token_type_ids carry non-zero values {values} and type_vocab_size "
            f"is {size}, so a token-type embedding exists and Wang's implicit "
            "all-zero default is not equivalent to what this repository passes. "
            "This is a candidate cause; A1 vs A3 measures its actual size."
        ),
    }
