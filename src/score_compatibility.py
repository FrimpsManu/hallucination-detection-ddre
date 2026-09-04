"""Pre-write 398-pair score-compatibility probe for NLI cache completion.

The reference bundle can only ever report ``checkpoint_identity_established =
false``: the formal v2 batch-1 Gate run did not record a resolved Hugging Face
revision, so no artifact establishes which checkpoint produced the cached
scores. That limitation is historical and cannot be repaired retrospectively.

But it can be *bounded empirically*. If the currently pinned checkpoint and
runtime reproduce, exactly, every score the formal v2 cache already holds for a
fixed set of pairs, then rows added now are numerically compatible with rows
added then -- whatever revision string was in play. That is a different and
weaker claim than checkpoint identity, and it is reported under a different
name so the two are never conflated:

    checkpoint_identity_established:  false   (historical, unrepairable)
    score_compatibility_established:  true/false  (measured, right now)

The sentinel set is the 398 released NBC pairs (199 factual + 199 nonfactual).
They were fixed long before this analysis, they are already part of the formal
v2 scoring path, and they are the pairs the histograms are built from, so they
are not selected to make the probe pass.

Two of the 398 released pairs are exact duplicates of two others, so the set
collapses to 396 distinct cache keys. Comparison is therefore done **per pair**,
not per key: all 398 are probed, and a duplicated pair is satisfied by the one
row its shared key names. Counting distinct keys instead would report 396 and
fail a perfectly compatible cache.

Protocol
--------
For each of the 398 pairs:

1. Build the exact production v2 cache key -- SHA-256 over the same
   ``model_name \\0 SCORE_VERSION \\0 premise \\0 hypothesis`` payload
   ``EntailmentScorer._cache_key`` uses.
2. Read the stored raw score from the ORIGINAL formal cache, opened READ-ONLY.
3. Recompute the pair with the already provenance-gated, revision-pinned model
   at batch size 1, through the ordinary production scorer, with the cache
   read AND the cache write both disabled.
4. Compare fresh against stored.

The source digest is taken before step 1 and again after step 3 has finished
for every pair and the scratch scorer is closed. The two must be equal, or
compatibility is not established -- a formal artifact that changed while it was
being read cannot certify anything.

Equality is **exact float equality**. The provenance guard has already required
the same pinned model, dtype, device, torch/transformers/tokenizers versions,
batch size 1, and the same v2 extraction and scaling path, so anything less
than bit-equality is a real difference rather than tolerable noise. Rounded and
bucketed agreement are computed and reported, but they are DIAGNOSTIC ONLY:
they never substitute for exact equality in the verdict, because two scores can
share a bucket and still be different numbers.

The probe performs 398 forward passes, which is the cost of the evidence. It
performs ZERO cache writes: the recompute runs against a scratch database with
``write_cache=False``, and the scratch row count is asserted to be zero
afterwards and reported. The source cache is never opened for writing.

Standard library only at module scope, so the comparison and verdict logic is
testable without torch or a model.
"""

import hashlib
import sqlite3

# The released NBC evidence sets. Fixed before this analysis.
EXPECTED_POSITIVE_PAIRS = 199
EXPECTED_NEGATIVE_PAIRS = 199
EXPECTED_PAIR_COUNT = EXPECTED_POSITIVE_PAIRS + EXPECTED_NEGATIVE_PAIRS

# Wang's released inference scores one pair at a time.
PROBE_BATCH_SIZE = 1

# Enough to see the shape of a failure without dumping 398 rows into a report.
MAX_REPORTED_MISMATCHES = 5


def production_cache_key(model_name, score_version, premise, hypothesis):
    """The exact key ``EntailmentScorer._cache_key`` computes.

    Kept as a separate implementation deliberately: a probe that asked the
    scorer for its own key could not detect a key-construction change, and the
    key is what decides whether a stored row is the row we think it is. The
    equality of the two constructions is pinned by test.
    """
    payload = "\0".join(
        [str(model_name), str(score_version), str(premise), str(hypothesis)]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_stored_scores(cache_path, keys):
    """Read raw scores for ``keys`` from a cache opened READ-ONLY.

    ``mode=ro`` is not a convention here, it is the mechanism: SQLite refuses a
    write on this connection, so the formal artifact cannot be modified by the
    probe even by accident.
    """
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        found = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), 500):
            chunk = unique[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                "SELECT cache_key, score FROM nli_scores "
                f"WHERE cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            found.update({key: float(score) for key, score in rows})
        return found
    finally:
        connection.close()


def source_integrity_check(source_cache, sha256_before, sha256_after):
    """Compare the source digest taken before the probe against the one after.

    The "after" digest must be taken once every forward pass has run and the
    scratch scorer is closed. A digest read before the work has happened proves
    nothing about the work.
    """
    unchanged = (
        sha256_before is not None
        and sha256_after is not None
        and sha256_before == sha256_after
    )
    return {
        "source_cache": str(source_cache),
        "sha256_before_probe": sha256_before,
        "sha256_after_probe": sha256_after,
        "unchanged": unchanged,
        "message": (
            "The source cache is byte-identical after the 398 forward passes."
            if unchanged
            else (
                "SOURCE CACHE CHANGED DURING THE COMPATIBILITY PROBE. A formal "
                "Gate 1 artifact was modified while it was being read. Do not "
                "extend anything derived from it until this is understood."
            )
            if sha256_before is not None and sha256_after is not None
            else (
                "The source digest could not be measured on both sides of the "
                "probe, so integrity is unverified. Unverified is treated as "
                "failed."
            )
        ),
    }


def recompute_scores(scorer, pairs):
    """Recompute every pair through the production scorer, cache bypassed.

    ``use_cache=False`` is what makes this evidence rather than a tautology: a
    cached read would return the stored value and compare it to itself.
    ``write_cache=False`` is what keeps the probe read-only.
    """
    return scorer.score_pairs(
        pairs,
        use_cache=False,
        write_cache=False,
        show_progress=True,
        description="398-pair compatibility probe",
    )


def _round_one_decimal(score):
    return round(float(score), 1)


def _nbc_bucket(score):
    """Released NBC_feature.py:34 discretizer, clamped to a valid index."""
    return max(0, min(int(_round_one_decimal(score) / 10.0), 9))


def compare_pair_scores(pairs, polarities, keys, stored, fresh):
    """One entry per pair: present, exact, and the diagnostic agreements."""
    entries = []
    for index, (pair, polarity, key) in enumerate(zip(pairs, polarities, keys)):
        stored_score = stored.get(key)
        fresh_score = float(fresh[index])
        present = stored_score is not None
        entry = {
            "index": index,
            "polarity": polarity,
            "cache_key": key,
            "present_in_source_cache": present,
            "stored_score": stored_score,
            "fresh_score": fresh_score,
            "exact_match": present and float(stored_score) == fresh_score,
            "absolute_delta": None
            if not present
            else abs(float(stored_score) - fresh_score),
            "one_decimal_match": present
            and _round_one_decimal(stored_score) == _round_one_decimal(fresh_score),
            "nbc_bucket_match": present
            and _nbc_bucket(stored_score) == _nbc_bucket(fresh_score),
            "stored_bucket": None if not present else _nbc_bucket(stored_score),
            "fresh_bucket": _nbc_bucket(fresh_score),
            "hypothesis_preview": str(pair[1])[:120],
        }
        entries.append(entry)
    return entries


def compatibility_report(
    entries,
    *,
    source_cache,
    model_name,
    score_version,
    positive_pairs,
    negative_pairs,
    batch_size=PROBE_BATCH_SIZE,
    scratch_cache_rows_after=None,
    source_check=None,
):
    """Aggregate the probe into a verdict the caller may gate on."""
    probed = len(entries)
    present = [entry for entry in entries if entry["present_in_source_cache"]]
    exact = [entry for entry in present if entry["exact_match"]]
    mismatches = [
        entry
        for entry in entries
        if not entry["present_in_source_cache"] or not entry["exact_match"]
    ]
    deltas = [entry["absolute_delta"] for entry in present]

    counts_expected = (
        probed == EXPECTED_PAIR_COUNT
        and positive_pairs == EXPECTED_POSITIVE_PAIRS
        and negative_pairs == EXPECTED_NEGATIVE_PAIRS
    )
    all_present = len(present) == probed
    all_exact = len(exact) == probed
    # Fail closed on the integrity measurements too. An absent measurement is
    # not a passing one: ``None`` here means nobody looked, and a probe that
    # did not look cannot certify what it did not observe. In particular
    # ``scratch_cache_rows_after is None`` must NOT count as "zero writes".
    no_writes = scratch_cache_rows_after == 0
    source_intact = bool(source_check) and source_check.get("unchanged") is True

    established = bool(
        counts_expected
        and all_present
        and all_exact
        and no_writes
        and source_intact
        and probed > 0
    )

    if established:
        reason = (
            f"All {probed} released NBC sentinel pairs are present in the formal "
            "v2 cache and the pinned scorer reproduced every stored raw score "
            "exactly. Rows added now are numerically compatible with the rows "
            "already there. This does NOT establish the historical checkpoint "
            "revision; see checkpoint_identity_established."
        )
    elif not counts_expected:
        reason = (
            f"Expected {EXPECTED_PAIR_COUNT} sentinel pairs "
            f"({EXPECTED_POSITIVE_PAIRS} positive + {EXPECTED_NEGATIVE_PAIRS} "
            f"negative); probed {probed} ({positive_pairs} + {negative_pairs}). "
            "The sentinel set is fixed and must be used whole."
        )
    elif not all_present:
        reason = (
            f"{probed - len(present)} of {probed} sentinel pairs are absent from "
            "the formal v2 cache, so there is nothing to compare them against. "
            "Compatibility is not established."
        )
    elif not all_exact:
        reason = (
            f"{len(present) - len(exact)} of {probed} sentinel pairs differ from "
            "the stored raw score. The pinned scorer does not reproduce the "
            "cached numbers, so extending the cache would mix two scoring "
            "behaviours. Bucket or one-decimal agreement does not substitute."
        )
    elif scratch_cache_rows_after is None:
        reason = (
            "The probe's scratch cache row count was never measured, so it "
            "cannot be shown that the probe wrote nothing. Unmeasured is "
            "treated as failed."
        )
    elif not no_writes:
        reason = (
            f"The probe's scratch cache holds {scratch_cache_rows_after} rows; "
            "it must hold zero. The probe must write nothing."
        )
    elif not source_check:
        reason = (
            "No post-probe source-integrity check was performed, so it cannot "
            "be shown that the formal cache is unchanged. Unmeasured is treated "
            "as failed."
        )
    else:
        reason = source_check.get("message") or (
            "The source cache did not survive the probe unchanged."
        )

    return {
        "probe": "nbc-398-score-compatibility",
        "source_cache": str(source_cache),
        "model_name": model_name,
        "score_version": score_version,
        "batch_size": batch_size,
        "expected_pairs": EXPECTED_PAIR_COUNT,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "pairs_probed": probed,
        "cached_rows_found": len(present),
        "missing_from_source_cache": probed - len(present),
        "exact_raw_matches": len(exact),
        "raw_mismatches": len(present) - len(exact),
        "maximum_absolute_raw_delta": max(deltas) if deltas else None,
        "one_decimal_matches": sum(
            1 for entry in present if entry["one_decimal_match"]
        ),
        "nbc_bucket_matches": sum(1 for entry in present if entry["nbc_bucket_match"]),
        "mismatch_examples": [
            {
                key: entry[key]
                for key in (
                    "index",
                    "polarity",
                    "present_in_source_cache",
                    "stored_score",
                    "fresh_score",
                    "absolute_delta",
                    "stored_bucket",
                    "fresh_bucket",
                    "hypothesis_preview",
                )
            }
            for entry in mismatches[:MAX_REPORTED_MISMATCHES]
        ],
        "cache_writes_performed": 0,
        "scratch_cache_rows_after": scratch_cache_rows_after,
        "source_cache_check": source_check,
        "source_sha256_before_probe": (source_check or {}).get("sha256_before_probe"),
        "source_sha256_after_probe": (source_check or {}).get("sha256_after_probe"),
        "source_cache_unchanged": source_intact,
        "score_compatibility_established": established,
        "verdict_reason": reason,
        "equality_rule": (
            "Exact float equality. The provenance guard already required the "
            "same pinned model, dtype, device, library versions, batch size 1 "
            "and v2 extraction path, so a difference of any size is a real "
            "difference. One-decimal and NBC-bucket agreement are reported as "
            "diagnostics and never substitute for exact equality."
        ),
        "scope_note": (
            "This probe bounds numerical compatibility with the rows already in "
            "the formal v2 cache. It does not, and cannot, establish which "
            "Hugging Face revision produced them."
        ),
    }


def scratch_scorer(
    tokenizer, model, model_name, scratch_path, batch_size=PROBE_BATCH_SIZE
):
    """A production ``EntailmentScorer`` bound to a throwaway database.

    ``EntailmentScorer.__init__`` opens its cache path for writing, so the
    probe must never be handed the formal cache path -- not even with
    ``write_cache=False``. Binding it to a scratch file makes that structural:
    the source cache is only ever opened through ``read_stored_scores``, which
    uses ``mode=ro``.

    Imported lazily so this module stays importable without torch.
    """
    from src.utils import EntailmentScorer

    return EntailmentScorer(
        tokenizer, model, model_name, cache_path=scratch_path, batch_size=batch_size
    )


def run_compatibility_probe(
    *,
    pairs,
    polarities,
    positive_pairs,
    negative_pairs,
    source_cache,
    model_name,
    score_version,
    scorer,
    finalize,
    source_digest,
):
    """Key, read, recompute, finalize, re-hash, compare, report -- in that order.

    The order is the point. ``source_digest`` is called once before anything
    happens and once *after* every one of the 398 forward passes has run and
    ``finalize`` has closed the scratch scorer. A digest taken before the work
    would only restate the file's starting state; taken after, it is evidence
    that reading the formal cache 398 times changed nothing.

    ``finalize`` closes the scratch scorer and returns its row count, so
    "the probe wrote nothing" is a measurement carried into the verdict rather
    than a claim in prose. Both measurements fail closed if absent.
    """
    keys = [
        production_cache_key(model_name, score_version, premise, hypothesis)
        for premise, hypothesis in pairs
    ]
    sha256_before = source_digest()
    stored = read_stored_scores(source_cache, keys)
    fresh = recompute_scores(scorer, pairs)

    # Every forward pass is done; close the scratch scorer, then measure.
    scratch_rows_after = finalize()
    sha256_after = source_digest()

    entries = compare_pair_scores(pairs, polarities, keys, stored, fresh)
    return compatibility_report(
        entries,
        source_cache=source_cache,
        model_name=model_name,
        score_version=score_version,
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        scratch_cache_rows_after=scratch_rows_after,
        source_check=source_integrity_check(
            source_cache, sha256_before, sha256_after
        ),
    )


def format_compatibility(report):
    lines = [
        "-" * 100,
        "PRE-WRITE SCORE-COMPATIBILITY PROBE (398 released NBC sentinel pairs)",
        "-" * 100,
        f"  source cache (read-only):   {report['source_cache']}",
        f"  total pairs:                {report['pairs_probed']} "
        f"({report['positive_pairs']} positive + {report['negative_pairs']} negative)",
        f"  cached rows found:          {report['cached_rows_found']}",
        f"  exact raw matches:          {report['exact_raw_matches']}",
        f"  raw mismatches:             {report['raw_mismatches']}",
        f"  missing from source cache:  {report['missing_from_source_cache']}",
        f"  maximum absolute delta:     {report['maximum_absolute_raw_delta']}",
        f"  one-decimal matches:        {report['one_decimal_matches']}  (diagnostic only)",
        f"  NBC bucket matches:         {report['nbc_bucket_matches']}  (diagnostic only)",
        f"  cache writes performed:     {report['cache_writes_performed']}",
        f"  scratch cache rows after:   {report['scratch_cache_rows_after']}",
        f"  source sha256 before probe: {report['source_sha256_before_probe']}",
        f"  source sha256 after probe:  {report['source_sha256_after_probe']}",
        f"  source cache unchanged:     {report['source_cache_unchanged']}",
    ]
    if report["mismatch_examples"]:
        lines.append("  first mismatches:")
        for example in report["mismatch_examples"]:
            lines.append(
                f"    [{example['index']:>3}] {example['polarity']:<9} "
                f"stored={example['stored_score']!r} fresh={example['fresh_score']!r} "
                f"delta={example['absolute_delta']!r} "
                f"bucket {example['stored_bucket']}->{example['fresh_bucket']}"
            )
            lines.append(f"          {example['hypothesis_preview']}")
    lines.append(
        f"  score_compatibility_established: "
        f"{report['score_compatibility_established']}"
    )
    lines.append(f"  {report['verdict_reason']}")
    if not report["score_compatibility_established"]:
        lines.append(
            "  ABORTING: no derived cache created, no completion inference "
            "performed, zero cache rows written."
        )
    lines.append("-" * 100)
    return "\n".join(lines)
