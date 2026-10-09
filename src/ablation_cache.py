"""Full-coverage held-out NLI cache for the preregistered ablation (§5).

``docs/ablation_preregistration.md`` §5 approves one inference pass whose only
purpose is a complete replay surface: every span of every document in the first
``max_docs`` positions of every held-out subclaim's evidence list, scored with
the frozen run's exact protocol, in a NEW derived cache. The canonical held-out
cache is never modified, and rows it already holds are never re-scored.

This module holds the pieces that do not need torch -- coverage enumeration,
key-presence checks, row-preservation verification, the environment gate and
the manifest -- plus ``complete_cache``, the ordered pipeline, which takes the
model-dependent steps as injected callables. The script wires the real ones.

What it deliberately cannot see: no detector, no metric, and never the VALUES
of newly scored rows. Coverage is checked by key presence; preservation compares
only rows that were already in the source. The manifest records counts, hashes,
identity and status, nothing about score distributions.
"""

import hashlib
import sqlite3
import stat
from pathlib import Path

from src.cache_completion import (
    bind_compatibility_to_source,
    cache_row_count,
    discard_invalid_derived_cache,
    prepare_derived_cache,
    sha256_file,
)
from src.score_compatibility import production_cache_key

PROTOCOL = "docs/ablation_preregistration.md §5"
MAX_DOCS = 10
SEGMENT_LENGTH = 400
OVERLAP_LENGTH = 100
BATCH_SIZE = 1

STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"


class CompletionAborted(RuntimeError):
    """A precondition failed before any row was written to a derived cache."""


# ------------------------------------------------------------------ coverage

def enumerate_coverage(records, split_text, *, model_name, score_version,
                       max_docs=MAX_DOCS):
    """Every (span, subclaim) pair the full replay surface needs, in fixed order.

    Order: records as given, subclaims in order, document positions 1..max_docs,
    spans in ``split_text`` order. Independent of any detector: a cell's
    stopping behaviour plays no part. Keys are deduplicated in first-seen
    order, and ``coverage_sha256`` fingerprints that ordered key list.
    """
    keys, pairs, seen = [], [], set()
    subclaims = documents = spans = empty_documents = 0
    for record in records:
        for subclaim in record.subclaims:
            subclaims += 1
            for document in subclaim.documents[:max_docs]:
                documents += 1
                segments = split_text(
                    document.page_content,
                    segment_length=SEGMENT_LENGTH,
                    overlap_length=OVERLAP_LENGTH,
                )
                if not segments:
                    empty_documents += 1
                for segment in segments:
                    spans += 1
                    key = production_cache_key(model_name, score_version, segment, subclaim.text)
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
                        pairs.append((segment, subclaim.text))
    digest = hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest()
    return {
        "keys": keys,
        "pairs": pairs,
        "counts": {
            "records": len(records),
            "subclaims": subclaims,
            "document_occurrences": documents,
            "documents_without_spans": empty_documents,
            "span_occurrences": spans,
            "unique_pairs": len(keys),
        },
        "coverage_sha256": digest,
    }


def present_keys(cache_path, keys):
    """Which keys a cache holds. Read-only, and reads keys only, never scores."""
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        found = set()
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), 500):
            chunk = unique[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            found.update(
                row[0] for row in connection.execute(
                    f"SELECT cache_key FROM nli_scores WHERE cache_key IN ({placeholders})",
                    chunk,
                )
            )
        return found
    finally:
        connection.close()


def verify_preserved_rows(source, derived):
    """Every source row must exist in the derived cache, byte-for-byte equal.

    Compares only rows the source already had. Rows added by completion are
    counted, never read.
    """
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        source_rows = src.execute(
            "SELECT cache_key, model_name, score_version, score FROM nli_scores "
            "ORDER BY cache_key"
        ).fetchall()
    finally:
        src.close()
    dst = sqlite3.connect(f"file:{derived}?mode=ro", uri=True)
    try:
        changed = missing = 0
        for start in range(0, len(source_rows), 500):
            chunk = source_rows[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            stored = {
                row[0]: row for row in dst.execute(
                    "SELECT cache_key, model_name, score_version, score FROM nli_scores "
                    f"WHERE cache_key IN ({placeholders})",
                    [row[0] for row in chunk],
                )
            }
            for row in chunk:
                other = stored.get(row[0])
                if other is None:
                    missing += 1
                elif tuple(other) != tuple(row):
                    changed += 1
        derived_rows = int(dst.execute("SELECT COUNT(*) FROM nli_scores").fetchone()[0])
    finally:
        dst.close()
    return {
        "source_rows": len(source_rows),
        "derived_rows": derived_rows,
        "source_rows_missing_from_derived": missing,
        "source_rows_changed_in_derived": changed,
        "preserved": missing == 0 and changed == 0,
    }


# ------------------------------------------------------------------ environment

def _dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# (field in the live snapshot, expected value or a path into the frozen run's
# recorded environment). A field that cannot be read fails: unverifiable is not
# a pass.
def environment_expectations(frozen_environment, *, revision, score_version):
    def recorded(path):
        return _dig(frozen_environment, path)

    return {
        "checkpoint_identity.resolved_revision": revision,
        "checkpoint_identity.model_config_commit_hash": revision,
        "checkpoint_identity.tokenizer_commit_hash": revision,
        "model.dtype": "torch.float16",
        "model.training_mode": False,
        "device_placement.selected_device": "mps",
        "device_placement.matches": True,
        "tokenizer.model_max_length": recorded("tokenizer.model_max_length"),
        "tokenizer.class": recorded("tokenizer.class"),
        "model.class": recorded("model.class"),
        "score_version": score_version,
        "tokenizer_emission_probe.input_ids_identical": True,
        "libraries.torch": recorded("libraries.torch"),
        "libraries.transformers": recorded("libraries.transformers"),
        "libraries.tokenizers": recorded("libraries.tokenizers"),
    }


def check_environment(observed, expectations):
    problems = []
    for field, expected in expectations.items():
        actual = _dig(observed, field)
        if expected is None:
            problems.append(f"{field}: the frozen run's reference does not record it")
        elif actual is None:
            problems.append(f"{field}: not observable in this environment")
        elif actual != expected:
            problems.append(f"{field}: observed {actual!r}, required {expected!r}")
    return problems


# ------------------------------------------------------------------ pipeline

def freeze_file(path):
    """Make the completed cache read-only for everyone. Returns the new mode."""
    path = Path(path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return oct(stat.S_IMODE(path.stat().st_mode))


def complete_cache(*, source, destination, source_sha256, coverage,
                   run_probe, open_scorer, environment, overwrite=False):
    """Probe, copy, bind, score the missing pairs, verify, freeze. In that order.

    ``run_probe()`` returns the 398-pair compatibility report against
    ``source``. ``open_scorer(path)`` returns an object with ``score_pairs`` and
    ``close`` writing to ``path``. Neither is called before the steps that
    must precede it, and nothing is written anywhere until the probe has passed
    and the copy is faithful and bound.

    Returns the manifest. Raises CompletionAborted before any write when a
    precondition fails; returns an ``incomplete`` manifest (and does not
    freeze) when scoring did not achieve full coverage.
    """
    source = Path(source)
    destination = Path(destination)
    observed_sha = sha256_file(source)
    if observed_sha != source_sha256:
        raise CompletionAborted(
            f"source cache digest {observed_sha} is not the canonical {source_sha256}"
        )
    if destination.exists() and not overwrite:
        raise CompletionAborted(f"derived cache {destination} already exists")

    keys = coverage["keys"]
    present_before = present_keys(source, keys)
    missing_before = [i for i, k in enumerate(keys) if k not in present_before]

    compatibility = run_probe()
    if compatibility.get("score_compatibility_established") is not True:
        raise CompletionAborted(
            "398-pair score compatibility was not established; nothing written. "
            f"{compatibility.get('verdict_reason')}"
        )

    identity = prepare_derived_cache(source, destination, overwrite=overwrite)
    if identity.get("copy_faithful") is not True:
        discard_invalid_derived_cache(identity)
        raise CompletionAborted(f"derived copy not faithful: {identity.get('copy_faithful_note')}")
    binding = bind_compatibility_to_source(compatibility, identity)
    if binding["bound"] is not True:
        discard_invalid_derived_cache(identity)
        raise CompletionAborted(binding["message"])

    to_score = [coverage["pairs"][i] for i in missing_before]
    scorer = open_scorer(str(destination))
    try:
        if to_score:
            # Return values are discarded: completion never looks at new scores.
            scorer.score_pairs(to_score, use_cache=True, write_cache=True,
                               show_progress=True, description="held-out completion")
    finally:
        scorer.close()

    present_after = present_keys(destination, keys)
    missing_after = [keys[i] for i in range(len(keys)) if keys[i] not in present_after]
    preserved = verify_preserved_rows(source, destination)
    source_after = sha256_file(source)
    rows_after = cache_row_count(destination)
    added = rows_after - identity["source_rows"]

    complete = (
        not missing_after
        and preserved["preserved"]
        and source_after == source_sha256
        and added == len(missing_before)
    )
    frozen_mode = freeze_file(destination) if complete else None
    return {
        "protocol": PROTOCOL,
        "status": STATUS_COMPLETE if complete else STATUS_INCOMPLETE,
        "source_cache": str(source),
        "source_sha256": source_sha256,
        "source_sha256_after": source_after,
        "source_unchanged": source_after == source_sha256,
        "derived_cache": str(destination),
        "derived_sha256": sha256_file(destination),
        "derived_frozen_mode": frozen_mode,
        "source_row_count": identity["source_rows"],
        "resulting_row_count": rows_after,
        "added_rows": added,
        "coverage": {
            **coverage["counts"],
            "coverage_sha256": coverage["coverage_sha256"],
            "expected_pairs": len(keys),
            "available_before": len(keys) - len(missing_before),
            "missing_before": len(missing_before),
            "missing_after": len(missing_after),
            "missing_after_keys": missing_after[:50],
        },
        "preserved_rows": preserved,
        "compatibility": {
            k: compatibility.get(k) for k in (
                "score_compatibility_established", "pairs_probed", "exact_raw_matches",
                "raw_mismatches", "source_sha256_after_probe", "verdict_reason",
            )
        },
        "source_binding": binding,
        "copy": {k: identity.get(k) for k in (
            "copy_faithful", "source_sha256_before", "destination_sha256_after_copy",
            "destination_rows_after_copy",
        )},
        "environment": environment,
        "inspected_only": [
            "compatibility", "pair counts", "missing coverage", "row counts",
            "environment", "hashes",
        ],
        "new_score_values_inspected": False,
        "detectors_run": False,
    }
