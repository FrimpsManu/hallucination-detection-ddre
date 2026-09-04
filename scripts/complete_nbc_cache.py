"""Complete the NLI cache for the three incomplete CM=14/CFA=24 placements.

The formal NBC sensitivity run against the corrected Wang-fidelity v2 cache
completed 97 of 100 CM=14/CFA=24 placements. Three could not be evaluated,
because the recorded Gate 1 run never consumed the documents those histograms
cause the policy to retrieve:

    (positive_bin=0, negative_bin=4)
    (positive_bin=8, negative_bin=4)
    (positive_bin=9, negative_bin=4)

Under the sensitivity analysis's own branch logic that is
INCONCLUSIVE_PARTIAL_CACHE_COVERAGE, not a negative result. This script adds
exactly the missing span scores so the full grid can be evaluated.

What it does
------------
Replays the unmodified ``bse_official`` loop for those three placements under
CM=14/CFA=24 only, using the ordinary production ``EntailmentScorer`` at batch
size 1, with the ordinary read-through / write-through cache mechanism. A span
already cached is reused; only a span that is genuinely missing is evaluated and
written back.

The formal v2 cache is an immutable completed Gate 1 artifact and is NEVER
written to. It is copied to a derived cache, and only the copy is extended. The
source SHA-256 is recorded before the copy and recomputed afterwards to prove
the original is byte-identical. The sensitivity rerun uses the derived cache.

Before the derived cache is opened for writes and before any inference, a
provenance guard compares this environment against previously recorded formal
provenance: model name, resolved checkpoint revision, SCORE_VERSION, Wang source
commit, batch size, truncation equivalence, dtype, device and the
score-affecting library versions. On any mismatch the run aborts having
performed zero inference and written zero cache rows. Unverifiable is treated as
failed.

The reference is a bundle of two artifacts, because no single one records
everything. ``--formal-provenance`` is the final formal v2 batch-1 Gate report
and is authoritative wherever it records a field. ``--checkpoint-provenance``
supplements only the fields it does not record at all -- the resolved Hugging
Face revision, the model/tokenizer commit hashes, the dtype, the GPU identity,
and the tokenizers/sentencepiece versions. Fields recorded by both must agree.

The presence of a resolved revision is a STATIC check: without one the run
aborts before ``from_pretrained`` rather than falling back to moving Hugging
Face main. Both the model and the tokenizer are then pinned to it.

The formal v2 Gate report does not itself record a revision, so pinning makes
this completion internally consistent and reproducible but does NOT prove the v2
cache rows were produced at that revision. The run reports that limitation
explicitly rather than claiming an identity no artifact establishes.

Because that identity cannot be established, it is bounded empirically instead.
Before the derived cache is created, a score-compatibility probe rescores all
398 released NBC sentinel pairs with the pinned model at batch size 1, reading
the source cache READ-ONLY and writing nothing, and requires every stored raw
score to be reproduced by EXACT float equality. Rounded and bucketed agreement
are reported as diagnostics and never substitute. If any sentinel is absent or
any score differs, the run aborts with no derived cache and no completion
inference. The result is reported separately from checkpoint identity:

    checkpoint_identity_established:  false        (historical, unrepairable)
    score_compatibility_established:  true/false   (measured, right now)

Immediately after the copy and before the scorer is constructed, the derived
cache is required to be a byte-faithful copy of the source. A copy that does not
match the source in digest and row count is never extended.

The work has to be interleaved rather than precomputed: retrieval is adaptive,
so which document comes next depends on the scores of the documents already
consumed, and the required spans cannot be enumerated in advance.

What it does not do
-------------------
It computes no metric, reaches no verdict, and reinterprets nothing. It does not
evaluate CM=28/CFA=96, whose retrieval path would pull in spans this step was not
asked for. It changes no baseline behaviour, histogram, cost, metric, threshold,
tolerance, or published reference value.

After it finishes, rerun ``scripts/diagnose_nbc_sensitivity.py`` unchanged
against the expanded cache.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_core import BSEDetector  # noqa: E402
from src.cache_completion import (  # noqa: E402
    INCOMPLETE_PRIMARY_PLACEMENTS,
    PRIMARY_CONFIGURATION,
    WANG_BATCH_SIZE,
    UnsafeCacheTarget,
    build_counting_scorer,
    completion_report,
    parse_placement,
    placement_label,
    prepare_derived_cache,
    verification_summary,
    verify_source_unchanged,
)
from src.provenance_guard import (  # noqa: E402
    OFFICIAL_NLI_MODEL,
    check_reference_bundle,
    check_runtime_preconditions,
    check_static_preconditions,
    extract_reference,
    format_bundle,
    format_guard,
    guard_report,
    merge_reference,
)
from src.cached_document_scores import (  # noqa: E402
    CachedDocumentScorer,
    MissingDocumentScore,
)
from src.nbc_sensitivity import combination_histograms  # noqa: E402
from src.reproduction_gate import PUBLISHED_TABLE1  # noqa: E402
from src.score_compatibility import (  # noqa: E402
    PROBE_BATCH_SIZE,
    format_compatibility,
    run_compatibility_probe,
    scratch_scorer,
)

OFFICIAL_MODEL = OFFICIAL_NLI_MODEL
DEFAULT_SOURCE_CACHE = "results/wang_nli_cache_fidelity_v2_batch1.sqlite"
DEFAULT_DERIVED_CACHE = (
    "results/wang_nli_cache_fidelity_v2_batch1_nbc_complete.sqlite"
)
DEFAULT_OUTPUT = "results/diagnostics/nbc_cache_completion.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add the missing NLI span scores required by the incomplete "
            "CM=14/CFA=24 NBC sensitivity placements. Cache completion only."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument(
        "--source-cache",
        default=DEFAULT_SOURCE_CACHE,
        help=(
            "The corrected Wang-fidelity v2 cache. Treated as an immutable "
            "completed Gate 1 artifact: read and hashed, never written to."
        ),
    )
    parser.add_argument(
        "--output-cache",
        default=DEFAULT_DERIVED_CACHE,
        help=(
            "The derived cache to create and extend. The sensitivity rerun must "
            "use this file, not the source."
        ),
    )
    parser.add_argument(
        "--formal-provenance",
        required=True,
        metavar="PATH",
        help=(
            "The final formal v2 batch-1 Gate report. AUTHORITATIVE wherever it "
            "records a field: corrected score version, Wang source commit, batch "
            "size, model name, truncation precondition, and the runtime/library "
            "fields it actually records."
        ),
    )
    parser.add_argument(
        "--checkpoint-provenance",
        default=None,
        metavar="PATH",
        help=(
            "A checkpoint-provenance artifact (the Step 2 scoring-path "
            "diagnostic). SUPPLEMENTS only the fields the formal Gate report "
            "does not record: resolved Hugging Face revision, model/tokenizer "
            "commit hashes, model dtype, GPU identity, tokenizers and "
            "sentencepiece versions. Fields recorded by both must agree."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace an existing derived cache instead of refusing.",
    )
    parser.add_argument(
        "--unsafe-allow-in-place",
        action="store_true",
        help=(
            "DEBUG ONLY. Permit source == destination, modifying the source "
            "cache. Never used by the documented formal command."
        ),
    )
    parser.add_argument(
        "--unsafe-allow-local-checkpoint",
        action="store_true",
        help=(
            "DEBUG ONLY. Downgrade the checkpoint-identity checks (model name "
            "and revision) to advisory, for exercising the tool against a local "
            "checkpoint. Score version, batch size, Wang commit, truncation, "
            "dtype, device and library versions still gate. Never used by the "
            "documented formal command."
        ),
    )
    parser.add_argument(
        "--placement",
        action="append",
        default=None,
        metavar="POS,NEG",
        help=(
            "A placement to complete, e.g. --placement 0,4. Repeatable. "
            "Defaults to the three placements the formal v2 run reported "
            "incomplete."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the placements and cache state, then exit without a model.",
    )
    return parser.parse_args()


def resolve_placements(args):
    if not args.placement:
        return [tuple(p) for p in INCOMPLETE_PRIMARY_PLACEMENTS]
    return [parse_placement(text) for text in args.placement]


def primary_detector(positive_bin, negative_bin):
    """bse_official under the published primary costs. Unmodified."""
    reference = PUBLISHED_TABLE1[PRIMARY_CONFIGURATION]
    pos_hist, neg_hist = combination_histograms(positive_bin, negative_bin)
    detector = BSEDetector(
        pos_hist,
        neg_hist,
        mode="official",
        p0=0.5,
        c_miss=reference["c_miss"],
        c_false_alarm=reference["c_false_alarm"],
        c_retrieve=1,
        max_docs=10,
    )
    return detector, pos_hist, neg_hist


def complete_placement(records, scorer, positive_bin, negative_bin):
    """Replay one placement, filling cache misses as they arise."""
    from tqdm import tqdm

    detector, pos_hist, neg_hist = primary_detector(positive_bin, negative_bin)
    scorer.reset_counts()
    label = placement_label(positive_bin, negative_bin)
    for record in tqdm(records, desc=f"completing {label}", unit="sentence"):
        detector.detect_sentence(record, scorer)
    entry = {
        "placement": label,
        "positive_bin": positive_bin,
        "negative_bin": negative_bin,
        "positive_histogram": pos_hist,
        "negative_histogram": neg_hist,
        **scorer._counts(),
    }
    return entry


def verify_placement(records, cache_path, model_name, score_version, split_text,
                     positive_bin, negative_bin):
    """Confirm the placement now completes with a READ-ONLY cache replay.

    Uses the same read-only scorer the sensitivity analysis uses, so a pass here
    is evidence that the rerun will complete rather than a claim about it.
    """
    scorer = CachedDocumentScorer(cache_path, model_name, score_version, split_text)
    detector, _, _ = primary_detector(positive_bin, negative_bin)
    entry = {
        "placement": placement_label(positive_bin, negative_bin),
        "positive_bin": positive_bin,
        "negative_bin": negative_bin,
        "complete": False,
        "reason": None,
    }
    try:
        for record in records:
            detector.detect_sentence(record, scorer)
        entry["complete"] = True
    except MissingDocumentScore as exc:
        entry["reason"] = str(exc)
    finally:
        entry["span_lookups"] = scorer.lookups
        entry["span_misses"] = scorer.misses
        scorer.close()
    return entry


def load_reference(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def wang_source_commit(data_root):
    """The released-artifact commit recorded beside the local Wang data."""
    source = Path(data_root) / "SOURCE.json"
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("source_commit")
    except Exception:  # noqa: BLE001 - unreadable is as disqualifying as absent
        return None


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    placements = resolve_placements(args)
    source_cache = Path(args.source_cache)
    derived_cache = Path(args.output_cache)

    print("=" * 100)
    print("NBC SENSITIVITY CACHE COMPLETION")
    print("=" * 100)
    print(f"  configuration completed: {PRIMARY_CONFIGURATION} only")
    print(f"  batch size:              {WANG_BATCH_SIZE} (released Wang semantics)")
    print(f"  source cache (immutable): {source_cache}")
    print(f"  derived cache (written):  {derived_cache}")
    print(f"  formal provenance:        {args.formal_provenance}")
    print(f"  checkpoint provenance:    {args.checkpoint_provenance}")
    print(f"  placements: {', '.join(placement_label(*p) for p in placements)}")
    print("  cached spans are reused; only genuinely missing spans are evaluated")
    if args.unsafe_allow_in_place:
        print("  *** --unsafe-allow-in-place: the SOURCE cache may be modified ***")
    if args.unsafe_allow_local_checkpoint:
        print("  *** --unsafe-allow-local-checkpoint: checkpoint-identity "
              "checks are advisory ***")

    if not source_cache.exists():
        print(f"\nERROR: source cache not found at {source_cache}")
        return 1

    if args.dry_run:
        print("\n--dry-run: placements listed only. No model loaded, nothing scored.")
        print("=" * 100)
        return 0

    # ---------------------------------------------------------------- guard
    # Everything below runs BEFORE the derived cache is created and BEFORE any
    # inference. An abort here leaves zero rows written and no derived cache.
    bundle = merge_reference(
        extract_reference(load_reference(args.formal_provenance)),
        extract_reference(load_reference(args.checkpoint_provenance))
        if args.checkpoint_provenance
        else None,
        formal_path=args.formal_provenance,
        checkpoint_path=args.checkpoint_provenance,
    )
    reference = bundle["reference"]
    print()
    print(format_bundle(bundle))

    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_sentence_records

    static_checks = check_static_preconditions(
        reference,
        score_version=SCORE_VERSION,
        batch_size=WANG_BATCH_SIZE,
        wang_source_commit=wang_source_commit(args.data_root),
        model_name=args.model_name,
        allow_local_checkpoint=args.unsafe_allow_local_checkpoint,
    ) + check_reference_bundle(bundle)
    static = guard_report(static_checks, reference_path=args.formal_provenance)
    if not static["passed"]:
        print()
        print(format_guard(static))
        _write_aborted(output_path, args, placements, static, stage="static",
                       bundle=bundle)
        return 1

    # Guaranteed non-None by the resolved_revision_present static check above,
    # except under the local-checkpoint debug override where there is no Hub
    # revision to pin. There is no fallback to moving Hugging Face main.
    revision = reference.get("resolved_revision")
    print(f"\n  pinning checkpoint revision: {revision!r}")
    if revision is None:
        print("  (unpinned: --unsafe-allow-local-checkpoint, debug only)")

    import torch  # noqa: F401  (imported for the device probe below)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import collect_live_environment

    load_kwargs = {"revision": revision} if revision else {}
    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, **load_kwargs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    observed = collect_live_environment(
        model_name=args.model_name, model=model, tokenizer=tokenizer,
        repo_root=PROJECT_ROOT, score_version=SCORE_VERSION,
    )
    runtime_checks = check_runtime_preconditions(
        reference,
        observed,
        allow_local_checkpoint=args.unsafe_allow_local_checkpoint,
    )
    guard = guard_report(
        static_checks + runtime_checks, reference_path=args.formal_provenance
    )
    guard["reference_bundle"] = bundle
    print()
    print(format_guard(guard))
    if not guard["passed"]:
        _write_aborted(output_path, args, placements, guard, stage="runtime",
                       environment=observed, bundle=bundle)
        return 1

    # ------------------------------------- 398-pair compatibility probe
    # Step 5. The provenance guard has established that this environment
    # matches what was recorded; this establishes that it reproduces what was
    # *scored*. It runs before the derived cache exists, so a failure leaves
    # nothing behind. It reads the source cache read-only and writes nothing.
    compatibility = _run_compatibility_probe(
        args, tokenizer, model, SCORE_VERSION, source_cache
    )
    print()
    print(format_compatibility(compatibility))
    if not compatibility["score_compatibility_established"]:
        _write_compatibility_abort(
            output_path, args, placements, compatibility, guard, bundle, observed
        )
        return 1

    # ------------------------------------------------------- derived cache
    try:
        identity = prepare_derived_cache(
            source_cache,
            derived_cache,
            allow_in_place=args.unsafe_allow_in_place,
            overwrite=args.overwrite_output,
        )
    except (UnsafeCacheTarget, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}")
        print("Zero inference performed, zero cache rows written.")
        return 1

    print()
    print(f"  source sha256 before: {identity['source_sha256_before']}")
    print(f"  source rows:          {identity['source_rows']}")
    if identity["copied"]:
        print(f"  derived cache created: {identity['destination_cache']}")
    print(f"  copy faithful:         {identity.get('copy_faithful')}")
    if identity.get("warning"):
        print(f"  *** {identity['warning']} ***")

    # HARD PRE-INFERENCE GATE. A derived cache whose digest or row count does
    # not match the source immediately after copying is already wrong before a
    # single row is added, and extending it would build correct new scores on a
    # corrupt base. Refuse before the scorer exists, so zero inference runs and
    # zero rows are written.
    if identity.get("copy_faithful") is not True:
        print()
        print("ABORTED: the derived cache is not a faithful copy of the source.")
        print(f"  {identity.get('copy_faithful_note')}")
        print("  Zero inference performed. Zero cache rows written.")
        _write_copy_abort(output_path, args, placements, identity, guard, bundle)
        return 1

    records = load_sentence_records(args.data_root, strict=True)
    print(f"  sentences: {len(records)}")

    scorer = build_counting_scorer(
        tokenizer, model, args.model_name, str(derived_cache),
        batch_size=WANG_BATCH_SIZE, cache_identity=identity,
    )
    rows_before = scorer.cache_size()
    print(f"  derived cache rows before: {rows_before}")

    per_placement = []
    try:
        for positive_bin, negative_bin in placements:
            print()
            entry = complete_placement(records, scorer, positive_bin, negative_bin)
            per_placement.append(entry)
            print(
                f"  {entry['placement']}: {entry['spans_evaluated_now']} spans "
                f"evaluated, {entry['spans_served_from_cache']} served from cache, "
                f"{entry['documents_scored']} documents"
            )
        rows_after = scorer.cache_size()
    finally:
        scorer.close()

    print(f"\n  derived cache rows after: {rows_after}")

    source_check = verify_source_unchanged(
        source_cache, identity["source_sha256_before"]
    )
    print(f"  source cache unchanged: {source_check['unchanged']}")
    if not source_check["unchanged"]:
        print(f"  *** {source_check['message']} ***")

    print("\nVerifying each placement now completes from the derived cache alone...")
    verification = [
        verify_placement(
            records, str(derived_cache), args.model_name, SCORE_VERSION, split_text,
            positive_bin, negative_bin,
        )
        for positive_bin, negative_bin in placements
    ]

    identity["destination_sha256_after_completion"] = None
    identity["destination_rows_after_completion"] = rows_after
    from src.cache_completion import sha256_file

    identity["destination_sha256_after_completion"] = sha256_file(derived_cache)

    report = completion_report(
        placements, rows_before, rows_after, per_placement, verification,
        cache_identity=identity, source_check=source_check, guard=guard,
        compatibility=compatibility,
    )
    report["environment"] = observed
    report["score_version"] = SCORE_VERSION
    report["pinned_revision"] = revision
    report["reference_bundle"] = bundle
    # Deliberately reported side by side and never merged. The first is a
    # historical fact that cannot be repaired; the second is a measurement made
    # just now. Matching scores do not turn the first one true.
    report["checkpoint_identity_established"] = bundle[
        "checkpoint_identity_established"
    ]
    report["provenance_interpretation"] = (
        "The exact historical Hugging Face revision cannot be established "
        "retrospectively, but the pinned scorer reproduced all "
        f"{compatibility['exact_raw_matches']} of "
        f"{compatibility['pairs_probed']} fixed v2 sentinel scores exactly."
    )
    report["provenance_limitations"] = bundle["limitations"]
    report["dataset"] = {
        "sentences": len(records),
        "subclaims": sum(len(r.subclaims) for r in records),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print()
    print("-" * 100)
    print("CACHE COMPLETION REPORT")
    print("-" * 100)
    print(f"  previously missing span scores:   {report['previously_missing_span_scores']}")
    print(f"  new NLI evaluations performed:    {report['new_nli_evaluations_performed']}")
    print(f"  spans served from existing cache: {report['spans_served_from_existing_cache']}")
    print(f"  derived cache rows before:        {report['cache_rows_before']}")
    print(f"  derived cache rows after:         {report['cache_rows_after']}")
    print(f"  derived cache rows added:         {report['cache_rows_added']}")
    print(f"  accounting consistent:            {report['accounting_consistent']}")
    if not report["accounting_consistent"]:
        print(f"  *** {report['accounting_note']} ***")

    print()
    print("-" * 100)
    print("CACHE IDENTITY")
    print("-" * 100)
    print(f"  source cache:      {identity['source_cache']}")
    print(f"  source sha256:     {identity['source_sha256_before']}")
    print(f"  source rows:       {identity['source_rows']}")
    print(f"  destination cache: {identity['destination_cache']}")
    print(f"  destination sha256 after completion: "
          f"{identity['destination_sha256_after_completion']}")
    print(f"  destination rows after completion:   {rows_after}")
    print(f"  source unchanged:  {source_check['unchanged']}")
    print(f"  copy faithful:     {identity.get('copy_faithful')}")

    print()
    print("-" * 100)
    print("PROVENANCE")
    print("-" * 100)
    print(
        "  checkpoint_identity_established:  "
        f"{bundle['checkpoint_identity_established']}"
    )
    print(
        "  score_compatibility_established:  "
        f"{compatibility['score_compatibility_established']}"
    )
    print(f"  {report['provenance_interpretation']}")
    for limitation in bundle["limitations"]:
        print(f"  - {limitation}")

    print()
    print("-" * 100)
    print("PLACEMENT COMPLETENESS (read-only replay of the derived cache)")
    print("-" * 100)
    for line in verification_summary(verification):
        print(line)
    print(
        f"  all requested placements complete: "
        f"{report['all_requested_placements_complete']}"
    )

    print()
    print(f"Written to {output_path}")
    print(f"  {report['scope_note']}")
    print("=" * 100)
    return 0 if report["run_sound"] else 1


def _run_compatibility_probe(args, tokenizer, model, score_version, source_cache):
    """Score the 398 fixed NBC sentinel pairs and compare against the source.

    The scratch database lives in a temporary directory that is removed
    afterwards, so the only databases this function can touch are the source
    (read-only) and a file that no longer exists when it returns.
    """
    import tempfile

    from src.cache_completion import sha256_file, verify_source_unchanged
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.data_root, per_class=None)
    pairs = [(item["premise"], item["hypothesis"]) for item in positive]
    pairs += [(item["premise"], item["hypothesis"]) for item in negative]
    polarities = ["positive"] * len(positive) + ["negative"] * len(negative)

    print()
    print(
        f"Score-compatibility probe: {len(pairs)} released NBC pairs "
        f"({len(positive)} positive + {len(negative)} negative), batch size "
        f"{PROBE_BATCH_SIZE}, cache read and cache write both disabled."
    )
    source_sha_before = sha256_file(source_cache)

    with tempfile.TemporaryDirectory(prefix="compat-probe-") as scratch_dir:
        scratch_path = str(Path(scratch_dir) / "scratch.sqlite")
        scorer = scratch_scorer(
            tokenizer, model, args.model_name, scratch_path,
            batch_size=PROBE_BATCH_SIZE,
        )
        try:
            report = run_compatibility_probe(
                pairs=pairs,
                polarities=polarities,
                positive_pairs=len(positive),
                negative_pairs=len(negative),
                source_cache=str(source_cache),
                model_name=args.model_name,
                score_version=score_version,
                scorer=scorer,
                scratch_row_count=scorer.cache_size,
                source_check=verify_source_unchanged(source_cache, source_sha_before),
            )
        finally:
            scorer.close()
    return report


def _write_compatibility_abort(output_path, args, placements, compatibility, guard,
                               bundle, environment):
    """Record an abort on score incompatibility: nothing created, nothing written."""
    report = {
        "analysis": "nbc-cache-completion",
        "aborted": True,
        "abort_stage": "score-compatibility",
        "abort_reason": "the pinned scorer does not reproduce the cached scores",
        "provenance_guard": guard,
        "reference_bundle": bundle,
        "checkpoint_identity_established": bundle["checkpoint_identity_established"],
        "score_compatibility": compatibility,
        "score_compatibility_established": False,
        "placements_requested": [list(p) for p in placements],
        "source_cache": args.source_cache,
        "destination_cache": args.output_cache,
        "derived_cache_created": False,
        "new_nli_evaluations_performed": 0,
        "cache_rows_added": 0,
        "run_sound": False,
        "environment": environment,
        "scope_note": (
            "Aborted after the read-only compatibility probe and before the "
            "derived cache was created. The probe's 398 forward passes are the "
            "only inference performed; zero cache rows were written and the "
            "source cache was not touched."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print()
    print("ABORTED: score compatibility not established.")
    print(f"  {compatibility['verdict_reason']}")
    print("  No derived cache created. No completion inference performed. "
          "Zero cache rows written.")
    print(f"\nWritten to {output_path}")
    print("=" * 100)


def _write_aborted(output_path, args, placements, guard, stage, environment=None,
                   bundle=None):
    """Record an aborted run: zero inference, zero rows, no derived cache."""
    report = {
        "analysis": "nbc-cache-completion",
        "aborted": True,
        "abort_stage": stage,
        "abort_reason": "provenance guard failed",
        "provenance_guard": guard,
        "reference_bundle": bundle,
        "placements_requested": [list(p) for p in placements],
        "source_cache": args.source_cache,
        "destination_cache": args.output_cache,
        "derived_cache_created": False,
        "new_nli_evaluations_performed": 0,
        "cache_rows_added": 0,
        "run_sound": False,
        "environment": environment,
        "scope_note": (
            "Aborted before any inference and before the derived cache was "
            "created. Zero cache rows were written and the source cache was "
            "not touched."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print()
    print("ABORTED: provenance guard failed.")
    for message in guard["failure_messages"]:
        print(f"  - {message}")
    print("  Zero inference performed. Zero cache rows written. "
          "No derived cache created.")
    print(f"\nWritten to {output_path}")
    print("=" * 100)


def _write_copy_abort(output_path, args, placements, identity, guard, bundle):
    """Record an abort on an unfaithful copy: zero inference, zero rows."""
    report = {
        "analysis": "nbc-cache-completion",
        "aborted": True,
        "abort_stage": "derived-cache-copy",
        "abort_reason": "derived cache is not a faithful copy of the source",
        "provenance_guard": guard,
        "reference_bundle": bundle,
        "cache_identity": identity,
        "derived_cache_copy_faithful": False,
        "placements_requested": [list(p) for p in placements],
        "source_cache": args.source_cache,
        "destination_cache": args.output_cache,
        "new_nli_evaluations_performed": 0,
        "cache_rows_added": 0,
        "run_sound": False,
        "scope_note": (
            "Aborted after copying and before the scorer was constructed. The "
            "derived cache did not match the source in digest and row count, so "
            "it was not extended. Zero inference was performed, zero cache rows "
            "were written, and the source cache was not touched."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nWritten to {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    sys.exit(main())
