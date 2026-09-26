#!/usr/bin/env python3
"""Run the frozen D-03 density-ratio support/stability diagnostic.

The methodology lives in ``src/density_ratio_diagnostics.py`` so it can be
reviewed and unit-tested without a GPU. This file is the runner: provenance,
scoring, and assembling the JSON verdict.

**What this measures.** Not "are the ratios large" -- a large ratio is what a
density-ratio method should produce where the classes separate. It measures
whether the evidence DDRE consumes is *supported* by the training score regions
and *stable* across the production uLSIF hyperparameter surface.

**Population.** ALL candidate documents for ALL validation subclaims, up to
``max_docs = 10``. Deliberately threshold-independent: running DDRE first and
inspecting only the documents it happened to retrieve would condition the
diagnostic on the very stopping policy that has not been tuned yet, and the
first-document behaviour -- the most direct D-03 concern -- would be selected
on rather than measured.

**Safety.** Held-out test evidence is never touched. The formal v2 cache is
opened read-only; new validation document scores go to a derived cache bound to
the exact certified source digest. The run aborts before any inference if
provenance, the 398-pair compatibility probe, the split identity or the
199/199 training counts fail.

**The historical limitation is preserved.** ``checkpoint_identity_established``
may remain false, because the historical v2 Gate report did not record its Hub
revision. That is not rewritten as solved. What D-03 needs is the weaker but
DIRECTLY MEASURED statement ``score_compatibility_established == true`` over all
398 released NBC pairs.

``--dry-run`` reports the exact measurement population -- split identity,
subclaim and document-occurrence counts, threshold-pair count, cache paths --
without loading a model, running inference, or writing anything.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_completion import (  # noqa: E402
    UnsafeCacheTarget,
    bind_compatibility_to_source,
    discard_invalid_derived_cache,
    prepare_derived_cache,
    sha256_file,
)
from src.density_ratio_diagnostics import (  # noqa: E402
    DIAGNOSTIC_MAX_DOCS,
    DIAGNOSTIC_NAME,
    DIAGNOSTIC_P0,
    DIAGNOSTIC_SPLIT_SEED,
    DIAGNOSTIC_VALIDATION_FRACTION,
    EXPECTED_FACTUAL_TRAINING,
    EXPECTED_HALLUCINATED_TRAINING,
    MAGNITUDE_NOTE,
    PROTOCOL_VERSION,
    DiagnosticIncomplete,
    completion_accounting,
    distribution_shift,
    escalation_triggers,
    hyperparameter_surface,
    one_document_stopping,
    population_ratio_report,
    production_hyperparameter_pairs,
    range_status,
    require_complete,
    selected_fit_ratios,
    stop_decision_stability,
    stops_under_selected_fit,
    support_counts,
    support_strata_report,
)
from src.provenance_guard import (  # noqa: E402
    OFFICIAL_NLI_MODEL,
    REQUIRED_BATCH_SIZE,
    check_reference_bundle,
    check_runtime_preconditions,
    check_static_preconditions,
    extract_reference,
    format_bundle,
    format_guard,
    guard_report,
    merge_reference,
)
from src.score_compatibility import (  # noqa: E402
    EXPECTED_PAIR_COUNT,
    PROBE_BATCH_SIZE,
    format_compatibility,
    run_compatibility_probe,
    scratch_scorer,
)

DEFAULT_SOURCE_CACHE = "results/wang_nli_cache.sqlite"
DEFAULT_DERIVED_CACHE = "results/d03_diagnostic_cache.sqlite"
DEFAULT_OUTPUT = "results/d03_density_ratio_diagnostic.json"

# Wang's released scorer evaluates one pair at a time, and the formal v2
# provenance machinery already treats batch size 1 as the canonical
# Wang-fidelity path. GPU batch 8 is NOT substituted here.
WANG_BATCH_SIZE = REQUIRED_BATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "D-03 density-ratio support/stability diagnostic. Validation "
            "evidence only; the held-out test split is never scored."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_NLI_MODEL)
    parser.add_argument(
        "--source-cache",
        default=DEFAULT_SOURCE_CACHE,
        help="The formal v2 cache. Read and hashed; never opened for writing.",
    )
    parser.add_argument(
        "--output-cache",
        default=DEFAULT_DERIVED_CACHE,
        help="The derived diagnostic cache. New validation scores go here only.",
    )
    parser.add_argument(
        "--formal-provenance",
        required=True,
        metavar="PATH",
        help="The formal v2 batch-1 Gate report. Authoritative where it records.",
    )
    parser.add_argument(
        "--checkpoint-provenance",
        default=None,
        metavar="PATH",
        help="Supplements only fields the formal Gate report does not record.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report the exact measurement population and exit. No model is "
            "downloaded or loaded, no inference runs, nothing is written."
        ),
    )
    return parser.parse_args()


def load_reference(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def wang_source_commit(data_root):
    source = Path(data_root) / "SOURCE.json"
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("source_commit")
    except Exception:  # noqa: BLE001 - unreadable is as disqualifying as absent
        return None


def frozen_validation_split(data_root):
    """The validation half of the frozen split, verified by passage IDENTITY.

    Re-derived from the released records with the frozen fraction and seed. The
    check is on the actual passage IDs, not on their count: a *different* 48
    passages would be a different diagnostic population.
    """
    from src.wang_data import group_split_records, load_sentence_records

    records = load_sentence_records(data_root, strict=True)
    validation, test, metadata = group_split_records(
        records,
        validation_fraction=DIAGNOSTIC_VALIDATION_FRACTION,
        random_state=DIAGNOSTIC_SPLIT_SEED,
    )
    expected = sorted(int(x) for x in metadata["validation_passage_ids"])
    observed = sorted({int(r.passage_index) for r in validation})
    identity = {
        "validation_fraction": DIAGNOSTIC_VALIDATION_FRACTION,
        "split_seed": DIAGNOSTIC_SPLIT_SEED,
        "validation_passage_ids": expected,
        "observed_validation_passage_ids": observed,
        "validation_passages": len(expected),
        "test_passages": int(metadata["test_passages"]),
        "identity_matches": expected == observed,
        "sha256": __import__("hashlib")
        .sha256(json.dumps(expected, sort_keys=True).encode("utf-8"))
        .hexdigest(),
    }
    return validation, test, identity


def validation_population(validation_records, max_docs=DIAGNOSTIC_MAX_DOCS):
    """Every candidate document occurrence, with its full identity.

    Occurrences are NOT deduplicated. The same page reached from two subclaims
    is two occurrences of evidence in the application, and collapsing them
    would silently reweight the population.
    """
    occurrences = []
    for record in validation_records:
        for subclaim_index, subclaim in enumerate(record.subclaims):
            documents = subclaim.documents[:max_docs]
            for document_index, document in enumerate(documents):
                occurrences.append(
                    {
                        "passage_index": int(record.passage_index),
                        "sentence_index": int(record.sentence_index),
                        "subclaim_index": int(subclaim_index),
                        "document_index": int(document_index),
                        "is_first_document": document_index == 0,
                        "subclaim_text": subclaim.text,
                        "page_content": document.page_content,
                    }
                )
    return occurrences


def population_counts(validation_records, occurrences):
    subclaims = sum(len(r.subclaims) for r in validation_records)
    return {
        "passages": len({r.passage_index for r in validation_records}),
        "sentences": len(validation_records),
        "subclaims": subclaims,
        "document_occurrences": len(occurrences),
        "first_document_occurrences": sum(
            1 for o in occurrences if o["is_first_document"]
        ),
        "max_docs": DIAGNOSTIC_MAX_DOCS,
        "deduplicated": False,
        "threshold_independent": True,
        "note": (
            "ALL candidate validation documents up to max_docs, not only those "
            "a tuned DDRE would have retrieved."
        ),
    }


def threshold_pairs(c_miss=28.0, c_false_alarm=96.0):
    """The actual cost-consistent search space, from the production helper."""
    from src.ddre_core import (
        CANDIDATE_LOWER_GRID,
        CANDIDATE_UPPER_GRID,
        cost_consistent_thresholds,
    )

    space = cost_consistent_thresholds(
        c_miss, c_false_alarm, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID
    )
    pairs = [
        (float(lower), float(upper))
        for lower in space["effective_lower_grid"]
        for upper in space["effective_upper_grid"]
        if lower < upper
    ]
    return pairs, space


def dry_run_report(args):
    """Everything reviewable before paying for a GPU. Writes nothing."""
    validation, test, identity = frozen_validation_split(args.data_root)
    occurrences = validation_population(validation)
    counts = population_counts(validation, occurrences)
    pairs, space = threshold_pairs()

    print("=" * 100)
    print(f"{DIAGNOSTIC_NAME} -- DRY RUN")
    print("=" * 100)
    print(f"  protocol version:        {PROTOCOL_VERSION}")
    print(f"  model:                   {args.model_name}")
    print(f"  batch size:              {WANG_BATCH_SIZE} (released Wang semantics)")
    print(f"  formal provenance:       {args.formal_provenance}")
    print(f"  checkpoint provenance:   {args.checkpoint_provenance}")
    print(f"  source cache (read-only): {args.source_cache}")
    print(f"  derived cache (written):  {args.output_cache}")
    print(f"  output:                  {args.output}")
    print()
    print("  FROZEN VALIDATION SPLIT (identity, not count)")
    print(f"    validation_fraction:   {identity['validation_fraction']}")
    print(f"    split seed:            {identity['split_seed']}")
    print(f"    validation passages:   {identity['validation_passages']}")
    print(f"    held-out passages:     {identity['test_passages']} (NEVER scored)")
    print(f"    passage-id sha256:     {identity['sha256']}")
    print(f"    identity matches:      {identity['identity_matches']}")
    print()
    print("  MEASUREMENT POPULATION (threshold-independent)")
    print(f"    sentences:             {counts['sentences']}")
    print(f"    subclaims:             {counts['subclaims']}")
    print(f"    document occurrences:  {counts['document_occurrences']} "
          f"(up to max_docs={DIAGNOSTIC_MAX_DOCS})")
    print(f"    first documents:       {counts['first_document_occurrences']}")
    print()
    print("  ESTIMATOR AND STOPPING")
    print(f"    NBC training required: {EXPECTED_FACTUAL_TRAINING} factual + "
          f"{EXPECTED_HALLUCINATED_TRAINING} hallucinated")
    print(f"    cost-consistent pairs: {len(pairs)} "
          f"(CM=28, CFA=96, P0={DIAGNOSTIC_P0})")
    print(f"    lower grid:            {list(space['effective_lower_grid'])}")
    print(f"    upper grid:            {list(space['effective_upper_grid'])}")
    print()
    print(f"  compatibility probe:     {EXPECTED_PAIR_COUNT} forward passes "
          f"at batch size {PROBE_BATCH_SIZE} (exact equality required)")
    print()
    print("  --dry-run: no model loaded, no inference, no cache written.")
    print("=" * 100)
    return {
        "diagnostic": DIAGNOSTIC_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "dry_run": True,
        "split_identity": identity,
        "population": counts,
        "threshold_pairs": len(pairs),
        "expected_compatibility_forward_passes": EXPECTED_PAIR_COUNT,
        "source_cache": args.source_cache,
        "derived_cache": args.output_cache,
        "output": args.output,
        "model_name": args.model_name,
        "batch_size": WANG_BATCH_SIZE,
        "held_out_scored": False,
    }


def main():
    args = parse_args()

    if args.dry_run:
        dry_run_report(args)
        return 0

    print("=" * 100)
    print(DIAGNOSTIC_NAME)
    print("=" * 100)
    print(f"  protocol version: {PROTOCOL_VERSION}")
    print(f"  batch size:       {WANG_BATCH_SIZE} (released Wang semantics)")
    print(f"  source cache:     {args.source_cache} (READ-ONLY)")
    print(f"  derived cache:    {args.output_cache}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_cache = Path(args.source_cache)
    if not source_cache.exists():
        print(f"\nERROR: source cache not found at {source_cache}")
        return 1

    # ------------------------------------------------------------- split
    # Before anything expensive: the population must be the frozen one.
    validation, _, identity = frozen_validation_split(args.data_root)
    if not identity["identity_matches"]:
        print("\nABORTED: the validation passage IDs are not the frozen split.")
        return 1
    occurrences = validation_population(validation)
    counts = population_counts(validation, occurrences)
    pairs, space = threshold_pairs()
    print(f"  validation passages: {counts['passages']} "
          f"(sha256 {identity['sha256'][:16]}...)")
    print(f"  document occurrences: {counts['document_occurrences']}")

    # -------------------------------------------------------- provenance
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

    from src.utils import SCORE_VERSION

    static_checks = check_static_preconditions(
        reference,
        score_version=SCORE_VERSION,
        batch_size=WANG_BATCH_SIZE,
        wang_source_commit=wang_source_commit(args.data_root),
        model_name=args.model_name,
    ) + check_reference_bundle(bundle)
    static = guard_report(static_checks, reference_path=args.formal_provenance)
    if not static["passed"]:
        print()
        print(format_guard(static))
        print("\nABORTED at the static provenance gate. Nothing scored.")
        return 1

    revision = reference.get("resolved_revision")
    print(f"\n  pinning checkpoint revision: {revision!r}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import collect_live_environment, select_device

    load_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, **load_kwargs
    )
    # CUDA -> MPS -> CPU. On Apple silicon the previous cuda-or-cpu expression
    # selected the CPU, which is both slow and -- because the formal v2 Gate run
    # recorded "mps" -- a device the runtime provenance gate would reject. The
    # selected string is passed to the snapshot so the recorded device is the
    # device the weights are actually on.
    selected_device = select_device(torch)
    device = torch.device(selected_device)
    model = model.to(device)
    model.eval()
    print(f"  device: {selected_device}")

    observed = collect_live_environment(
        model_name=args.model_name, model=model, tokenizer=tokenizer,
        repo_root=PROJECT_ROOT, score_version=SCORE_VERSION,
        selected_device=selected_device,
    )
    placement = observed.get("device_placement") or {}
    if placement.get("matches") is not True:
        print(
            "\nABORTED: the model is on "
            f"{placement.get('model_device')!r} but the run selected "
            f"{placement.get('selected_device')!r}. Provenance would record a "
            "device the arithmetic did not run on. Nothing scored."
        )
        return 1
    guard = guard_report(
        static_checks + check_runtime_preconditions(reference, observed),
        reference_path=args.formal_provenance,
    )
    guard["reference_bundle"] = bundle
    print()
    print(format_guard(guard))
    if not guard["passed"]:
        print("\nABORTED at the runtime provenance gate. Nothing scored.")
        return 1

    # --------------------------------------- 398-pair compatibility probe
    compatibility = _run_compatibility_probe(
        args, tokenizer, model, SCORE_VERSION, source_cache
    )
    print()
    print(format_compatibility(compatibility))
    if not compatibility["score_compatibility_established"]:
        print("\nABORTED: score compatibility not established. Nothing scored.")
        return 1

    # ------------------------------------------------------ derived cache
    try:
        cache_identity = prepare_derived_cache(
            source_cache, Path(args.output_cache), overwrite=args.overwrite_output
        )
    except (UnsafeCacheTarget, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    if cache_identity.get("copy_faithful") is not True:
        print("\nABORTED: the derived cache is not a faithful copy of the source.")
        print(f"  {cache_identity.get('copy_faithful_note')}")
        print(f"  {discard_invalid_derived_cache(cache_identity)['reason']}")
        return 1

    binding = bind_compatibility_to_source(compatibility, cache_identity)
    print(f"  compatibility source bound: {binding['bound']}")
    if not binding["bound"]:
        print(f"\nABORTED: {binding['message']}")
        print(f"  {discard_invalid_derived_cache(cache_identity)['reason']}")
        return 1

    # ------------------------------------------------------------ scoring
    from src.utils import EntailmentScorer
    from src.wang_data import load_nbc_pairs

    scorer = EntailmentScorer(
        tokenizer, model, args.model_name, cache_path=str(args.output_cache),
        batch_size=WANG_BATCH_SIZE,
    )
    try:
        positive, negative = load_nbc_pairs(args.data_root, per_class=None)
        if (len(positive), len(negative)) != (
            EXPECTED_FACTUAL_TRAINING, EXPECTED_HALLUCINATED_TRAINING
        ):
            print(
                f"\nABORTED: NBC training counts are {len(positive)}/"
                f"{len(negative)}; the production estimator requires "
                f"{EXPECTED_FACTUAL_TRAINING}/{EXPECTED_HALLUCINATED_TRAINING}."
            )
            return 1

        factual_scores = scorer.score_pairs(
            [(x["premise"], x["hypothesis"]) for x in positive],
            use_cache=True, write_cache=True, show_progress=True,
            description="NBC factual",
        )
        hallucinated_scores = scorer.score_pairs(
            [(x["premise"], x["hypothesis"]) for x in negative],
            use_cache=True, write_cache=True, show_progress=True,
            description="NBC hallucinated",
        )

        document_scores = []
        span_evaluations = 0
        for occurrence in occurrences:
            score, spans = scorer.score_document(
                occurrence["subclaim_text"], occurrence["page_content"],
                use_cache=True, write_cache=True,
            )
            document_scores.append(float(score))
            span_evaluations += int(spans)
    finally:
        scorer.close()

    accounting = require_complete(
        completion_accounting(
            expected_occurrences=counts["document_occurrences"],
            scored_occurrences=len(document_scores),
            first_documents=sum(
                1 for o in occurrences if o["is_first_document"]
            ),
            expected_first_documents=counts["first_document_occurrences"],
        )
    )

    report = build_report(
        args=args,
        identity=identity,
        counts=counts,
        occurrences=occurrences,
        document_scores=document_scores,
        factual_scores=factual_scores,
        hallucinated_scores=hallucinated_scores,
        pairs=pairs,
        space=space,
        guard=guard,
        bundle=bundle,
        compatibility=compatibility,
        cache_identity=cache_identity,
        binding=binding,
        environment=observed,
        accounting=accounting,
        span_evaluations=span_evaluations,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nWrote {output_path}")
    print(
        "  additional_sensitivity_required: "
        f"{report['escalation']['additional_sensitivity_required']}"
    )
    print(f"  {report['escalation']['meaning']}")
    return 0


def build_report(**kwargs):
    """Assemble the JSON verdict from the frozen diagnostic helpers."""
    from src.ddre_core import ULSIFDensityRatio

    args = kwargs["args"]
    occurrences = kwargs["occurrences"]
    document_scores = kwargs["document_scores"]
    factual_scores = kwargs["factual_scores"]
    hallucinated_scores = kwargs["hallucinated_scores"]
    pairs = kwargs["pairs"]

    # The production fit, on the canonically scored NBC pairs. Not histogram
    # midpoints, not scores synthesised from released counts, and nothing from
    # the held-out split.
    estimator = ULSIFDensityRatio(max_centers=100, random_state=42)
    estimator.fit(factual_scores, hallucinated_scores)

    first_indices = [
        i for i, o in enumerate(occurrences) if o["is_first_document"]
    ]
    first_scores = [document_scores[i] for i in first_indices]

    selected_documents = selected_fit_ratios(estimator, document_scores)
    selected_first = selected_fit_ratios(estimator, first_scores)

    surface_first = hyperparameter_surface(
        estimator, factual_scores, hallucinated_scores, first_scores
    )
    stability = stop_decision_stability(
        list(selected_first["returned"]), surface_first["returned_matrix"], pairs
    )
    stopped = stops_under_selected_fit(list(selected_first["returned"]), pairs)
    support_first = support_counts(
        first_scores, factual_scores, hallucinated_scores, estimator.sigma
    )
    document_report = population_ratio_report(
        estimator, document_scores, name="validation_document_max"
    )

    escalation = escalation_triggers(
        selected_document_clip_activity=document_report["clip_activity"],
        first_document_stopped=stopped,
        first_document_direction=surface_first["direction"],
        first_document_min_class_support=support_first["min_class_support"],
        first_document_unanimous=stability["unanimous_per_score"],
    )

    return {
        "diagnostic": DIAGNOSTIC_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "provenance": {
            "formal_provenance_path": args.formal_provenance,
            "formal_provenance_sha256": _sha(args.formal_provenance),
            "checkpoint_provenance_path": args.checkpoint_provenance,
            "checkpoint_provenance_sha256": _sha(args.checkpoint_provenance),
            "source_cache": str(args.source_cache),
            "source_sha256_before": kwargs["cache_identity"]["source_sha256_before"],
            "source_sha256_after_probe": kwargs["compatibility"].get(
                "source_integrity", {}
            ).get("sha256_after"),
            "derived_diagnostic_cache": str(args.output_cache),
            "compatibility_source_bound": kwargs["binding"]["bound"],
            "guard": kwargs["guard"],
            "reference_bundle": kwargs["bundle"],
            "environment": kwargs["environment"],
            "batch_size": WANG_BATCH_SIZE,
            "wang_source_commit": wang_source_commit(args.data_root),
        },
        "checkpoint_identity_established": kwargs["bundle"][
            "checkpoint_identity_established"
        ],
        "checkpoint_identity_note": (
            "May be false: the historical v2 Gate report did not record its Hub "
            "revision. That limitation is NOT rewritten as solved. D-03 relies "
            "on the weaker but directly measured score_compatibility_established."
        ),
        "score_compatibility_established": kwargs["compatibility"][
            "score_compatibility_established"
        ],
        "score_compatibility": kwargs["compatibility"],
        "split": kwargs["identity"],
        "held_out_scored": False,
        "population": kwargs["counts"],
        "completion": kwargs["accounting"],
        "span_evaluations": kwargs["span_evaluations"],
        "training": {
            "factual_count": len(factual_scores),
            "hallucinated_count": len(hallucinated_scores),
            "expected_factual": EXPECTED_FACTUAL_TRAINING,
            "expected_hallucinated": EXPECTED_HALLUCINATED_TRAINING,
        },
        "ulsif_selected_fit": {
            "fit_diagnostics": estimator.fit_diagnostics,
            "cv_table": estimator.cv_table,
            "sigma": float(estimator.sigma),
            "lambda": float(estimator.lam),
            "n_centers": int(estimator.centers.size),
        },
        "selected_fit_populations": {
            "nbc_factual_training": population_ratio_report(
                estimator, factual_scores, name="nbc_factual_training"
            ),
            "nbc_hallucinated_training": population_ratio_report(
                estimator, hallucinated_scores, name="nbc_hallucinated_training"
            ),
            "validation_document_max": document_report,
            "validation_first_document": population_ratio_report(
                estimator, first_scores, name="validation_first_document"
            ),
        },
        "support": {
            "first_document": support_strata_report(
                min_class_support=list(support_first["min_class_support"]),
                returned_ratios=list(selected_first["returned"]),
                log_ratio_span=list(surface_first["log_ratio_span"]),
                directions=surface_first["direction"],
                stopped_flags=stopped,
                unanimous_flags=stability["unanimous_per_score"],
            ),
            "radius_sigma": support_first["radius_sigma"],
            "space": support_first["space"],
            "range_status": range_status(
                first_scores, factual_scores, hallucinated_scores
            ),
        },
        "hyperparameter_sensitivity": {
            key: (list(value) if hasattr(value, "tolist") else value)
            for key, value in surface_first.items()
            if key != "returned_matrix"
        },
        "one_document_stopping": {
            key: value
            for key, value in one_document_stopping(
                list(selected_first["returned"]), pairs
            ).items()
            if key != "states"
        },
        "stop_decision_stability": {
            key: value
            for key, value in stability.items()
            if key != "unanimous_per_score"
        },
        "threshold_search_space": kwargs["space"],
        "d13_distribution_shift": distribution_shift(
            list(factual_scores) + list(hallucinated_scores), document_scores
        ),
        "escalation": escalation,
        "additional_sensitivity_required": escalation[
            "additional_sensitivity_required"
        ],
        "selected_cap": None,
        "selected_calibration": None,
        "method_change_made": False,
        "magnitude_note": MAGNITUDE_NOTE,
        "d03_status": (
            "DIAGNOSTIC PROTOCOL FROZEN AND MEASURED. D-03 is not resolved by "
            "this run alone; the result requires review before any tuning."
        ),
    }


def _sha(path):
    return sha256_file(path) if path and Path(path).exists() else None


def _git_commit():
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _run_compatibility_probe(args, tokenizer, model, score_version, source_cache):
    """The fixed 398-pair probe. Reads the source read-only; writes nothing.

    The scratch database lives in a temporary directory removed afterwards, so
    the only databases this touches are the source (read-only) and a file that
    no longer exists when it returns.
    """
    import tempfile

    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.data_root, per_class=None)
    pairs = [(x["premise"], x["hypothesis"]) for x in positive]
    pairs += [(x["premise"], x["hypothesis"]) for x in negative]
    polarities = ["positive"] * len(positive) + ["negative"] * len(negative)

    print()
    print(
        f"Score-compatibility probe: {len(pairs)} released NBC pairs at batch "
        f"size {PROBE_BATCH_SIZE}; exact equality required."
    )
    with tempfile.TemporaryDirectory(prefix="d03-compat-") as scratch_dir:
        scratch_path = str(Path(scratch_dir) / "scratch.sqlite")
        probe_scorer = scratch_scorer(
            tokenizer, model, args.model_name, scratch_path,
            batch_size=PROBE_BATCH_SIZE,
        )
        closed = {"done": False}

        def finalize():
            rows = probe_scorer.cache_size()
            probe_scorer.close()
            closed["done"] = True
            return rows

        try:
            return run_compatibility_probe(
                pairs=pairs,
                polarities=polarities,
                positive_pairs=len(positive),
                negative_pairs=len(negative),
                source_cache=str(source_cache),
                model_name=args.model_name,
                score_version=score_version,
                scorer=probe_scorer,
                finalize=finalize,
                source_digest=lambda: sha256_file(source_cache),
            )
        finally:
            if not closed["done"]:
                probe_scorer.close()


if __name__ == "__main__":
    raise SystemExit(main())
