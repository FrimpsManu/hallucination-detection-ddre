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
size 1 against the existing v2 cache, with the ordinary read-through /
write-through cache mechanism. A span already cached is reused; only a span that
is genuinely missing is evaluated and written back.

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
    build_counting_scorer,
    completion_report,
    parse_placement,
    placement_label,
    verification_summary,
)
from src.cached_document_scores import (  # noqa: E402
    CachedDocumentScorer,
    MissingDocumentScore,
)
from src.nbc_sensitivity import combination_histograms  # noqa: E402
from src.reproduction_gate import PUBLISHED_TABLE1  # noqa: E402

OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_CACHE = "results/wang_nli_cache_fidelity_v2_batch1.sqlite"
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
        "--cache-path",
        default=DEFAULT_CACHE,
        help=(
            "The corrected Wang-fidelity v2 cache to extend IN PLACE. Must be "
            "the same cache the sensitivity analysis replays."
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


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    placements = resolve_placements(args)
    cache_path = Path(args.cache_path)

    print("=" * 100)
    print("NBC SENSITIVITY CACHE COMPLETION")
    print("=" * 100)
    print(f"  configuration completed: {PRIMARY_CONFIGURATION} only")
    print(f"  batch size:              {WANG_BATCH_SIZE} (released Wang semantics)")
    print(f"  cache (extended in place): {cache_path}")
    print(f"  placements: {', '.join(placement_label(*p) for p in placements)}")
    print("  cached spans are reused; only genuinely missing spans are evaluated")

    if not cache_path.exists():
        print(f"\nERROR: cache not found at {cache_path}")
        print("Point --cache-path at the corrected Wang-fidelity v2 cache.")
        return 1

    if args.dry_run:
        print("\n--dry-run: placements listed only. No model loaded, nothing scored.")
        print("=" * 100)
        return 0

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import collect_live_environment
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_sentence_records

    records = load_sentence_records(args.data_root, strict=True)
    print(f"  sentences: {len(records)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    environment = collect_live_environment(
        model_name=args.model_name, model=model, tokenizer=tokenizer,
        repo_root=PROJECT_ROOT, score_version=SCORE_VERSION,
    )
    model_block = environment.get("model") or {}
    if model_block.get("training_mode"):
        raise RuntimeError("model.eval() did not take effect; refusing to score")

    scorer = build_counting_scorer(
        tokenizer, model, args.model_name, str(cache_path), batch_size=WANG_BATCH_SIZE
    )
    rows_before = scorer.cache_size()
    print(f"  score_version: {SCORE_VERSION}")
    print(f"  cache rows before: {rows_before}")

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

    print(f"\n  cache rows after: {rows_after}")

    print("\nVerifying each placement now completes from the cache alone...")
    verification = [
        verify_placement(
            records, str(cache_path), args.model_name, SCORE_VERSION, split_text,
            positive_bin, negative_bin,
        )
        for positive_bin, negative_bin in placements
    ]

    report = completion_report(
        placements, rows_before, rows_after, per_placement, verification
    )
    report["environment"] = environment
    report["score_version"] = SCORE_VERSION
    report["cache_path"] = str(cache_path)
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
    print(f"  cache rows before:                {report['cache_rows_before']}")
    print(f"  cache rows after:                 {report['cache_rows_after']}")
    print(f"  cache rows added:                 {report['cache_rows_added']}")
    print(f"  accounting consistent:            {report['accounting_consistent']}")
    if not report["accounting_consistent"]:
        print(f"  *** {report['accounting_note']} ***")

    print()
    print("-" * 100)
    print("PLACEMENT COMPLETENESS (read-only cache replay)")
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
    return 0 if report["all_requested_placements_complete"] and report[
        "accounting_consistent"
    ] else 1


if __name__ == "__main__":
    sys.exit(main())
