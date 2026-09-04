"""Gate 1 sensitivity analysis: can the two unreleased NBC examples close the gap?

Wang et al. report s = 200 factual and s = 200 nonfactual NBC examples. The
released files contain 199 each, so two examples described in the paper are
absent from the artifacts and from every histogram this repository builds.

This script enumerates all 10 x 10 = 100 ways one extra positive and one extra
negative example could fall across the ten discretized bins, and reports how the
frozen Gate 1 verdict responds under both published cost configurations.

It is a SENSITIVITY ANALYSIS ONLY. It never selects a favourable histogram, it
never touches the released NBC files, and the released 199 + 199 data remain the
experiment's baseline whatever the result.

Everything it evaluates is the existing, unmodified machinery: ``BSEDetector`` in
``mode="official"``, ``evaluate_detector``, and ``evaluate_configuration``
against the published Table 1 values and the frozen tolerances.

No inference. No Hugging Face downloads. No model is constructed. Document
scores are replayed from an existing NLI cache opened READ-ONLY.

A document the recorded run never consumed has no cached score. Rather than
inventing one, completeness is tracked PER COST CONFIGURATION: the miss marks
only the configuration that hit it, and a result the other configuration already
produced is preserved and still counted. Retrieval is adaptive, so CM=14/CFA=24
and CM=28/CFA=96 consume different documents from the same placement, and one
can complete while the other cannot. A completed CM=14/CFA=24 evaluation is a
real measurement and is never discarded because CM=28/CFA=96 later ran short of
cached scores. Only ``both_configurations_pass`` requires both to have completed.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_core import BSEDetector  # noqa: E402
from src.cached_document_scores import (  # noqa: E402
    CachedDocumentScorer,
    MissingDocumentScore,
)
from src.evaluation import summarize_method  # noqa: E402
from src.nbc_sensitivity import (  # noqa: E402
    MISSING_EXAMPLES_PER_CLASS,
    PAPER_EXAMPLES_PER_CLASS,
    RELEASED_EXAMPLES_PER_CLASS,
    RELEASED_NEGATIVE_HISTOGRAM,
    RELEASED_POSITIVE_HISTOGRAM,
    closest_combinations,
    combination_histograms,
    enumerate_combinations,
    interpretation,
    raw_count,
    summarize,
)
from src.reproduction_gate import (  # noqa: E402
    PUBLISHED_TABLE1,
    evaluate_configuration,
)

OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_OUTPUT = "results/diagnostics/nbc_count_sensitivity.json"
PRIMARY = "CM_14_CFA_24"
SECONDARY = "CM_28_CFA_96"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis: could the two unreleased NBC examples explain "
            "the remaining Gate 1 CM=14/CFA=24 gap? Replays cached scores only; "
            "runs no inference."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument(
        "--cache-path",
        default="results/wang_nli_cache_fidelity_v2_batch1.sqlite",
        help=(
            "Existing NLI cache to replay. Opened READ-ONLY; never written to. "
            "For the formal result this MUST be the corrected Wang-fidelity v2 "
            "cache, not the historical v1 cache: v1 rows were scaled inside the "
            "tensor and carry the half-precision rounding this experiment is "
            "downstream of."
        ),
    )
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument(
        "--score-version",
        default=None,
        help=(
            "Cache score_version to replay. Defaults to src.utils.SCORE_VERSION. "
            "Set explicitly to replay a cache written under an older version."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write the report; e.g. a Google Drive diagnostics path.",
    )
    parser.add_argument(
        "--limit-combinations",
        type=int,
        default=None,
        help="Evaluate only the first N combinations (debugging aid).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the grid and the released histograms, then exit.",
    )
    return parser.parse_args()


def metrics_for_gate(metrics):
    """Map evaluation output onto the metric names the gate compares.

    Transcribed from ``scripts/reproduce_wang_baseline.py`` so the sensitivity
    analysis is judged by exactly the quantities Gate 1 judges.
    """
    return {
        "nonfactual_auc_pr": metrics["nonfactual"]["auc_pr"],
        "factual_auc_pr": metrics["factual"]["auc_pr"],
        "accuracy": metrics["accuracy"],
        "pearson": metrics["passage_level"]["pearson"],
        "spearman": metrics["passage_level"]["spearman"],
        "evidence_num_per_sentence": metrics["efficiency"][
            "avg_retrieved_documents_per_sentence"
        ],
        "avg_retrieved_documents_per_subclaim": metrics["efficiency"][
            "avg_retrieved_documents_per_subclaim"
        ],
        "total_retrieved_documents": metrics["efficiency"]["total_retrieved_documents"],
        "total_nli_span_calls": metrics["efficiency"]["total_nli_span_calls"],
    }


def evaluate_one_configuration(records, scorer, pos_hist, neg_hist, config_name):
    """Run unmodified bse_official under one histogram and one cost setting."""
    reference = PUBLISHED_TABLE1[config_name]
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
    results = [detector.detect_sentence(record, scorer) for record in records]
    metrics = summarize_method(records, results)
    return evaluate_configuration(config_name, metrics_for_gate(metrics))


def evaluate_combination(records, scorer, positive_bin, negative_bin):
    """Evaluate one placement under both cost configurations, independently.

    The two configurations are evaluated in separate try blocks on purpose. A
    completed CM_14_CFA_24 result is a real measurement, and a cache miss that
    only occurs while evaluating CM_28_CFA_96 must not discard it. Completeness
    is therefore recorded per configuration, never once for the whole row.
    """
    pos_hist, neg_hist = combination_histograms(positive_bin, negative_bin)
    row = {
        "positive_bin": positive_bin,
        "negative_bin": negative_bin,
        "positive_histogram": pos_hist,
        "negative_histogram": neg_hist,
        "configurations": {},
    }
    for config_name in (PRIMARY, SECONDARY):
        try:
            block = evaluate_one_configuration(
                records, scorer, pos_hist, neg_hist, config_name
            )
            block["complete"] = True
            block["incomplete_reason"] = None
        except MissingDocumentScore as exc:
            block = {
                "configuration": config_name,
                "complete": False,
                "incomplete_reason": (
                    f"required document score absent from the cache ({exc})"
                ),
                "verdict": None,
                "metrics": [],
            }
        row["configurations"][config_name] = block
    return row


def print_header(args, records=None):
    print("=" * 100)
    print("GATE 1 SENSITIVITY ANALYSIS: THE TWO UNRELEASED NBC EXAMPLES")
    print("=" * 100)
    print(
        f"  paper protocol: s = {PAPER_EXAMPLES_PER_CLASS} per class; released "
        f"files contain {RELEASED_EXAMPLES_PER_CLASS}; "
        f"{MISSING_EXAMPLES_PER_CLASS} missing per class"
    )
    print(f"  released positive histogram: {list(RELEASED_POSITIVE_HISTOGRAM)}"
          f"  (raw {raw_count(RELEASED_POSITIVE_HISTOGRAM)})")
    print(f"  released negative histogram: {list(RELEASED_NEGATIVE_HISTOGRAM)}"
          f"  (raw {raw_count(RELEASED_NEGATIVE_HISTOGRAM)})")
    print(f"  combinations: {len(enumerate_combinations())} (10 positive bins x 10 negative bins)")
    print(f"  configurations per combination: {PRIMARY}, {SECONDARY}")
    print(f"  cache (read-only): {args.cache_path}")
    if records is not None:
        print(f"  sentences: {len(records)}")
    print("  no model is constructed; no inference is run; no download occurs")


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print_header(args)
        print("\n--dry-run: grid described only. Nothing evaluated.")
        print("=" * 100)
        return 0

    # src.utils imports torch, but only for tensor work in EntailmentScorer,
    # which is never constructed here. split_text is pure Python.
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_sentence_records

    score_version = args.score_version or SCORE_VERSION
    records = load_sentence_records(args.data_root, strict=True)
    print_header(args, records)
    print(f"  score_version replayed: {score_version}")

    scorer = CachedDocumentScorer(
        args.cache_path, args.model_name, score_version, split_text
    )
    print(f"  cache rows available: {scorer.cache_rows()}")

    combinations = enumerate_combinations()
    if args.limit_combinations:
        combinations = combinations[: args.limit_combinations]

    rows = []
    try:
        for index, (positive_bin, negative_bin) in enumerate(combinations, start=1):
            row = evaluate_combination(records, scorer, positive_bin, negative_bin)
            rows.append(row)
            marker = "/".join(
                row["configurations"][name]["verdict"]
                if row["configurations"][name]["complete"]
                else "INCOMPLETE"
                for name in (PRIMARY, SECONDARY)
            )
            print(
                f"  [{index:3d}/{len(combinations)}] pos_bin={positive_bin} "
                f"neg_bin={negative_bin}  {marker}",
                flush=True,
            )
    finally:
        scorer.close()

    summary = summarize(
        rows,
        primary=PRIMARY,
        secondary=SECONDARY,
        expected_combinations=len(enumerate_combinations()),
    )
    reading = interpretation(summary, primary=PRIMARY)
    closest = closest_combinations(rows, configuration=PRIMARY, limit=5)

    report = {
        "analysis": "nbc-missing-examples-sensitivity",
        "purpose": (
            "Determine whether the two NBC examples described in the paper but "
            "absent from the released files could plausibly explain the "
            "remaining Gate 1 CM=14/CFA=24 reproduction gap."
        ),
        "scope": {
            "sensitivity_analysis_only": True,
            "released_nbc_files_unmodified": True,
            "baseline_unchanged": True,
            "no_inference_performed": True,
            "cache_opened_read_only": True,
        },
        "released_histograms": {
            "positive": list(RELEASED_POSITIVE_HISTOGRAM),
            "negative": list(RELEASED_NEGATIVE_HISTOGRAM),
            "raw_examples_per_class": RELEASED_EXAMPLES_PER_CLASS,
            "paper_examples_per_class": PAPER_EXAMPLES_PER_CLASS,
        },
        "scorer": scorer.stats(),
        "dataset": {
            "sentences": len(records),
            "passages": len({r.passage_index for r in records}),
            "subclaims": sum(len(r.subclaims) for r in records),
        },
        "combinations": rows,
        "summary": summary,
        "closest_to_published_primary": closest,
        "interpretation": reading,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print()
    print("-" * 100)
    print("SUMMARY")
    print("-" * 100)
    print(
        f"  combinations evaluated:      {summary['combinations_evaluated']}"
        f" of {summary['expected_combinations']}"
        f"   (full grid: {summary['grid_fully_enumerated']})"
    )
    print()
    print(f"  {PRIMARY} (primary)")
    print(
        f"    complete / incomplete:     {summary[f'{PRIMARY}_complete']}"
        f" / {summary[f'{PRIMARY}_incomplete']}"
    )
    print(f"    PASS:                      {summary[f'{PRIMARY}_pass']}")
    print(f"    WARN (not FAIL):           {summary[f'{PRIMARY}_warn_not_fail']}")
    print(f"    FAIL:                      {summary[f'{PRIMARY}_fail']}")
    print()
    print(f"  {SECONDARY} (secondary)")
    print(
        f"    complete / incomplete:     {summary[f'{SECONDARY}_complete']}"
        f" / {summary[f'{SECONDARY}_incomplete']}"
    )
    print(f"    PASS:                      {summary[f'{SECONDARY}_pass']}")
    print()
    print(
        f"  both complete:               {summary['both_configurations_complete']}"
    )
    print(f"  BOTH configurations PASS:    {summary['both_configurations_pass']}")

    if closest:
        print()
        print("-" * 100)
        print(f"CLOSEST COMBINATIONS TO PUBLISHED {PRIMARY}")
        print("-" * 100)
        for entry in closest:
            print(
                f"  pos_bin={entry['positive_bin']} neg_bin={entry['negative_bin']}  "
                f"summed |delta| = {entry['summed_absolute_metric_delta']:.4f}  "
                f"verdict {entry['verdict']}"
            )
            for name, values in entry["metrics"].items():
                reproduced = (
                    "n/a" if values["reproduced"] is None else f"{values['reproduced']:.4f}"
                )
                signed = (
                    "n/a" if values["signed_delta"] is None else f"{values['signed_delta']:+.4f}"
                )
                print(
                    f"      {name:<30}{values['published']:>10.4f}{reproduced:>12}"
                    f"{signed:>12}  {values['status']}"
                )

    print()
    print("=" * 100)
    print(f"INTERPRETATION: {reading['headline']}")
    print("=" * 100)
    print(f"  {reading['message']}")
    print()
    print(
        "  This is a sensitivity analysis. The released NBC files are unmodified "
        "and remain\n  the experiment's baseline. No combination here may be "
        "adopted as the histogram."
    )
    print()
    print(f"Written to {output_path}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
