"""Gate 1: reproduce the released Wang et al. BSE baseline and judge the result.

Two modes:

``--probe``
    Cheap diagnostic. Scores only the ~400 released NBC evidence pairs, builds
    the Laplace-smoothed histograms, and reports whether ``bse_official`` will
    retrieve any documents at all. Because every subclaim starts at P = P0 and
    the histograms are global, the first-iteration ``stop_cost > search_cost``
    test is a dataset-wide constant: either every subclaim retrieves at least
    one document, or none does. Running this first avoids committing to a full
    NLI pass against a baseline that would be degenerate.

default (full gate)
    Reproduces both published cost configurations on the complete released
    evidence set, compares every metric against Table 1 using the predeclared
    tolerances in ``src/reproduction_gate.py``, records full provenance, and
    exits non-zero on FAIL.

This script must not change ``bse_official`` behaviour. It observes and judges.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow this file to be executed directly from the repository root with:
#   python scripts/reproduce_wang_baseline.py
# Python otherwise puts only scripts/ on sys.path, so the sibling src/ package
# is not importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline_core import BSEDetector, build_nbc_histograms, stop_cost
from src.diagnostic_probe import select_device
from src.evaluation import evaluate_detector
from src.provenance import collect_provenance
from src.reproduction_gate import (
    OFFICIAL_NLI_MODEL,
    PUBLISHED_TABLE1,
    STATUS_FAIL,
    STATUS_PASS,
    evaluate_gate,
    evaluate_protocol_preconditions,
    format_preconditions,
    format_report,
)
from src.utils import SCORE_VERSION, EntailmentScorer
from src.wang_data import load_nbc_pairs, load_sentence_records


# Single source of truth lives in src/reproduction_gate.py, where the protocol
# preconditions are enforced.
OFFICIAL_MODEL = OFFICIAL_NLI_MODEL

# Wang et al. report sampling s = 200 factual and s = 200 nonfactual examples.
# The released NBC files do not necessarily contain exactly that many; the
# actual count is recorded dynamically alongside this protocol value.
PAPER_PROTOCOL_NBC_PER_CLASS = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gate 1: reproduce Wang et al. BSE on their full released SelfCheckGPT "
            "evidence set and judge the result against published Table 1 values."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument("--cache-path", default="results/wang_nli_cache.sqlite")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--nbc-per-class", type=int, default=200)
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Only score the released NBC pairs and report whether bse_official "
            "will retrieve documents. Exits non-zero if it would not."
        ),
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def retrieval_condition(p0, c_miss, c_false_alarm, c_retrieve):
    """Describe the first-iteration retrieve/stop condition at P = p0.

    Wang's released loop retrieves only while ``stop_cost > search_cost``, and
    checks this before the first document. At P = p0 that comparison depends
    only on the NBC histograms, so it is the same for every subclaim.
    """
    initial_stop_cost = stop_cost(p0, c_miss, c_false_alarm)
    budget = initial_stop_cost - c_retrieve
    return {
        "p0": p0,
        "initial_stop_cost": initial_stop_cost,
        "c_retrieve": c_retrieve,
        "expected_next_stop_cost_budget": budget,
        "retrieves_if_expected_posterior_above": (
            None if c_miss == 0 else 1.0 - budget / float(c_miss)
        ),
        "retrieves_if_expected_posterior_below": (
            None if c_false_alarm == 0 else budget / float(c_false_alarm)
        ),
    }


def build_scorer(args):
    # CUDA -> MPS -> CPU. CUDA behaviour is unchanged; Apple Metal is consulted
    # only where CUDA is absent. The selected name is what collect_provenance
    # records below, so the artifact states the backend actually used.
    device = torch.device(select_device(torch))
    # Batch size is deliberately NOT changed for MPS. The canonical
    # Wang-fidelity path is batch size 1 and the formal run passes it
    # explicitly; inventing a new default here would be a scientific decision
    # dressed up as a hardware fix.
    batch_size = args.batch_size or (8 if torch.cuda.is_available() else 2)
    print(f"NLI model: {args.model_name}")
    print(f"Device: {device}; batch size: {batch_size}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()
    scorer = EntailmentScorer(
        tokenizer,
        model,
        args.model_name,
        cache_path=args.cache_path,
        batch_size=batch_size,
    )
    return scorer, tokenizer, model, device, batch_size


def load_nbc_with_counts(data_root, per_class):
    """Load NBC pairs and record both the on-disk and actually-used counts."""
    available_pos, available_neg = load_nbc_pairs(data_root, per_class=None)
    used_pos, used_neg = load_nbc_pairs(data_root, per_class=per_class)
    metadata = {
        "paper_protocol_s_per_class": PAPER_PROTOCOL_NBC_PER_CLASS,
        "requested_per_class": per_class,
        "available_in_released_files": {
            "positive": len(available_pos),
            "negative": len(available_neg),
        },
        "actually_loaded": {"positive": len(used_pos), "negative": len(used_neg)},
    }
    metadata["matches_paper_protocol"] = (
        metadata["actually_loaded"]["positive"] == PAPER_PROTOCOL_NBC_PER_CLASS
        and metadata["actually_loaded"]["negative"] == PAPER_PROTOCOL_NBC_PER_CLASS
    )
    return used_pos, used_neg, metadata


def reproduced_metrics_for_gate(metrics):
    """Map evaluation output onto the metric names the gate compares.

    Table 1's evidence count is documents per *sentence*. Documents per
    subclaim is carried through as a diagnostic only.
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


def run_probe(provenance, preconditions, nbc_metadata, pos_hist, neg_hist):
    print("\nProbing first-iteration retrieval behaviour of bse_official...")
    configurations = {}
    any_zero = False
    for name, reference in PUBLISHED_TABLE1.items():
        c_miss = reference["c_miss"]
        c_false_alarm = reference["c_false_alarm"]
        detector = BSEDetector(
            pos_hist,
            neg_hist,
            mode="official",
            p0=0.5,
            c_miss=c_miss,
            c_false_alarm=c_false_alarm,
            c_retrieve=1,
            max_docs=10,
        )
        # Private accessor used deliberately: the expected next posterior is the
        # quantity that decides the dataset-wide retrieve/stop constant, and is
        # exactly what must be inspected when the baseline retrieves nothing.
        expected_posterior = detector._official_expected_next_posterior(0.5)
        will_retrieve = detector.should_continue(0.5)
        any_zero = any_zero or not will_retrieve
        configurations[name] = {
            "c_miss": c_miss,
            "c_false_alarm": c_false_alarm,
            "expected_next_posterior_at_p0": expected_posterior,
            "continue_cost_at_p0": detector.continue_cost(0.5),
            "condition": retrieval_condition(0.5, c_miss, c_false_alarm, 1),
            "will_retrieve_first_document": bool(will_retrieve),
        }

    report = {
        "mode": "probe",
        "purpose": (
            "Cheap NBC-only diagnostic: does the released bse_official control "
            "flow retrieve any evidence on the real NBC histograms?"
        ),
        "provenance": provenance,
        "protocol_preconditions": preconditions,
        "formal_gate1_run": preconditions["overall"] == STATUS_PASS,
        "nbc": nbc_metadata,
        "nbc_histograms_laplace_smoothed": {
            "positive": list(pos_hist),
            "negative": list(neg_hist),
        },
        "configurations": configurations,
        "any_configuration_retrieves_nothing": bool(any_zero),
    }

    # Preconditions were printed by main() moments ago; they are carried in the
    # JSON report rather than reprinted here.
    print("\n" + "=" * 100)
    print("BASELINE RETRIEVAL PROBE")
    print("=" * 100)
    print(f"Positive NBC histogram (Laplace-smoothed): {list(pos_hist)}")
    print(f"Negative NBC histogram (Laplace-smoothed): {list(neg_hist)}")
    for name, result in configurations.items():
        condition = result["condition"]
        print(f"\n[{name}] C_M={result['c_miss']} C_FA={result['c_false_alarm']}")
        print(f"  stop cost at P0=0.5:            {condition['initial_stop_cost']:.4f}")
        print(f"  continue cost at P0=0.5:        {result['continue_cost_at_p0']:.4f}")
        print(f"  expected next posterior E[P1]:  {result['expected_next_posterior_at_p0']:.4f}")
        print(
            "  retrieves if E[P1] > "
            f"{condition['retrieves_if_expected_posterior_above']:.4f} "
            f"or E[P1] < {condition['retrieves_if_expected_posterior_below']:.4f}"
        )
        print(f"  WILL RETRIEVE FIRST DOCUMENT:   {result['will_retrieve_first_document']}")
    print("-" * 100)
    if any_zero:
        print(
            "RESULT: at least one configuration retrieves nothing. bse_official would be\n"
            "degenerate (P = P0 for every sentence). This is a stop condition: investigate\n"
            "before running the full gate. Do NOT change bse_official control flow to fix it."
        )
    else:
        print("RESULT: both configurations retrieve evidence.")
    if preconditions["overall"] != STATUS_PASS:
        print(
            "PROTOCOL: preconditions failed "
            f"({', '.join(preconditions['failures'])}); this run cannot support a "
            "formal Gate 1 PASS."
        )
    elif not any_zero:
        print("Proceed to the full gate.")
    print("=" * 100)
    failed = any_zero or preconditions["overall"] != STATUS_PASS
    return report, (1 if failed else 0)


def run_gate(scorer, provenance, preconditions, nbc_metadata, pos_hist, neg_hist, records):
    total_subclaims = sum(len(record.subclaims) for record in records)
    reproduced = {}
    raw_metrics = {}
    for name, reference in PUBLISHED_TABLE1.items():
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
        metrics, _ = evaluate_detector(
            detector,
            records,
            scorer,
            description=f"Wang reproduction {name}",
            use_cache=True,
        )
        raw_metrics[name] = metrics
        reproduced[name] = reproduced_metrics_for_gate(metrics)

    gate_report = evaluate_gate(reproduced, preconditions)

    report = {
        "mode": "gate",
        "purpose": (
            "Gate 1: does this repository reproduce Wang et al. closely enough to "
            "use bse_official as the primary baseline?"
        ),
        "provenance": provenance,
        "dataset": {
            "sentences": len(records),
            "passages": len({record.passage_index for record in records}),
            "subclaims": total_subclaims,
        },
        "nbc": nbc_metadata,
        "nbc_histograms_laplace_smoothed": {
            "positive": list(pos_hist),
            "negative": list(neg_hist),
        },
        "gate": gate_report,
        "full_metrics": raw_metrics,
    }

    print("\n" + format_report(gate_report))
    return report, (1 if gate_report["overall_verdict"] == STATUS_FAIL else 0)


def main():
    args = parse_args()
    Path("results").mkdir(exist_ok=True)

    scorer, tokenizer, model, device, batch_size = build_scorer(args)
    try:
        provenance = collect_provenance(
            model_name=args.model_name,
            data_root=args.data_root,
            device=device,
            batch_size=batch_size,
            score_version=SCORE_VERSION,
            tokenizer=tokenizer,
            model=model,
            repo_root=PROJECT_ROOT,
        )
        preconditions = evaluate_protocol_preconditions(provenance)
        print()
        print(format_preconditions(preconditions))

        default_output = (
            "results/wang_probe.json" if args.probe else "results/wang_reproduction.json"
        )
        output_path = Path(args.output or default_output)

        # A truncation mismatch means every entailment score would be computed
        # over different inputs than the released implementation used, so there
        # is nothing worth scoring. Abort before spending any NLI compute. The
        # fix belongs at the source of the mismatch, never in src/utils.py to
        # make this check pass.
        if preconditions["truncation_equivalence"] != STATUS_PASS:
            print(
                "\nABORTING: truncation equivalence to Wang's truncation=True is not "
                "established.\n"
                + "\n".join(
                    f"  - {message}"
                    for message in preconditions["failure_messages"]
                )
                + "\nNo NLI inference was run. Investigate the mismatch; do not edit "
                "src/utils.py to silence this check."
            )
            report = {
                "mode": "probe" if args.probe else "gate",
                "aborted": True,
                "abort_reason": "truncation_equivalence precondition failed",
                "provenance": provenance,
                "protocol_preconditions": preconditions,
                "formal_gate1_run": False,
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            print(f"\nWritten to {output_path}")
            return 1

        pos_pairs, neg_pairs, nbc_metadata = load_nbc_with_counts(
            args.data_root, args.nbc_per_class
        )
        print(
            "NBC pairs: paper protocol s="
            f"{nbc_metadata['paper_protocol_s_per_class']} per class; "
            f"released files contain {nbc_metadata['available_in_released_files']}; "
            f"actually loaded {nbc_metadata['actually_loaded']}"
        )
        pos_hist, neg_hist, _, _ = build_nbc_histograms(pos_pairs, neg_pairs, scorer)

        if args.probe:
            report, exit_code = run_probe(
                provenance, preconditions, nbc_metadata, pos_hist, neg_hist
            )
        else:
            records = load_sentence_records(args.data_root, strict=True)
            report, exit_code = run_gate(
                scorer,
                provenance,
                preconditions,
                nbc_metadata,
                pos_hist,
                neg_hist,
                records,
            )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWritten to {output_path}")
        return exit_code
    finally:
        scorer.close()


if __name__ == "__main__":
    sys.exit(main())
