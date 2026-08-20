import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline_core import BSEDetector, build_nbc_histograms
from src.ddre_core import DDREDetector, ULSIFDensityRatio
from src.evaluation import evaluate_detector, prediction_rows, summarize_method
from src.utils import EntailmentScorer
from src.wang_data import (
    group_split_records,
    load_nbc_pairs,
    load_sentence_records,
)


RANDOM_STATE = 42
OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper experiment: Wang et al. Bayesian sequential estimation versus "
            "retrieval-aware direct density-ratio estimation (uLSIF)."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument("--cache-path", default="results/wang_nli_cache.sqlite")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--live-inference",
        action="store_true",
        help=(
            "Bypass the NLI cache during final test evaluation. This is much slower "
            "but gives hardware-dependent wall-clock inference measurements."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--c-miss", type=float, default=28.0)
    parser.add_argument("--c-false-alarm", type=float, default=96.0)
    parser.add_argument("--c-retrieve", type=float, default=1.0)
    parser.add_argument("--p0", type=float, default=0.5)
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--nbc-per-class", type=int, default=200)
    parser.add_argument(
        "--quality-tolerance",
        type=float,
        default=0.005,
        help="Validation tolerance when seeking a DDRE configuration that preserves BSE quality.",
    )
    parser.add_argument(
        "--retrieval-penalty",
        type=float,
        default=0.05,
        help="Fallback DDRE validation penalty on normalized document cost when no dominating threshold pair exists.",
    )
    parser.add_argument(
        "--no-push-results",
        action="store_true",
        help="Do not automatically commit/push full-run result artifacts to the current GitHub branch.",
    )
    return parser.parse_args()


def clear_cache(cache_path):
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(cache_path) + suffix)
        if path.exists():
            path.unlink()


def stratified_subset(records, n_samples, seed):
    if len(records) <= n_samples:
        return list(records)
    rng = np.random.default_rng(seed)
    by_label = {0: [], 1: []}
    for record in records:
        by_label[record.label].append(record)

    selected = []
    for label in (0, 1):
        target = max(1, int(round(n_samples * len(by_label[label]) / len(records))))
        target = min(target, len(by_label[label]))
        indices = rng.choice(len(by_label[label]), size=target, replace=False)
        selected.extend(by_label[label][int(i)] for i in indices)

    if len(selected) > n_samples:
        rng.shuffle(selected)
        selected = selected[:n_samples]
    return sorted(selected, key=lambda r: (r.passage_index, r.sentence_index))


def evaluate_quiet(detector, records, scorer):
    results = [
        detector.detect_sentence(record, scorer, use_cache=True)
        for record in records
    ]
    return summarize_method(records, results), results


def tune_ddre_thresholds(
    ratio_estimator,
    validation_records,
    scorer,
    baseline_metrics,
    *,
    p0,
    c_miss,
    c_false_alarm,
    max_docs,
    quality_tolerance,
    retrieval_penalty,
):
    """Select a retrieval stopping interval on validation data only.

    Primary rule: among configurations that preserve the BSE official baseline's
    factual AUC-PR and balanced PR-AUC within a small tolerance, choose the one
    using the fewest documents. If none qualifies, use a predeclared penalized
    quality/cost objective and explicitly record that the dominance condition was
    not achieved on validation data.
    """
    lower_grid = np.round(np.arange(0.05, 0.41, 0.05), 2)
    upper_grid = np.round(np.arange(0.60, 0.96, 0.05), 2)
    candidates = []

    baseline_factual = baseline_metrics["factual"]["auc_pr"]
    baseline_balanced = baseline_metrics["balanced_pr_auc"]

    for lower in lower_grid:
        for upper in upper_grid:
            if lower >= upper:
                continue
            detector = DDREDetector(
                ratio_estimator,
                lower_threshold=float(lower),
                upper_threshold=float(upper),
                p0=p0,
                c_miss=c_miss,
                c_false_alarm=c_false_alarm,
                max_docs=max_docs,
            )
            metrics, _ = evaluate_quiet(detector, validation_records, scorer)
            avg_docs = metrics["efficiency"]["avg_retrieved_documents_per_sentence"]
            normalized_docs = avg_docs / max(1.0, float(max_docs))
            qualifies = (
                metrics["factual"]["auc_pr"]
                >= baseline_factual - quality_tolerance
                and metrics["balanced_pr_auc"]
                >= baseline_balanced - quality_tolerance
            )
            candidate = {
                "lower": float(lower),
                "upper": float(upper),
                "factual_auc_pr": metrics["factual"]["auc_pr"],
                "nonfactual_auc_pr": metrics["nonfactual"]["auc_pr"],
                "balanced_pr_auc": metrics["balanced_pr_auc"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "avg_documents": avg_docs,
                "avg_nli_span_calls": metrics["efficiency"]["avg_nli_span_calls_per_sentence"],
                "preserves_baseline_quality": bool(qualifies),
                "fallback_objective": float(
                    metrics["balanced_pr_auc"] - retrieval_penalty * normalized_docs
                ),
            }
            candidates.append(candidate)

    feasible = [c for c in candidates if c["preserves_baseline_quality"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda c: (
                c["avg_documents"],
                -c["balanced_pr_auc"],
                -c["factual_auc_pr"],
            ),
        )
        selection_rule = (
            "minimum retrieval cost among validation configurations preserving "
            "BSE-official factual and balanced PR-AUC within tolerance"
        )
    else:
        selected = max(
            candidates,
            key=lambda c: (
                c["fallback_objective"],
                c["balanced_pr_auc"],
                -c["avg_documents"],
            ),
        )
        selection_rule = (
            "fallback penalized balanced-PR-AUC/retrieval objective; no DDRE "
            "threshold pair preserved BSE-official validation quality"
        )

    return selected, candidates, selection_rule


def hypothesis_comparison(ddre, baseline):
    ddre_docs = ddre["efficiency"]["avg_retrieved_documents_per_sentence"]
    base_docs = baseline["efficiency"]["avg_retrieved_documents_per_sentence"]
    ddre_nli = ddre["efficiency"]["avg_nli_span_calls_per_sentence"]
    base_nli = baseline["efficiency"]["avg_nli_span_calls_per_sentence"]

    retrieval_reduction = (
        1.0 - ddre_docs / base_docs if base_docs > 0 else None
    )
    nli_reduction = 1.0 - ddre_nli / base_nli if base_nli > 0 else None
    factual_delta = ddre["factual"]["auc_pr"] - baseline["factual"]["auc_pr"]
    nonfactual_delta = (
        ddre["nonfactual"]["auc_pr"] - baseline["nonfactual"]["auc_pr"]
    )
    balanced_delta = ddre["balanced_pr_auc"] - baseline["balanced_pr_auc"]

    supported = (
        retrieval_reduction is not None
        and retrieval_reduction > 0
        and factual_delta > 0
        and balanced_delta >= 0
    )

    return {
        "primary_baseline": "bse_official",
        "factual_auc_pr_delta": factual_delta,
        "nonfactual_auc_pr_delta": nonfactual_delta,
        "balanced_pr_auc_delta": balanced_delta,
        "retrieved_documents_reduction_fraction": retrieval_reduction,
        "nli_span_calls_reduction_fraction": nli_reduction,
        "hypothesis_supported_on_test": bool(supported),
        "support_rule": (
            "DDRE must use fewer retrieved documents, improve factual AUC-PR, "
            "and not reduce balanced PR-AUC versus BSE official."
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def auto_push_results(paths):
    """Commit/push only generated result artifacts; never raw data or NLI cache."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "add", *[str(path) for path in paths]], check=True)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            check=False,
        )
        if staged.returncode == 0:
            print("No result changes to commit.")
            return {"pushed": False, "reason": "no changes"}

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(
            ["git", "commit", "-m", f"research: update full experiment results ({stamp})"],
            check=True,
        )
        subprocess.run(["git", "push", "origin", branch], check=True)
        print(f"Automatically pushed result artifacts to origin/{branch}.")
        return {"pushed": True, "branch": branch}
    except Exception as exc:
        print(f"WARNING: experiment succeeded but automatic result push failed: {exc}")
        return {"pushed": False, "reason": str(exc)}


def main():
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    Path("results").mkdir(exist_ok=True)
    if args.rebuild_cache:
        clear_cache(args.cache_path)

    print("Loading Wang et al. released experimental artifacts...")
    records = load_sentence_records(args.data_root, strict=True)
    validation_records, test_records, split_metadata = group_split_records(
        records,
        validation_fraction=args.validation_fraction,
        random_state=RANDOM_STATE,
    )

    if args.smoke_test:
        validation_records = stratified_subset(validation_records, 20, RANDOM_STATE)
        test_records = stratified_subset(test_records, 40, RANDOM_STATE + 1)
        print("*** SMOKE TEST: outputs are debugging-only and are not paper results. ***")

    pos_pairs, neg_pairs = load_nbc_pairs(
        args.data_root,
        per_class=args.nbc_per_class,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    try:
        print("\nBuilding the published BSE NBC distributions from Wang's separate NBC data...")
        pos_hist, neg_hist, factual_nbc_scores, hallucinated_nbc_scores = build_nbc_histograms(
            pos_pairs,
            neg_pairs,
            scorer,
        )

        print("\nFitting genuine direct density-ratio estimator (uLSIF)...")
        ratio_estimator = ULSIFDensityRatio(random_state=RANDOM_STATE).fit(
            factual_nbc_scores,
            hallucinated_nbc_scores,
        )
        print(
            f"uLSIF selected sigma={ratio_estimator.sigma:.6f}, "
            f"lambda={ratio_estimator.lam:.6g}"
        )

        bse_official = BSEDetector(
            pos_hist,
            neg_hist,
            mode="official",
            p0=args.p0,
            c_miss=args.c_miss,
            c_false_alarm=args.c_false_alarm,
            c_retrieve=args.c_retrieve,
            max_docs=args.max_docs,
        )
        bse_eq8 = BSEDetector(
            pos_hist,
            neg_hist,
            mode="eq8",
            p0=args.p0,
            c_miss=args.c_miss,
            c_false_alarm=args.c_false_alarm,
            c_retrieve=args.c_retrieve,
            max_docs=args.max_docs,
        )

        print("\nEvaluating BSE official implementation on validation data...")
        bse_val_metrics, _ = evaluate_detector(
            bse_official,
            validation_records,
            scorer,
            description="BSE official validation",
            use_cache=True,
        )

        print("\nTuning DDRE stopping thresholds on validation data only...")
        selected, threshold_table, selection_rule = tune_ddre_thresholds(
            ratio_estimator,
            validation_records,
            scorer,
            bse_val_metrics,
            p0=args.p0,
            c_miss=args.c_miss,
            c_false_alarm=args.c_false_alarm,
            max_docs=args.max_docs,
            quality_tolerance=args.quality_tolerance,
            retrieval_penalty=args.retrieval_penalty,
        )
        print(
            f"Selected DDRE interval: [{selected['lower']:.2f}, {selected['upper']:.2f}] "
            f"({selection_rule})"
        )

        ddre = DDREDetector(
            ratio_estimator,
            lower_threshold=selected["lower"],
            upper_threshold=selected["upper"],
            p0=args.p0,
            c_miss=args.c_miss,
            c_false_alarm=args.c_false_alarm,
            max_docs=args.max_docs,
        )

        use_cache_for_test = not args.live_inference
        print("\nFinal held-out test evaluation: BSE official...")
        bse_official_metrics, bse_official_results = evaluate_detector(
            bse_official,
            test_records,
            scorer,
            description="BSE official test",
            use_cache=use_cache_for_test,
        )
        print("\nFinal held-out test evaluation: BSE Equation 8...")
        bse_eq8_metrics, bse_eq8_results = evaluate_detector(
            bse_eq8,
            test_records,
            scorer,
            description="BSE Eq8 test",
            use_cache=use_cache_for_test,
        )
        print("\nFinal held-out test evaluation: DDRE/uLSIF...")
        ddre_metrics, ddre_results = evaluate_detector(
            ddre,
            test_records,
            scorer,
            description="DDRE uLSIF test",
            use_cache=use_cache_for_test,
        )

        comparison = hypothesis_comparison(ddre_metrics, bse_official_metrics)
        run_timestamp = datetime.now(timezone.utc).isoformat()

        summary = {
            "experiment_version": "wang-aligned-ddre-v1",
            "run_timestamp_utc": run_timestamp,
            "run_mode": "smoke-test" if args.smoke_test else "full-paper-experiment",
            "research_question": (
                "Can retrieval-aware direct density-ratio estimation reduce the "
                "computational cost of hallucination detection while improving "
                "factuality detection performance compared with Bayesian sequential estimation?"
            ),
            "source_baseline": {
                "paper": "Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation",
                "authors": "Wang et al.",
                "venue": "EMNLP 2023",
                "repository": "https://github.com/xhwang22/HallucinationDetection",
            },
            "controlled_comparison": {
                "same_selfcheckgpt_sentences": True,
                "same_released_subclaims": True,
                "same_released_retrieved_web_documents": True,
                "same_nli_model": True,
                "same_nbc_training_pairs": True,
                "only_statistical_decision_mechanism_changes": True,
            },
            "config": {
                "model_name": args.model_name,
                "device": str(device),
                "batch_size": batch_size,
                "p0": args.p0,
                "C_M": args.c_miss,
                "C_FA": args.c_false_alarm,
                "C_retrieve": args.c_retrieve,
                "max_documents_per_subclaim": args.max_docs,
                "nbc_examples_per_class": args.nbc_per_class,
                "test_uses_nli_cache": use_cache_for_test,
                "wall_clock_note": (
                    "cached wall-clock is not a live model latency benchmark"
                    if use_cache_for_test
                    else "live uncached NLI inference"
                ),
            },
            "split": split_metadata,
            "execution_sentences": {
                "validation": len(validation_records),
                "test": len(test_records),
            },
            "bse_official_validation": bse_val_metrics,
            "ddre": {
                "estimator": "uLSIF direct density-ratio estimation",
                "ratio_definition": "p(entailment_score | factual) / p(entailment_score | hallucinated)",
                "sigma": ratio_estimator.sigma,
                "lambda": ratio_estimator.lam,
                "cv_table": ratio_estimator.cv_table,
                "selected_lower_threshold": selected["lower"],
                "selected_upper_threshold": selected["upper"],
                "threshold_selection_rule": selection_rule,
                "threshold_validation_table": threshold_table,
            },
            "test_metrics": {
                "bse_official": bse_official_metrics,
                "bse_equation8": bse_eq8_metrics,
                "ddre_ulsif": ddre_metrics,
            },
            "hypothesis_test": comparison,
        }

        if args.smoke_test:
            summary_path = Path("results/smoke_summary.json")
            predictions_path = Path("results/smoke_predictions.csv")
            model_path = Path("results/smoke_ddre_model.json")
        else:
            summary_path = Path("results/latest_summary.json")
            predictions_path = Path("results/latest_predictions.csv")
            model_path = Path("results/latest_ddre_model.json")

        all_rows = []
        all_rows.extend(prediction_rows("bse_official", test_records, bse_official_results))
        all_rows.extend(prediction_rows("bse_equation8", test_records, bse_eq8_results))
        all_rows.extend(prediction_rows("ddre_ulsif", test_records, ddre_results))

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_csv(predictions_path, all_rows)
        with model_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "estimator": "uLSIF",
                    "sigma": ratio_estimator.sigma,
                    "lambda": ratio_estimator.lam,
                    "centers": ratio_estimator.centers.ravel().tolist(),
                    "alpha": ratio_estimator.alpha.tolist(),
                    "lower_threshold": selected["lower"],
                    "upper_threshold": selected["upper"],
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 88)
        print("HELD-OUT TEST SUMMARY")
        print("=" * 88)
        for name, metrics in (
            ("BSE official", bse_official_metrics),
            ("BSE Eq.8", bse_eq8_metrics),
            ("DDRE/uLSIF", ddre_metrics),
        ):
            print(name)
            print(f"  Factual AUC-PR:       {metrics['factual']['auc_pr']:.4f}")
            print(f"  Nonfactual AUC-PR:    {metrics['nonfactual']['auc_pr']:.4f}")
            print(f"  Balanced PR-AUC:      {metrics['balanced_pr_auc']:.4f}")
            print(f"  Accuracy:             {metrics['accuracy']:.4f}")
            print(
                f"  Avg retrieved docs:   "
                f"{metrics['efficiency']['avg_retrieved_documents_per_sentence']:.3f}"
            )
            print(
                f"  Avg NLI span calls:   "
                f"{metrics['efficiency']['avg_nli_span_calls_per_sentence']:.3f}"
            )
        print("-" * 88)
        print(
            "Hypothesis supported on held-out test: "
            f"{comparison['hypothesis_supported_on_test']}"
        )
        if comparison["retrieved_documents_reduction_fraction"] is not None:
            print(
                "DDRE document-cost reduction vs BSE official: "
                f"{100 * comparison['retrieved_documents_reduction_fraction']:.2f}%"
            )
        print(
            "DDRE factual AUC-PR delta vs BSE official: "
            f"{comparison['factual_auc_pr_delta']:+.4f}"
        )
        print("=" * 88)
        print(f"Summary: {summary_path}")
        print(f"Predictions: {predictions_path}")
        print(f"DDRE model: {model_path}")

        if not args.smoke_test and not args.no_push_results:
            push_status = auto_push_results(
                [summary_path, predictions_path, model_path]
            )
            print(f"Result push status: {push_status}")

    finally:
        scorer.close()


if __name__ == "__main__":
    main()
