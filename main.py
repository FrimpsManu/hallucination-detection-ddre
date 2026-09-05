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
from src.ddre_core import (
    CANDIDATE_LOWER_GRID,
    CANDIDATE_UPPER_GRID,
    DDREDetector,
    ULSIFDensityRatio,
    cost_consistent_thresholds,
)
from src.evaluation import (
    evaluate_detector,
    evaluate_detector_with_identity,
    prediction_rows,
    summarize_method,
)
from src.threshold_selection import (
    SAFEGUARD_NOTE,
    candidate_record,
    select_threshold_configuration,
)
from src.paired_bootstrap import (
    CONFIRMATORY_SPLIT_SEED,
    CONFIRMATORY_VALIDATION_FRACTION,
    BootstrapUnavailable,
    PairedInputMismatch,
    assess_claim,
    confirmatory_protocol,
    paired_passage_bootstrap,
)
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
    # D-12. Publishing is OPT-IN. A normal full run writes its artifacts to
    # disk and stops there; committing and pushing them is a separate,
    # explicit request. --no-push-results is kept only so existing invocations
    # keep working -- it is now a no-op, because not pushing is the default.
    publication = parser.add_mutually_exclusive_group()
    publication.add_argument(
        "--push-results",
        action="store_true",
        default=False,
        help=(
            "Explicitly allow committing and pushing generated full-run result "
            "artifacts to the current branch. Off by default: without this flag "
            "results are written locally and nothing is staged, committed or "
            "pushed. Protected branches (main/master) are never auto-pushed, "
            "with or without this flag."
        ),
    )
    publication.add_argument(
        "--no-push-results",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # deprecated no-op; not pushing is the default
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

    The search space is derived from the configured costs before any candidate
    is evaluated: a stopping threshold that contradicts the final cost rule is
    never scored, so it cannot be selected. See ``cost_consistent_thresholds``.

    Primary rule: among configurations that preserve the BSE-official baseline's
    nonfactual, factual AND balanced PR-AUC within the validation tolerance,
    choose the one using the fewest documents. If none qualifies, use the
    predeclared penalized quality/cost objective and record that the selection
    is NOT confirmatory. The safeguards and the selection live in
    ``src/threshold_selection.py`` so the decision rule is reviewable on its own.
    """
    search_space = cost_consistent_thresholds(
        c_miss, c_false_alarm, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID
    )
    lower_grid = search_space["effective_lower_grid"]
    upper_grid = search_space["effective_upper_grid"]
    candidates = []

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
            candidates.append(
                candidate_record(
                    lower, upper, metrics, baseline_metrics,
                    quality_tolerance=quality_tolerance,
                    retrieval_penalty=retrieval_penalty,
                    max_docs=max_docs,
                )
            )

    selected, selection_rule, confirmatory_selection = (
        select_threshold_configuration(candidates)
    )
    feasible = [c for c in candidates if c["preserves_baseline_quality"]]

    selected["confirmatory_validation_selection"] = confirmatory_selection
    search_space["threshold_pairs_scored"] = len(candidates)
    search_space["feasible_pairs"] = len(feasible)
    search_space["quality_tolerance"] = float(quality_tolerance)
    search_space["safeguards"] = SAFEGUARD_NOTE
    search_space["confirmatory_validation_selection"] = confirmatory_selection
    return selected, candidates, selection_rule, search_space


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

    # D-08. These are DESCRIPTIVE point estimates only. They carry no
    # uncertainty, so they cannot and do not decide whether the scientific
    # claim holds; that is settled by the frozen paired bootstrap in
    # src/paired_bootstrap.py. The old point-estimate "supported" boolean,
    # computed from these numbers alone, is deliberately gone.
    return {
        "primary_baseline": "bse_official",
        "estimate_kind": "descriptive point estimates; NOT a scientific claim",
        "factual_auc_pr_delta": factual_delta,
        "nonfactual_auc_pr_delta": nonfactual_delta,
        "balanced_pr_auc_delta": balanced_delta,
        "retrieved_documents_reduction_fraction": retrieval_reduction,
        "nli_span_calls_reduction_fraction": nli_reduction,
        "sign_conventions": {
            "performance_delta": "DDRE - BSE; positive favours DDRE",
            "reduction_fraction": "1 - DDRE/BSE; positive favours DDRE",
        },
        "note": (
            "Effect sizes for description only. The confirmatory decision lives "
            "in claim_assessment, which is driven by the pre-registered paired "
            "passage-level bootstrap. A point estimate can never set "
            "primary_claim_supported."
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


PROTECTED_BRANCHES = frozenset({"main", "master"})


def auto_push_results(paths):
    """Commit/push only generated result artifacts; never raw data or NLI cache.

    The branch is resolved and checked BEFORE anything is staged. This
    repository's workflow is branch -> PR -> review -> merge, and an experiment
    helper must never bypass that by committing generated results straight to
    the default branch. Staging first and refusing afterwards would still leave
    the index dirty on a protected branch, so the check comes first.

    A detached or otherwise ambiguous HEAD is refused for the same reason: there
    is no branch to push to, and guessing one is exactly the kind of helpfulness
    that produces a commit nobody asked for.

    A pre-existing dirty index is refused too. ``git commit -m`` commits
    EVERYTHING already staged, so someone who had run ``git add
    src/unrelated_work.py`` before starting the experiment would find that file
    swept into a commit labelled "update full experiment results" -- which this
    function's own docstring says it never does. Nothing is unstaged to work
    around it, and no partial commit is attempted: the index belongs to whoever
    staged it, and it is theirs to review and commit.

    The order is: resolve branch -> protected/detached -> pre-existing index ->
    add -> post-add no-change -> commit -> push. Every refusal happens before
    the index is touched.
    """
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        if branch in PROTECTED_BRANCHES:
            print(
                f"Refusing to auto-push: '{branch}' is a protected branch. "
                "Results are on disk; open a branch and a pull request to "
                "publish them."
            )
            return {"pushed": False, "reason": "protected branch", "branch": branch}
        if not branch or branch == "HEAD":
            print(
                "Refusing to auto-push: HEAD is detached, so there is no branch "
                "to push to. Results are on disk."
            )
            return {"pushed": False, "reason": "detached HEAD", "branch": branch}
        # Fail closed on ANY answer but a clean index: 0 is clean, 1 is dirty,
        # and anything else means the check itself did not work, which is not a
        # licence to proceed.
        preexisting = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            check=False,
        )
        if preexisting.returncode != 0:
            if preexisting.returncode == 1:
                detail = (
                    "the index already contains staged changes that must be "
                    "reviewed and committed separately"
                )
                reason = "pre-existing staged changes"
            else:
                detail = (
                    "the staged-changes check itself failed "
                    f"(git exited {preexisting.returncode})"
                )
                reason = "index check failed"
            print(
                f"Refusing to auto-push: {detail}. Result artifacts remain on "
                "disk; nothing was staged, committed or pushed, and the "
                "existing index was not modified."
            )
            return {"pushed": False, "reason": reason, "branch": branch}

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
        fit_diagnostics = ratio_estimator.fit_diagnostics
        print(
            f"uLSIF final-fit sanity check passed: "
            f"{fit_diagnostics['sanity_check_passed']} "
            f"({fit_diagnostics['n_positive_alpha']} of "
            f"{fit_diagnostics['n_alpha']} coefficients strictly positive; "
            f"raw fitted ratio on the training support in "
            f"[{fit_diagnostics['raw_ratio_min_on_all_train']:.6g}, "
            f"{fit_diagnostics['raw_ratio_max_on_all_train']:.6g}])"
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
        selected, threshold_table, selection_rule, threshold_search_space = tune_ddre_thresholds(
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
            f"Cost decision threshold t = C_M/(C_M+C_FA) = "
            f"{threshold_search_space['cost_decision_threshold']:.10f}"
        )
        print(
            f"  candidate lower grid: {threshold_search_space['candidate_lower_grid']}"
        )
        print(
            f"  effective lower grid: {threshold_search_space['effective_lower_grid']} "
            f"(excluded {threshold_search_space['excluded_lower_grid']}: lower > t)"
        )
        print(
            f"  effective upper grid: {threshold_search_space['effective_upper_grid']} "
            f"(excluded {threshold_search_space['excluded_upper_grid']}: upper <= t)"
        )
        print(
            f"  threshold pairs evaluated: "
            f"{threshold_search_space['threshold_pairs_evaluated']} of "
            f"{threshold_search_space['candidate_pairs_before_filtering']} candidates"
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
        bse_official_metrics, bse_official_results, bse_official_observations = (
            evaluate_detector_with_identity(
                bse_official,
                test_records,
                scorer,
                description="BSE official test",
                use_cache=use_cache_for_test,
            )
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
        ddre_metrics, ddre_results, ddre_observations = (
            evaluate_detector_with_identity(
                ddre,
                test_records,
                scorer,
                description="DDRE uLSIF test",
                use_cache=use_cache_for_test,
            )
        )

        comparison = hypothesis_comparison(ddre_metrics, bse_official_metrics)

        # Confirmatory analysis, under the protocol frozen before any held-out
        # DDRE result was inspected. A failure here makes the claim
        # NOT_CONFIRMATORY -- never NOT_SUPPORTED, which would misreport an
        # unavailable analysis as a negative result.
        print("\nRunning the pre-registered paired passage-level bootstrap...")
        bootstrap = None
        bootstrap_error = None
        try:
            # Identity-bearing observations, built inside the evaluation loop,
            # so "paired" is verified rather than assumed.
            bootstrap = paired_passage_bootstrap(
                ddre_observations, bse_official_observations
            )
        except (BootstrapUnavailable, PairedInputMismatch) as exc:
            bootstrap_error = f"{type(exc).__name__}: {exc}"
            print(f"  CONFIRMATORY BOOTSTRAP UNAVAILABLE: {bootstrap_error}")

        # Re-derive the canonical split from the released records with the
        # FROZEN fraction and seed, and compare passage IDs. "190 passages" is
        # not the frozen test set; a different 190 would be a different
        # experiment. No held-out result is inspected to do this.
        _, _, frozen_split = group_split_records(
            records,
            validation_fraction=CONFIRMATORY_VALIDATION_FRACTION,
            random_state=CONFIRMATORY_SPLIT_SEED,
        )

        claim_assessment = assess_claim(
            bootstrap,
            validation_selection_confirmatory=selected[
                "confirmatory_validation_selection"
            ],
            quality_tolerance=args.quality_tolerance,
            smoke_test=args.smoke_test,
            bootstrap_error=bootstrap_error,
            run_configuration={
                "c_miss": args.c_miss,
                "c_false_alarm": args.c_false_alarm,
                "c_retrieve": args.c_retrieve,
                "p0": args.p0,
                "max_docs": args.max_docs,
                "validation_fraction": args.validation_fraction,
                "split_seed": RANDOM_STATE,
            },
            split_metadata=split_metadata,
            expected_validation_passage_ids=frozen_split["validation_passage_ids"],
            expected_test_passage_ids=frozen_split["test_passage_ids"],
        )
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
            "result_publication": {
                "automatic_push_requested": bool(args.push_results),
                "policy": "opt-in; protected branches are never auto-pushed",
                "note": (
                    "Records the intent expressed on the command line, not "
                    "whether a push was later accepted by the remote."
                ),
            },
            "split": split_metadata,
            "confirmatory_split_identity": claim_assessment["split_identity"],
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
                "ulsif_fit_diagnostics": ratio_estimator.fit_diagnostics,
                "cv_table": ratio_estimator.cv_table,
                "selected_lower_threshold": selected["lower"],
                "selected_upper_threshold": selected["upper"],
                "threshold_selection_rule": selection_rule,
                "confirmatory_validation_selection": selected[
                    "confirmatory_validation_selection"
                ],
                "threshold_search_space": threshold_search_space,
                "threshold_validation_table": threshold_table,
            },
            "test_metrics": {
                "bse_official": bse_official_metrics,
                "bse_equation8": bse_eq8_metrics,
                "ddre_ulsif": ddre_metrics,
            },
            "descriptive_point_estimates": comparison,
            "confirmatory_statistical_protocol": confirmatory_protocol(),
            "confirmatory_bootstrap": bootstrap,
            "confirmatory_bootstrap_error": bootstrap_error,
            "claim_assessment": claim_assessment,
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
        print("DESCRIPTIVE POINT ESTIMATES (no uncertainty; not the claim)")
        if comparison["retrieved_documents_reduction_fraction"] is not None:
            print(
                "  DDRE document-cost reduction vs BSE official: "
                f"{100 * comparison['retrieved_documents_reduction_fraction']:.2f}%"
            )
        for label, key in (
            ("nonfactual", "nonfactual_auc_pr_delta"),
            ("factual", "factual_auc_pr_delta"),
            ("balanced", "balanced_pr_auc_delta"),
        ):
            print(f"  DDRE {label} PR-AUC delta vs BSE official: "
                  f"{comparison[key]:+.4f}")

        print("-" * 88)
        print("CONFIRMATORY ASSESSMENT (pre-registered paired passage bootstrap)")
        if bootstrap is not None:
            print(
                f"  {bootstrap['n_resamples']} resamples over "
                f"{bootstrap['unique_passages']} passages, seed "
                f"{bootstrap['seed']}, {int(100 * bootstrap['ci_level'])}% "
                f"{bootstrap['ci_method']} intervals"
            )
            for name, endpoint in bootstrap["endpoints"].items():
                print(
                    f"  {name:<48}{endpoint['observed']:+.4f}  "
                    f"[{endpoint['ci_lower']:+.4f}, {endpoint['ci_upper']:+.4f}]"
                )
        performance = claim_assessment["performance_noninferiority"]
        print(f"  performance non-inferiority (margin "
              f"{performance['margin']}): nonfactual="
              f"{performance['nonfactual_pass']} factual="
              f"{performance['factual_pass']} balanced="
              f"{performance['balanced_pass']}")
        print("  retrieval-efficiency superiority: "
              f"{claim_assessment['retrieval_efficiency_superiority_pass']}")
        print("  NLI-efficiency superiority (secondary): "
              f"{claim_assessment['nli_efficiency_superiority_pass']}")
        print("  validation selection confirmatory: "
              f"{claim_assessment['validation_selection_confirmatory']}")
        print("  bootstrap provenance matches frozen protocol: "
              f"{claim_assessment['bootstrap_provenance_matches_frozen_protocol']}")
        print("  run configuration matches frozen: "
              f"{claim_assessment['run_configuration_matches_frozen']}")
        print("  held-out split matches frozen: "
              f"{claim_assessment['split_matches_frozen']}")
        for disqualifier in claim_assessment["confirmatory_disqualifiers"]:
            print(f"    ! {disqualifier}")
        print(f"  CLAIM STATUS: {claim_assessment['claim_status']}")
        print(f"  {claim_assessment['interpretation']}")
        print("=" * 88)
        print(f"Summary: {summary_path}")
        print(f"Predictions: {predictions_path}")
        print(f"DDRE model: {model_path}")

        # D-12. Opt-in. A smoke test never auto-pushes, even when the flag is
        # given: its outputs are debugging-only and are not paper results.
        if not args.smoke_test and args.push_results:
            push_status = auto_push_results(
                [summary_path, predictions_path, model_path]
            )
            print(f"Result push status: {push_status}")
        elif args.smoke_test and args.push_results:
            print(
                "Result artifacts written locally; smoke-test outputs are "
                "debugging-only and are never auto-pushed."
            )
        else:
            print(
                "Result artifacts written locally; automatic push not requested."
            )

    finally:
        scorer.close()


if __name__ == "__main__":
    main()
