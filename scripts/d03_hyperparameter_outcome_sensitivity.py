#!/usr/bin/env python3
"""Validation-only follow-up to the completed frozen D-03 diagnostic.

D-03 trigger 2 fired: the evidence direction is unstable across the 20
``(sigma, lambda)`` pairs the production cross-validation already scored. This
asks the narrower downstream question -- does the paper-level DDRE
quality/efficiency conclusion move across those same 20 pairs? -- and answers it
on the frozen validation half only.

**This is a sensitivity analysis.** It selects no hyperparameter, no cap, no
calibration, no evidence transform, no held-out result and no post-hoc test. The
production pair remains the uLSIF-CV-selected ``sigma=0.2269287109375,
lambda=1.0``. The output records ``sensitivity_only=true`` and
``production_hyperparameters_changed=false``.

**No model and no NLI inference.** Every score is replayed read-only
(SQLite ``mode=ro``) from the completed D-03 derived cache via
``src.cached_document_scores.CachedDocumentScorer``. A missing score fails
closed rather than being defaulted.

**Held-out passages are never scored.** The split helper returns both halves;
only the validation half reaches a detector, a metric or the tuner, and
``assert_validation_only`` refuses to proceed otherwise.

The validation rule is the repository's existing one --
``src.threshold_selection.candidate_record`` and
``select_threshold_configuration`` -- not a new quality criterion.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_completion import sha256_file  # noqa: E402
from src.density_ratio_diagnostics import (  # noqa: E402
    production_hyperparameter_pairs,
)
from src.evaluation import summarize_method  # noqa: E402
from src.hyperparameter_outcome_sensitivity import (  # noqa: E402
    ANALYSIS_NAME,
    C_FALSE_ALARM,
    C_MISS,
    C_RETRIEVE,
    D03_BASE_COMMIT,
    EXPECTED_FACTUAL_TRAINING,
    EXPECTED_HALLUCINATED_TRAINING,
    EXPECTED_HELD_OUT_PASSAGES,
    EXPECTED_HYPERPARAMETER_PAIRS,
    EXPECTED_SPLIT_SEED,
    EXPECTED_THRESHOLD_PAIRS,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VALIDATION_PASSAGES,
    EXPECTED_VALIDATION_SENTENCES,
    EXPECTED_VALIDATION_SUBCLAIMS,
    MAX_DOCS,
    P0,
    PROTOCOL_VERSION,
    ArtifactVerificationFailed,
    ProductionFitChanged,
    assert_validation_only,
    candidate_estimator,
    passage_ids_sha256,
    robustness_summary,
    verify_d03_artifact,
    verify_production_fit,
)
from src.threshold_selection import (  # noqa: E402
    SAFEGUARD_NOTE,
    candidate_record,
    select_threshold_configuration,
)

DEFAULT_REPORT_SHA = "f2468e5bb1265e107f4060d36782f6f21ba9f5f55141dde4814d5ffd11aa7f6d"
DEFAULT_CACHE_SHA = "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"
OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Matches main.py's production defaults. Not tunable from this script: the
# validation rule must be the one the experiment already uses.
QUALITY_TOLERANCE = 0.005
RETRIEVAL_PENALTY = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only sensitivity of the DDRE outcome to the 20 "
            "production uLSIF hyperparameter pairs. Selects nothing."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--d03-report", required=True, metavar="PATH")
    parser.add_argument("--derived-cache", required=True, metavar="PATH")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument("--expected-report-sha256", default=DEFAULT_REPORT_SHA)
    parser.add_argument("--expected-cache-sha256", default=DEFAULT_CACHE_SHA)
    parser.add_argument("--output", default=None, metavar="PATH")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify artifacts and report the population, then exit. No scoring.",
    )
    return parser.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def frozen_validation_split(data_root):
    """The frozen split, verified by passage IDENTITY rather than by count."""
    from src.wang_data import group_split_records, load_sentence_records

    validation, held_out, metadata = group_split_records(
        load_sentence_records(data_root, strict=True),
        validation_fraction=EXPECTED_VALIDATION_FRACTION,
        random_state=EXPECTED_SPLIT_SEED,
    )
    expected = sorted(int(x) for x in metadata["validation_passage_ids"])
    observed = sorted({int(r.passage_index) for r in validation})
    identity = {
        "validation_fraction": EXPECTED_VALIDATION_FRACTION,
        "split_seed": EXPECTED_SPLIT_SEED,
        "validation_passages": len(expected),
        "held_out_passages": int(metadata["test_passages"]),
        "validation_sentences": len(validation),
        "validation_subclaims": sum(len(r.subclaims) for r in validation),
        "validation_passage_ids": expected,
        "observed_validation_passage_ids": observed,
        "identity_matches": expected == observed,
        "sha256": passage_ids_sha256(expected),
    }
    held_out_ids = sorted(int(x) for x in metadata["test_passage_ids"])
    return validation, held_out, held_out_ids, identity


def evaluate_on_validation(detector, records, scorer):
    """Metrics for one detector on the validation records. Cache-only scorer."""
    results = [
        detector.detect_sentence(record, scorer, use_cache=True)
        for record in records
    ]
    return summarize_method(records, results)


def efficiency_deltas(metrics, baseline):
    """Retrieval and NLI-call reductions relative to BSE validation."""
    ddre_docs = metrics["efficiency"]["avg_retrieved_documents_per_sentence"]
    base_docs = baseline["efficiency"]["avg_retrieved_documents_per_sentence"]
    ddre_nli = metrics["efficiency"]["avg_nli_span_calls_per_sentence"]
    base_nli = baseline["efficiency"]["avg_nli_span_calls_per_sentence"]
    return {
        "avg_retrieved_documents_per_sentence": ddre_docs,
        "avg_retrieved_documents_per_subclaim": metrics["efficiency"][
            "avg_retrieved_documents_per_subclaim"
        ],
        "avg_nli_span_calls_per_sentence": ddre_nli,
        "retrieved_documents_reduction_fraction": (
            None if base_docs <= 0 else 1.0 - ddre_docs / base_docs
        ),
        "nli_span_calls_reduction_fraction": (
            None if base_nli <= 0 else 1.0 - ddre_nli / base_nli
        ),
        "accuracy_delta_vs_bse": metrics["accuracy"] - baseline["accuracy"],
        "macro_f1_delta_vs_bse": metrics["macro_f1"] - baseline["macro_f1"],
    }


def main():
    args = parse_args()

    # ---------------------------------------------- 1. verify the inputs
    report_path = Path(args.d03_report).expanduser()
    cache_path = Path(args.derived_cache).expanduser()
    for path, label in ((report_path, "D-03 report"), (cache_path, "derived cache")):
        if not path.exists():
            print(f"\nERROR: {label} not found: {path}")
            return 1

    report_sha = sha256_file(report_path)
    cache_sha = sha256_file(cache_path)
    with report_path.open("r", encoding="utf-8") as handle:
        d03 = json.load(handle)

    try:
        verification = verify_d03_artifact(
            d03,
            report_sha256=report_sha,
            expected_report_sha256=args.expected_report_sha256,
            cache_sha256=cache_sha,
            expected_cache_sha256=args.expected_cache_sha256,
        )
    except ArtifactVerificationFailed as exc:
        print(f"\nABORTED at artifact verification.\n{exc}")
        return 1

    print("=" * 100)
    print("D-03 FOLLOW-UP: HYPERPARAMETER OUTCOME SENSITIVITY  (validation only)")
    print("=" * 100)
    print(f"  D-03 report        {report_path}")
    print(f"    sha256           {report_sha}")
    print(f"  derived cache      {cache_path}")
    print(f"    sha256           {cache_sha}")
    print(f"  D-03 git commit    {d03.get('git_commit')}")
    print("  artifact checks    all passed")

    # ---------------------------------------------- 2. frozen split
    validation, _held_out, held_out_ids, identity = frozen_validation_split(
        args.data_root
    )
    d03_split_sha = verification["d03_split_sha256"]
    if not identity["identity_matches"] or identity["sha256"] != d03_split_sha:
        print(
            "\nABORTED: the reconstructed validation split is not D-03's.\n"
            f"  reconstructed sha256 = {identity['sha256']}\n"
            f"  D-03 recorded sha256 = {d03_split_sha}"
        )
        return 1
    assert_validation_only(validation, held_out_ids, where="the validation population")

    print("\n  FROZEN VALIDATION SPLIT (identity, not count)")
    print(f"    passage-id sha256  {identity['sha256']}")
    print(f"    matches D-03       True")
    print(f"    passages           {identity['validation_passages']}")
    print(f"    sentences          {identity['validation_sentences']}")
    print(f"    subclaims          {identity['validation_subclaims']}")
    print(f"    held-out passages  {identity['held_out_passages']}  (never scored)")

    if args.dry_run:
        print("\n--dry-run: inputs verified and population reported. Nothing scored.")
        return 0

    # ---------------------------------------------- 3. cache-only scorer
    from src.baseline_core import BSEDetector, build_nbc_histograms
    from src.cached_document_scores import (
        CachedDocumentScorer,
        MissingDocumentScore,
        MissingPairScore,
    )
    from src.ddre_core import (
        DDREDetector,
        ULSIFDensityRatio,
        cost_consistent_thresholds,
        CANDIDATE_LOWER_GRID,
        CANDIDATE_UPPER_GRID,
    )
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_nbc_pairs

    scorer = CachedDocumentScorer(
        cache_path, args.model_name, SCORE_VERSION, split_text
    )
    try:
        positive, negative = load_nbc_pairs(args.data_root, per_class=None)
        if (len(positive), len(negative)) != (
            EXPECTED_FACTUAL_TRAINING, EXPECTED_HALLUCINATED_TRAINING
        ):
            print(
                f"\nABORTED: NBC counts are {len(positive)}/{len(negative)}; "
                f"expected {EXPECTED_FACTUAL_TRAINING}/{EXPECTED_HALLUCINATED_TRAINING}."
            )
            return 1
        try:
            factual_scores = scorer.score_pairs(
                [(x["premise"], x["hypothesis"]) for x in positive]
            )
            hallucinated_scores = scorer.score_pairs(
                [(x["premise"], x["hypothesis"]) for x in negative]
            )
        except MissingPairScore as exc:
            print(f"\nABORTED: the derived cache is missing an NBC score: {exc}")
            return 1

        pos_hist, neg_hist, _, _ = build_nbc_histograms(positive, negative, scorer)

        # -------------------------------------- 4. production fit, once
        production = ULSIFDensityRatio(random_state=EXPECTED_SPLIT_SEED).fit(
            factual_scores, hallucinated_scores
        )
        try:
            production_fit = verify_production_fit(production)
        except ProductionFitChanged as exc:
            print(f"\nABORTED: {exc}")
            return 1
        print("\n  PRODUCTION uLSIF FIT (reproduced, not re-selected)")
        print(f"    sigma              {production_fit['sigma']!r}")
        print(f"    lambda             {production_fit['lambda']!r}")
        print(f"    cv pairs           {production_fit['unique_hyperparameter_pairs']}")
        print(f"    centers (fixed)    {production_fit['n_centers']}")

        factual_x = production._as_column(factual_scores)
        hallucinated_x = production._as_column(hallucinated_scores)
        pairs = production_hyperparameter_pairs(production)

        space = cost_consistent_thresholds(
            C_MISS, C_FALSE_ALARM, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID
        )
        threshold_pairs = [
            (lower, upper)
            for lower in space["effective_lower_grid"]
            for upper in space["effective_upper_grid"]
            if lower < upper
        ]
        if len(threshold_pairs) != EXPECTED_THRESHOLD_PAIRS:
            print(
                f"\nABORTED: {len(threshold_pairs)} cost-consistent threshold "
                f"pairs, expected {EXPECTED_THRESHOLD_PAIRS}."
            )
            return 1

        # -------------------------------------- 5. BSE official on the same records
        bse = BSEDetector(
            pos_hist, neg_hist, mode="official", p0=P0, c_miss=C_MISS,
            c_false_alarm=C_FALSE_ALARM, c_retrieve=C_RETRIEVE, max_docs=MAX_DOCS,
        )
        assert_validation_only(validation, held_out_ids, where="BSE validation")
        baseline = evaluate_on_validation(bse, validation, scorer)
        print("\n  BSE OFFICIAL CM28/CFA96 ON THE SAME 48 VALIDATION PASSAGES")
        print(f"    nonfactual PR-AUC  {baseline['nonfactual']['auc_pr']:.6f}")
        print(f"    factual PR-AUC     {baseline['factual']['auc_pr']:.6f}")
        print(f"    balanced PR-AUC    {baseline['balanced_pr_auc']:.6f}")
        print(f"    accuracy           {baseline['accuracy']:.6f}")
        print(
            f"    docs/sentence      "
            f"{baseline['efficiency']['avg_retrieved_documents_per_sentence']:.6f}"
        )

        # -------------------------------------- 6. the 20-pair sweep
        rows = []
        print(f"\n  SWEEPING {len(pairs)} HYPERPARAMETER PAIRS x "
              f"{len(threshold_pairs)} THRESHOLD PAIRS (validation only)")
        cv_lookup = {
            (float(r["sigma"]), float(r["lambda"])): float(r["cv_objective"])
            for r in production.cv_table
        }
        for sigma, lam in pairs:
            estimator = candidate_estimator(
                production, factual_x, hallucinated_x, sigma, lam
            )
            candidates = []
            for lower, upper in threshold_pairs:
                detector = DDREDetector(
                    estimator, lower_threshold=float(lower),
                    upper_threshold=float(upper), p0=P0, c_miss=C_MISS,
                    c_false_alarm=C_FALSE_ALARM, max_docs=MAX_DOCS,
                )
                assert_validation_only(
                    validation, held_out_ids, where="a DDRE candidate evaluation"
                )
                metrics = evaluate_on_validation(detector, validation, scorer)
                record = candidate_record(
                    lower, upper, metrics, baseline,
                    quality_tolerance=QUALITY_TOLERANCE,
                    retrieval_penalty=RETRIEVAL_PENALTY,
                    max_docs=MAX_DOCS,
                )
                record.update(efficiency_deltas(metrics, baseline))
                candidates.append(record)

            selected, rule, confirmatory = select_threshold_configuration(candidates)
            feasible = [c for c in candidates if c["preserves_baseline_quality"]]
            is_production = (
                float(sigma) == production_fit["sigma"]
                and float(lam) == production_fit["lambda"]
            )
            rows.append({
                "sigma": float(sigma),
                "lambda": float(lam),
                "ulsif_cv_objective": cv_lookup.get((float(sigma), float(lam))),
                "is_production_pair": bool(is_production),
                "threshold_candidates": candidates,
                "threshold_pairs_scored": len(candidates),
                "feasible_threshold_count": len(feasible),
                "selected": selected,
                "selection_rule": rule,
                "confirmatory_threshold_selection": bool(confirmatory),
            })
            mark = " <- production" if is_production else ""
            print(
                f"    sigma={float(sigma):.10f} lambda={float(lam):<8g} "
                f"feasible={len(feasible):>2}/{len(candidates)} "
                f"confirmatory={str(confirmatory):<5}{mark}"
            )

        summary = robustness_summary(rows)
    finally:
        scorer.close()

    # ---------------------------------------------- 7. report
    short = (_git_commit() or "unknown")[:7]
    output_path = Path(
        args.output
        or Path.home()
        / "ddre-artifacts"
        / "gate1-2026-09-20"
        / f"d03_hyperparameter_outcome_sensitivity_main_{short}.json"
    ).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "d03_base_commit": D03_BASE_COMMIT,
        "question": (
            "Across the 20 (sigma, lambda) pairs the production uLSIF CV already "
            "scored, how much does the validation-level DDRE quality/efficiency "
            "outcome move?"
        ),
        "sensitivity_only": True,
        "production_hyperparameters_changed": False,
        "selected_cap": None,
        "selected_calibration": None,
        "method_change_made": False,
        "held_out_scored": False,
        "held_out_passages": identity["held_out_passages"],
        "expected_held_out_passages": EXPECTED_HELD_OUT_PASSAGES,
        "inputs": {
            "d03_report_path": str(report_path),
            "d03_report_sha256": report_sha,
            "derived_cache_path": str(cache_path),
            "derived_cache_sha256": cache_sha,
            "derived_cache_opened_read_only": True,
            "model_loaded": False,
            "nli_inference_performed": False,
        },
        "artifact_verification": verification,
        "split": identity,
        "expected_population": {
            "validation_passages": EXPECTED_VALIDATION_PASSAGES,
            "validation_sentences": EXPECTED_VALIDATION_SENTENCES,
            "validation_subclaims": EXPECTED_VALIDATION_SUBCLAIMS,
        },
        "production_fit": production_fit,
        "threshold_search_space": space,
        "threshold_pairs_per_hyperparameter": len(threshold_pairs),
        "validation_rule": {
            "source": "src.threshold_selection",
            "candidate_record": "src.threshold_selection.candidate_record",
            "selection": "src.threshold_selection.select_threshold_configuration",
            "quality_tolerance": QUALITY_TOLERANCE,
            "retrieval_penalty": RETRIEVAL_PENALTY,
            "safeguards": SAFEGUARD_NOTE,
            "new_quality_criterion_introduced": False,
        },
        "bse_official_validation": baseline,
        "hyperparameter_rows": rows,
        "robustness_summary": summary,
        "scorer": scorer.stats(),
        "status": (
            "SENSITIVITY MEASURED. No hyperparameter, cap, calibration or "
            "method change is selected by this run, and no held-out record was "
            "scored. Interpretation requires review."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    print_table(rows, baseline)
    print_summary(summary)
    print(f"\nWritten to {output_path}")
    print(f"  sha256 {sha256_file(output_path)}")
    return 0


def print_table(rows, baseline):
    print("\n" + "=" * 132)
    print("PER-HYPERPARAMETER OUTCOME (validation only; selection by the existing rule)")
    print("=" * 132)
    header = (
        f"{'sigma':>14} {'lambda':>8} {'CV obj':>11} {'feas':>5} "
        f"{'sel L/U':>12} {'nonfact D':>10} {'factual D':>10} "
        f"{'balanced D':>11} {'docs red':>9} {'NLI red':>9} {'conf':>5}"
    )
    print(header)
    print("-" * 132)
    for row in rows:
        s = row["selected"]
        def pct(v):
            return "   n/a" if v is None else f"{100.0 * v:>7.2f}%"
        print(
            f"{row['sigma']:>14.10f} {row['lambda']:>8g} "
            f"{row['ulsif_cv_objective']:>11.6f} "
            f"{row['feasible_threshold_count']:>2}/{row['threshold_pairs_scored']:<2} "
            f"{s['lower']:>5.2f}/{s['upper']:<6.2f} "
            f"{s['nonfactual_auc_pr_delta_vs_bse']:>+10.4f} "
            f"{s['factual_auc_pr_delta_vs_bse']:>+10.4f} "
            f"{s['balanced_pr_auc_delta_vs_bse']:>+11.4f} "
            f"{pct(s.get('retrieved_documents_reduction_fraction')):>9} "
            f"{pct(s.get('nli_span_calls_reduction_fraction')):>9} "
            f"{str(row['confirmatory_threshold_selection']):>5}"
            + ("  <- production" if row["is_production_pair"] else "")
        )


def print_summary(summary):
    print("\n" + "=" * 100)
    print("AGGREGATE ACROSS THE 20 PRODUCTION HYPERPARAMETER PAIRS")
    print("=" * 100)
    print(f"  pairs                                      {summary['hyperparameter_pairs']}")
    print(f"  with >=1 confirmatorily feasible threshold {summary['pairs_with_a_confirmatorily_feasible_threshold']}")
    print(f"  without one                                {summary['pairs_without_a_confirmatorily_feasible_threshold']}")
    print(f"  confirmatory threshold selection           {summary['pairs_with_confirmatory_threshold_selection']}")
    print(f"  selection reduces retrieval                {summary['pairs_whose_selection_reduces_retrieval']}")
    print(f"  selection reduces NLI calls                {summary['pairs_whose_selection_reduces_nli_calls']}")
    print(f"  preserves quality AND reduces retrieval    {summary['pairs_preserving_quality_and_reducing_retrieval']}")
    for label, block in (
        ("all 20 pairs", summary["spread_all_pairs"]),
        ("confirmatory only", summary["spread_confirmatory_pairs_only"]),
    ):
        print(f"\n  spread ({label}):")
        print(f"    {'field':<44}{'min':>12}{'median':>12}{'max':>12}")
        for field, spread in block.items():
            def f(v):
                return "         n/a" if v is None else f"{v:>12.6f}"
            print(f"    {field:<44}{f(spread['min'])}{f(spread['median'])}{f(spread['max'])}")
    print(f"\n  interpretation: {summary['interpretation']!r}  (not asserted by this run)")


if __name__ == "__main__":
    sys.exit(main())
