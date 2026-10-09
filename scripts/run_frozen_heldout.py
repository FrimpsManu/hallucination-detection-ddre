#!/usr/bin/env python3
"""THE held-out experiment. BSE official versus the frozen DDRE configuration.

One run, two methods, 190 held-out passages, nothing selected.

The DDRE configuration was frozen on validation BEFORE any held-out access and
is hardcoded below. There is no CLI argument for sigma, lambda, lower or upper:
the only held-out configuration is the frozen one, and this runner cannot be
asked to try another. No tuner, no threshold grid, no sigma or lambda search,
no cap, no calibration and no fallback selection is reachable from here.

What is reused rather than reimplemented: the frozen split, the D-03 cache and
its NBC scores, the D-03 candidate-estimator reconstruction, BSEDetector,
DDREDetector, evaluate_detector_with_traces, paired_passage_bootstrap,
assess_claim, retrieval_protocol_asymmetry and prediction_rows.

Honesty constraints this runner enforces rather than documents:

* The selection history is recorded as validation-tuned across 640
  configurations. It is NOT the original pre-registered uLSIF-CV-only
  selection, so it is assessed as ``post_d03_validation_sweep`` (recorded as
  ``validation_selection_confirmatory`` False) and a resulting NOT_CONFIRMATORY
  status is preserved, not engineered away.
* The tuning-budget asymmetry against the fixed published BSE baseline is
  written into the artifact.
* The historical checkpoint-identity limitation is carried through from D-03
  unchanged.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_completion import prepare_derived_cache, sha256_file  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_detector_with_traces,
    prediction_rows,
    retrieval_protocol_asymmetry,
    summarize_subclaim_efficiency,
)
from src.hyperparameter_outcome_sensitivity import (  # noqa: E402
    EXPECTED_FACTUAL_TRAINING,
    EXPECTED_HALLUCINATED_TRAINING,
    EXPECTED_HELD_OUT_PASSAGES,
    EXPECTED_SPLIT_SEED,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VALIDATION_PASSAGES,
    PRODUCTION_LAMBDA,
    PRODUCTION_SIGMA,
    ProductionFitChanged,
    candidate_estimator,
    passage_ids_sha256,
    verify_production_fit,
)
from src.paired_bootstrap import (  # noqa: E402
    CONFIRMATORY_PR_AUC_MARGIN,
    VALIDATION_SELECTION_POST_D03_SWEEP,
    assess_claim,
    confirmatory_protocol,
    paired_passage_bootstrap,
)

# ----------------------------------------------------------------- FROZEN
# Selected on validation before any held-out access. Immutable: these are
# literals, not defaults, and no argument overrides them.
FROZEN_SIGMA = 0.11346435546875
FROZEN_LAMBDA = 1.0
FROZEN_LOWER = 0.20
FROZEN_UPPER = 0.80

FREEZE_SHA256 = "744abf4afc05f586ed138dbbb5a503ac8014af40ab14c2a6d13b5f512ecdf9fa"
SENSITIVITY_SHA256 = (
    "fc5604854b18f3e3e712fe7ef01859b58dac569e99696041713abb11daa6dcc0"
)
D03_CACHE_SHA256 = (
    "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"
)

VALIDATION_CONFIGURATIONS_CONSIDERED = 640
ELIGIBLE_VALIDATION_CONFIGURATIONS = 102

# Production cost configuration. Published BSE, not tunable here.
C_MISS = 28.0
C_FALSE_ALARM = 96.0
C_RETRIEVE = 1.0
P0 = 0.5
MAX_DOCS = 10

OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ARTIFACTS = Path("~/ddre-artifacts/gate1-2026-09-20").expanduser()

TUNING_BUDGET_DISCLOSURE = (
    "DDRE received validation-based model/threshold selection across 640 "
    "configurations whereas BSE remains the fixed published baseline. This "
    "tuning-budget asymmetry must be disclosed in the paper."
)
SELECTION_HISTORY_NOTE = (
    "This configuration was selected on validation AFTER D-03, from 640 "
    "candidate configurations. It is NOT the original pre-registered "
    "uLSIF-CV-selected configuration, and no claim is made that the CV-selected "
    "configuration succeeded. The frozen confirmatory protocol predates this "
    "selection history, so validation_selection_confirmatory is False and any "
    "resulting NOT_CONFIRMATORY status is reported as-is."
)


class Aborted(RuntimeError):
    """A precondition failed. Nothing was scored."""


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _first(payload, *paths):
    for path in paths:
        value = _dig(payload, path)
        if value is not None:
            return value
    return None


def verify_freeze(freeze_path):
    """The freeze artifact must be exactly the one that was frozen."""
    path = Path(freeze_path).expanduser()
    if not path.exists():
        raise Aborted(f"freeze artifact not found: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != FREEZE_SHA256:
        raise Aborted(
            "the freeze artifact is not the expected one.\n"
            f"  expected {FREEZE_SHA256}\n  observed {observed_sha}"
        )
    with path.open("r", encoding="utf-8") as handle:
        freeze = json.load(handle)

    selected = freeze.get("selected_configuration") or {}
    problems = []
    if freeze.get("held_out_scored") is not False:
        problems.append("held_out_scored is not false")
    if freeze.get("validation_tuned") is not True:
        problems.append("validation_tuned is not true")
    if freeze.get("source_sensitivity_artifact_sha256") != SENSITIVITY_SHA256:
        problems.append(
            "source sensitivity SHA is "
            f"{freeze.get('source_sensitivity_artifact_sha256')!r}, expected "
            f"{SENSITIVITY_SHA256!r}"
        )
    for name, expected in (
        ("sigma", FROZEN_SIGMA),
        ("lambda", FROZEN_LAMBDA),
        ("lower_threshold", FROZEN_LOWER),
        ("upper_threshold", FROZEN_UPPER),
    ):
        if selected.get(name) != expected:
            problems.append(
                f"selected {name} is {selected.get(name)!r}, expected {expected!r}"
            )
    if problems:
        raise Aborted(
            "the freeze artifact does not describe the frozen configuration:\n  "
            + "\n  ".join(problems)
        )
    return {
        "freeze_artifact_path": str(path),
        "freeze_artifact_sha256": observed_sha,
        "source_sensitivity_artifact_sha256": SENSITIVITY_SHA256,
        "selected_configuration": selected,
        "held_out_scored_in_freeze": False,
        "validation_tuned": True,
        "verified": True,
    }


def verify_d03_cache(cache_path):
    path = Path(cache_path).expanduser()
    if not path.exists():
        raise Aborted(f"D-03 cache not found: {path}")
    observed = sha256_file(path)
    if observed != D03_CACHE_SHA256:
        raise Aborted(
            "the D-03 cache is not the expected one.\n"
            f"  expected {D03_CACHE_SHA256}\n  observed {observed}"
        )
    return {"d03_cache_path": str(path), "d03_cache_sha256": observed}


def frozen_split(data_root):
    """The frozen split, verified by passage IDENTITY on both halves."""
    from src.wang_data import group_split_records, load_sentence_records

    validation, held_out, metadata = group_split_records(
        load_sentence_records(data_root, strict=True),
        validation_fraction=EXPECTED_VALIDATION_FRACTION,
        random_state=EXPECTED_SPLIT_SEED,
    )
    expected_validation = sorted(int(x) for x in metadata["validation_passage_ids"])
    expected_held_out = sorted(int(x) for x in metadata["test_passage_ids"])
    observed_held_out = sorted({int(r.passage_index) for r in held_out})

    identity = {
        "validation_fraction": EXPECTED_VALIDATION_FRACTION,
        "split_seed": EXPECTED_SPLIT_SEED,
        "validation_passages": len(expected_validation),
        "held_out_passages": len(expected_held_out),
        "held_out_sentences": len(held_out),
        "held_out_subclaims": sum(len(r.subclaims) for r in held_out),
        "validation_passage_ids": expected_validation,
        "held_out_passage_ids": expected_held_out,
        "observed_held_out_passage_ids": observed_held_out,
        "validation_passage_ids_sha256": passage_ids_sha256(expected_validation),
        "held_out_passage_ids_sha256": passage_ids_sha256(expected_held_out),
        "identity_matches": expected_held_out == observed_held_out,
    }
    problems = []
    if not identity["identity_matches"]:
        problems.append("the held-out passage IDs are not the frozen ones")
    if identity["validation_passages"] != EXPECTED_VALIDATION_PASSAGES:
        problems.append(
            f"{identity['validation_passages']} validation passages, expected "
            f"{EXPECTED_VALIDATION_PASSAGES}"
        )
    if identity["held_out_passages"] != EXPECTED_HELD_OUT_PASSAGES:
        problems.append(
            f"{identity['held_out_passages']} held-out passages, expected "
            f"{EXPECTED_HELD_OUT_PASSAGES}"
        )
    if set(expected_validation) & set(expected_held_out):
        problems.append("the two halves overlap")
    if problems:
        raise Aborted("the frozen split did not reconstruct:\n  " + "\n  ".join(problems))
    return held_out, identity


def d03_checkpoint(d03_report_path):
    """The revision D-03 actually used, plus its identity limitation."""
    from src.provenance_guard import extract_reference

    path = Path(d03_report_path).expanduser()
    if not path.exists():
        raise Aborted(f"D-03 report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    revision = _first(
        report,
        "provenance.reference_bundle.reference.resolved_revision",
        "provenance.environment.checkpoint_identity.resolved_revision",
    ) or extract_reference(report.get("provenance") or {}).get("resolved_revision")
    if not revision:
        raise Aborted(
            "the D-03 report records no resolved Hugging Face revision, so the "
            "held-out run cannot pin the checkpoint D-03 used. Refusing to fall "
            "back to an unpinned newer checkpoint silently."
        )
    established = _first(report, "checkpoint_identity_established")
    return {
        "d03_report_path": str(path),
        "d03_report_sha256": sha256_file(path),
        "resolved_revision": revision,
        "checkpoint_identity_established": established,
        "checkpoint_identity_limitation": _first(report, "checkpoint_identity_note"),
        "limitation_preserved": True,
    }


def claim_split_metadata(split):
    """What assess_claim compares: the passages actually evaluated on held-out.

    The validation half is never evaluated here, so its recorded IDs are the
    frozen ones. The held-out half is the set of passages the evaluated records
    actually came from, so a drift between the records and the frozen split is
    a disqualifier rather than something assumed away.
    """
    return {
        "validation_fraction": split["validation_fraction"],
        "random_state": split["split_seed"],
        "validation_passage_ids": split["validation_passage_ids"],
        "test_passage_ids": split["observed_held_out_passage_ids"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "The held-out experiment: BSE official versus the frozen DDRE "
            "configuration. The configuration is hardcoded and cannot be "
            "changed from the command line."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--freeze", default=str(ARTIFACTS / "ddre_validation_freeze.json"))
    parser.add_argument(
        "--d03-cache", default=str(ARTIFACTS / "d03_diagnostic_cache_main_2661b04.sqlite")
    )
    parser.add_argument(
        "--d03-report",
        default=str(ARTIFACTS / "d03_density_ratio_diagnostic_main_2661b04.json"),
    )
    parser.add_argument(
        "--held-out-cache", default=str(ARTIFACTS / "frozen_heldout_nli_cache.sqlite")
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default=str(ARTIFACTS))
    parser.add_argument(
        "--overwrite-held-out-cache", action="store_true",
        help="Replace an existing held-out cache with a fresh copy of the D-03 cache.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Verify the freeze, the D-03 cache, the split and the checkpoint, "
            "then exit BEFORE any held-out inference. Scores nothing."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        freeze = verify_freeze(args.freeze)
        cache_identity_source = verify_d03_cache(args.d03_cache)
        held_out, split = frozen_split(args.data_root)
        checkpoint = d03_checkpoint(args.d03_report)
    except Aborted as exc:
        print(f"\nABORTED before any inference.\n{exc}")
        return 1

    print("=" * 104)
    print("FROZEN HELD-OUT EXPERIMENT   BSE official  vs  frozen DDRE")
    print("=" * 104)
    print(f"  freeze artifact      {freeze['freeze_artifact_path']}")
    print(f"    sha256             {freeze['freeze_artifact_sha256']}  (verified)")
    print(f"  sensitivity source   {SENSITIVITY_SHA256}  (verified)")
    print(f"  D-03 cache sha256    {cache_identity_source['d03_cache_sha256']}  (verified)")
    print(f"  checkpoint revision  {checkpoint['resolved_revision']}")
    print(f"    identity established {checkpoint['checkpoint_identity_established']}"
          "   (historical limitation preserved)")
    print("\n  FROZEN CONFIGURATION (immutable; no CLI override exists)")
    print(f"    sigma              {FROZEN_SIGMA!r}")
    print(f"    lambda             {FROZEN_LAMBDA!r}")
    print(f"    lower threshold    {FROZEN_LOWER!r}")
    print(f"    upper threshold    {FROZEN_UPPER!r}")
    print("\n  FROZEN SPLIT (identity, not count)")
    print(f"    validation passages {split['validation_passages']}  (not used here)")
    print(f"    held-out passages   {split['held_out_passages']}")
    print(f"    held-out sentences  {split['held_out_sentences']}")
    print(f"    held-out subclaims  {split['held_out_subclaims']}")
    print(f"    held-out id sha256  {split['held_out_passage_ids_sha256']}")

    if args.dry_run:
        print(
            "\n--dry-run: freeze, cache, split and checkpoint verified. "
            "NO held-out evidence was scored and no model was loaded."
        )
        return 0

    # ------------------------------------------------- held-out cache
    held_out_cache = Path(args.held_out_cache).expanduser()
    if held_out_cache.exists() and not args.overwrite_held_out_cache:
        derived = {
            "held_out_cache": str(held_out_cache),
            "reused_existing": True,
            "sha256_before": sha256_file(held_out_cache),
        }
    else:
        derived = prepare_derived_cache(
            Path(args.d03_cache).expanduser(), held_out_cache,
            overwrite=args.overwrite_held_out_cache,
        )
        if derived.get("copy_faithful") is not True:
            print(f"\nABORTED: {derived.get('copy_faithful_note')}")
            return 1
        derived["reused_existing"] = False
    print(f"\n  held-out cache       {held_out_cache}  (writable, distinct artifact)")

    # ------------------------------------------------- NBC scores, read-only
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.baseline_core import BSEDetector, build_nbc_histograms
    from src.cached_document_scores import CachedDocumentScorer, MissingPairScore
    from src.ddre_core import DDREDetector, ULSIFDensityRatio
    from src.diagnostic_probe import collect_live_environment, select_device
    from src.provenance import collect_provenance
    from src.utils import SCORE_VERSION, EntailmentScorer, split_text
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.data_root, per_class=None)
    if (len(positive), len(negative)) != (
        EXPECTED_FACTUAL_TRAINING, EXPECTED_HALLUCINATED_TRAINING
    ):
        print(
            f"\nABORTED: NBC counts are {len(positive)}/{len(negative)}; expected "
            f"{EXPECTED_FACTUAL_TRAINING}/{EXPECTED_HALLUCINATED_TRAINING}."
        )
        return 1

    nbc_reader = CachedDocumentScorer(
        Path(args.d03_cache).expanduser(), OFFICIAL_MODEL, SCORE_VERSION, split_text
    )
    try:
        factual_scores = nbc_reader.score_pairs(
            [(x["premise"], x["hypothesis"]) for x in positive]
        )
        hallucinated_scores = nbc_reader.score_pairs(
            [(x["premise"], x["hypothesis"]) for x in negative]
        )
        pos_hist, neg_hist, _, _ = build_nbc_histograms(positive, negative, nbc_reader)
    except MissingPairScore as exc:
        print(f"\nABORTED: the D-03 cache is missing an NBC score: {exc}")
        return 1
    finally:
        nbc_reader.close()

    # ------------------------------- the frozen estimator, reconstructed
    production = ULSIFDensityRatio(random_state=EXPECTED_SPLIT_SEED).fit(
        factual_scores, hallucinated_scores
    )
    try:
        production_fit = verify_production_fit(production)
    except ProductionFitChanged as exc:
        print(f"\nABORTED: {exc}")
        return 1
    estimator = candidate_estimator(
        production,
        production._as_column(factual_scores),
        production._as_column(hallucinated_scores),
        FROZEN_SIGMA,
        FROZEN_LAMBDA,
    )
    assert estimator.sigma == FROZEN_SIGMA and estimator.lam == FROZEN_LAMBDA
    print(
        f"  production fit       sigma={production_fit['sigma']!r} "
        f"lambda={production_fit['lambda']!r}  (reproduced)"
    )
    print(
        f"  frozen estimator     sigma={estimator.sigma!r} lambda={estimator.lam!r}, "
        f"centers held fixed ({int(estimator.centers.size)})"
    )

    # ------------------------------------------------- the two detectors
    device = select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        OFFICIAL_MODEL, revision=checkpoint["resolved_revision"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        OFFICIAL_MODEL, revision=checkpoint["resolved_revision"]
    ).to(torch.device(device))
    model.eval()
    scorer = EntailmentScorer(
        tokenizer, model, OFFICIAL_MODEL,
        cache_path=str(held_out_cache), batch_size=args.batch_size,
    )
    print(f"  device               {device}   batch size {args.batch_size}")

    try:
        environment = collect_live_environment(
            model_name=OFFICIAL_MODEL, model=model, tokenizer=tokenizer,
            repo_root=PROJECT_ROOT, score_version=SCORE_VERSION,
            selected_device=device,
        )
        provenance = collect_provenance(
            model_name=OFFICIAL_MODEL, data_root=args.data_root, device=device,
            batch_size=args.batch_size, score_version=SCORE_VERSION,
            tokenizer=tokenizer, model=model, repo_root=PROJECT_ROOT,
        )

        bse = BSEDetector(
            pos_hist, neg_hist, mode="official", p0=P0, c_miss=C_MISS,
            c_false_alarm=C_FALSE_ALARM, c_retrieve=C_RETRIEVE, max_docs=MAX_DOCS,
        )
        ddre = DDREDetector(
            estimator, lower_threshold=FROZEN_LOWER, upper_threshold=FROZEN_UPPER,
            p0=P0, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM, max_docs=MAX_DOCS,
        )

        print("\n  HELD-OUT EVALUATION (each method exactly once)")
        bse_metrics, bse_results, bse_obs = evaluate_detector_with_traces(
            bse, held_out, scorer, description="BSE official held-out", use_cache=True
        )
        ddre_metrics, ddre_results, ddre_obs = evaluate_detector_with_traces(
            ddre, held_out, scorer, description="frozen DDRE held-out", use_cache=True
        )
    finally:
        scorer.close()

    # ------------------------------------------------- deltas
    def eff(metrics, key):
        return metrics["efficiency"][key]

    deltas = {
        "nonfactual_auc_pr_delta": ddre_metrics["nonfactual"]["auc_pr"]
        - bse_metrics["nonfactual"]["auc_pr"],
        "factual_auc_pr_delta": ddre_metrics["factual"]["auc_pr"]
        - bse_metrics["factual"]["auc_pr"],
        "balanced_pr_auc_delta": ddre_metrics["balanced_pr_auc"]
        - bse_metrics["balanced_pr_auc"],
        "accuracy_delta": ddre_metrics["accuracy"] - bse_metrics["accuracy"],
        "macro_f1_delta": ddre_metrics["macro_f1"] - bse_metrics["macro_f1"],
        "sign_convention": "DDRE - BSE",
    }
    base_docs = eff(bse_metrics, "avg_retrieved_documents_per_sentence")
    base_nli = eff(bse_metrics, "avg_nli_span_calls_per_sentence")
    reductions = {
        "retrieval_reduction_fraction": (
            None if base_docs <= 0
            else 1.0 - eff(ddre_metrics, "avg_retrieved_documents_per_sentence") / base_docs
        ),
        "nli_call_reduction_fraction": (
            None if base_nli <= 0
            else 1.0 - eff(ddre_metrics, "avg_nli_span_calls_per_sentence") / base_nli
        ),
        "definition": "1 - DDRE / BSE, per sentence",
    }

    # ------------------------------------------------- frozen bootstrap
    bootstrap = paired_passage_bootstrap(ddre_obs, bse_obs)
    claim = assess_claim(
        bootstrap,
        # This selection is NOT the pre-registered CV-only one, and it is not
        # the fallback either: it is the post-D-03 validation sweep. Naming it
        # exactly is what makes the NOT_CONFIRMATORY reason true. Saying
        # otherwise to obtain a SUPPORTED label would be rewriting the history.
        validation_selection=VALIDATION_SELECTION_POST_D03_SWEEP,
        quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN,
        run_configuration={
            "c_miss": C_MISS, "c_false_alarm": C_FALSE_ALARM,
            "c_retrieve": C_RETRIEVE, "p0": P0, "max_docs": MAX_DOCS,
            "validation_fraction": EXPECTED_VALIDATION_FRACTION,
            "split_seed": EXPECTED_SPLIT_SEED,
        },
        # The evaluated held-out IDs are compared against the split re-derived
        # at the frozen fraction and seed. frozen_split already verified this;
        # passing both here lets assess_claim record that verification instead
        # of reporting the split as unverified.
        split_metadata=claim_split_metadata(split),
        expected_validation_passage_ids=split["validation_passage_ids"],
        expected_test_passage_ids=split["held_out_passage_ids"],
    )
    asymmetry = retrieval_protocol_asymmetry(bse_obs, ddre_obs, max_docs=MAX_DOCS)

    # ------------------------------------------------- artifacts
    short = (_git_commit() or "unknown")[:7]
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"frozen_heldout_result_{short}.json"
    csv_path = out_dir / f"frozen_heldout_predictions_{short}.csv"

    result = {
        "experiment": "frozen-heldout-bse-official-vs-frozen-ddre",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runner_git_commit": _git_commit(),
        "held_out_evaluated_once": True,
        "held_out_configuration_changed": False,
        "held_out_passages": split["held_out_passages"],
        "validation_tuned": True,
        "validation_configurations_considered": VALIDATION_CONFIGURATIONS_CONSIDERED,
        "eligible_validation_configurations": ELIGIBLE_VALIDATION_CONFIGURATIONS,
        "frozen_sigma": FROZEN_SIGMA,
        "frozen_lambda": FROZEN_LAMBDA,
        "frozen_lower": FROZEN_LOWER,
        "frozen_upper": FROZEN_UPPER,
        "freeze_artifact_sha256": FREEZE_SHA256,
        "freeze": freeze,
        "d03_cache": cache_identity_source,
        "held_out_cache": derived,
        "checkpoint": checkpoint,
        "split": split,
        "provenance": provenance,
        "environment": environment,
        "production_fit": production_fit,
        "frozen_estimator": {
            "sigma": estimator.sigma, "lambda": estimator.lam,
            "n_centers": int(estimator.centers.size),
            "centers_held_fixed_from_production_fit": True,
            "reconstruction_note": (
                "Production CV fit reproduced, its final centre set preserved, "
                "and only alpha re-solved at the frozen sigma/lambda. No centres "
                "were resampled and no CV chose the frozen candidate."
            ),
        },
        "methods_evaluated": ["bse_official", "frozen_ddre"],
        "bse_official": bse_metrics,
        "frozen_ddre": ddre_metrics,
        "point_estimate_deltas": deltas,
        "efficiency_reductions": reductions,
        "bootstrap": bootstrap,
        "confirmatory_protocol": confirmatory_protocol(),
        "claim_assessment": claim,
        "selection_history_note": SELECTION_HISTORY_NOTE,
        "tuning_budget_disclosure": TUNING_BUDGET_DISCLOSURE,
        "retrieval_protocol_asymmetry": asymmetry,
        "subclaim_efficiency": {
            "bse_official": summarize_subclaim_efficiency(bse_obs),
            "frozen_ddre": summarize_subclaim_efficiency(ddre_obs),
        },
        "no_fairness_correction_applied": True,
        "tuning_performed_in_this_run": False,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    rows = prediction_rows("bse_official", held_out, bse_results, bse_obs)
    rows += prediction_rows("frozen_ddre", held_out, ddre_results, ddre_obs)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_report(bse_metrics, ddre_metrics, deltas, reductions, bootstrap, claim, asymmetry)
    print(f"\n  {json_path}")
    print(f"    sha256 {sha256_file(json_path)}")
    print(f"  {csv_path}")
    print(f"    sha256 {sha256_file(csv_path)}")
    return 0


def print_report(bse, ddre, deltas, reductions, bootstrap, claim, asymmetry):
    print("\n" + "=" * 104)
    print("HELD-OUT RESULT")
    print("=" * 104)
    print(f"  {'metric':<40}{'BSE official':>16}{'frozen DDRE':>16}{'delta':>14}")
    print("  " + "-" * 86)
    for label, get in (
        ("nonfactual PR-AUC", lambda m: m["nonfactual"]["auc_pr"]),
        ("factual PR-AUC", lambda m: m["factual"]["auc_pr"]),
        ("balanced PR-AUC", lambda m: m["balanced_pr_auc"]),
        ("accuracy", lambda m: m["accuracy"]),
        ("macro F1", lambda m: m["macro_f1"]),
        ("docs / sentence",
         lambda m: m["efficiency"]["avg_retrieved_documents_per_sentence"]),
        ("docs / subclaim",
         lambda m: m["efficiency"]["avg_retrieved_documents_per_subclaim"]),
        ("NLI spans / sentence",
         lambda m: m["efficiency"]["avg_nli_span_calls_per_sentence"]),
    ):
        b, d = get(bse), get(ddre)
        print(f"  {label:<40}{b:>16.6f}{d:>16.6f}{d - b:>+14.6f}")

    for label, key in (
        ("retrieval reduction", "retrieval_reduction_fraction"),
        ("NLI-call reduction", "nli_call_reduction_fraction"),
    ):
        value = reductions[key]
        print(f"  {label:<40}{'':>16}{'':>16}"
              f"{'n/a' if value is None else f'{100.0 * value:>13.2f}%'}")

    print("\n  PAIRED PASSAGE BOOTSTRAP "
          f"({bootstrap['n_resamples']} resamples, seed {bootstrap['seed']}, "
          f"{bootstrap['bootstrap_unit']} unit)")
    print(f"  {'endpoint':<44}{'observed':>13}{'95% CI low':>13}{'95% CI high':>13}  sign")
    print("  " + "-" * 92)
    for name, entry in bootstrap["endpoints"].items():
        print(
            f"  {name:<44}{entry['observed']:>13.6f}{entry['ci_lower']:>13.6f}"
            f"{entry['ci_upper']:>13.6f}  {entry['sign_convention']}"
        )

    print(f"\n  claim status              {claim['claim_status']}")
    print(f"  validation selection               {claim['validation_selection']}")
    print(f"  validation selection confirmatory  {claim['validation_selection_confirmatory']}")
    print(f"  interpretation            {claim['interpretation']}")
    print("\n  D-02 RETRIEVAL PROTOCOL ASYMMETRY (descriptive; no correction applied)")
    # Indexed, not probed: a renamed key in src.evaluation must break this
    # report loudly rather than silently drop the disclosure.
    print(
        "    bse_zero_retrieval_nonempty_subclaims       "
        f"{asymmetry['bse_official']['zero_retrieval_nonempty_subclaims']}"
    )
    print(
        "    ddre_zero_retrieval_nonempty_subclaims      "
        f"{asymmetry['ddre_ulsif']['zero_retrieval_nonempty_subclaims']}"
    )
    print(
        "    ddre_first_retrieval_floor_documents        "
        f"{asymmetry['ddre_first_retrieval_floor_documents']}"
    )
    print(
        "    ddre_documents_above_first_retrieval_floor  "
        f"{asymmetry['ddre_documents_above_first_retrieval_floor']}"
    )
    print(f"    {asymmetry['confirmatory_endpoint_unchanged']}")
    print(f"\n  {TUNING_BUDGET_DISCLOSURE}")


if __name__ == "__main__":
    sys.exit(main())
