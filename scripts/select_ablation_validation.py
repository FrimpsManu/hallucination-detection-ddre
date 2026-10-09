#!/usr/bin/env python3
"""Validation selection for ablation cells B, C and D0, frozen once (prereg §4).

Runs on the frozen 48-passage validation split only, replayed from the D-03
validation cache opened read-only. It never opens a held-out cache, never
evaluates a held-out record, and never computes a held-out metric.

Order, each step fail-closed:

1. pinned inputs: the D-03 validation cache, the D-03 report, the sensitivity
   artifact (which stores the BSE validation reference and the recorded D0
   rows), and the preregistration document itself; the freeze artifact must not
   already exist;
2. the validation split is re-derived and must match the recorded split by
   passage identity; ``--dry-run`` stops here;
3. NBC histograms and the production uLSIF fit are rebuilt from cached scores;
   the fit must reproduce the recorded production selection;
4. BSE official on validation must reproduce the stored reference PR-AUCs with
   exact float equality;
5. B (32), C (224) and D0 (32) candidates are evaluated, each evaluation behind
   the validation-only guard;
6. consistency: C at kappa = 1 must equal B exactly, and D0 must re-derive the
   recorded sensitivity rows exactly;
7. each cell is selected by the preregistered rule, with no fallback, and one
   freeze artifact is written and hashed.
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

from src import ablation_selection as S  # noqa: E402
from src.cache_completion import sha256_file  # noqa: E402
from src.hyperparameter_outcome_sensitivity import (  # noqa: E402
    EXPECTED_FACTUAL_TRAINING,
    HeldOutLeak,
    ProductionFitChanged,
    EXPECTED_HALLUCINATED_TRAINING,
    EXPECTED_SPLIT_SEED,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VALIDATION_PASSAGES,
    EXPECTED_VALIDATION_SENTENCES,
    EXPECTED_VALIDATION_SUBCLAIMS,
    assert_validation_only,
    candidate_estimator,
    passage_ids_sha256,
    verify_production_fit,
)

ARTIFACTS = Path("~/ddre-artifacts/gate1-2026-09-20").expanduser()
OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
P0, C_MISS, C_FALSE_ALARM, C_RETRIEVE, MAX_DOCS = 0.5, 28.0, 96.0, 1.0, 10

# The only cache this stage reads: the D-03 validation cache.
VALIDATION_CACHE_NAME = "d03_diagnostic_cache_main_2661b04.sqlite"
VALIDATION_CACHE_SHA256 = "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"
D03_REPORT_NAME = "d03_density_ratio_diagnostic_main_2661b04.json"
D03_REPORT_SHA256 = "f2468e5bb1265e107f4060d36782f6f21ba9f5f55141dde4814d5ffd11aa7f6d"
SENSITIVITY_NAME = "d03_hyperparameter_outcome_sensitivity_main_59f4f53.json"
SENSITIVITY_SHA256 = "fc5604854b18f3e3e712fe7ef01859b58dac569e99696041713abb11daa6dcc0"
FREEZE_NAME = "ablation_validation_freeze.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify inputs and the split. Evaluates nothing, writes nothing.")
    return parser.parse_args()


def paths(artifacts):
    return {
        "validation_cache": artifacts / VALIDATION_CACHE_NAME,
        "d03_report": artifacts / D03_REPORT_NAME,
        "sensitivity": artifacts / SENSITIVITY_NAME,
        "preregistration": PROJECT_ROOT / S.PREREGISTRATION_PATH,
        "freeze": artifacts / FREEZE_NAME,
    }


def verify_inputs(p):
    pinned = {
        "validation_cache": VALIDATION_CACHE_SHA256,
        "d03_report": D03_REPORT_SHA256,
        "sensitivity": SENSITIVITY_SHA256,
        "preregistration": S.PREREGISTRATION_SHA256,
    }
    problems = []
    digests = {}
    for name, want in pinned.items():
        got = sha256_file(p[name]) if p[name].exists() else None
        digests[name] = got
        if got != want:
            problems.append(f"{name}: sha256 {got}, pinned {want}")
    if p["freeze"].exists():
        problems.append(f"freeze artifact already exists at {p['freeze']}; selection runs once")
    if problems:
        raise S.SelectionRefused("inputs:\n  " + "\n  ".join(problems))
    return digests


def validation_split(data_root, sensitivity):
    """Validation records only. Held-out records are dropped here, never returned."""
    from src.wang_data import group_split_records, load_sentence_records

    validation, _held_out, metadata = group_split_records(
        load_sentence_records(data_root, strict=True),
        validation_fraction=EXPECTED_VALIDATION_FRACTION,
        random_state=EXPECTED_SPLIT_SEED,
    )
    del _held_out
    held_out_ids = sorted(int(x) for x in metadata["test_passage_ids"])
    ids = sorted({int(r.passage_index) for r in validation})
    identity = {
        "validation_passages": len(ids),
        "validation_sentences": len(validation),
        "validation_subclaims": sum(len(r.subclaims) for r in validation),
        "validation_passage_ids_sha256": passage_ids_sha256(ids),
        "held_out_passages_excluded": len(held_out_ids),
    }
    expected = (EXPECTED_VALIDATION_PASSAGES, EXPECTED_VALIDATION_SENTENCES,
                EXPECTED_VALIDATION_SUBCLAIMS, sensitivity["split"]["sha256"])
    observed = (identity["validation_passages"], identity["validation_sentences"],
                identity["validation_subclaims"], identity["validation_passage_ids_sha256"])
    if observed != expected:
        raise S.SelectionRefused(f"validation split {observed} != recorded {expected}")
    assert_validation_only(validation, held_out_ids, where="the validation population")
    return validation, held_out_ids, identity


def evaluate(detector, records, held_out_ids, scorer, where):
    """One validation evaluation, behind the validation-only guard."""
    from src.evaluation import summarize_method

    assert_validation_only(records, held_out_ids, where=where)
    results = [detector.detect_sentence(r, scorer, use_cache=True) for r in records]
    return summarize_method(records, results)


def run_cells(records, held_out_ids, scorer, pos_hist, neg_hist, d0_estimator, reference):
    from src.ddre_core import DDREDetector

    def band(estimator, lower, upper):
        return DDREDetector(estimator, lower_threshold=lower, upper_threshold=upper,
                            p0=P0, c_miss=C_MISS, c_false_alarm=C_FALSE_ALARM,
                            max_docs=MAX_DOCS)

    rows = {"B": [], "C": [], "D0": []}
    for params in S.cell_parameters("B"):
        est = S.HistogramRatio(pos_hist, neg_hist, kappa=1.0)
        m = evaluate(band(est, params["lower"], params["upper"]), records, held_out_ids,
                     scorer, "cell B")
        rows["B"].append(S.candidate_row("B", params, m, reference))
    for params in S.cell_parameters("C"):
        est = S.HistogramRatio(pos_hist, neg_hist, kappa=params["kappa"])
        m = evaluate(band(est, params["lower"], params["upper"]), records, held_out_ids,
                     scorer, "cell C")
        rows["C"].append(S.candidate_row("C", params, m, reference))
    for params in S.cell_parameters("D0"):
        m = evaluate(band(d0_estimator, params["lower"], params["upper"]), records,
                     held_out_ids, scorer, "cell D0")
        rows["D0"].append(S.candidate_row("D0", params, m, reference))
    return rows


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=str(PROJECT_ROOT), text=True).strip())
    except Exception:
        return None


def main():
    args = parse_args()
    p = paths(Path(args.artifacts_dir).expanduser())
    digests = verify_inputs(p)
    sensitivity = json.loads(p["sensitivity"].read_text())
    reference = S.stored_reference(sensitivity)
    recorded_d0 = S.recorded_production_rows(sensitivity)
    records, held_out_ids, identity = validation_split(args.data_root, sensitivity)
    print(f"validation split: {identity}")
    if args.dry_run:
        print("--dry-run: inputs and split verified. Nothing evaluated, nothing written.")
        return 0

    from src.baseline_core import BSEDetector, build_nbc_histograms
    from src.cached_document_scores import CachedDocumentScorer
    from src.ddre_core import (
        CANDIDATE_LOWER_GRID,
        CANDIDATE_UPPER_GRID,
        ULSIFDensityRatio,
        cost_consistent_thresholds,
    )
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_nbc_pairs

    S.verify_band_grid(cost_consistent_thresholds(
        C_MISS, C_FALSE_ALARM, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID))

    scorer = CachedDocumentScorer(p["validation_cache"], OFFICIAL_MODEL, SCORE_VERSION,
                                  split_text)
    try:
        positive, negative = load_nbc_pairs(args.data_root, per_class=None)
        if (len(positive), len(negative)) != (EXPECTED_FACTUAL_TRAINING,
                                              EXPECTED_HALLUCINATED_TRAINING):
            raise S.SelectionRefused(f"NBC counts {len(positive)}/{len(negative)}")
        factual = scorer.score_pairs([(x["premise"], x["hypothesis"]) for x in positive])
        hallucinated = scorer.score_pairs([(x["premise"], x["hypothesis"]) for x in negative])
        pos_hist, neg_hist, _, _ = build_nbc_histograms(positive, negative, scorer)

        production = ULSIFDensityRatio(random_state=EXPECTED_SPLIT_SEED).fit(
            factual, hallucinated)
        production_fit = verify_production_fit(production)
        d0_estimator = candidate_estimator(
            production, production._as_column(factual), production._as_column(hallucinated),
            S.D0_SIGMA, S.D0_LAMBDA)

        bse = BSEDetector(pos_hist, neg_hist, mode="official", p0=P0, c_miss=C_MISS,
                          c_false_alarm=C_FALSE_ALARM, c_retrieve=C_RETRIEVE,
                          max_docs=MAX_DOCS)
        baseline = evaluate(bse, records, held_out_ids, scorer, "the BSE reference")
        replayed_reference = S.verify_reference(baseline, reference)

        rows = run_cells(records, held_out_ids, scorer, pos_hist, neg_hist,
                         d0_estimator, reference)
        cache_misses = scorer.misses
    finally:
        scorer.close()
    if cache_misses:
        raise S.SelectionRefused(f"{cache_misses} validation cache misses")

    problems = S.c_kappa_one_matches_b(rows["B"], rows["C"])
    problems += S.d0_matches_record(rows["D0"], recorded_d0)
    if problems:
        raise S.SelectionRefused("consistency checks failed:\n  " + "\n  ".join(problems[:20]))

    outcomes = {cell: S.select(cell, rows[cell]) for cell in ("B", "C", "D0")}
    if sha256_file(p["validation_cache"]) != VALIDATION_CACHE_SHA256:
        raise S.SelectionRefused("the validation cache changed during selection")

    freeze = {
        "artifact": "ablation-validation-freeze",
        "protocol": S.PREREGISTRATION_PATH,
        "preregistration_sha256": digests["preregistration"],
        "preregistration_merge_commit": S.PREREGISTRATION_MERGE_COMMIT,
        "selection_rule_version": S.SELECTION_RULE_VERSION,
        "selection_rule": S.SELECTION_RULE,
        "quality_tolerance": S.QUALITY_TOLERANCE,
        "code_commit": git_commit(),
        "code_tree_dirty": git_dirty(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "held_out_records_evaluated": 0,
        "held_out_cache_opened": False,
        "inputs": {
            "validation_cache_sha256": digests["validation_cache"],
            "validation_cache_unchanged": True,
            "d03_report_sha256": digests["d03_report"],
            "sensitivity_artifact_sha256": digests["sensitivity"],
        },
        "validation_split": identity,
        "validation_reference": {
            "source": "sensitivity artifact bse_official_validation (full precision)",
            "stored": reference,
            "replayed": replayed_reference,
            "exact_match": True,
        },
        "production_fit": production_fit,
        "grids": {"bands": [list(b) for b in S.BAND_GRID], "kappa": list(S.KAPPA_GRID),
                  "d0": {"sigma": S.D0_SIGMA, "lambda": S.D0_LAMBDA}},
        "candidate_counts": {cell: len(rows[cell]) for cell in rows},
        "expected_candidate_counts": S.EXPECTED_CANDIDATES,
        "eligible_counts": {cell: outcomes[cell]["eligible"] for cell in outcomes},
        "consistency": {"c_kappa_one_equals_b": True, "d0_matches_recorded_rows": True},
        "B": outcomes["B"],
        "C": outcomes["C"],
        "D0": outcomes["D0"],
        "candidates": rows,
    }
    p["freeze"].write_text(json.dumps(freeze, indent=1) + "\n")
    freeze_sha = sha256_file(p["freeze"])
    for cell in ("B", "C", "D0"):
        o = outcomes[cell]
        chosen = o["selected"]
        params = ({k: chosen[k] for k in ("kappa", "lower", "upper") if k in chosen}
                  if chosen else None)
        print(f"{cell:2s} candidates {o['candidates']:3d} eligible {o['eligible']:3d} "
              f"outcome {o['outcome']} {params or ''}")
    print(f"freeze artifact {p['freeze']}\n  sha256 {freeze_sha}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (S.SelectionRefused, HeldOutLeak, ProductionFitChanged) as exc:
        print(f"\nREFUSED: {exc}\nNothing frozen.")
        sys.exit(1)
