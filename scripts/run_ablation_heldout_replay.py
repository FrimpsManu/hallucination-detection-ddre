#!/usr/bin/env python3
"""Held-out ablation replay of the frozen cells A, B, C and D (prereg §6).

EXPLORATORY / POST-HOC held-out ablation, not confirmatory.

Replays the four frozen cells once on the 190 held-out passages from the
completed full-coverage held-out cache (read-only). Every configuration comes
from constants checked against the frozen validation artifact; there is no
option to change any of them, and no selection or tuning code is reachable.
D0 is not evaluated (no eligible validation configuration).

Order, each step fail-closed:

1. pinned inputs: validation freeze, full held-out cache and its manifest,
   preregistration, canonical predictions CSV and result, D-03 cache (NBC
   training scores); the outputs must not already exist;
2. the frozen cells are read from the validation freeze and verified;
3. the held-out split is re-derived and matched to the canonical run;
   ``--dry-run`` stops here;
4. A and D are replayed and must reproduce the canonical predictions exactly
   (the consistency gate); otherwise no ablation result is written;
5. B and C are replayed;
6. metrics, the paired passage bootstrap and the preregistered interpretation
   rules are computed, and one result JSON plus a predictions CSV are written
   and hashed.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_frozen_heldout as frozen  # noqa: E402
from src import ablation_replay as R  # noqa: E402
from src import ablation_selection as S  # noqa: E402
from src.cache_completion import sha256_file  # noqa: E402

PINNED = {
    "validation_freeze": ("ablation_validation_freeze.json",
                          "a1824f663653b66df9deec590146de67992bd6a63320ccfc5f18e9a7bd7967ac"),
    "full_cache": ("ablation_heldout_full_cache.sqlite",
                   "2497487664e2df552ba9bf28947c4ccae7b5f66d2feccbada05ffa5aaea88f28"),
    "full_cache_manifest": ("ablation_heldout_full_cache_manifest.json",
                            "5163c8a2ed8b8730e245964e65ce669d7bb30c6603e40de7be6d25d42d2c12d7"),
    "canonical_predictions": ("frozen_heldout_predictions_fae3eee.csv",
                              "a78b09b7aa09803a21a761b3bede7600180014b97142a381b509d7a1267211c1"),
    "canonical_result": ("frozen_heldout_result_fae3eee.json",
                         "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67"),
    "d03_cache": ("d03_diagnostic_cache_main_2661b04.sqlite",
                  "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"),
}
RESULT_NAME = "ablation_heldout_result.json"
PREDICTIONS_NAME = "ablation_heldout_predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts-dir", default=str(frozen.ARTIFACTS))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify inputs, cells and split. Replays nothing, writes nothing.")
    return parser.parse_args()


def paths(artifacts):
    p = {name: artifacts / filename for name, (filename, _) in PINNED.items()}
    p["preregistration"] = PROJECT_ROOT / S.PREREGISTRATION_PATH
    p["result"] = artifacts / RESULT_NAME
    p["predictions"] = artifacts / PREDICTIONS_NAME
    return p


def verify_inputs(p):
    expected = {name: sha for name, (_, sha) in PINNED.items()}
    expected["preregistration"] = S.PREREGISTRATION_SHA256
    digests, problems = {}, []
    for name, want in expected.items():
        got = sha256_file(p[name]) if p[name].exists() else None
        digests[name] = got
        if got != want:
            problems.append(f"{name}: sha256 {got}, pinned {want}")
    for name in ("result", "predictions"):
        if p[name].exists():
            problems.append(f"{name} already exists at {p[name]}; the replay runs once")
    if problems:
        raise R.ReplayRefused("inputs:\n  " + "\n  ".join(problems))
    manifest = json.loads(p["full_cache_manifest"].read_text())
    if (manifest.get("status") != "complete"
            or manifest.get("derived_sha256") != PINNED["full_cache"][1]
            or (manifest.get("coverage") or {}).get("missing_after") != 0):
        raise R.ReplayRefused("the full-cache manifest does not certify the pinned complete cache")
    return digests


def verify_unchanged(p, digests):
    """Every input verified at the start, the preregistration included, is unchanged."""
    changed = [name for name, sha in digests.items() if sha256_file(p[name]) != sha]
    if changed:
        raise R.ReplayRefused(f"inputs changed during the replay: {changed}")


def held_out(data_root, canonical_result):
    records, split = frozen.frozen_split(data_root)
    for field in ("held_out_passage_ids_sha256", "validation_passage_ids_sha256"):
        if split[field] != canonical_result["split"][field]:
            raise R.ReplayRefused(f"re-derived split {field} differs from the canonical run")
    return records, split


def build(args, p, cells):
    """The four frozen detectors. No other configuration can be constructed here."""
    from src.baseline_core import BSEDetector, build_nbc_histograms
    from src.cached_document_scores import CachedDocumentScorer
    from src.ddre_core import DDREDetector, ULSIFDensityRatio
    from src.hyperparameter_outcome_sensitivity import candidate_estimator, verify_production_fit
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.data_root, per_class=None)
    nbc = CachedDocumentScorer(p["d03_cache"], frozen.OFFICIAL_MODEL, SCORE_VERSION, split_text)
    try:
        factual = nbc.score_pairs([(x["premise"], x["hypothesis"]) for x in positive])
        hallucinated = nbc.score_pairs([(x["premise"], x["hypothesis"]) for x in negative])
        pos_hist, neg_hist, _, _ = build_nbc_histograms(positive, negative, nbc)
    finally:
        nbc.close()
    production = ULSIFDensityRatio(random_state=frozen.EXPECTED_SPLIT_SEED).fit(
        factual, hallucinated)
    verify_production_fit(production)
    d = cells["D"]
    d_estimator = candidate_estimator(
        production, production._as_column(factual), production._as_column(hallucinated),
        d["sigma"], d["lambda"])

    def band(estimator, cfg):
        return DDREDetector(estimator, lower_threshold=cfg["lower"],
                            upper_threshold=cfg["upper"], p0=R.COSTS["p0"],
                            c_miss=R.COSTS["c_miss"], c_false_alarm=R.COSTS["c_false_alarm"],
                            max_docs=R.COSTS["max_docs"])

    detectors = {
        "A": BSEDetector(pos_hist, neg_hist, mode="official", p0=R.COSTS["p0"],
                         c_miss=R.COSTS["c_miss"], c_false_alarm=R.COSTS["c_false_alarm"],
                         c_retrieve=R.COSTS["c_retrieve"], max_docs=R.COSTS["max_docs"]),
        "B": band(S.HistogramRatio(pos_hist, neg_hist, kappa=cells["B"]["kappa"]), cells["B"]),
        "C": band(S.HistogramRatio(pos_hist, neg_hist, kappa=cells["C"]["kappa"]), cells["C"]),
        "D": band(d_estimator, d),
    }
    scorer = CachedDocumentScorer(p["full_cache"], frozen.OFFICIAL_MODEL, SCORE_VERSION,
                                  split_text)
    return detectors, scorer


def replay(cell, detector, records, scorer):
    from src.evaluation import (
        evaluate_detector_with_traces,
        prediction_rows,
        summarize_subclaim_efficiency,
    )

    metrics, results, obs = evaluate_detector_with_traces(
        detector, records, scorer, description=f"cell {cell}", use_cache=True)
    metrics["efficiency"].pop("wall_clock_seconds", None)
    metrics["efficiency"].pop("avg_wall_clock_seconds_per_sentence", None)
    metrics["stopping_depth"] = summarize_subclaim_efficiency(obs)
    return metrics, obs, prediction_rows(f"cell_{cell}", records, results, obs)


def reported_metrics(m):
    return {
        "nonfactual_pr_auc": m["nonfactual"]["auc_pr"],
        "factual_pr_auc": m["factual"]["auc_pr"],
        "balanced_pr_auc": m["balanced_pr_auc"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "factual_precision": m["factual"]["precision"],
        "factual_recall": m["factual"]["recall"],
        "nonfactual_precision": m["nonfactual"]["precision"],
        "nonfactual_recall": m["nonfactual"]["recall"],
        "documents_per_sentence": m["efficiency"]["avg_retrieved_documents_per_sentence"],
        "documents_per_subclaim": m["efficiency"]["avg_retrieved_documents_per_subclaim"],
        "nli_spans_per_sentence": m["efficiency"]["avg_nli_span_calls_per_sentence"],
        "stopping_depth_histogram": m["stopping_depth"]["retrieval_depth_histogram"],
        "full_summary": m,
    }


def git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=str(PROJECT_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    p = paths(Path(args.artifacts_dir).expanduser())
    digests = verify_inputs(p)
    cells = R.frozen_cells(json.loads(p["validation_freeze"].read_text()))
    canonical_result = json.loads(p["canonical_result"].read_text())
    records, split = held_out(args.data_root, canonical_result)
    print(f"cells: {json.dumps({k: v for k, v in cells.items() if k != 'D0'})}")
    print(f"D0: not evaluated ({cells['D0']['reason'][:60]}...)")
    print(f"held-out sentences: {len(records)}")
    if args.dry_run:
        print("--dry-run: inputs, cells and split verified. Nothing replayed, nothing written.")
        return 0

    canonical = list(csv.DictReader(p["canonical_predictions"].open()))
    by_method = {m: [r for r in canonical if r["method"] == m]
                 for m in ("bse_official", "frozen_ddre")}
    detectors, scorer = build(args, p, cells)
    metrics, observations, rows = {}, {}, {}
    try:
        gate = []
        for cell, method in (("A", "bse_official"), ("D", "frozen_ddre")):
            metrics[cell], observations[cell], rows[cell] = replay(
                cell, detectors[cell], records, scorer)
            gate.append(R.consistency_gate(cell, by_method[method], rows[cell]))
        for cell in ("B", "C"):
            metrics[cell], observations[cell], rows[cell] = replay(
                cell, detectors[cell], records, scorer)
        misses = scorer.misses
    finally:
        scorer.close()
    if misses:
        raise R.ReplayRefused(f"{misses} cache misses on the full-coverage cache")

    bootstrap = R.multi_cell_bootstrap(observations)
    balanced = {c: metrics[c]["balanced_pr_auc"] for c in R.CELLS}
    interpretation = R.interpret(bootstrap["endpoints"], balanced)

    verify_unchanged(p, digests)

    result = {
        "artifact": "ablation-heldout-replay",
        "status": R.STATUS,
        "confirmatory": False,
        "not_confirmatory_statement": R.NOT_CONFIRMATORY,
        "protocol": S.PREREGISTRATION_PATH,
        "preregistration_sha256": digests["preregistration"],
        "preregistration_merge_commit": S.PREREGISTRATION_MERGE_COMMIT,
        "code_commit": git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(git("status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": str(p[name]), "sha256": digests[name]} for name in digests},
        "inputs_unchanged": True,
        "split": {k: split[k] for k in ("held_out_passages", "held_out_sentences",
                                         "held_out_subclaims", "held_out_passage_ids_sha256")},
        "cells": cells,
        "tuning_budgets": R.TUNING_BUDGETS,
        "consistency_gate": {"passed": True, "cells": gate},
        "metrics": {c: reported_metrics(metrics[c]) for c in R.CELLS},
        "bootstrap": bootstrap,
        "interpretation": interpretation,
    }
    R.check_language(json.dumps(interpretation))
    p["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
    with p["predictions"].open("w", newline="") as handle:
        all_rows = [r for c in R.CELLS for r in rows[c]]
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"result {p['result']}\n  sha256 {sha256_file(p['result'])}")
    print(f"predictions {p['predictions']}\n  sha256 {sha256_file(p['predictions'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (R.ReplayRefused, frozen.Aborted) as exc:
        print(f"\nREFUSED: {exc}\nNo ablation result was written.")
        sys.exit(1)
