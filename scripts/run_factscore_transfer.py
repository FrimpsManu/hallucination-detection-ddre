#!/usr/bin/env python3
"""FActScore frozen-transfer replay of cells A, B, C and D, once (prereg §5-§8).

Runs the four SelfCheckGPT cells exactly as frozen on the preregistered
FActScore analysis set, reading every NLI score from the frozen FActScore
cache (read-only). Nothing is selected or tuned:
- B and C come from the SelfCheckGPT validation freeze;
- D and A are constants;
- the histograms and the uLSIF fit come from Wang's NBC pairs.

The detector builder and the traced replay are the reviewed SelfCheckGPT
replay functions, reused as they are.

Reported:
- every metric per cell (pooled, and per generator);
- the paired entity-cluster bootstrap;
- the RQ1 decision rules;
- the SelfCheckGPT RQ2 rules applied unchanged;
- the score-support diagnostic, computed before any interpretation is
  written.

The result JSON and the predictions CSV are written once and hashed.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_ablation_heldout_replay as RM  # noqa: E402
from src import ablation_replay as AR  # noqa: E402
from src import factscore_transfer as T  # noqa: E402
from src.factscore_data import (  # noqa: E402
    GENERATORS,
    PREREGISTRATION_MERGE_COMMIT,
    PREREGISTRATION_PATH,
    PREREGISTRATION_SHA256,
    load_analysis_set,
    sha256_path,
)
from src.factscore_records import build_records, read_evidence  # noqa: E402

RELEASE = Path("~/ddre-artifacts/factscore-release").expanduser()
ARTIFACTS = Path("~/ddre-artifacts/gate1-2026-09-20").expanduser()

PINNED_RELEASE = {
    "evidence": ("factscore_evidence_top10.jsonl",
                 "44f479128d41818fe5a056f125407dbde75a8735669c398a3b945cfeaa310d09"),
    "evidence_manifest": ("factscore_evidence_manifest.json",
                          "4983050b19eedcee5a3bbd2ac01bbda1a4f08bf3e38e8a6f46d812315eb2ced4"),
    # Pinned once the one-time scoring run has finished (prereg §9 step 3).
    "score_cache": ("factscore_nli_cache.sqlite", None),
    "score_cache_manifest": ("factscore_nli_cache_manifest.json", None),
}
PINNED_ARTIFACTS = {
    "validation_freeze": RM.PINNED["validation_freeze"],
    "d03_cache": RM.PINNED["d03_cache"],
}
RESULT_NAME = "factscore_transfer_result.json"
PREDICTIONS_NAME = "factscore_transfer_predictions.csv"


class TransferRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release-dir", default=str(RELEASE))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS))
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify inputs, cells and records. Replays nothing, writes nothing.")
    return parser.parse_args()


def paths(release, artifacts):
    p = {name: release / filename for name, (filename, _) in PINNED_RELEASE.items()}
    p.update({name: artifacts / filename for name, (filename, _) in PINNED_ARTIFACTS.items()})
    p["preregistration"] = PROJECT_ROOT / PREREGISTRATION_PATH
    p["result"] = release / RESULT_NAME
    p["predictions"] = release / PREDICTIONS_NAME
    return p


def expected_digests():
    out = {name: sha for name, (_, sha) in {**PINNED_RELEASE, **PINNED_ARTIFACTS}.items()}
    out["preregistration"] = PREREGISTRATION_SHA256
    return out


def verify_inputs(p):
    problems, digests = [], {}
    for name, want in expected_digests().items():
        if want is None:
            problems.append(f"{name}: no pinned digest yet")
            continue
        got = sha256_path(p[name]) if p[name].exists() else None
        digests[name] = got
        if got != want:
            problems.append(f"{name}: sha256 {got}, pinned {want}")
    for name in ("result", "predictions"):
        if p[name].exists():
            problems.append(f"{name} already exists; the transfer replay runs once")
    if problems:
        raise TransferRefused("inputs:\n  " + "\n  ".join(problems))
    manifest = json.loads(p["score_cache_manifest"].read_text())
    if (manifest.get("status") != "complete"
            or manifest.get("derived_sha256") != digests["score_cache"]
            or manifest.get("evidence_sha256") != digests["evidence"]
            or (manifest.get("coverage") or {}).get("missing_after") != 0):
        raise TransferRefused("the score-cache manifest does not certify the pinned complete cache")
    return digests


def verify_unchanged(p, digests):
    changed = [name for name, sha in digests.items() if sha256_path(p[name]) != sha]
    if changed:
        raise TransferRefused(f"inputs changed during the replay: {changed}")


def generator_of(item):
    return GENERATORS[item.sentence_index // 1_000_000]


def consumed_scores(records, observations, scorer):
    """NLI scores of the documents a cell actually read, and of all retrieved ones."""
    consumed, every = [], []
    for record, obs in zip(records, observations):
        for subclaim, trace in zip(record.subclaims, obs.subclaim_traces):
            scores = [scorer.score_document(subclaim.text, d.page_content)[0]
                      for d in subclaim.documents]
            every.extend(scores)
            consumed.extend(scores[:trace.documents_used])
    return consumed, every


def per_generator_metrics(records, observations):
    from src.evaluation import summarize_method

    out = {}
    for g in GENERATORS:
        idx = [i for i, r in enumerate(records) if generator_of(r) == g]
        out[g] = {c: summarize_method([records[i] for i in idx],
                                      [observations[c][i].result for i in idx])
                  for c in AR.CELLS}
    return out


def git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=str(PROJECT_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    p = paths(Path(args.release_dir).expanduser(), Path(args.artifacts_dir).expanduser())
    digests = verify_inputs(p)
    cells = AR.frozen_cells(json.loads(p["validation_freeze"].read_text()))
    sentences = load_analysis_set(Path(args.release_dir).expanduser() / "data")
    records, clusters = build_records(sentences, read_evidence(p["evidence"]))
    print(f"cells A-D frozen; D0 not evaluated; {len(records)} sentences, "
          f"{len(clusters)} entity clusters")
    if args.dry_run:
        print("--dry-run: inputs, cells and records verified. Nothing replayed, nothing written.")
        return 0

    detectors, scorer = RM.build(SimpleNamespace(data_root=args.wang_data_root),
                                 {"d03_cache": p["d03_cache"], "full_cache": p["score_cache"]},
                                 cells)
    metrics, observations, rows, consumed = {}, {}, {}, {}
    try:
        for cell in AR.CELLS:
            metrics[cell], observations[cell], rows[cell] = RM.replay(
                cell, detectors[cell], records, scorer)
        for cell in AR.CELLS:
            consumed[cell], every = consumed_scores(records, observations[cell], scorer)
        misses = scorer.misses
    finally:
        scorer.close()
    if misses:
        raise TransferRefused(f"{misses} cache misses on the frozen FActScore cache")

    support = T.support_diagnostic(consumed, every)
    pooled = T.bootstrap(observations)
    by_generator = {g: T.bootstrap(T.subset(observations, lambda o, g=g: generator_of(o) == g))
                    for g in GENERATORS}
    balanced = {c: metrics[c]["balanced_pr_auc"] for c in AR.CELLS}
    interpretation = {
        "status": T.STATUS,
        "rq1_pooled": T.rq1(pooled["endpoints"]),
        "rq1_by_generator_descriptive": {g: T.rq1(b["endpoints"]) for g, b in by_generator.items()},
        "rq2_selfcheckgpt_rules_pooled": T.rq2(pooled["endpoints"], balanced),
        "support": support,
    }
    AR.check_language(json.dumps(interpretation))
    verify_unchanged(p, digests)

    result = {
        "artifact": "factscore-frozen-transfer",
        "status": T.STATUS,
        "protocol": PREREGISTRATION_PATH,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_merge_commit": PREREGISTRATION_MERGE_COMMIT,
        "code_commit": git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(git("status", "--porcelain")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": str(p[name]), "sha256": sha} for name, sha in digests.items()},
        "inputs_unchanged": True,
        "analysis_set": {"sentences": len(records), "entity_clusters": len(clusters)},
        "cells": cells,
        "tuning_budgets": AR.TUNING_BUDGETS,
        "metrics_pooled": {c: RM.reported_metrics(metrics[c]) for c in AR.CELLS},
        "metrics_by_generator": per_generator_metrics(records, observations),
        "bootstrap_pooled": pooled,
        "bootstrap_by_generator": by_generator,
        "interpretation": interpretation,
    }
    p["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
    with p["predictions"].open("w", newline="") as handle:
        all_rows = [r for c in AR.CELLS for r in rows[c]]
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    for name in ("result", "predictions"):
        p[name].chmod(0o444)
        print(f"{name} {p[name]}\n  sha256 {sha256_path(p[name])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (TransferRefused, AR.ReplayRefused) as exc:
        print(f"\nREFUSED: {exc}\nNo transfer result was written.")
        sys.exit(1)
