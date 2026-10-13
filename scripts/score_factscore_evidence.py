#!/usr/bin/env python3
"""Score every retrieved FActScore document with the frozen NLI model, once.

Implements step 3 of ``docs/factscore_transfer_preregistration.md`` (§4): every
span of every retrieved document of every subclaim is scored by the frozen
SelfCheckGPT scorer into a new cache, after the existing 398-pair exact-score
probe passes. It reuses the reviewed cache-completion pipeline
(``src.ablation_cache.complete_cache``):
- source digest check;
- the probe against the source, writing nothing;
- a faithful, bound copy;
- scoring of missing pairs only;
- coverage, preservation and row checks;
- a read-only freeze and a manifest.

The source is the D-03 validation cache, which holds the 398 NBC sentinel
pairs; the FActScore rows are added to a derived copy. Only completion status,
counts, environment and hashes are inspected. No score value is read, and no
detector or metric runs.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import complete_ablation_heldout_cache as CA  # noqa: E402
from src import ablation_cache as AC  # noqa: E402
from src.factscore_data import (  # noqa: E402
    PREREGISTRATION_MERGE_COMMIT,
    PREREGISTRATION_PATH,
    PREREGISTRATION_SHA256,
    load_analysis_set,
    sha256_path,
)
from src.factscore_records import build_records, read_evidence  # noqa: E402

RELEASE = Path("~/ddre-artifacts/factscore-release").expanduser()
ARTIFACTS = Path("~/ddre-artifacts/gate1-2026-09-20").expanduser()
SOURCE_CACHE_NAME = "d03_diagnostic_cache_main_2661b04.sqlite"
SOURCE_CACHE_SHA256 = "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"
EVIDENCE_NAME = "factscore_evidence_top10.jsonl"
EVIDENCE_MANIFEST_NAME = "factscore_evidence_manifest.json"
CACHE_NAME = "factscore_nli_cache.sqlite"
MANIFEST_NAME = "factscore_nli_cache_manifest.json"
EXPECTED_SUBCLAIMS = 10_386


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release-dir", default=str(RELEASE))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS))
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify inputs and report coverage counts. No model, no writes.")
    return parser.parse_args()


def verify_inputs(release, artifacts):
    problems = []
    if sha256_path(PROJECT_ROOT / PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        problems.append("the preregistration differs from its pinned digest")
    manifest_path = release / EVIDENCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        problems.append("the evidence manifest is missing; run retrieval first")
    else:
        if manifest.get("preregistration_sha256") != PREREGISTRATION_SHA256:
            problems.append("the evidence was not retrieved under this preregistration")
        if manifest.get("labels_recorded") is not False or manifest.get("code_tree_dirty"):
            problems.append("the evidence manifest does not certify a clean, label-free retrieval")
        if manifest.get("subclaims") != EXPECTED_SUBCLAIMS:
            problems.append("the evidence manifest does not cover 10,386 subclaims")
        if sha256_path(release / EVIDENCE_NAME) != manifest.get("evidence_sha256"):
            problems.append("the evidence file differs from its manifest digest")
    for name, sha in ((SOURCE_CACHE_NAME, SOURCE_CACHE_SHA256),
                      (CA.RESULT_NAME, CA.RESULT_SHA256)):
        if sha256_path(artifacts / name) != sha:
            problems.append(f"{name} is not the pinned file")
    for name in (CACHE_NAME, MANIFEST_NAME):
        if (release / name).exists():
            problems.append(f"{name} already exists; scoring runs once")
    if problems:
        raise AC.CompletionAborted("inputs:\n  " + "\n  ".join(problems))
    return manifest, json.loads((artifacts / CA.RESULT_NAME).read_text())


def main():
    args = parse_args()
    release = Path(args.release_dir).expanduser()
    artifacts = Path(args.artifacts_dir).expanduser()
    evidence_manifest, canonical_result = verify_inputs(release, artifacts)
    records, clusters = build_records(load_analysis_set(release / "data"),
                                      read_evidence(release / EVIDENCE_NAME))

    from src.utils import SCORE_VERSION, split_text

    model_name = CA.frozen.OFFICIAL_MODEL
    coverage = AC.enumerate_coverage(records, split_text, model_name=model_name,
                                     score_version=SCORE_VERSION)
    source = artifacts / SOURCE_CACHE_NAME
    present = AC.present_keys(source, coverage["keys"])
    counts = dict(coverage["counts"], clusters=len(clusters),
                  coverage_sha256=coverage["coverage_sha256"],
                  available_in_source=len(present),
                  missing_from_source=len(coverage["keys"]) - len(present))
    print("FActScore coverage (top-10 evidence, every span):")
    for key, value in counts.items():
        print(f"  {key:26s} {value}")
    if args.dry_run:
        print("\n--dry-run: inputs verified. No model loaded, nothing written.")
        return 0

    tokenizer, model, device = CA.load_model(model_name)
    from src.diagnostic_probe import collect_live_environment

    observed = collect_live_environment(
        model_name=model_name, model=model, tokenizer=tokenizer,
        repo_root=PROJECT_ROOT, score_version=SCORE_VERSION, selected_device=device)
    expectations = AC.environment_expectations(
        canonical_result["environment"], revision=CA.REVISION, score_version=SCORE_VERSION)
    problems = AC.check_environment(observed, expectations)
    if problems:
        raise AC.CompletionAborted("environment differs from the frozen run:\n  "
                                   + "\n  ".join(problems))

    from src.utils import EntailmentScorer

    manifest = AC.complete_cache(
        source=source,
        destination=release / CACHE_NAME,
        source_sha256=SOURCE_CACHE_SHA256,
        coverage=coverage,
        run_probe=lambda: CA.compatibility_probe(
            args.wang_data_root, tokenizer, model, model_name, SCORE_VERSION, source),
        open_scorer=lambda path: EntailmentScorer(
            tokenizer, model, model_name, cache_path=path, batch_size=AC.BATCH_SIZE),
        environment={
            "model_name": model_name, "checkpoint_revision": CA.REVISION,
            "checkpoint_identity": observed.get("checkpoint_identity"),
            "tokenizer": observed.get("tokenizer"), "model": observed.get("model"),
            "device": device, "dtype": (observed.get("model") or {}).get("dtype"),
            "batch_size": AC.BATCH_SIZE, "score_version": SCORE_VERSION,
            "libraries": observed.get("libraries"),
            "environment_checks": sorted(expectations),
        },
    )
    manifest.update({
        "protocol": PREREGISTRATION_PATH + " §4",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_merge_commit": PREREGISTRATION_MERGE_COMMIT,
        "evidence_sha256": evidence_manifest["evidence_sha256"],
        "clusters": len(clusters),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": CA.frozen._git_commit(),
    })
    (release / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(f"\nstatus {manifest['status']}; added {manifest['added_rows']} rows; "
          f"missing after {manifest['coverage']['missing_after']}")
    print(f"cache sha256 {manifest['derived_sha256']}")
    print(f"manifest sha256 {sha256_path(release / MANIFEST_NAME)}")
    return 0 if manifest["status"] == AC.STATUS_COMPLETE else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AC.CompletionAborted as exc:
        print(f"\nABORTED: {exc}")
        sys.exit(1)
