#!/usr/bin/env python3
"""Derived correction of the frozen held-out run's NOT_CONFIRMATORY reasons.

The canonical result artifact (runner fae3eee, SHA-256 b5bc08cc...) records
claim_status NOT_CONFIRMATORY, which is correct, but gives two reasons that are
not:

1. "validation threshold selection used the fallback objective, so no
   configuration preserved ... within tolerance". The runner passed a bare
   ``validation_selection_confirmatory=False``, and assess_claim rendered every
   False as the fallback. The frozen configuration came from the post-D-03
   validation sweep, where 102 of 640 configurations were eligible.
2. "the frozen split was not verified: no expected passage IDs were supplied".
   The runner verified the split by identity, but did not pass the expected
   IDs into assess_claim.

This script does NOT modify, regenerate or rename the canonical artifact. It
writes a separate correction artifact that:

* names the canonical artifact by path and SHA-256, and quotes its original
  reasons and interpretation verbatim;
* re-runs assess_claim on the SAVED bootstrap record and the SAVED split and
  run configuration, with the selection history named exactly and the expected
  split re-derived from the released data at the frozen fraction and seed;
* refuses to write anything unless every field of the claim assessment other
  than the reason fields is identical to the original, and the claim status
  is still NOT_CONFIRMATORY.

No scores, predictions, bootstrap samples or metrics are recomputed: nothing is
resampled, no detector runs, no model is loaded.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_frozen_heldout as frozen  # noqa: E402
from src.cache_completion import sha256_file  # noqa: E402
from src.paired_bootstrap import (  # noqa: E402
    CLAIM_NOT_CONFIRMATORY,
    VALIDATION_SELECTION_POST_D03_SWEEP,
    assess_claim,
)

CANONICAL_RESULT_NAME = "frozen_heldout_result_fae3eee.json"
CANONICAL_RESULT_SHA256 = (
    "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67"
)
CORRECTION_NAME = "frozen_heldout_result_fae3eee.claim_reasons_correction.json"

# The only fields of the claim assessment this correction may change.
REGENERATED_FIELDS = (
    "confirmatory_disqualifiers",
    "interpretation",
    "split_identity",
    "split_matches_frozen",
    "split_mismatches",
)
# Fields the corrected assessment adds; the original had no such key.
ADDED_FIELDS = ("validation_selection",)

DEFECTS = (
    {
        "id": "selection-history-rendered-as-fallback",
        "original_reason_prefix": "validation threshold selection used the fallback objective",
        "description": (
            "assess_claim mapped every validation_selection_confirmatory=False to "
            "the fallback disqualifier. The frozen configuration was selected by "
            "the post-D-03 validation sweep (102 of 640 eligible), not by the "
            "fallback objective."
        ),
    },
    {
        "id": "verified-split-not-passed",
        "original_reason_prefix": "the frozen split was not verified",
        "description": (
            "run_frozen_heldout.py verified the held-out split by passage "
            "identity but did not pass the expected IDs to assess_claim, which "
            "therefore reported the split as unverified."
        ),
    },
)
UNCHANGED_STATEMENT = (
    "This artifact changes no scores, predictions, bootstrap samples or metrics. "
    "No detector was run, no model was loaded and nothing was resampled: the "
    "saved bootstrap record, run configuration and split were reused verbatim, "
    "and only the claim-assessment reasons were regenerated. The canonical "
    "result artifact is unmodified."
)


class CorrectionRefused(RuntimeError):
    """The correction would change more than the reasons. Nothing written."""


def split_metadata_from_result(result):
    split = result["split"]
    return {
        "validation_fraction": split["validation_fraction"],
        "random_state": split["split_seed"],
        "validation_passage_ids": split["validation_passage_ids"],
        "test_passage_ids": split["observed_held_out_passage_ids"],
    }


def build_correction(result, *, result_path, result_sha256,
                     expected_validation_ids, expected_test_ids, repository_commit):
    """The correction record, or CorrectionRefused. Pure: no I/O."""
    original = result["claim_assessment"]
    corrected = assess_claim(
        result["bootstrap"],
        validation_selection=VALIDATION_SELECTION_POST_D03_SWEEP,
        quality_tolerance=original["validation_quality_tolerance"],
        run_configuration=original["run_configuration"],
        split_metadata=split_metadata_from_result(result),
        expected_validation_passage_ids=expected_validation_ids,
        expected_test_passage_ids=expected_test_ids,
    )

    problems = []
    unexpected = set(corrected) - set(original) - set(ADDED_FIELDS)
    missing = set(original) - set(corrected)
    if unexpected or missing:
        problems.append(f"field sets differ: added {sorted(unexpected)}, missing {sorted(missing)}")
    identical = []
    for field in sorted(set(original) & set(corrected)):
        if field in REGENERATED_FIELDS:
            continue
        if json.dumps(original[field], sort_keys=True) != json.dumps(corrected[field], sort_keys=True):
            problems.append(
                f"{field} would change: {original[field]!r} -> {corrected[field]!r}"
            )
        else:
            identical.append(field)
    for name, record in (("original", original), ("corrected", corrected)):
        if record["claim_status"] != CLAIM_NOT_CONFIRMATORY:
            problems.append(f"{name} claim_status is {record['claim_status']!r}")
    if corrected["split_mismatches"]:
        problems.append(f"the split does not verify: {corrected['split_mismatches']}")
    for defect in DEFECTS:
        if not any(r.startswith(defect["original_reason_prefix"])
                   for r in original["confirmatory_disqualifiers"]):
            problems.append(f"original does not carry defect {defect['id']}")
        if any(r.startswith(defect["original_reason_prefix"])
               for r in corrected["confirmatory_disqualifiers"]):
            problems.append(f"corrected still carries defect {defect['id']}")
    if problems:
        raise CorrectionRefused("\n  ".join(["correction refused:"] + problems))

    return {
        "artifact": "frozen-heldout-claim-reasons-correction",
        "corrects": {"path": str(result_path), "sha256": result_sha256},
        "canonical_artifact_modified": False,
        "statement": UNCHANGED_STATEMENT,
        "claim_status": {
            "original": original["claim_status"],
            "corrected": corrected["claim_status"],
            "unchanged": True,
        },
        "defects_corrected": list(DEFECTS),
        "original": {
            "confirmatory_disqualifiers": original["confirmatory_disqualifiers"],
            "interpretation": original["interpretation"],
            "split_matches_frozen": original["split_matches_frozen"],
            "split_mismatches": original["split_mismatches"],
        },
        "corrected": {
            "validation_selection": corrected["validation_selection"],
            "confirmatory_disqualifiers": corrected["confirmatory_disqualifiers"],
            "interpretation": corrected["interpretation"],
            "split_identity": corrected["split_identity"],
            "split_matches_frozen": corrected["split_matches_frozen"],
            "split_mismatches": corrected["split_mismatches"],
        },
        "regenerated_fields": list(REGENERATED_FIELDS),
        "added_fields": list(ADDED_FIELDS),
        "fields_verified_identical": identical,
        "corrected_claim_assessment": corrected,
        "generator": {
            "script": "scripts/correct_frozen_heldout_claim_reasons.py",
            "repository_commit": repository_commit,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts-dir", default=str(frozen.ARTIFACTS))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--output", default=None,
                        help=f"default: <artifacts-dir>/{CORRECTION_NAME}")
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = Path(args.artifacts_dir).expanduser()
    result_path = artifacts / CANONICAL_RESULT_NAME
    output = Path(args.output).expanduser() if args.output else artifacts / CORRECTION_NAME
    if output.resolve() == result_path.resolve():
        raise CorrectionRefused("refusing to write over the canonical result artifact")
    if output.exists():
        raise CorrectionRefused(f"{output} exists; a correction is written once")

    before = sha256_file(result_path)
    if before != CANONICAL_RESULT_SHA256:
        raise CorrectionRefused(
            f"canonical result digest is {before}, expected {CANONICAL_RESULT_SHA256}"
        )
    result = json.loads(result_path.read_text())

    # The expected split, re-derived from the released records at the frozen
    # fraction and seed -- arithmetic over passage indices, no scoring.
    _, split = frozen.frozen_split(args.data_root)
    for field in ("validation_passage_ids_sha256", "held_out_passage_ids_sha256"):
        if split[field] != result["split"][field]:
            raise CorrectionRefused(f"re-derived split {field} differs from the result")

    record = build_correction(
        result,
        result_path=result_path,
        result_sha256=before,
        expected_validation_ids=split["validation_passage_ids"],
        expected_test_ids=split["held_out_passage_ids"],
        repository_commit=frozen._git_commit(),
    )
    if sha256_file(result_path) != before:
        raise CorrectionRefused("the canonical result changed during the correction")

    output.write_text(json.dumps(record, indent=1) + "\n")
    print(f"canonical result  {result_path}\n  sha256          {before}  (unchanged)")
    print(f"claim status      {record['claim_status']['original']} -> "
          f"{record['claim_status']['corrected']}")
    print("original reasons:")
    for reason in record["original"]["confirmatory_disqualifiers"]:
        print(f"  - {reason}")
    print("corrected reasons:")
    for reason in record["corrected"]["confirmatory_disqualifiers"]:
        print(f"  - {reason}")
    print(f"fields verified identical: {len(record['fields_verified_identical'])}")
    print(f"wrote {output}\n  sha256 {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CorrectionRefused, frozen.Aborted) as exc:
        print(f"\nREFUSED: {exc}\nNothing was written.")
        sys.exit(1)
