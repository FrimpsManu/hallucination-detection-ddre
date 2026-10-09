#!/usr/bin/env python3
"""Build the full-coverage held-out NLI cache for the preregistered ablation.

Implements ``docs/ablation_preregistration.md`` §5 and nothing else. It copies
the canonical frozen-run held-out cache to a NEW derived cache and scores every
span of every document in positions 1..10 of every held-out subclaim that the
source does not already hold, with the frozen run's exact scoring setup.

Order, each step fail-closed:

1. the source cache and the canonical result (the environment reference) must
   match their pinned SHA-256 digests;
2. the held-out split is re-derived and must match the frozen run by passage
   identity;
3. the full coverage set is enumerated (no detector involved);
   ``--dry-run`` stops here, having loaded no model;
4. the model and tokenizer are loaded at the pinned revision, on MPS, and the
   live environment must match the frozen run's (revision, commit hashes,
   float16, eval mode, tokenizer, score version, torch/transformers/tokenizers);
5. the existing 398-pair exact score-compatibility probe runs against the
   source, writing nothing; if it fails, the run stops with nothing written;
6. the source is copied, the copy must be faithful and bound to the digest the
   probe certified;
7. only the missing pairs are scored, at batch size 1, into the derived cache;
8. coverage (by key presence), preservation of every source row, the source
   digest and the row arithmetic are verified; a complete cache is made
   read-only and hashed, and the manifest is written.

It never runs a detector, computes a metric, or reads the value of a newly
scored row.
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_frozen_heldout as frozen  # noqa: E402
from src import ablation_cache as AC  # noqa: E402
from src.cache_completion import sha256_file  # noqa: E402

SOURCE_CACHE_NAME = "frozen_heldout_nli_cache.sqlite"
SOURCE_CACHE_SHA256 = "dce6c49f813cb0ebecaf15d813e35ef940d259f97ad246d42ab8208c7a85be25"
RESULT_NAME = "frozen_heldout_result_fae3eee.json"
RESULT_SHA256 = "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67"
REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
DERIVED_CACHE_NAME = "ablation_heldout_full_cache.sqlite"
MANIFEST_NAME = "ablation_heldout_full_cache_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts-dir", default=str(frozen.ARTIFACTS))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify inputs and report the coverage counts. Loads no model, writes nothing.",
    )
    return parser.parse_args()


def paths(artifacts):
    return {
        "source": artifacts / SOURCE_CACHE_NAME,
        "result": artifacts / RESULT_NAME,
        "derived": artifacts / DERIVED_CACHE_NAME,
        "manifest": artifacts / MANIFEST_NAME,
    }


def verify_inputs(p):
    problems = []
    for name, want in (("source", SOURCE_CACHE_SHA256), ("result", RESULT_SHA256)):
        got = sha256_file(p[name]) if p[name].exists() else None
        if got != want:
            problems.append(f"{name}: sha256 {got}, expected {want}")
    for name in ("derived", "manifest"):
        if p[name].exists():
            problems.append(f"{name} already exists at {p[name]}; completion runs once")
    if problems:
        raise AC.CompletionAborted("inputs:\n  " + "\n  ".join(problems))
    return json.loads(p["result"].read_text())


def held_out_records(data_root, result):
    records, split = frozen.frozen_split(data_root)
    for field in ("held_out_passage_ids_sha256", "validation_passage_ids_sha256"):
        if split[field] != result["split"][field]:
            raise AC.CompletionAborted(f"re-derived split {field} differs from the frozen run")
    if len(records) != result["split"]["held_out_sentences"]:
        raise AC.CompletionAborted("held-out sentence count differs from the frozen run")
    return records, split


def load_model(model_name):
    """The frozen runner's load, verbatim: pinned revision, default dtype, MPS."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import select_device

    device = select_device(torch)
    if device != "mps":
        raise AC.CompletionAborted(f"selected device is {device!r}; the frozen run used 'mps'")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=REVISION
    ).to(torch.device(device))
    model.eval()
    return tokenizer, model, device


def compatibility_probe(data_root, tokenizer, model, model_name, score_version, source):
    """The existing 398-pair probe. Reads the source read-only; writes only scratch."""
    from src.score_compatibility import (
        PROBE_BATCH_SIZE,
        run_compatibility_probe,
        scratch_scorer,
    )
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(data_root, per_class=None)
    pairs = [(x["premise"], x["hypothesis"]) for x in positive + negative]
    polarities = ["positive"] * len(positive) + ["negative"] * len(negative)
    with tempfile.TemporaryDirectory(prefix="ablation-compat-") as scratch:
        probe = scratch_scorer(tokenizer, model, model_name,
                               str(Path(scratch) / "scratch.sqlite"),
                               batch_size=PROBE_BATCH_SIZE)
        closed = {"done": False}

        def finalize():
            rows = probe.cache_size()
            probe.close()
            closed["done"] = True
            return rows

        try:
            return run_compatibility_probe(
                pairs=pairs, polarities=polarities,
                positive_pairs=len(positive), negative_pairs=len(negative),
                source_cache=str(source), model_name=model_name,
                score_version=score_version, scorer=probe, finalize=finalize,
                source_digest=lambda: sha256_file(source),
            )
        finally:
            if not closed["done"]:
                probe.close()


def main():
    args = parse_args()
    p = paths(Path(args.artifacts_dir).expanduser())
    result = verify_inputs(p)
    records, split = held_out_records(args.data_root, result)

    from src.utils import SCORE_VERSION, split_text

    model_name = frozen.OFFICIAL_MODEL
    coverage = AC.enumerate_coverage(
        records, split_text, model_name=model_name, score_version=SCORE_VERSION,
    )
    present = AC.present_keys(p["source"], coverage["keys"])
    counts = dict(coverage["counts"], coverage_sha256=coverage["coverage_sha256"],
                  available_in_source=len(present),
                  missing_from_source=len(coverage["keys"]) - len(present))
    print("held-out coverage (positions 1..10, every span):")
    for key, value in counts.items():
        print(f"  {key:26s} {value}")
    if args.dry_run:
        print("\n--dry-run: inputs verified. No model loaded, nothing written.")
        return 0

    tokenizer, model, device = load_model(model_name)
    from src.diagnostic_probe import collect_live_environment

    observed = collect_live_environment(
        model_name=model_name, model=model, tokenizer=tokenizer,
        repo_root=PROJECT_ROOT, score_version=SCORE_VERSION, selected_device=device,
    )
    expectations = AC.environment_expectations(
        result["environment"], revision=REVISION, score_version=SCORE_VERSION,
    )
    problems = AC.check_environment(observed, expectations)
    if problems:
        raise AC.CompletionAborted("environment differs from the frozen run:\n  "
                                   + "\n  ".join(problems))

    from src.utils import EntailmentScorer

    manifest = AC.complete_cache(
        source=p["source"],
        destination=p["derived"],
        source_sha256=SOURCE_CACHE_SHA256,
        coverage=coverage,
        run_probe=lambda: compatibility_probe(
            args.data_root, tokenizer, model, model_name, SCORE_VERSION, p["source"]
        ),
        open_scorer=lambda path: EntailmentScorer(
            tokenizer, model, model_name, cache_path=path, batch_size=AC.BATCH_SIZE
        ),
        environment={
            "model_name": model_name,
            "checkpoint_revision": REVISION,
            "checkpoint_identity": observed.get("checkpoint_identity"),
            "tokenizer": observed.get("tokenizer"),
            "model": observed.get("model"),
            "device": device,
            "dtype": (observed.get("model") or {}).get("dtype"),
            "batch_size": AC.BATCH_SIZE,
            "score_version": SCORE_VERSION,
            "segmentation": {"segment_length": AC.SEGMENT_LENGTH,
                             "overlap_length": AC.OVERLAP_LENGTH},
            "document_score": "max over spans",
            "libraries": observed.get("libraries"),
            "environment_checks": sorted(expectations),
        },
    )
    manifest.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": frozen._git_commit(),
        "canonical_result_sha256": RESULT_SHA256,
        "held_out_passage_ids_sha256": split["held_out_passage_ids_sha256"],
    })
    p["manifest"].write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(f"\nstatus {manifest['status']}; added {manifest['added_rows']} rows; "
          f"missing after {manifest['coverage']['missing_after']}")
    print(f"derived cache {p['derived']}\n  sha256 {manifest['derived_sha256']}")
    print(f"manifest {p['manifest']}\n  sha256 {sha256_file(p['manifest'])}")
    return 0 if manifest["status"] == AC.STATUS_COMPLETE else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AC.CompletionAborted, frozen.Aborted) as exc:
        print(f"\nABORTED: {exc}")
        sys.exit(1)
