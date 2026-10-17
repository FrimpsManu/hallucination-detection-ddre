#!/usr/bin/env python3
"""Score every (span, sentence) pair the HELM test needs, once
(docs/helm_preregistration.md §3, §8 step 3).

One float16 NLI pass with ``FeatureScorer`` over every span of the top-10
retrieved passages of every HELM unit, plus Wang's 398 NBC pairs (which the
BSE histogram needs). No detector is run.

Checks, all of which must pass:

1. **Gate P:** the HELM preregistration, the estimators, the multivariate
   feature cache and the HELM evidence file (with its manifest) match their
   pins; outputs do not exist yet.
2. **Gate G0:** the loader reproduces the preregistered totals, and the
   evidence file holds exactly one list per unit, in loader order.
3. **Gate F, before any bulk scoring:** the NBC pairs' entailment equals the
   float16 reference caches bit for bit.
4. **After scoring:** every pair also in the multivariate study's feature
   cache equals it on every field, and every pair in a float16 reference
   cache equals it on entailment.

``--dry-run`` runs P and G0 and enumerates pairs; no model is loaded and
nothing is written. The cache and manifest are written once, read-only.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import complete_ablation_heldout_cache as CA  # noqa: E402
import run_frozen_heldout as frozen  # noqa: E402
import score_confirmatory_features as SC  # noqa: E402
import score_mddre_features as SF  # noqa: E402
from src import ablation_cache as AC  # noqa: E402
from src import helm_data as H  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

DTYPE = "float16"  # preregistration §3
MAX_DOCS = 10
EVIDENCE = (H.ROOT / "helm_evidence_top10.jsonl",
            "c86f405a331788c160e02f7448b2543e58144fd1cb9f810dd59a07cc7a3fb9c0")
EVIDENCE_MANIFEST = H.ROOT / "helm_evidence_manifest.json"


class HelmScoringRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(H.ROOT))
    parser.add_argument("--dry-run", action="store_true",
                        help="Gates P and G0 and pair enumeration only. Nothing written.")
    return parser.parse_args()


def output_paths(out_dir):
    return {"cache": out_dir / f"helm_features_{DTYPE}.sqlite",
            "manifest": out_dir / f"helm_features_{DTYPE}_manifest.json"}


def verify_inputs(out, dry_run):
    problems = list(H.prereg_problems(PROJECT_ROOT))
    for path, sha in (SC.ESTIMATORS, SC.MDDRE_FEATURES, EVIDENCE):
        if not path.exists():
            problems.append(f"{path} is missing")
        elif sha256_path(path) != sha:
            problems.append(f"{path.name} differs from its pinned digest")
    manifest = json.loads(EVIDENCE_MANIFEST.read_text()) if EVIDENCE_MANIFEST.exists() else {}
    if manifest.get("evidence_sha256") != EVIDENCE[1] or manifest.get("labels_recorded"):
        problems.append("the evidence manifest does not certify the pinned evidence")
    digests = {}
    for ref, (path, pinned) in SF.FLOAT16_REFERENCES.items():
        digests[ref] = sha256_path(path) if path.exists() else None
        if digests[ref] != pinned:
            problems.append(f"reference {ref} differs from its pinned digest")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; scoring runs once")
    if problems:
        raise HelmScoringRefused("inputs:\n  " + "\n  ".join(problems))
    return digests, manifest


def evidence_records(units, evidence_path):
    """One record per unit (loader order) with its sentence as the only subclaim."""
    rows = [json.loads(line) for line in open(evidence_path, encoding="utf-8")]
    if [r["id"] for r in rows] != [u.id for u in units]:
        raise HelmScoringRefused("gate G0: the evidence does not hold one list per unit "
                                 "in loader order")
    for r, u in zip(rows, units):
        if r["sentence"] != u.sentence or r["title"] != u.title:
            raise HelmScoringRefused(f"gate G0: evidence for {u.id} does not match the unit")
    return [SimpleNamespace(subclaims=[SimpleNamespace(
        text=u.sentence,
        documents=[SimpleNamespace(page_content=d["text"]) for d in r["documents"]])])
        for r, u in zip(rows, units)]


def enumerate_pairs(units):
    from src.utils import SCORE_VERSION, split_text

    records = evidence_records(units, EVIDENCE[0])
    return AC.enumerate_coverage(records, split_text, model_name=frozen.OFFICIAL_MODEL,
                                 score_version=SCORE_VERSION, max_docs=MAX_DOCS)


def main():
    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser())
    digests, evidence_manifest = verify_inputs(out, args.dry_run)
    units, titles = H.load_helm()
    g0 = H.check_expected(units, titles)
    coverage = enumerate_pairs(units)
    print(f"gate G0: {g0}\n  coverage {coverage['counts']}")
    if args.dry_run:
        print("--dry-run: gates P and G0 passed, pairs enumerated. No model, nothing written.")
        return 0

    from tqdm import tqdm

    from src.feature_scorer import (FEATURE_VERSION, FeatureScorer, entailment_mismatches,
                                    feature_key)
    from src.utils import SCORE_VERSION
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.wang_data_root, per_class=None)
    nbc = [(x["premise"], x["hypothesis"]) for x in positive + negative]
    ref_paths = [path for path, _ in SF.FLOAT16_REFERENCES.values()]
    tokenizer, model, device, loaded = SF.load_model(DTYPE)
    scorer = FeatureScorer(tokenizer, model, frozen.OFFICIAL_MODEL, dtype=loaded,
                           cache_path=out["cache"], batch_size=1)
    keys, pairs = coverage["keys"], coverage["pairs"]
    status, mismatches = "complete", {}
    try:
        nbc_keys = [AC.production_cache_key(frozen.OFFICIAL_MODEL, SCORE_VERSION, p, h)
                    for p, h in nbc]
        nbc_ref = SF.reference_scores(ref_paths, nbc_keys)
        nbc_rows = dict(zip(nbc_keys, scorer.score_pairs(nbc)))
        mismatches["nbc"] = entailment_mismatches(nbc_rows, nbc_ref)
        if len(nbc_ref) != len(set(nbc_keys)) or mismatches["nbc"]:
            raise HelmScoringRefused(
                f"gate F: {len(nbc_ref)} reference rows, {len(mismatches['nbc'])} "
                "mismatches. No bulk scoring was done.")
        print(f"gate F: {len(nbc_ref)} / {len(nbc_ref)} NBC pairs exact", flush=True)

        with tqdm(total=len(pairs), desc="feature scoring", unit="pair") as bar:
            for start in range(0, len(pairs), 1000):
                scorer.score_pairs(pairs[start:start + 1000], progress=bar.update)

        rows = dict(zip(keys, scorer.score_pairs(pairs)))
        ref = SF.reference_scores(ref_paths, keys)
        mismatches["reference"] = entailment_mismatches({k: rows[k] for k in ref}, ref)
        by_feature_key = {feature_key(frozen.OFFICIAL_MODEL, loaded, p, h): rows[k]
                          for k, (p, h) in zip(keys, pairs)}
        checked, mismatches["feature_cache"] = SC.feature_cache_mismatches(by_feature_key)
        if mismatches["reference"] or mismatches["feature_cache"]:
            status = "failed_equality"
        cache_rows = scorer.cache_size()
    finally:
        scorer.close()

    for name, (path, _) in SF.FLOAT16_REFERENCES.items():
        if sha256_path(path) != digests[name]:
            raise HelmScoringRefused(f"reference {name} changed during scoring")
    out["cache"].chmod(0o444)
    manifest = {
        "artifact": "helm-nli-features",
        "protocol": H.PREREG_PATH + " §3, §8 step 3",
        "preregistration_sha256": H.PREREG_SHA256,
        "status": status,
        "dtype": loaded,
        "device": device,
        "batch_size": 1,
        "model_name": frozen.OFFICIAL_MODEL,
        "checkpoint_revision": CA.REVISION,
        "feature_version": FEATURE_VERSION,
        "gate_G0": g0,
        "coverage": dict(coverage["counts"], coverage_sha256=coverage["coverage_sha256"]),
        "evidence_sha256": EVIDENCE[1],
        "unique_pairs": len(pairs),
        "nbc_pairs": len(nbc),
        "cache_rows": cache_rows,
        "equality": {
            "nbc_exact": len(nbc_ref) - len(mismatches["nbc"]),
            "reference_pairs_checked": len(ref),
            "reference_mismatches": len(mismatches["reference"]),
            "feature_cache_pairs_checked": checked,
            "feature_cache_mismatches": len(mismatches["feature_cache"]),
        },
        "detectors_run": False,
        "cache_sha256": sha256_path(out["cache"]),
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["manifest"].write_text(json.dumps(manifest, indent=1) + "\n")
    out["manifest"].chmod(0o444)
    print(f"status {status}; cache {out['cache']}\n  sha256 {manifest['cache_sha256']}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HelmScoringRefused, H.HelmDataRefused, AC.CompletionAborted) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
