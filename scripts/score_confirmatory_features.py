#!/usr/bin/env python3
"""Score every document the confirmatory test needs, once
(``docs/confirmatory_preregistration.md`` §3, §8 steps 2-3).

One float16 NLI pass with ``FeatureScorer`` over every (span, subclaim) pair of:

- ExpertQA: all labelled sentences with atomic claims, including Unsure (the
  §6 sensitivity analysis needs them), at most 5 passages per atomic claim;
- BingCheck: every claim, including "not supported" (§6), at most 3 pages;
- Wang's 398 NBC pairs, which cell A's histogram needs.

Checks, all of which must pass:

1. **Gate P:** the preregistration, design and estimator files match their pins.
2. **Gate G0:** both loaders reproduce the preregistered primary totals.
3. **Gate F, before any bulk scoring:** the 398 NBC pairs' entailment equals the
   float16 reference caches bit for bit.
4. **After scoring:** every pair that is also in the multivariate study's
   feature cache equals it on every field, and every pair in a float16
   reference cache equals it on entailment.

``--dry-run`` runs gates P and G0 and enumerates pairs; no model is loaded and
nothing is written. Inputs are read-only. The cache and manifest are written
once and made read-only. No detector is run.
"""

import argparse
import json
import sqlite3
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
import score_mddre_features as SF  # noqa: E402
from src import ablation_cache as AC  # noqa: E402
from src import confirmatory_data as C  # noqa: E402
from src import mddre_data as M  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

DTYPE = "float16"  # preregistration §3
ESTIMATORS = (SF.MDDRE / "mddre_estimators_float16.json",
              "486296a2717b95a4ba661f15fbff8286613f7eeba508dab804614fa98d624014")
MDDRE_FEATURES = (SF.MDDRE / "mddre_features_float16.sqlite",
                  "47a0425fcd90944f47331c2dad8900fe8c05d8bd03eb4d4ca0f5d70b7e097353")
# Supersets of the primary mapping, so the §6 sensitivity analyses are covered.
SCORING_MAPPINGS = {"expertqa": "unsure_hallucinated",
                    "bingcheck": "not_supported_hallucinated"}


class ConfirmatoryScoringRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(C.ARTIFACTS))
    parser.add_argument("--dry-run", action="store_true",
                        help="Gates P and G0 and pair enumeration only. Nothing written.")
    return parser.parse_args()


def output_paths(out_dir):
    return {"cache": out_dir / f"confirmatory_features_{DTYPE}.sqlite",
            "manifest": out_dir / f"confirmatory_features_{DTYPE}_manifest.json"}


def verify_inputs(out, dry_run):
    problems = list(C.prereg_problems(PROJECT_ROOT))
    problems.extend(M.design_problems(PROJECT_ROOT, require_amendment_1=True))
    for path, sha in (ESTIMATORS, MDDRE_FEATURES):
        if sha256_path(path) != sha:
            problems.append(f"{path.name} differs from its pinned digest")
    digests = {}
    for ref, (path, pinned) in SF.FLOAT16_REFERENCES.items():
        digests[ref] = sha256_path(path)
        if digests[ref] != pinned:
            problems.append(f"reference {ref}: sha256 {digests[ref]}, pinned {pinned}")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; scoring runs once")
    if problems:
        raise ConfirmatoryScoringRefused("inputs:\n  " + "\n  ".join(problems))
    return digests


def records(units):
    return [SimpleNamespace(subclaims=[
        SimpleNamespace(text=s.text, documents=[SimpleNamespace(page_content=d)
                                                for d in s.documents])
        for s in u.subclaims]) for u in units]


def enumerate_pairs():
    from src.utils import SCORE_VERSION, split_text

    gate_g0, coverage = {}, {}
    for name, loader in C.LOADERS.items():
        gate_g0[name] = C.check_expected(name, loader())
        units = loader(mapping=SCORING_MAPPINGS[name])
        coverage[name] = AC.enumerate_coverage(
            records(units), split_text, model_name=frozen.OFFICIAL_MODEL,
            score_version=SCORE_VERSION, max_docs=C.MAX_DOCS[name])
        coverage[name]["counts"]["scoring_mapping"] = SCORING_MAPPINGS[name]
    return gate_g0, coverage


def feature_cache_mismatches(rows_by_feature_key):
    """Pairs also in the multivariate study's feature cache that differ on any field."""
    from src.feature_scorer import FIELDS

    con = sqlite3.connect(f"file:{MDDRE_FEATURES[0]}?mode=ro", uri=True)
    keys, checked, bad = list(rows_by_feature_key), 0, []
    try:
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            found = con.execute(
                f"SELECT cache_key, {', '.join(FIELDS)} FROM nli_features WHERE cache_key IN "
                f"({','.join('?' for _ in chunk)})", chunk).fetchall()
            for r in found:
                checked += 1
                ref = dict(zip(FIELDS, map(float, r[1:])))
                if any(ref[f] != rows_by_feature_key[r[0]][f] for f in FIELDS):
                    bad.append(r[0])
    finally:
        con.close()
    return checked, bad


def main():
    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser())
    digests = verify_inputs(out, args.dry_run)
    gate_g0, coverage = enumerate_pairs()

    all_pairs, all_keys, seen = [], [], set()
    for cov in coverage.values():
        for key, pair in zip(cov["keys"], cov["pairs"]):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)
                all_pairs.append(pair)
    for name in C.LOADERS:
        print(f"  {name:10s} G0 {gate_g0[name]}\n  {'':10s} scoring {coverage[name]['counts']}")
    print(f"  unique pairs overall  {len(all_pairs):,}")
    if args.dry_run:
        print("--dry-run: gates P and G0 passed, pairs enumerated. No model, nothing written.")
        return 0

    from tqdm import tqdm

    from src.feature_scorer import FEATURE_VERSION, FeatureScorer, entailment_mismatches, \
        feature_key
    from src.utils import SCORE_VERSION
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.wang_data_root, per_class=None)
    nbc = [(x["premise"], x["hypothesis"]) for x in positive + negative]
    ref_paths = [path for path, _ in SF.FLOAT16_REFERENCES.values()]
    tokenizer, model, device, loaded = SF.load_model(DTYPE)
    scorer = FeatureScorer(tokenizer, model, frozen.OFFICIAL_MODEL, dtype=loaded,
                           cache_path=out["cache"], batch_size=1)
    status, mismatches = "complete", {}
    try:
        # Gate F: NBC equality before any bulk scoring.
        nbc_keys = [AC.production_cache_key(frozen.OFFICIAL_MODEL, SCORE_VERSION, p, h)
                    for p, h in nbc]
        nbc_ref = SF.reference_scores(ref_paths, nbc_keys)
        nbc_rows = dict(zip(nbc_keys, scorer.score_pairs(nbc)))
        mismatches["nbc"] = entailment_mismatches(nbc_rows, nbc_ref)
        if len(nbc_ref) != len(set(nbc_keys)) or mismatches["nbc"]:
            raise ConfirmatoryScoringRefused(
                f"NBC equality failed: {len(nbc_ref)} reference rows, "
                f"{len(mismatches['nbc'])} mismatches. No bulk scoring was done.")
        print(f"NBC equality: {len(nbc_ref)} / {len(nbc_ref)} exact")

        with tqdm(total=len(all_pairs), desc="feature scoring", unit="pair") as bar:
            for start in range(0, len(all_pairs), 1000):
                scorer.score_pairs(all_pairs[start:start + 1000], progress=bar.update)

        rows = dict(zip(all_keys, scorer.score_pairs(all_pairs)))
        ref = SF.reference_scores(ref_paths, all_keys)
        mismatches["reference"] = entailment_mismatches({k: rows[k] for k in ref}, ref)
        by_feature_key = {feature_key(frozen.OFFICIAL_MODEL, loaded, p, h): rows[k]
                          for k, (p, h) in zip(all_keys, all_pairs)}
        feature_checked, mismatches["feature_cache"] = feature_cache_mismatches(by_feature_key)
        if mismatches["reference"] or mismatches["feature_cache"]:
            status = "failed_equality"
        cache_rows = scorer.cache_size()
    finally:
        scorer.close()

    for ref_name, (path, _) in SF.FLOAT16_REFERENCES.items():
        if sha256_path(path) != digests[ref_name]:
            raise ConfirmatoryScoringRefused(f"reference {ref_name} changed during scoring")
    out["cache"].chmod(0o444)
    manifest = {
        "artifact": "confirmatory-nli-features",
        "protocol": C.PREREG_PATH + " §3, §8 steps 2-3",
        "preregistration_sha256": C.PREREG_AMENDMENT_1_SHA256,
        "status": status,
        "dtype": loaded,
        "device": device,
        "batch_size": 1,
        "model_name": frozen.OFFICIAL_MODEL,
        "checkpoint_revision": CA.REVISION,
        "feature_version": FEATURE_VERSION,
        "gate_G0": gate_g0,
        "coverage": {name: dict(cov["counts"], coverage_sha256=cov["coverage_sha256"])
                     for name, cov in coverage.items()},
        "unique_pairs": len(all_pairs),
        "nbc_pairs": len(nbc),
        "cache_rows": cache_rows,
        "equality": {
            "nbc_exact": len(nbc_ref) - len(mismatches["nbc"]),
            "reference_pairs_checked": len(ref),
            "reference_mismatches": len(mismatches["reference"]),
            "feature_cache_pairs_checked": feature_checked,
            "feature_cache_mismatches": len(mismatches["feature_cache"]),
            "reference_caches": {k: {"path": str(p), "sha256": digests[k]}
                                 for k, (p, _) in SF.FLOAT16_REFERENCES.items()},
        },
        "inputs": {"expertqa_sha256": C.EXPERTQA_SHA256,
                   "bingcheck_listing_sha256": C.BINGCHECK_LISTING_SHA256},
        "detectors_run": False,
        "cache_sha256": sha256_path(out["cache"]),
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["manifest"].write_text(json.dumps(manifest, indent=1) + "\n")
    out["manifest"].chmod(0o444)
    print(f"status {status}; reference pairs {len(ref):,} ({len(mismatches['reference'])} "
          f"mismatches); feature-cache pairs {feature_checked:,} "
          f"({len(mismatches['feature_cache'])} mismatches)")
    print(f"cache {out['cache']}\n  sha256 {manifest['cache_sha256']}")
    print(f"manifest {out['manifest']}\n  sha256 {sha256_path(out['manifest'])}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConfirmatoryScoringRefused, C.ConfirmatoryDataRefused, AC.CompletionAborted) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
