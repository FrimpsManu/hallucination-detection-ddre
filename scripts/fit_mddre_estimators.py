#!/usr/bin/env python3
"""Fit the multivariate DDRE estimators once (design §3, §8 step 4).

On the training set only (FActScore training entities, all three
generators, every retrieved document at ranks 1-10), fits:

- A': BSE's histogram of e on the new training set;
- U: normalised uLSIF on e;
- M: normalised uLSIF on (e, k, log rank), the proposed method;
- L: logistic-regression ratio on (e, k, log rank).

Each training document is labelled by its fact's S/NS label. Hyperparameters
come only from each estimator's own cross-validation on a grid fixed in
``src/mddre_estimators.py``. No test set is read. The fitted parameters and
the normalisation diagnostics are written once and made read-only.
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

import run_frozen_heldout as frozen  # noqa: E402
import score_mddre_features as SF  # noqa: E402
from src import mddre_data as M  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

E_GRID = tuple(range(0, 101, 5))


class FitRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dtype", choices=SF.DTYPES, required=True)
    parser.add_argument("--out-dir", default=str(SF.MDDRE))
    return parser.parse_args()


def verify_features(out_dir, dtype):
    paths = SF.output_paths(out_dir, dtype)
    manifest = json.loads(paths["manifest"].read_text())
    problems = []
    if sha256_path(PROJECT_ROOT / M.DESIGN_PATH) != M.DESIGN_SHA256:
        problems.append("the design preregistration differs from its pinned digest")
    if manifest.get("status") != "complete":
        problems.append(f"feature cache status is {manifest.get('status')!r}")
    if sha256_path(paths["cache"]) != manifest.get("cache_sha256"):
        problems.append("the feature cache differs from its manifest digest")
    if manifest.get("equality", {}).get("overlap_mismatches") != 0:
        problems.append("the feature cache did not pass the equality check")
    out = out_dir / f"mddre_estimators_{dtype}.json"
    if out.exists():
        problems.append(f"{out} already exists; the fit runs once")
    if problems:
        raise FitRefused("inputs:\n  " + "\n  ".join(problems))
    return paths["cache"], manifest, out


def training_documents(reader, training, evidence):
    factual, hallucinated = [], []
    for s in training:
        for i, fact in enumerate(s.facts):
            texts = evidence[(s.generator, s.response_index, s.sentence_index, i)]
            for rank, text in enumerate(texts, start=1):
                doc = reader.document(fact.text, text, rank)
                (factual if fact.label == "S" else hallucinated).append(doc)
    return factual, hallucinated


def curve(estimator, k, rank):
    return {str(e): estimator.log_ratio({"e": float(e), "k": k, "rank": rank}) for e in E_GRID}


def main():
    from src.mddre_features import FeatureReader

    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    cache, manifest, out = verify_features(out_dir, args.dtype)
    training, _, split = M.factscore_split(SF.FACTSCORE / "data")
    evidence = SF.evidence_rows(SF.FACTSCORE / SF.FACTSCORE_EVIDENCE[0])
    evidence.update(SF.evidence_rows(SF.MDDRE / SF.TRAINING_EVIDENCE))
    reader = FeatureReader(cache, frozen.OFFICIAL_MODEL, manifest["dtype"])
    try:
        factual, hallucinated = training_documents(reader, training, evidence)
    finally:
        reader.close()
    print(f"training documents: {len(factual):,} factual-claim, "
          f"{len(hallucinated):,} hallucinated-claim")

    fitted = {
        "A_prime": E.HistogramEvidence.fit([d["e"] for d in factual],
                                           [d["e"] for d in hallucinated]),
        "U": E.MultiULSIF(E.FEATURE_SETS["U"]).fit(factual, hallucinated),
        "M": E.MultiULSIF(E.FEATURE_SETS["M"]).fit(factual, hallucinated),
        "L": E.LogisticRatio(E.FEATURE_SETS["M"]).fit(factual, hallucinated),
    }
    k_med = float(sorted(d["k"] for d in factual + hallucinated)[
        len(factual + hallucinated) // 2])
    result = {
        "artifact": "mddre-estimators",
        "protocol": M.DESIGN_PATH + " §3, §8 step 4",
        "design_sha256": M.DESIGN_SHA256,
        "dtype": manifest["dtype"],
        "feature_cache_sha256": manifest["cache_sha256"],
        "split": split,
        "training": {"sentences": len(training), "factual_documents": len(factual),
                     "hallucinated_documents": len(hallucinated)},
        "estimators": {name: est.to_json() for name, est in fitted.items()},
        "log_ratio_curves": {
            "grid": "e = 0..100 step 5; k at its training median, rank 1",
            "k_median": k_med,
            **{name: curve(est, k_med, 1) for name, est in fitted.items()},
        },
        "test_data_read": False,
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    out.chmod(0o444)
    for name, est in fitted.items():
        print(f"  {name:8s} {getattr(est, 'diagnostics', {})}")
    print(f"written {out}\n  sha256 {sha256_path(out)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FitRefused, M.MDDREDataRefused, E.EstimatorError) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
