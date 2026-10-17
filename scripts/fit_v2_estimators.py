#!/usr/bin/env python3
"""Fit every plan-v2 estimator and its Eq. 8 look-ahead support, once.

Research plan v2 §3.3-§3.4, Phase 3. No test set is read.

Tier 1, Wang's setting (Wang's 199 + 199 NBC pairs, feature e):
- BSE: Wang's histogram, built exactly as BSE builds it;
- DDRE-1D: self-normalised uLSIF on e, CV folds grouped by NBC hypothesis.

Tier 2, retriever-aware (FActScore training entities, BM25 Wikipedia, ranks
1-10; CV folds grouped by entity):
- Hist-1D: BSE's histogram of e;
- DDRE-1D': self-normalised uLSIF on e;
- NB-M: Naive-Bayes histograms on (e, k, rank), normalised;
- DDRE-M: self-normalised uLSIF on (e, k, log rank);
- Logit-M: logistic ratio on (e, k, log rank), normalised.

Hyperparameters come only from each estimator's own cross-validation
(sigma grid ``SIGMA_MULTIPLIERS_V2``). Each estimator's look-ahead support is
built from its own hallucinated training documents: histogram bins for the
1-D histograms, otherwise the estimator's log ratios on those documents,
per rank for estimators that read rank.

Gate X1: the BSE histogram rebuilt from the feature cache must equal the
Laplace-smoothed histogram recorded by the Gate 1 reproduction.

Inputs are read-only. The result is written once and made read-only.
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
import run_mddre_replay as RR  # noqa: E402
import score_mddre_features as SF  # noqa: E402
from src import mddre_data as M  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src import sequential as S  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

DTYPE = "float16"
PLAN = ("docs/research_plan_v2.md",
        "29a80fa2fcb1e4afd2a7a60d582d6f611ab6fdff3181b8335c82dfa43c1df98c")
FEATURES = (SF.MDDRE / "mddre_features_float16.sqlite",
            "47a0425fcd90944f47331c2dad8900fe8c05d8bd03eb4d4ca0f5d70b7e097353")
GATE1 = (SF.GATE1 / "wang_reproduction_fidelity_v2_batch1_mps.json",
         "38a4a1521f89b94a2d16d2d21c2d8b18f236f86fba9ca7478612149d171a8f74")
OUT_DIR = Path("~/ddre-artifacts/v2").expanduser()
OUT_NAME = f"estimators_v2_{DTYPE}.json"
E_GRID = tuple(range(0, 101, 5))


class FitRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def verify_inputs(out):
    problems = []
    if sha256_path(PROJECT_ROOT / PLAN[0]) != PLAN[1]:
        problems.append(f"{PLAN[0]} differs from the approved plan")
    for path, sha in (FEATURES, GATE1):
        if sha256_path(path) != sha:
            problems.append(f"{path.name} differs from its pinned digest")
    manifest = json.loads(SF.output_paths(SF.MDDRE, DTYPE)["manifest"].read_text())
    if manifest.get("status") != "complete" or manifest.get("cache_sha256") != FEATURES[1]:
        problems.append("the feature manifest does not certify the pinned cache")
    if out.exists():
        problems.append(f"{out} already exists; the fit runs once")
    if problems:
        raise FitRefused("inputs:\n  " + "\n  ".join(problems))
    return manifest


def gate_bse_histogram(bse):
    recorded = json.loads(GATE1[0].read_text())["nbc_histograms_laplace_smoothed"]
    if [bse.pos_hist, bse.neg_hist] != [recorded["positive"], recorded["negative"]]:
        raise FitRefused(f"gate X1: BSE histogram {bse.pos_hist} / {bse.neg_hist} differs "
                         f"from Gate 1's {recorded['positive']} / {recorded['negative']}")
    return {"positive": bse.pos_hist, "negative": bse.neg_hist, "passed": True}


def nbc_documents(reader, wang_root):
    """Wang's NBC pairs as one-document features (whole premise, rank 1)."""
    from src.wang_data import load_nbc_pairs

    out = {}
    for name, pairs in zip(("factual", "hallucinated"),
                           load_nbc_pairs(wang_root, per_class=None)):
        rows = [reader.pair(x["premise"], x["hypothesis"]) for x in pairs]
        out[name] = ([{"e": r["entailment"], "k": r["contradiction"], "rank": 1}
                      for r in rows], [x["hypothesis"] for x in pairs])
    return out


def training_documents(reader, training, evidence):
    """FActScore training documents with their entity, as the multivariate fit."""
    docs = {"factual": ([], []), "hallucinated": ([], [])}
    for s in training:
        for i, fact in enumerate(s.facts):
            texts = evidence[(s.generator, s.response_index, s.sentence_index, i)]
            side = docs["factual" if fact.label == "S" else "hallucinated"]
            for rank, text in enumerate(texts, start=1):
                side[0].append(reader.document(fact.text, text, rank))
                side[1].append(s.entity)
    return docs


def fit_tier(docs, *, multivariate):
    (f, fg), (h, hg) = docs["factual"], docs["hallucinated"]
    kw = {"factual_groups": fg, "hallucinated_groups": hg}
    fitted = {}
    if multivariate:
        fitted["Hist-1D"] = E.HistogramEvidence.fit([d["e"] for d in f], [d["e"] for d in h])
        fitted["DDRE-1D'"] = E.MultiULSIF(("e",)).fit(
            f, h, sigma_multipliers=E.SIGMA_MULTIPLIERS_V2, **kw)
        fitted["NB-M"] = E.NaiveBayesEvidence(E.NB_FEATURES).fit(f, h, normalise=True)
        fitted["DDRE-M"] = E.MultiULSIF(E.FEATURE_SETS["M"]).fit(
            f, h, sigma_multipliers=E.SIGMA_MULTIPLIERS_V2, **kw)
        fitted["Logit-M"] = E.LogisticRatio(E.FEATURE_SETS["M"]).fit(f, h, normalise=True, **kw)
    else:
        fitted["DDRE-1D"] = E.MultiULSIF(("e",)).fit(
            f, h, sigma_multipliers=E.SIGMA_MULTIPLIERS_V2, **kw)
    supports = {}
    for name, est in fitted.items():
        if isinstance(est, E.HistogramEvidence):
            supports[name] = S.Support.from_histogram(est.pos_hist, est.neg_hist)
        else:
            supports[name] = S.build_support(est, h)
    return fitted, supports


def support_summary(support):
    if isinstance(support, S.RankedSupport):
        return {"ranked": True, "atoms_by_rank": {str(r): int(s.log_ratios.size)
                                                  for r, s in support.by_rank.items()},
                "mean_ratio_by_rank": {str(r): s.mean_ratio
                                       for r, s in support.by_rank.items()}}
    return {"ranked": False, "atoms": int(support.log_ratios.size),
            "mean_ratio": support.mean_ratio}


def curve(estimator, k, rank):
    return {str(e): estimator.log_ratio({"e": float(e), "k": k, "rank": rank}) for e in E_GRID}


def main():
    from src.mddre_features import FeatureReader

    args = parse_args()
    out = Path(args.out_dir).expanduser() / OUT_NAME
    manifest = verify_inputs(out)
    training, _, split = M.factscore_split(SF.FACTSCORE / "data")
    evidence = SF.evidence_rows(SF.FACTSCORE / SF.FACTSCORE_EVIDENCE[0])
    evidence.update(SF.evidence_rows(SF.MDDRE / SF.TRAINING_EVIDENCE))
    reader = FeatureReader(FEATURES[0], frozen.OFFICIAL_MODEL, manifest["dtype"])
    try:
        bse = RR.nbc_histogram(reader, args.wang_data_root)
        x1 = gate_bse_histogram(bse)
        print(f"gate X1 (BSE histogram): {x1}")
        nbc = nbc_documents(reader, args.wang_data_root)
        tier2_docs = training_documents(reader, training, evidence)
    finally:
        reader.close()

    tier1, supports1 = fit_tier(nbc, multivariate=False)
    tier1 = {"BSE": bse, **tier1}
    supports1 = {"BSE": S.Support.from_histogram(bse.pos_hist, bse.neg_hist), **supports1}
    tier2, supports2 = fit_tier(tier2_docs, multivariate=True)

    all_h = tier2_docs["factual"][0] + tier2_docs["hallucinated"][0]
    k_med = float(sorted(d["k"] for d in all_h)[len(all_h) // 2])
    result = {
        "artifact": "v2-estimators",
        "protocol": PLAN[0] + " §3.3-§3.4 (Phase 3)",
        "plan_sha256": PLAN[1],
        "dtype": manifest["dtype"],
        "feature_cache_sha256": FEATURES[1],
        "gates": {"X1_bse_histogram": x1},
        "tier1": {
            "training": {"source": "Wang NBC pairs (whole premise, as Wang)",
                         "factual": len(nbc["factual"][0]),
                         "hallucinated": len(nbc["hallucinated"][0]),
                         "cv_groups": "NBC hypothesis text"},
            "estimators": {n: e.to_json() for n, e in tier1.items()},
            "supports": {n: S.support_to_json(s) for n, s in supports1.items()},
            "support_summary": {n: support_summary(s) for n, s in supports1.items()},
            "log_ratio_curves": {n: curve(e, 0.0, 1) for n, e in tier1.items()},
        },
        "tier2": {
            "training": {"source": "FActScore training entities, BM25 Wikipedia, ranks 1-10",
                         "split": split,
                         "factual": len(tier2_docs["factual"][0]),
                         "hallucinated": len(tier2_docs["hallucinated"][0]),
                         "cv_groups": "entity"},
            "estimators": {n: e.to_json() for n, e in tier2.items()},
            "supports": {n: S.support_to_json(s) for n, s in supports2.items()},
            "support_summary": {n: support_summary(s) for n, s in supports2.items()},
            "log_ratio_curves": {"grid": "e = 0..100 step 5; k at its training median, rank 1",
                                 "k_median": k_med,
                                 **{n: curve(e, k_med, 1) for n, e in tier2.items()}},
        },
        "test_data_read": False,
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    out.chmod(0o444)
    for tier, fitted in (("tier1", tier1), ("tier2", tier2)):
        for name, est in fitted.items():
            diag = getattr(est, "diagnostics", None)
            extra = {k: getattr(est, k) for k in ("sigma", "lam") if hasattr(est, k)}
            print(f"  {tier} {name:9s} {extra} {diag}")
    print(f"written {out}\n  sha256 {sha256_path(out)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FitRefused, M.MDDREDataRefused, E.EstimatorError, S.SequentialError) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
