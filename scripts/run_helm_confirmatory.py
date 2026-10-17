#!/usr/bin/env python3
"""The preregistered confirmatory test on HELM, once
(docs/helm_preregistration.md §3-§7, §8 step 4).

From the frozen HELM features: BSE, DDRE-1D, NB-M and DDRE-M under backward
induction over every (K, C_retrieve) point (K = 1..10, C_retrieve in
{0.125, 0.25, 0.5, 1}) at C_M/C_FA 28/96 and 14/24, plus Wang's fixed-k max
entailment for K = 1..10 and BSE-official at the operating points. Each
sentence is its own single subclaim. One paired cluster bootstrap over
articles (10,000 replicates, seed 42) gives every endpoint;
``src.helm_decision`` applies §5's rule to H1a, H1b, H2a and H2b. §6's
secondary and sensitivity analyses are descriptive.

Gates, any failure stops the run before anything is written:

- **P:** the HELM preregistration, plan v2 at amendment 2, the estimators,
  the HELM evidence and features (with manifests) and the SelfCheckGPT
  references match their pins; outputs do not exist yet.
- **G0:** the HELM loader reproduces the preregistered totals.
- **X2:** on SelfCheckGPT held-out, BSE-official reproduces the frozen
  replays row for row.
- **X4:** on SelfCheckGPT held-out, every method's (K = 10, C_retrieve = 1)
  point reproduces Phase 5's operating points.

Inputs are read-only; the result and predictions are written once, read-only.
"""

import argparse
import csv
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
import run_v2_budget as RB  # noqa: E402
import run_v2_replay as RV  # noqa: E402
import score_mddre_features as SF  # noqa: E402
from src import ablation_replay as AR  # noqa: E402
from src import helm_data as H  # noqa: E402
from src import helm_decision as D  # noqa: E402
from src import metrics_v2 as V  # noqa: E402
from src import plan_v2  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_budget as B  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

MAX_DOCS = 10
SIX_GENERATORS = tuple(g for g in H.GENERATORS if g != "mpt7b")
PINS = {
    "estimators": RV.PINS["estimators"],
    "helm_evidence": (H.ROOT / "helm_evidence_top10.jsonl",
                      "c86f405a331788c160e02f7448b2543e58144fd1cb9f810dd59a07cc7a3fb9c0"),
    "helm_features": (H.ROOT / "helm_features_float16.sqlite",
                      "a02ef22579c80dc661e0fef296a3ef4af695467e5cf548171e23a09c35d9a54d"),
    "mddre_features": RV.PINS["mddre_features"],
    "mddre_manifest": RV.PINS["mddre_manifest"],
    "mddre_predictions": RV.PINS["mddre_predictions"],
    "phase5_predictions": RB.PINS["phase5_predictions"],
}
HELM_FEATURES_MANIFEST = H.ROOT / "helm_features_float16_manifest.json"
N_RESAMPLES, SEED = 10_000, 42


class HelmConfirmatoryRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(H.ROOT))
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Gates P, G0, X2, X4 and loading every HELM document's features; "
                             "no detector is run on HELM and nothing is written.")
    return parser.parse_args()


def output_paths(out_dir):
    return {"result": out_dir / "helm_confirmatory_result_float16.json",
            "predictions": out_dir / "helm_confirmatory_predictions_float16.csv"}


def verify_inputs(out, n_resamples, dry_run=False):
    problems = list(H.prereg_problems(PROJECT_ROOT))
    problems.extend(plan_v2.plan_problems(PROJECT_ROOT, require_amendment_2=True))
    if n_resamples != N_RESAMPLES:
        problems.append(f"n_resamples is fixed at {N_RESAMPLES}")
    for name, (path, sha) in PINS.items():
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
        elif sha256_path(path) != sha:
            problems.append(f"{name}: {path.name} differs from its pinned digest")
    fm = (json.loads(HELM_FEATURES_MANIFEST.read_text())
          if HELM_FEATURES_MANIFEST.exists() else {})
    eq = fm.get("equality", {})
    if (fm.get("status") != "complete" or fm.get("cache_sha256") != PINS["helm_features"][1]
            or fm.get("evidence_sha256") != PINS["helm_evidence"][1]
            or eq.get("reference_mismatches") or eq.get("feature_cache_mismatches")):
        problems.append("the HELM feature manifest does not certify a complete, equal cache")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; the confirmatory test runs once")
    if problems:
        raise HelmConfirmatoryRefused("inputs:\n  " + "\n  ".join(problems))
    return fm


# ---------------------------------------------------------------- data


def normalise_title(title):
    return "".join(title.split()).casefold()


def prepare_helm(units, evidence_path, reader, exclude_source=False):
    """Prepared sentences (src.v2_replay format): one subclaim, top-10 documents.

    With ``exclude_source`` the source article's passages are dropped and the
    remaining passages keep their order (ranks renumbered 1..n), §6.
    """
    rows = [json.loads(line) for line in open(evidence_path, encoding="utf-8")]
    if [r["id"] for r in rows] != [u.id for u in units]:
        raise HelmConfirmatoryRefused("the evidence does not hold one list per unit in order")
    out = []
    for r, u in zip(rows, units):
        docs = r["documents"][:MAX_DOCS]
        if exclude_source:
            docs = [d for d in docs if normalise_title(d["title"]) != normalise_title(u.title)]
        out.append({"id": u.id, "cluster": u.cluster, "label": u.label,
                    "subclaims": [[reader.document(u.sentence, d["text"], rank)
                                   for rank, d in enumerate(docs, start=1)]]})
    return out


# ---------------------------------------------------------------- gates


def gate_selfcheckgpt(est, sup, wang_root):
    """X2 and X4 on SelfCheckGPT held-out (the engine is the one Phase 5 used)."""
    from src.mddre_features import FeatureReader

    evidence = SF.evidence_rows(SF.FACTSCORE / SF.FACTSCORE_EVIDENCE[0])
    sets, _ = RR.test_sets(wang_root, evidence)
    reader = FeatureReader(PINS["mddre_features"][0], frozen.OFFICIAL_MODEL,
                           RV.cache_dtype("mddre_features"))
    try:
        prepared = RV.prepare(sets["selfcheckgpt_heldout"], reader, MAX_DOCS)
    finally:
        reader.close()
    official = R.run_bse_official(prepared, est["BSE"], S.Costs(28, 96, 1), MAX_DOCS)
    x2 = RV.gate_x2("selfcheckgpt_heldout", prepared, official,
                    RV.frozen_official_rows(PINS["mddre_predictions"][0]))
    op = B.operating_key(MAX_DOCS)
    curves = {}
    for m in B.METHODS:
        lrs = R.log_ratios(prepared, est[m])
        for cm, cfa in R.COST_PAIRS:
            costs = S.Costs(cm, cfa, 1.0)
            curves[(m, (cm, cfa))] = {op: R.run_backward_induction(
                prepared, lrs, S.PolicyByHorizon(sup[m], costs, 10), costs, MAX_DOCS)}
    x4 = RB.gate_x4("selfcheckgpt_heldout", prepared, curves, MAX_DOCS,
                    RB.phase5_rows(PINS["phase5_predictions"][0]))
    if x2 or x4:
        raise HelmConfirmatoryRefused(f"gates on SelfCheckGPT: X2 {x2}, X4 {x4}")
    return {"X2_official_rows_differing": 0, "X4_operating_point_rows_differing": 0,
            "sentences": len(prepared)}


# ---------------------------------------------------------------- analysis


def bootstrap(curves, labels, clusters, idx_map, n_resamples):
    """Endpoints over the units ``idx_map`` (global indices), resampled by cluster."""
    import numpy as np

    idx_map = np.asarray(idx_map)
    sub_clusters = [clusters[i] for i in idx_map]
    return V.cluster_bootstrap(
        lambda idx: B.contrast_endpoints(curves, labels, idx_map[idx], MAX_DOCS, R.COST_PAIRS),
        sub_clusters, n_resamples, SEED)


def summaries(observed, draws, n_resamples):
    out = {}
    for k, v in observed.items():
        s = V.summarise(v, draws[k])
        s["p_greater_text"] = V.format_p(s["p_greater"], n_resamples)
        s["p_less_text"] = V.format_p(s["p_less"], n_resamples)
        out[k] = s
    return out


def operating_tables(curves, official, labels, passages):
    """Wang's table and calibration at both operating points, every method."""
    op = B.operating_key(MAX_DOCS)
    out = {}
    for m in B.METHODS:
        for pair in R.COST_PAIRS:
            r = curves[(m, pair)][op]
            row = V.wang_table(labels, r["p_factual"], r["prediction"], r["documents"],
                               r["nli_calls"], passages=passages)
            row["calibration"] = V.calibration(labels, r["p_factual"])
            out[f"{m}|{pair[0]:g}/{pair[1]:g}"] = row
    for pair, r in official.items():
        row = V.wang_table(labels, r["p_factual"], r["prediction"], r["documents"],
                           r["nli_calls"], passages=passages)
        row["calibration"] = V.calibration(labels, r["p_factual"])
        out[f"BSE-official|{pair[0]:g}/{pair[1]:g}"] = row
    return out


def prediction_rows(prepared, units, curves, official):
    op = B.operating_key(MAX_DOCS)
    rows = []
    sources = {f"{m}|{p[0]:g}/{p[1]:g}": curves[(m, p)][op]
               for m in B.METHODS for p in R.COST_PAIRS}
    sources.update({f"BSE-official|{p[0]:g}/{p[1]:g}": r for p, r in official.items()})
    for name, r in sources.items():
        method, setting = name.split("|")
        for j, (s, u) in enumerate(zip(prepared, units)):
            rows.append({"method": method, "setting": setting, "id": s["id"],
                         "cluster": s["cluster"], "generator": u.generator,
                         "gold_label": s["label"], "p_factual": float(r["p_factual"][j]),
                         "prediction": int(r["prediction"][j]),
                         "documents": int(r["documents"][j]),
                         "nli_calls": int(r["nli_calls"][j])})
    return rows


def main():
    import numpy as np

    from src.mddre_features import FeatureReader

    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser())
    fm = verify_inputs(out, args.n_resamples, args.dry_run)
    units, titles = H.load_helm()
    g0 = H.check_expected(units, titles)
    est, sup = RV.load_estimators(json.loads(PINS["estimators"][0].read_text()))
    gates = {"G0": g0, **gate_selfcheckgpt(est, sup, args.wang_data_root)}
    print(f"gates: {gates}", flush=True)

    reader = FeatureReader(PINS["helm_features"][0], frozen.OFFICIAL_MODEL, fm["dtype"])
    try:
        prepared = prepare_helm(units, PINS["helm_evidence"][0], reader)
        prepared_ex = prepare_helm(units, PINS["helm_evidence"][0], reader, exclude_source=True)
    finally:
        reader.close()
    if args.dry_run:
        docs = sum(len(s["subclaims"][0]) for s in prepared)
        kept = sum(len(s["subclaims"][0]) for s in prepared_ex)
        print(f"--dry-run: gates passed; {len(prepared)} units, {docs} documents with "
              f"features ({docs - kept} source-article passages). No detector run, "
              "nothing written.")
        return 0
    labels = np.array([u.label for u in units])
    clusters = [u.cluster for u in units]
    passages = [f"{u.generator}:{u.cluster}" for u in units]
    policies = RB.build_policies(sup)

    curves = RB.budget_curves(prepared, est, sup, policies, MAX_DOCS)
    official = {pair: R.run_bse_official(prepared, est["BSE"], S.Costs(*pair, 1.0), MAX_DOCS)
                for pair in R.COST_PAIRS}
    everyone = np.arange(len(units))
    observed, draws, n_clusters = bootstrap(curves, labels, clusters, everyone,
                                            args.n_resamples)
    decision = D.decide(observed, draws)
    AR.check_language(json.dumps(decision["statements"]))
    print(json.dumps(decision["statements"], indent=1), flush=True)

    sensitivity = {}
    six = [i for i, u in enumerate(units) if u.generator in SIX_GENERATORS]
    o6, d6, _ = bootstrap(curves, labels, clusters, six, args.n_resamples)
    sensitivity["six_generators"] = {"units": len(six), "decision": D.decide(o6, d6),
                                     "endpoints": summaries(o6, d6, args.n_resamples)}
    curves_ex = RB.budget_curves(prepared_ex, est, sup, policies, MAX_DOCS)
    ox, dx, _ = bootstrap(curves_ex, labels, clusters, everyone, args.n_resamples)
    removed = sum(MAX_DOCS - len(s["subclaims"][0]) for s in prepared_ex)
    sensitivity["source_article_excluded"] = {
        "passages_removed": int(removed), "decision": D.decide(ox, dx),
        "endpoints": summaries(ox, dx, args.n_resamples)}
    by_generator = {}
    for g in H.GENERATORS:
        idx = np.array([i for i, u in enumerate(units) if u.generator == g])
        by_generator[g] = {"units": len(idx), "hallucinated": int(np.sum(labels[idx] == 0)),
                           "observed": B.contrast_endpoints(curves, labels, idx, MAX_DOCS,
                                                            R.COST_PAIRS)}
    print("sensitivity analyses done", flush=True)

    result = {
        "artifact": "helm-confirmatory",
        "status": "preregistered confirmatory test",
        "protocol": H.PREREG_PATH,
        "preregistration_sha256": H.PREREG_SHA256,
        "plan_prefix_sha256": plan_v2.AMENDMENT_2_SHA256,
        "pins": {k: v[1] for k, v in PINS.items()},
        "dtype": fm["dtype"],
        "gates": gates,
        "settings": {"cost_pairs": [list(c) for c in R.COST_PAIRS],
                     "c_retrieve": list(B.BUDGET_C_RETRIEVE), "k": [1, MAX_DOCS],
                     "stopping": "backward induction over min(available documents, K)",
                     "subclaims": "each sentence is its own single subclaim"},
        "decision": decision,
        "units": len(units), "hallucinated": int(np.sum(labels == 0)), "clusters": n_clusters,
        "endpoints": summaries(observed, draws, args.n_resamples),
        "curves": B.observed_curves(curves, labels),
        "operating_tables": operating_tables(curves, official, labels, passages),
        "sensitivity": sensitivity,
        "by_generator": by_generator,
        "bootstrap": {"n_resamples": args.n_resamples, "seed": SEED, "unit": "article"},
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
    rows = prediction_rows(prepared, units, curves, official)
    with out["predictions"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for k in ("result", "predictions"):
        out[k].chmod(0o444)
        print(f"{k} {out[k]}\n  sha256 {sha256_path(out[k])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HelmConfirmatoryRefused, H.HelmDataRefused, D.DecisionError, RB.BudgetRefused,
            RV.V2ReplayRefused, S.SequentialError, R.ReplayError, V.MetricsError,
            AR.ReplayRefused, ValueError) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
