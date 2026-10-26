#!/usr/bin/env python3
"""Data for the paper's worked example (Figure 1), from the frozen HELM artifacts,
and the two fitted likelihood ratios on a fine grid (Figure 2).

The example is chosen by a fixed rule, stated in the figure caption: the first
HELM sentence, in file order, that BSE and DDRE-1D both classify correctly at
C_M/C_FA = 28/96 (K = 10, C_retrieve = 1) and for which they read a different
number of documents. For that sentence the script replays both detectors
document by document, records the posterior after each document and each
stage's continue region, and checks the replay against the frozen
predictions (documents read and final P(factual)). Nothing is refit and no
artifact is written to; the output is a small JSON file for plotting.

The ratio curves evaluate the frozen BSE histogram and DDRE-1D estimator at
e = 0, 0.25, ..., 100 with the estimators' own code, and store the raw
training counts of BSE's histogram (its stored counts minus Laplace's +1).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_frozen_heldout as frozen  # noqa: E402
import run_helm_confirmatory as HC  # noqa: E402
import run_v2_replay as RV  # noqa: E402
from src import helm_data as H  # noqa: E402
from src import sequential as S  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

PREDICTIONS = (H.ROOT / "helm_confirmatory_predictions_float16.csv",
               "994d85f223b45eaa0afe5740ea5cd98e72f8193d05c50266935dd79302d7a8e8")
METHODS = ("BSE", "DDRE-1D")
SETTING, COSTS = "28/96", S.Costs(28, 96, 1.0)
MAX_DOCS = HC.MAX_DOCS
SNIPPET = 160


def choose(path):
    """The fixed rule over the frozen predictions, in file order."""
    rows = {}
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            if r["setting"] == SETTING and r["method"] in METHODS:
                rows.setdefault(r["id"], {})[r["method"]] = r
    for uid, by in rows.items():
        if len(by) < 2:
            continue
        correct = all(by[m]["prediction"] == by[m]["gold_label"] for m in METHODS)
        if correct and by["BSE"]["documents"] != by["DDRE-1D"]["documents"]:
            return uid, by
    raise SystemExit("no sentence satisfies the rule")


def trajectory(log_ratios, spans, policy, p0=0.5):
    """``S.run_subclaim`` step by step: the posterior after every document read."""
    p, x = float(p0), S._logit(p0)
    path = [p]
    for rank, lr in enumerate(log_ratios[:MAX_DOCS], start=1):
        if not policy.should_continue(p, rank):
            break
        x = min(max(x + float(lr), -S.LOG_ODDS_CLIP), S.LOG_ODDS_CLIP)
        p = S._sigmoid(x)
        path.append(p)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "paper/figures/data/helm_example.json"))
    ap.add_argument("--curves-out",
                    default=str(PROJECT_ROOT / "paper/figures/data/ratio_curves.json"))
    args = ap.parse_args()
    for name, (path, sha) in {"predictions": PREDICTIONS, "estimators": RV.PINS["estimators"],
                              "helm_evidence": HC.PINS["helm_evidence"],
                              "helm_features": HC.PINS["helm_features"]}.items():
        if sha256_path(path) != sha:
            raise SystemExit(f"{name} does not match its pin")

    uid, frozen_rows = choose(PREDICTIONS[0])
    units, _ = H.load_helm()
    j = [u.id for u in units].index(uid)
    unit = units[j]
    record = json.loads(RV.PINS["estimators"][0].read_text())
    est, sup = RV.load_estimators(record)
    dtype = json.loads(HC.HELM_FEATURES_MANIFEST.read_text())["dtype"]

    from src.mddre_features import FeatureReader

    reader = FeatureReader(HC.PINS["helm_features"][0], frozen.OFFICIAL_MODEL, dtype)
    try:
        evidence = None
        with open(HC.PINS["helm_evidence"][0], encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i == j:
                    evidence = json.loads(line)
                    break
        if evidence["id"] != uid:
            raise SystemExit("evidence order does not match the units")
        docs = evidence["documents"][:MAX_DOCS]
        feats = [reader.document(unit.sentence, d["text"], rank)
                 for rank, d in enumerate(docs, start=1)]
    finally:
        reader.close()

    out = {"rule": ("first HELM sentence in file order that BSE and DDRE-1D both classify "
                    "correctly at C_M/C_FA 28/96 (K 10, C_retrieve 1) and for which they "
                    "read a different number of documents"),
           "id": uid, "generator": unit.generator, "title": unit.title,
           "sentence": unit.sentence, "label": unit.label,
           "tau": COSTS.c_miss / (COSTS.c_miss + COSTS.c_false_alarm),
           "documents": [{"rank": f["rank"], "title": d["title"],
                          "snippet": " ".join(d["text"].split())[:SNIPPET],
                          "entailment": float(f["e"]), "spans": int(f["spans"])}
                         for d, f in zip(docs, feats)],
           "methods": {}}
    spans = [f["spans"] for f in feats]
    for m in METHODS:
        lrs = [est[m].log_ratio(f) for f in feats]
        policy = S.PolicyByHorizon(sup[m], COSTS, MAX_DOCS).for_documents(len(feats))
        path = trajectory(lrs, spans, policy)
        fr = frozen_rows[m]
        if len(path) - 1 != int(fr["documents"]) or abs(path[-1] - float(fr["p_factual"])) > 1e-9:
            raise SystemExit(f"{m}: replay differs from the frozen predictions")
        out["methods"][m] = {
            "log_ratios": [float(x) for x in lrs], "posterior": path,
            "documents": len(path) - 1, "p_factual": path[-1],
            "prediction": int(fr["prediction"]),
            "continue_regions": {str(n): [list(iv) for iv in policy.intervals[n]]
                                 for n in sorted(policy.intervals)}}
    out["pins"] = {"predictions": PREDICTIONS[1], "estimators": RV.PINS["estimators"][1],
                   "helm_evidence": HC.PINS["helm_evidence"][1],
                   "helm_features": HC.PINS["helm_features"][1]}
    grid = [i / 4 for i in range(401)]
    hist = record["tier1"]["estimators"]["BSE"]
    curves = {"grid": grid, "pins": {"estimators": RV.PINS["estimators"][1]},
              "log_ratio": {m: [float(est[m].log_ratio({"e": e, "k": 0.0, "rank": 1, "spans": 1}))
                                for e in grid] for m in METHODS},
              "bse_training_counts": {"factual": [c - 1 for c in hist["pos_hist"]],
                                      "hallucinated": [c - 1 for c in hist["neg_hist"]]}}
    for m in METHODS:  # the fit script's stored curve, at its own grid points
        for e, v in record["tier1"]["log_ratio_curves"][m].items():
            if abs(curves["log_ratio"][m][grid.index(float(e))] - v) > 1e-9:
                raise SystemExit(f"{m}: curve differs from the stored one at e = {e}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.curves_out).write_text(json.dumps(curves) + "\n")
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{uid}: BSE {out['methods']['BSE']['documents']} documents, "
          f"DDRE-1D {out['methods']['DDRE-1D']['documents']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
