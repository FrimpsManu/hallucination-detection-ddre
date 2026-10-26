#!/usr/bin/env python3
"""The preregistered second confirmatory test, on FACTORY, once
(docs/factory_preregistration.md §3-§7, §8 step 4).

HELM's confirmatory replay (``scripts/run_helm_confirmatory.py``) on the
FACTORY features: BSE, DDRE-1D, NB-M and DDRE-M under backward induction over
every (K, C_retrieve) point at C_M/C_FA 28/96 and 14/24, Wang's fixed-k max
entailment for K = 1..10, and BSE-official at the operating points. Each
sentence is its own single subclaim. One paired cluster bootstrap over
questions (10,000 replicates, seed 42) gives every endpoint;
``src.factory_decision`` applies §5 and §5.1. §6 is descriptive: Rule B
labels, and HELM and FACTORY pooled.

Gates, any failure stops the run before anything is written:

- **P:** the FACTORY preregistration, plan v2 at amendment 2, the estimators,
  the FACTORY evidence and features (with manifests), the HELM inputs for the
  pooled analysis and the SelfCheckGPT references match their pins; outputs
  do not exist yet.
- **G0:** the FACTORY loader reproduces the preregistered totals, for Rule A
  and Rule B, and Rule B's units are Rule A's with the same labels.
- **X2, X4:** as on HELM, on SelfCheckGPT held-out.

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
import run_helm_confirmatory as HC  # noqa: E402
import run_v2_budget as RB  # noqa: E402
import run_v2_replay as RV  # noqa: E402
from src import ablation_replay as AR  # noqa: E402
from src import factory_data as F  # noqa: E402
from src import factory_decision as FD  # noqa: E402
from src import helm_data as H  # noqa: E402
from src import helm_decision as D  # noqa: E402
from src import metrics_v2 as V  # noqa: E402
from src import plan_v2  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_budget as B  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

MAX_DOCS = HC.MAX_DOCS
PINS = {
    "factory_evidence": (F.ROOT / "factory_evidence_top10.jsonl",
                         "c9bcf9cf42338b07653025a9f0010f04d30fa18a65d6aa7a25314090337f7ac8"),
    "factory_features": (F.ROOT / "factory_features_float16.sqlite",
                         "d00eb4ecf0041147e470f5c0e55fe604dbf7a4f810772023461f2f652a97b66b"),
    "helm_evidence": HC.PINS["helm_evidence"],
    "helm_features": HC.PINS["helm_features"],
    **{k: v for k, v in HC.PINS.items() if k not in ("helm_evidence", "helm_features")},
}
FACTORY_FEATURES_MANIFEST = F.ROOT / "factory_features_float16_manifest.json"
N_RESAMPLES, SEED = HC.N_RESAMPLES, HC.SEED


class FactoryConfirmatoryRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(F.ROOT))
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--dry-run", action="store_true",
                        help="Gates P (without the output check) and G0 only. Nothing written.")
    return parser.parse_args()


def output_paths(out_dir):
    return {"result": out_dir / "factory_confirmatory_result_float16.json",
            "predictions": out_dir / "factory_confirmatory_predictions_float16.csv"}


def verify_inputs(out, n_resamples, dry_run=False):
    problems = list(F.prereg_problems(PROJECT_ROOT))
    problems.extend(plan_v2.plan_problems(PROJECT_ROOT, require_amendment_2=True))
    if n_resamples != N_RESAMPLES:
        problems.append(f"n_resamples is fixed at {N_RESAMPLES}")
    for name, (path, sha) in PINS.items():
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
        elif sha256_path(path) != sha:
            problems.append(f"{name}: {path.name} differs from its pinned digest")
    fm = (json.loads(FACTORY_FEATURES_MANIFEST.read_text())
          if FACTORY_FEATURES_MANIFEST.exists() else {})
    eq = fm.get("equality", {})
    if (fm.get("status") != "complete" or fm.get("cache_sha256") != PINS["factory_features"][1]
            or fm.get("evidence_sha256") != PINS["factory_evidence"][1]
            or eq.get("reference_mismatches") or eq.get("feature_cache_mismatches")):
        problems.append("the FACTORY feature manifest does not certify a complete, equal cache")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; the confirmatory test runs once")
    if problems:
        raise FactoryConfirmatoryRefused("inputs:\n  " + "\n  ".join(problems))
    return fm


def load_units():
    """Rule A units (gate G0) and the indices of Rule B's units within them."""
    units, counts = F.load_factory()
    g0 = F.check_expected(units, counts)
    units_b, counts_b = F.load_factory(rule=F.rule_b)
    g0_b = {k: v for k, v in F.totals(units_b, counts_b).items() if k in F.EXPECTED_RULE_B}
    if g0_b != F.EXPECTED_RULE_B:
        raise FactoryConfirmatoryRefused(f"gate G0, Rule B: {g0_b} vs {F.EXPECTED_RULE_B}")
    position = {u.id: i for i, u in enumerate(units)}
    idx_b = []
    for u in units_b:
        i = position.get(u.id)
        if i is None or units[i].label != u.label:
            raise FactoryConfirmatoryRefused(f"gate G0: Rule B unit {u.id} is not a Rule A "
                                             "unit with the same label")
        idx_b.append(i)
    return units, idx_b, {"rule_a": g0, "rule_b": g0_b}


# ---------------------------------------------------------------- analysis


def informativeness_bootstrap(curves, labels, clusters, idx_map, n_resamples):
    """§5.1's statistic, on the same cluster draws as ``HC.bootstrap``."""
    import numpy as np

    idx_map = np.asarray(idx_map)
    score = curves[(B.FIXED, None)][f"K{MAX_DOCS}"]["p_rounded"]
    obs, draws, _ = V.cluster_bootstrap(
        lambda idx: {FD.INFORMATIVE_KEY: V.balanced_ap_fast(labels[idx_map[idx]],
                                                            score[idx_map[idx]])},
        [clusters[i] for i in idx_map], n_resamples, SEED)
    return FD.informativeness(obs[FD.INFORMATIVE_KEY], draws[FD.INFORMATIVE_KEY])


def analyse(curves, labels, clusters, idx, n_resamples):
    observed, draws, n_clusters = HC.bootstrap(curves, labels, clusters, idx, n_resamples)
    info = informativeness_bootstrap(curves, labels, clusters, idx, n_resamples)
    return {"units": len(idx), "clusters": n_clusters,
            "hallucinated": int(sum(labels[i] == 0 for i in idx)),
            "decision": FD.decide(observed, draws, info),
            "endpoints": HC.summaries(observed, draws, n_resamples)}


def concat_curves(a, b):
    """Two datasets' curves as one, units of ``a`` first (the pooled analysis)."""
    import numpy as np

    if set(a) != set(b):
        raise FactoryConfirmatoryRefused("the two datasets' curves have different cells")
    out = {}
    for key in a:
        if set(a[key]) != set(b[key]):
            raise FactoryConfirmatoryRefused(f"{key}: different points")
        out[key] = {}
        for point, va in a[key].items():
            vb = b[key][point]
            out[key][point] = {f: (np.concatenate([va[f], vb[f]]) if isinstance(va[f], np.ndarray)
                                   else list(va[f]) + list(vb[f])) for f in va}
    return out


def helm_curves(est, sup, policies):
    from src.mddre_features import FeatureReader

    units, titles = H.load_helm()
    H.check_expected(units, titles)
    fm = json.loads(HC.HELM_FEATURES_MANIFEST.read_text())
    reader = FeatureReader(PINS["helm_features"][0], frozen.OFFICIAL_MODEL, fm["dtype"])
    try:
        prepared = HC.prepare_helm(units, PINS["helm_evidence"][0], reader)
    finally:
        reader.close()
    return units, RB.budget_curves(prepared, est, sup, policies, MAX_DOCS)


def main():
    import numpy as np

    from src.mddre_features import FeatureReader

    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser())
    fm = verify_inputs(out, args.n_resamples, args.dry_run)
    units, idx_b, g0 = load_units()
    print(f"gate G0: {g0}", flush=True)
    if args.dry_run:
        print("--dry-run: gates P and G0 passed. No detector run, nothing written.")
        return 0
    est, sup = RV.load_estimators(json.loads(PINS["estimators"][0].read_text()))
    gates = {"G0": g0, **HC.gate_selfcheckgpt(est, sup, args.wang_data_root)}
    print(f"gates: {gates}", flush=True)

    reader = FeatureReader(PINS["factory_features"][0], frozen.OFFICIAL_MODEL, fm["dtype"])
    try:
        prepared = HC.prepare_helm(units, PINS["factory_evidence"][0], reader)
    finally:
        reader.close()
    labels = np.array([u.label for u in units])
    clusters = [u.cluster for u in units]
    policies = RB.build_policies(sup)
    curves = RB.budget_curves(prepared, est, sup, policies, MAX_DOCS)
    official = {pair: R.run_bse_official(prepared, est["BSE"], S.Costs(*pair, 1.0), MAX_DOCS)
                for pair in R.COST_PAIRS}

    primary = analyse(curves, labels, clusters, np.arange(len(units)), args.n_resamples)
    AR.check_language(json.dumps(primary["decision"]["statements"]))
    print(json.dumps({"informativeness": primary["decision"]["informativeness"],
                      "statements": primary["decision"]["statements"]}, indent=1), flush=True)

    sensitivity = {"rule_b": analyse(curves, labels, clusters, idx_b, args.n_resamples)}
    helm_units, helm = helm_curves(est, sup, policies)
    pooled = concat_curves(helm, curves)
    pooled_labels = np.concatenate([np.array([u.label for u in helm_units]), labels])
    pooled_clusters = ([f"helm:{u.cluster}" for u in helm_units]
                       + [f"factory:{c}" for c in clusters])
    sensitivity["pooled_helm_factory"] = analyse(pooled, pooled_labels, pooled_clusters,
                                                 np.arange(len(pooled_labels)),
                                                 args.n_resamples)
    print("sensitivity analyses done", flush=True)

    result = {
        "artifact": "factory-confirmatory",
        "status": "preregistered confirmatory test",
        "protocol": F.PREREG_PATH,
        "preregistration_sha256": F.PREREG_SHA256,
        "plan_prefix_sha256": plan_v2.AMENDMENT_2_SHA256,
        "pins": {k: v[1] for k, v in PINS.items()},
        "dtype": fm["dtype"],
        "gates": gates,
        "settings": {"cost_pairs": [list(c) for c in R.COST_PAIRS],
                     "c_retrieve": list(B.BUDGET_C_RETRIEVE), "k": [1, MAX_DOCS],
                     "stopping": "backward induction over min(available documents, K)",
                     "subclaims": "each sentence is its own single subclaim",
                     "labels": "Rule A (§2.2)"},
        "decision": primary["decision"],
        "units": primary["units"], "hallucinated": primary["hallucinated"],
        "clusters": primary["clusters"],
        "endpoints": primary["endpoints"],
        "curves": B.observed_curves(curves, labels),
        "operating_tables": HC.operating_tables(curves, official, labels, clusters),
        "sensitivity": sensitivity,
        "bootstrap": {"n_resamples": args.n_resamples, "seed": SEED, "unit": "question"},
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
    rows = HC.prediction_rows(prepared, units, curves, official)
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
    except (FactoryConfirmatoryRefused, HC.HelmConfirmatoryRefused, F.FactoryDataRefused,
            H.HelmDataRefused, D.DecisionError, RB.BudgetRefused, RV.V2ReplayRefused,
            S.SequentialError, R.ReplayError, V.MetricsError, AR.ReplayRefused,
            ValueError) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
