#!/usr/bin/env python3
"""Plan-v2 amendment 2: budget-knob curves and the fixed-k contrasts, once.

Exploratory (plan §11). On the five used datasets, from cached features only:
BSE, DDRE-1D, NB-M and DDRE-M under backward induction over every
(K, C_retrieve) point, K = 1 .. the dataset's maximum documents and
C_retrieve in {0.125, 0.25, 0.5, 1}, at C_M/C_FA 28/96 and 14/24; and Wang's
fixed-k maximum-entailment baseline for K = 1 .. maximum. A method's curve is
the step envelope over its points. The contrasts in ``src.v2_budget`` are
bootstrapped (paired by cluster, 10,000 replicates, seed 42).

Gates, any failure stops the run before anything is written:

- **X0:** the plan at amendment 2 and every input match their pins; the
  output does not exist yet.
- **X2:** BSE-official reproduces the frozen earlier replays row for row.
- **X4:** every method's (K = maximum, C_retrieve = 1) point reproduces the
  Phase 5 operating-point predictions exactly, at both cost pairs.

Inputs are read-only; the result is written once and made read-only.
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
import run_v2_replay as RV  # noqa: E402
from src import confirmatory_data as C  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src import metrics_v2 as V  # noqa: E402
from src import plan_v2  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_budget as B  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

PINS = dict(RV.PINS, **{
    "phase5_result": (RV.V2 / "v2_replay_result_float16.json",
                      "a7fe078169b0625eddffb8991bdd52b113c499ffb26a77b63bcafb9c2531af93"),
    "phase5_predictions": (RV.V2 / "v2_replay_predictions_float16.csv",
                           "bd75f4864ee72695c8e95bb3791daea126d6496d78e3f329e7a34a41db32feb5"),
})
STATUS = "exploratory: plan v2 amendment 2, written after Phase 5's results were seen"
N_RESAMPLES, SEED = RV.N_RESAMPLES, RV.SEED


class BudgetRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(RV.V2))
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES, help=argparse.SUPPRESS)
    return parser.parse_args()


def output_path(out_dir):
    return out_dir / f"v2_budget_result_{RV.DTYPE}.json"


def verify_inputs(out, n_resamples):
    problems = list(plan_v2.plan_problems(PROJECT_ROOT, require_amendment_2=True))
    if n_resamples != N_RESAMPLES:
        problems.append(f"n_resamples is fixed at {N_RESAMPLES}")
    for name, (path, sha) in PINS.items():
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
        elif sha256_path(path) != sha:
            problems.append(f"{name}: {path.name} differs from its pinned digest")
    if out.exists():
        problems.append(f"{out} already exists; the budget run runs once")
    if problems:
        raise BudgetRefused("inputs:\n  " + "\n  ".join(problems))


def phase5_rows(path):
    with open(path, newline="") as handle:
        return {(r["dataset"], r["method"], r["setting"], r["id"]): r
                for r in csv.DictReader(handle)}


def gate_x4(name, prepared, curves, max_docs, rows):
    """Operating-point rows that differ from Phase 5's predictions."""
    bad = 0
    op = B.operating_key(max_docs)
    for method in B.METHODS:
        for cm, cfa in R.COST_PAIRS:
            r = curves[(method, (cm, cfa))][op]
            setting = R.setting_key(cm, cfa, R.OPERATING_C_RETRIEVE)
            for j, s in enumerate(prepared):
                row = rows.get((name, method, setting, s["id"]))
                bad += (row is None
                        or str(float(r["p_factual"][j])) != row["p_factual"]
                        or int(r["documents"][j]) != int(row["documents"])
                        or int(r["nli_calls"][j]) != int(row["nli_calls"]))
    return bad


def budget_curves(prepared, est, sup, policies, max_docs):
    """{(method, cost pair): {point key: arrays}}, plus the fixed-k baseline."""
    import numpy as np

    lrs = {m: R.log_ratios(prepared, est[m]) for m in B.METHODS}
    curves = {}
    for cm, cfa in R.COST_PAIRS:
        for m in B.METHODS:
            points = {}
            for k, cr in B.grid(max_docs):
                costs = S.Costs(cm, cfa, cr)
                points[B.point_key(k, cr)] = R.run_backward_induction(
                    prepared, lrs[m], policies[(m, cm, cfa, cr)], costs, k)
            curves[(m, (cm, cfa))] = points
    fixed = {}
    for k in range(1, max_docs + 1):
        r = R.run_fixed_k(prepared, None, k, "max", max_docs)
        fixed[f"K{k}"] = {"p_factual": r["score"], "documents": r["documents"],
                          "nli_calls": r["nli_calls"]}
    curves[(B.FIXED, None)] = fixed
    for points in curves.values():
        for v in points.values():
            v["p_rounded"] = np.round(v["p_factual"], V.RANK_DECIMALS)
    return curves


def build_policies(sup):
    return {(m, cm, cfa, cr): S.PolicyByHorizon(sup[m], S.Costs(cm, cfa, cr), 10)
            for m in B.METHODS for cm, cfa in R.COST_PAIRS for cr in B.BUDGET_C_RETRIEVE}


def main():
    import numpy as np

    from src.mddre_features import FeatureReader

    args = parse_args()
    out = output_path(Path(args.out_dir).expanduser())
    verify_inputs(out, args.n_resamples)
    record = json.loads(PINS["estimators"][0].read_text())
    est, sup = RV.load_estimators(record)
    sets, _ = RV.load_sets(args.wang_data_root)
    frozen_rows = {k: RV.frozen_official_rows(PINS[k][0])
                   for k in ("mddre_predictions", "confirmatory_predictions")}
    p5_rows = phase5_rows(PINS["phase5_predictions"][0])
    readers = {k: FeatureReader(PINS[k][0], frozen.OFFICIAL_MODEL, RV.cache_dtype(k))
               for k in ("mddre_features", "confirmatory_features")}
    prepared = {}
    try:
        for name, (max_docs, cache, preds, _) in RV.DATASETS.items():
            prepared[name] = RV.prepare(sets[name], readers[cache], max_docs)
            official = R.run_bse_official(prepared[name], est["BSE"], S.Costs(28, 96, 1),
                                          max_docs)
            x2 = RV.gate_x2(name, prepared[name], official, frozen_rows[preds])
            if x2:
                raise BudgetRefused(f"gate X2 failed on {name}: {x2} rows differ")
    finally:
        for r in readers.values():
            r.close()

    policies = build_policies(sup)
    results = {}
    for name, (max_docs, _, _, _) in RV.DATASETS.items():
        data = prepared[name]
        labels = np.array([s["label"] for s in data])
        clusters = [s["cluster"] for s in data]
        curves = budget_curves(data, est, sup, policies, max_docs)
        x4 = gate_x4(name, data, curves, max_docs, p5_rows)
        print(f"{name}: X2 0, X4 {x4}", flush=True)
        if x4:
            raise BudgetRefused(f"gate X4 failed on {name}: {x4} operating-point rows differ")
        observed, draws, n_clusters = V.cluster_bootstrap(
            lambda idx: B.contrast_endpoints(curves, labels, idx, max_docs, R.COST_PAIRS),
            clusters, args.n_resamples, SEED)
        endpoints = {}
        for k, v in observed.items():
            summary = V.summarise(v, draws[k])
            summary["p_greater_text"] = V.format_p(summary["p_greater"], args.n_resamples)
            summary["p_less_text"] = V.format_p(summary["p_less"], args.n_resamples)
            endpoints[k] = summary
        results[name] = {"status": STATUS, "max_documents": max_docs, "sentences": len(data),
                         "clusters": n_clusters,
                         "gates": {"X2_official_rows_differing": 0,
                                   "X4_operating_point_rows_differing": 0},
                         "curves": B.observed_curves(curves, labels), "endpoints": endpoints}
        print(f"{name}: done", flush=True)

    result = {
        "artifact": "v2-budget",
        "status": STATUS,
        "protocol": plan_v2.PLAN_PATH + " §11 (amendment 2)",
        "plan_prefix_sha256": plan_v2.AMENDMENT_2_SHA256,
        "pins": {k: v[1] for k, v in PINS.items()},
        "settings": {"cost_pairs": [list(c) for c in R.COST_PAIRS],
                     "c_retrieve": list(B.BUDGET_C_RETRIEVE),
                     "k": "1 .. dataset maximum documents per subclaim",
                     "stopping": "backward induction over min(available documents, K)"},
        "contrasts": [{"reference": a, "method": b, "anchor": c} for a, b, c in B.CONTRASTS],
        "bootstrap": {"n_resamples": args.n_resamples, "seed": SEED, "ci_level": 0.95},
        "results": results,
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    out.chmod(0o444)
    print(f"result {out}\n  sha256 {sha256_path(out)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BudgetRefused, RV.V2ReplayRefused, C.ConfirmatoryDataRefused, E.EstimatorError,
            S.SequentialError, R.ReplayError, V.MetricsError, B.BudgetError) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
