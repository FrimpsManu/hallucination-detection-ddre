#!/usr/bin/env python3
"""Plan-v2 exploratory replay on the five used datasets, once (Phase 5).

Research plan v2 §3-§5 and amendment 1. Every dataset here has been used
before, so every result is **exploratory** (plan §1).

For SelfCheckGPT held-out, FActScore test entities, Factcheck-Bench, ExpertQA
and BingCheck, from cached NLI features only:

- Tier 1 (Wang's NBC data): BSE and DDRE-1D; Tier 2 (FActScore training
  entities): Hist-1D, DDRE-1D', NB-M, DDRE-M, Logit-M. All are stopped by
  backward induction over each subclaim's available documents, for
  C_M/C_FA in {28/96, 14/24} and C_retrieve on a 15-point log grid
  (0.125 .. 16, operating point 1).
- References: BSE-official (Wang's released rule) on the same sweep, and the
  one-step Eq. 8 for every method at the operating points.
- Wang's Figure 2 fixed-k baselines (max, avg, NBC with Wang's histogram).
- Observed: Wang's Table 1, balanced AP and calibration for every method and
  setting. Bootstrapped (paired by cluster, 10,000 replicates, seed 42): the
  endpoints of every contrast in ``src.v2_replay.CONTRASTS``.

Gates, any failure stops the run before anything is written:

- **X0:** the plan at amendment 1, the estimators, both feature caches and
  both frozen prediction files match their pins; outputs do not exist yet.
- **X2:** BSE-official at 28/96, C_retrieve 1, reproduces the frozen earlier
  replays row for row (P, documents, NLI calls, per-subclaim documents) on
  every dataset, so the data and features are the ones used before.
- **X3:** BSE-eq8 through the generic engine reads exactly the documents of
  ``BSEDetector(mode="eq8")`` on every dataset, at both cost pairs.

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
import score_mddre_features as SF  # noqa: E402
from src import confirmatory_data as C  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src import metrics_v2 as V  # noqa: E402
from src import plan_v2  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

DTYPE = "float16"
V2 = Path("~/ddre-artifacts/v2").expanduser()
PINS = {
    "estimators": (V2 / "estimators_v2_float16.json",
                   "6a007e379a3fc84fabb6d367ede41a3c81c53accc4f7f86b0a29e2ec546fadec"),
    "mddre_features": (SF.MDDRE / "mddre_features_float16.sqlite",
                       "47a0425fcd90944f47331c2dad8900fe8c05d8bd03eb4d4ca0f5d70b7e097353"),
    "confirmatory_features": (C.ARTIFACTS / "confirmatory_features_float16.sqlite",
                              "5ad38eb54a343493aebeb7ec5e07075e006b3e037c158c5110ae21350b95348b"),
    "mddre_manifest": (SF.MDDRE / "mddre_features_float16_manifest.json",
                       "32271a5584f53aa7be385f2da40899a27e14b6c23e8f5aaaed385eaca92518d9"),
    "confirmatory_manifest": (C.ARTIFACTS / "confirmatory_features_float16_manifest.json",
                              "4a562b14b945946bdab3ffe56d722fbc3d13c2309de3be053310a671a2aac354"),
    "mddre_predictions": (SF.MDDRE / "mddre_replay_predictions_float16_amendment1.csv",
                          "c874d05f7bab41dfe0c033eb36ce2fb0f5648a452348e6ac5e30aee22526423a"),
    "confirmatory_predictions": (C.ARTIFACTS / "confirmatory_replay_predictions_float16.csv",
                                 "6c7a7371fd587589e0e6ac951691ef00924c144c4c82a99126a99ffcadfbfcd1"),
}
DATASETS = {  # name: (max documents, feature cache, frozen predictions, passages)
    "selfcheckgpt_heldout": (10, "mddre_features", "mddre_predictions", True),
    "factscore_test": (10, "mddre_features", "mddre_predictions", False),
    "factcheck_bench": (5, "mddre_features", "mddre_predictions", False),
    "expertqa": (5, "confirmatory_features", "confirmatory_predictions", False),
    "bingcheck": (3, "confirmatory_features", "confirmatory_predictions", False),
}
STATUS = ("exploratory: every dataset was used before plan v2 (plan §1); "
          "no confirmatory claim is made from this run")
N_RESAMPLES, SEED = 10_000, 42
FIXED_K_AGGREGATES = ("max", "avg", "nbc")


class V2ReplayRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(V2))
    parser.add_argument("--dry-run", action="store_true",
                        help="X0, data loading and gates X2/X3 only. Nothing written.")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES, help=argparse.SUPPRESS)
    return parser.parse_args()


def output_paths(out_dir):
    return {"result": out_dir / f"v2_replay_result_{DTYPE}.json",
            "predictions": out_dir / f"v2_replay_predictions_{DTYPE}.csv"}


def verify_inputs(out, n_resamples, dry_run):
    problems = list(plan_v2.plan_problems(PROJECT_ROOT, require_amendment_1=True))
    if n_resamples != N_RESAMPLES:
        problems.append(f"n_resamples is fixed at {N_RESAMPLES}")
    for name, (path, sha) in PINS.items():
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
        elif sha256_path(path) != sha:
            problems.append(f"{name}: {path.name} differs from its pinned digest")
    for cache in ("mddre", "confirmatory"):
        manifest, features = PINS[f"{cache}_manifest"][0], PINS[f"{cache}_features"][1]
        if manifest.exists() and (json.loads(manifest.read_text()).get("cache_sha256")
                                  != features):
            problems.append(f"{manifest.name} does not certify the pinned {cache} cache")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; the replay runs once")
    if problems:
        raise V2ReplayRefused("inputs:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- data


def load_sets(wang_root):
    evidence = SF.evidence_rows(SF.FACTSCORE / SF.FACTSCORE_EVIDENCE[0])
    sets, split = RR.test_sets(wang_root, evidence)
    for name in ("expertqa", "bingcheck"):
        units = C.LOADERS[name]()
        C.check_expected(name, units)
        sets[name] = [{"id": u.id, "cluster": u.cluster, "label": u.label,
                       "subclaims": [(s.text, list(s.documents)) for s in u.subclaims]}
                      for u in units]
    return sets, split


def prepare(sentences, reader, max_docs):
    """Documents' features in rank order, at most ``max_docs`` per subclaim."""
    out = []
    for s in sentences:
        subs = []
        for text, pages in s["subclaims"]:
            subs.append([reader.document(text, page, rank)
                         for rank, page in enumerate(pages[:max_docs], start=1)])
        out.append({"id": s["id"], "cluster": s["cluster"], "label": int(s["label"]),
                    "subclaims": subs})
    return out


def load_estimators(record):
    est, sup = {}, {}
    for tier in ("tier1", "tier2"):
        for name, rec in record[tier]["estimators"].items():
            est[name] = E.from_json(rec)
            sup[name] = S.support_from_json(record[tier]["supports"][name])
    return est, sup


# ---------------------------------------------------------------- gates


def cache_dtype(cache):
    """The dtype string the feature cache's keys were written with (its manifest)."""
    return json.loads(PINS[f"{cache.split('_')[0]}_manifest"][0].read_text())["dtype"]


def frozen_official_rows(path):
    with open(path, newline="") as handle:
        return {(r["dataset"], r["id"]): r for r in csv.DictReader(handle)
                if r["cell"] == "A_official" and r.get("analysis", "primary") == "primary"}


def gate_x2(name, prepared, official, frozen_rows):
    """Rows of BSE-official (28/96/1) that differ from the frozen replay."""
    bad = 0
    for j, s in enumerate(prepared):
        row = frozen_rows.get((name, s["id"]))
        if row is None:
            bad += 1
            continue
        same = (str(float(official["p_factual"][j])) == row["p_factual"]
                and int(official["documents"][j]) == int(row["documents"])
                and int(official["nli_calls"][j]) == int(row["nli_calls"])
                and official["subclaim_documents"][j] == json.loads(row["subclaim_documents"]))
        bad += not same
    expected = sum(1 for k in frozen_rows if k[0] == name)
    return bad + abs(expected - len(prepared))


def gate_x3(prepared, histogram, max_docs):
    """Sentences where BSE-eq8 via the engine and via BSEDetector disagree."""
    from src.baseline_core import BSEDetector
    from src.mddre_replay import bse_subclaim

    lrs = R.log_ratios(prepared, histogram)
    bad = 0
    for cm, cfa in R.COST_PAIRS:
        costs = S.Costs(cm, cfa, R.OPERATING_C_RETRIEVE)
        policy = S.Eq8Policy(S.Support.from_histogram(histogram.pos_hist, histogram.neg_hist),
                             costs)
        got = R.run_eq8(prepared, lrs, policy, costs, max_docs)
        bse = BSEDetector(histogram.pos_hist, histogram.neg_hist, mode="eq8", c_miss=cm,
                          c_false_alarm=cfa, c_retrieve=R.OPERATING_C_RETRIEVE)
        for j, s in enumerate(prepared):
            want = [bse_subclaim([histogram.likelihoods(d) for d in sub],
                                 [d["spans"] for d in sub], bse, max_docs)[1]
                    for sub in s["subclaims"]]
            bad += got["subclaim_documents"][j] != want
    return bad


# ---------------------------------------------------------------- run


def run_dataset(prepared, est, sup, policies, eq8_policies, max_docs):
    lrs = {m: R.log_ratios(prepared, est[m]) for m in R.TIER1 + R.TIER2}
    results = {m: {} for m in ("BSE-official",) + R.TIER1 + R.TIER2}
    for cm, cfa, cr in R.settings():
        costs, key = S.Costs(cm, cfa, cr), R.setting_key(cm, cfa, cr)
        results["BSE-official"][key] = R.run_bse_official(prepared, est["BSE"], costs, max_docs)
        for m in R.TIER1 + R.TIER2:
            results[m][key] = R.run_backward_induction(prepared, lrs[m], policies[(m, key)],
                                                       costs, max_docs)
    eq8 = {}
    for cm, cfa in R.COST_PAIRS:
        costs = S.Costs(cm, cfa, R.OPERATING_C_RETRIEVE)
        key = R.setting_key(cm, cfa, R.OPERATING_C_RETRIEVE)
        for m in R.TIER1 + R.TIER2:
            eq8.setdefault(f"{m}-eq8", {})[key] = R.run_eq8(
                prepared, lrs[m], eq8_policies[(m, key)], costs, max_docs)
    fixed = {}
    labels = [s["label"] for s in prepared]
    for k in range(1, max_docs + 1):
        for how in FIXED_K_AGGREGATES:
            r = R.run_fixed_k(prepared, lrs["BSE"] if how == "nbc" else None, k, how, max_docs)
            fixed[f"{how}@{k}"] = {
                "documents_per_sentence": float(r["documents"].mean()),
                "nli_calls_per_sentence": float(r["nli_calls"].mean()),
                "balanced_ap": V.balanced_ap(labels, r["score"]),
                "balanced_pr_auc": V.balanced_wang_pr_auc(labels, r["score"])}
    return R.with_rounded(results), eq8, fixed


def build_policies(sup):
    policies, eq8 = {}, {}
    for cm, cfa, cr in R.settings():
        costs, key = S.Costs(cm, cfa, cr), R.setting_key(cm, cfa, cr)
        for m in R.TIER1 + R.TIER2:
            policies[(m, key)] = S.PolicyByHorizon(sup[m], costs, 10)
            if cr == R.OPERATING_C_RETRIEVE:
                eq8[(m, key)] = S.Eq8Policy(sup[m], costs)
    return policies, eq8


def prediction_rows(name, prepared, by_method):
    rows = []
    for method, by_setting in by_method.items():
        for cm, cfa in R.COST_PAIRS:
            key = R.setting_key(cm, cfa, R.OPERATING_C_RETRIEVE)
            r = by_setting[key]
            for j, s in enumerate(prepared):
                rows.append({"dataset": name, "method": method, "setting": key, "id": s["id"],
                             "cluster": s["cluster"], "gold_label": s["label"],
                             "p_factual": float(r["p_factual"][j]),
                             "prediction": int(r["prediction"][j]),
                             "documents": int(r["documents"][j]),
                             "nli_calls": int(r["nli_calls"][j]),
                             "subclaim_documents": json.dumps(r["subclaim_documents"][j])})
    return rows


def main():
    import numpy as np

    from src.mddre_features import FeatureReader

    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser())
    verify_inputs(out, args.n_resamples, args.dry_run)
    record = json.loads(PINS["estimators"][0].read_text())
    est, sup = load_estimators(record)
    sets, split = load_sets(args.wang_data_root)
    frozen_rows = {k: frozen_official_rows(PINS[k][0])
                   for k in ("mddre_predictions", "confirmatory_predictions")}
    readers = {k: FeatureReader(PINS[k][0], frozen.OFFICIAL_MODEL, cache_dtype(k))
               for k in ("mddre_features", "confirmatory_features")}
    prepared, gates = {}, {}
    try:
        for name, (max_docs, cache, preds, _) in DATASETS.items():
            prepared[name] = prepare(sets[name], readers[cache], max_docs)
            official = R.run_bse_official(prepared[name], est["BSE"],
                                          S.Costs(28, 96, 1), max_docs)
            x2 = gate_x2(name, prepared[name], official, frozen_rows[preds])
            x3 = gate_x3(prepared[name], est["BSE"], max_docs)
            gates[name] = {"X2_official_rows_differing": x2, "X3_eq8_sentences_differing": x3}
            print(f"{name}: {len(prepared[name])} sentences; X2 {x2}, X3 {x3}", flush=True)
            if x2 or x3:
                raise V2ReplayRefused(f"gate failed on {name}: {gates[name]}")
    finally:
        for r in readers.values():
            r.close()
    if args.dry_run:
        print("--dry-run: X0, X2 and X3 passed. Nothing written.")
        return 0

    policies, eq8_policies = build_policies(sup)
    results, rows = {}, []
    for name, (max_docs, _, _, passages) in DATASETS.items():
        data = prepared[name]
        labels = np.array([s["label"] for s in data])
        clusters = [s["cluster"] for s in data]
        res, eq8, fixed = run_dataset(data, est, sup, policies, eq8_policies, max_docs)
        tables = R.observed_tables({**res, **eq8}, labels,
                                   passages=clusters if passages else None)
        observed, draws, n_clusters = V.cluster_bootstrap(
            lambda idx: R.contrast_endpoints(res, labels, idx), clusters,
            args.n_resamples, SEED)
        endpoints = {}
        for k, v in observed.items():
            summary = V.summarise(v, draws[k])
            summary["p_greater_text"] = V.format_p(summary["p_greater"], args.n_resamples)
            summary["p_less_text"] = V.format_p(summary["p_less"], args.n_resamples)
            endpoints[k] = summary
        results[name] = {
            "status": STATUS, "max_documents": max_docs, "sentences": len(data),
            "hallucinated": int(np.sum(labels == 0)), "clusters": n_clusters,
            "gates": gates[name], "tables": tables, "fixed_k": fixed,
            "endpoints": endpoints}
        rows.extend(prediction_rows(name, data, {**res, **eq8}))
        print(f"{name}: done", flush=True)

    result = {
        "artifact": "v2-replay",
        "status": STATUS,
        "protocol": plan_v2.PLAN_PATH + " §3-§5, amendment 1 (Phase 5)",
        "plan_sha256": plan_v2.AMENDMENT_1_SHA256,
        "pins": {k: v[1] for k, v in PINS.items()},
        "factscore_split": split,
        "settings": {"cost_pairs": [list(c) for c in R.COST_PAIRS],
                     "c_retrieve_grid": list(R.C_RETRIEVE_GRID),
                     "operating_c_retrieve": R.OPERATING_C_RETRIEVE,
                     "stopping": "backward induction over each subclaim's available documents"},
        "contrasts": [{"reference": a, "method": b, "role": c} for a, b, c in R.CONTRASTS],
        "bootstrap": {"n_resamples": args.n_resamples, "seed": SEED, "ci_level": 0.95,
                      "unit": "cluster (passage, entity, response or question author)"},
        "policies": {f"{m}|{key}": p.to_json() for (m, key), p in eq8_policies.items()},
        "results": results,
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out["result"].parent.mkdir(parents=True, exist_ok=True)
    out["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
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
    except (V2ReplayRefused, C.ConfirmatoryDataRefused, E.EstimatorError,
            S.SequentialError, R.ReplayError, V.MetricsError) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
