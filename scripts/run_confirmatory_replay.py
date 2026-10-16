#!/usr/bin/env python3
"""The confirmatory replay, run once (``docs/confirmatory_preregistration.md``
§3-§8 step 4).

On ExpertQA (primary) and BingCheck, cells A (BSE official, Wang's NBC
histogram), A', U, M and L are replayed with the frozen estimators, under
BSE's own band and along the Amendment-1 curve. One paired cluster bootstrap
per dataset gives E1 (net matched-quality saving), E2 (matched-cost gain), the
same-band differences and the RQ3 chain. §5's Holm rule over the four primary
tests gives the conclusions. §6's sensitivity analyses and the ExpertQA
per-system breakdown are descriptive.

Gates, any failure stops the run before anything is written:

- **P:** preregistration, design, estimators and feature cache match their pins;
  the feature manifest is complete with zero equality mismatches.
- **G0:** both loaders reproduce the preregistered totals.
- **G1:** the engine reproduces the canonical SelfCheckGPT ablation predictions
  of cells A and D.
- **G2:** on each dataset, cell A under BSE's band uses exactly BSE's official
  documents.

Inputs are read-only. The result and predictions are written once, read-only.
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
import score_confirmatory_features as SC  # noqa: E402
from src import ablation_replay as AR  # noqa: E402
from src import confirmatory_data as C  # noqa: E402
from src import confirmatory_replay as CR  # noqa: E402
from src import mddre_data as M  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src import mddre_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

CELLS = RR.CELLS
CHAIN = RR.CHAIN
DATASETS = ("expertqa", "bingcheck")
SENSITIVITY = (("expertqa", "unsure_hallucinated"),
               ("bingcheck", "refuted_only"),
               ("bingcheck", "not_supported_hallucinated"))


class ConfirmatoryReplayRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(C.ARTIFACTS))
    parser.add_argument("--n-resamples", type=int, default=R.N_RESAMPLES,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def output_paths(out_dir):
    features = SC.output_paths(out_dir)
    return {"result": out_dir / f"confirmatory_replay_result_{SC.DTYPE}.json",
            "predictions": out_dir / f"confirmatory_replay_predictions_{SC.DTYPE}.csv",
            "features_cache": features["cache"], "features_manifest": features["manifest"]}


def verify_inputs(p, n_resamples):
    problems = list(C.prereg_problems(PROJECT_ROOT))
    problems.extend(M.design_problems(PROJECT_ROOT, require_amendment_1=True))
    if n_resamples != R.N_RESAMPLES:
        problems.append(f"n_resamples is fixed at {R.N_RESAMPLES}")
    if sha256_path(SC.ESTIMATORS[0]) != SC.ESTIMATORS[1]:
        problems.append("the estimators differ from their pinned digest")
    fm = json.loads(p["features_manifest"].read_text())
    eq = fm.get("equality", {})
    if (fm.get("status") != "complete"
            or sha256_path(p["features_cache"]) != fm.get("cache_sha256")
            or fm.get("preregistration_sha256") != C.PREREG_AMENDMENT_1_SHA256
            or eq.get("reference_mismatches") or eq.get("feature_cache_mismatches")
            or eq.get("nbc_exact") != 396):
        problems.append("the feature cache is not the complete, manifest-certified one")
    for k in ("result", "predictions"):
        if p[k].exists():
            problems.append(f"{p[k]} already exists; the replay runs once")
    if problems:
        raise ConfirmatoryReplayRefused("inputs:\n  " + "\n  ".join(problems))
    return fm, json.loads(SC.ESTIMATORS[0].read_text())


def sentences(units):
    return [{"id": u.id, "cluster": u.cluster, "label": u.label,
             "subclaims": [(s.text, list(s.documents)) for s in u.subclaims]}
            for u in units]


def expertqa_systems():
    """{row index: answering system}, read after the loader's hash check."""
    with open(C.EXPERTQA_DIR / C.EXPERTQA_FILE) as handle:
        return {i: next(iter(json.loads(line)["answers"])) for i, line in enumerate(handle)}


def replay(name, units, estimators, reader):
    from src.baseline_core import BSEDetector

    max_docs = C.MAX_DOCS[name]
    bse = BSEDetector(estimators["A"].pos_hist, estimators["A"].neg_hist, mode="official",
                      p0=R.P0, c_miss=R.C_MISS, c_false_alarm=R.C_FALSE_ALARM,
                      c_retrieve=1, max_docs=max_docs)
    band = R.bse_band(bse)
    curves, reference, rows, band_mismatch = RR.replay_dataset(
        sentences(units), estimators, bse, band, max_docs, reader)
    if band_mismatch:
        raise ConfirmatoryReplayRefused(f"gate G2 failed on {name}: {band_mismatch} units")
    return band, curves, reference, rows


def subset(curves, reference, idx):
    import numpy as np

    idx = np.asarray(idx)
    sub = {c: {k: (v[:, idx] if v.ndim == 2 else v[idx]) for k, v in d.items()}
           for c, d in curves.items()}
    return sub, {k: v[idx] for k, v in reference.items()}


def main():
    import numpy as np

    from src.evaluation import wang_pr_auc
    from src.mddre_features import FeatureReader

    args = parse_args()
    p = output_paths(Path(args.out_dir).expanduser())
    fm, est = verify_inputs(p, args.n_resamples)
    gate_g0 = {name: C.check_expected(name, C.LOADERS[name]()) for name in DATASETS}
    print(f"gate G0: {gate_g0}")
    try:
        g1 = RR.gate_engine(args.wang_data_root)
    except RR.MDDREReplayRefused as exc:
        raise ConfirmatoryReplayRefused(str(exc)) from exc
    print(f"gate G1 (engine): {g1}")

    reader = FeatureReader(p["features_cache"], frozen.OFFICIAL_MODEL, fm["dtype"])
    results, sensitivity, by_system, all_rows = {}, {}, {}, []
    try:
        estimators = {name: E.from_json(est["estimators"][name])
                      for name in ("A_prime", "U", "M", "L")}
        estimators["A"] = RR.nbc_histogram(reader, args.wang_data_root)
        for name in DATASETS:
            units = C.LOADERS[name]()
            band, curves, reference, rows = replay(name, units, estimators, reader)
            labels = np.array([u.label for u in units])
            clusters = [u.cluster for u in units]
            boot = CR.bootstrap(curves, reference, CELLS, labels, clusters, wang_pr_auc,
                                chain=CHAIN, n_resamples=args.n_resamples)
            results[name] = {"status": "confirmatory (preregistered)",
                             "max_docs": C.MAX_DOCS[name], "units": len(units),
                             "hallucinated": int(np.sum(labels == 0)), "band": band,
                             "gate_G2_band_mismatches": 0, "bootstrap": boot}
            for r in rows:
                r["dataset"], r["analysis"] = name, "primary"
            all_rows.extend(rows)
            print(f"{name}: primary done ({len(units)} units, {boot['clusters']} clusters)")
            if name == "expertqa":
                systems = expertqa_systems()
                groups = {}
                for i, u in enumerate(units):
                    groups.setdefault(systems[int(u.id.split(":")[0])], []).append(i)
                for system, idx in sorted(groups.items()):
                    sub, ref = subset(curves, reference, idx)
                    obs, _ = CR.endpoints(sub, ref, CELLS, labels[idx], wang_pr_auc,
                                          np.arange(len(idx)), CHAIN)
                    by_system[system] = {"units": len(idx),
                                         "hallucinated": int(np.sum(labels[idx] == 0)),
                                         "observed": obs}

        for name, mapping in SENSITIVITY:
            units = C.LOADERS[name](mapping=mapping)
            _, curves, reference, _ = replay(name, units, estimators, reader)
            labels = np.array([u.label for u in units])
            boot = CR.bootstrap(curves, reference, CELLS, labels, [u.cluster for u in units],
                                wang_pr_auc, chain=CHAIN, n_resamples=args.n_resamples)
            sensitivity[f"{name}.{mapping}"] = {
                "status": "descriptive sensitivity analysis (§6)", "units": len(units),
                "hallucinated": int(np.sum(labels == 0)), "bootstrap": boot}
            print(f"{name}.{mapping}: done ({len(units)} units)")
    finally:
        reader.close()

    decision = CR.decide({name: results[name]["bootstrap"]["endpoints"] for name in DATASETS})
    AR.check_language(json.dumps(decision))
    result = {
        "artifact": "confirmatory-replay",
        "status": "preregistered confirmatory test",
        "protocol": C.PREREG_PATH,
        "preregistration_sha256": C.PREREG_AMENDMENT_1_SHA256,
        "dtype": fm["dtype"],
        "feature_cache_sha256": fm["cache_sha256"],
        "estimators_sha256": SC.ESTIMATORS[1],
        "gates": {"P": "passed", "G0": gate_g0, "G1_engine": g1,
                  "G2_band": "passed on every dataset and analysis"},
        "band_scales": list(R.BAND_SCALES),
        "cells": list(CELLS),
        "chain": [list(c) for c in CHAIN],
        "decision": decision,
        "results": results,
        "sensitivity": sensitivity,
        "expertqa_by_system": by_system,
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    p["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
    with p["predictions"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "analysis"] + [
            k for k in all_rows[0] if k not in ("dataset", "analysis")])
        writer.writeheader()
        writer.writerows(all_rows)
    for k in ("result", "predictions"):
        p[k].chmod(0o444)
        print(f"{k} {p[k]}\n  sha256 {sha256_path(p[k])}")
    print(json.dumps(decision["statements"], indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConfirmatoryReplayRefused, C.ConfirmatoryDataRefused, AR.ReplayRefused) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
