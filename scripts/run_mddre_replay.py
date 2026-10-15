#!/usr/bin/env python3
"""Replay cells A, A', U, M and L on every test set once (design §5-§8 step 5).

For each test set (SelfCheckGPT held-out, FActScore test entities,
Factcheck-Bench):

- cell A runs BSE's official look-ahead with Wang's NBC histogram;
- every cell runs the band rule at BSE's own band (the operating point) and
  at BSE's band scaled in log-odds by 0.25 .. 4.0 (the cost-quality curve,
  design amendment 1);
- one paired cluster bootstrap gives the matched-quality saving, the
  matched-cost gain, the same-band differences and the RQ3 chain;
- §7's thresholds are applied and reported. Under amendment 1, RQ1, RQ2 and
  the RQ3 chain are exploratory on every test set, because the first replay's
  results had been seen.

The first replay's outputs stay frozen; this one writes ``*_amendment1`` files.
Three gates run before any endpoint is reported:

- **G1, engine:** on the SelfCheckGPT held-out set with the frozen float16
  scores, this engine must reproduce the canonical ablation predictions of
  cells A (official look-ahead) and D (frozen DDRE, band) exactly.
- **G2, band:** on every test set, cell A's histogram under BSE's band must
  use exactly the same documents as the official look-ahead.
- **G3, frozen operating point:** the operating point never used the curve, so
  every row of the frozen first-replay predictions must be reproduced exactly.

All inputs are read-only. Results are written once and made read-only.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import complete_ablation_heldout_cache as CA  # noqa: E402
import run_ablation_heldout_replay as RM  # noqa: E402
import run_frozen_heldout as frozen  # noqa: E402
import score_mddre_features as SF  # noqa: E402
from src import ablation_replay as AR  # noqa: E402
from src import mddre_data as M  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src import mddre_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

CELLS = ("A", "A_prime", "U", "M", "L")
CHAIN = (("A", "A_prime"), ("A_prime", "U"), ("U", "M"), ("L", "M"))
ABLATION_PREDICTIONS = ("ablation_heldout_predictions.csv",
                        "ebdcf5c2090cffbabbad81a628e0c04d297d0f85193318aa9c0d5903a50e260b")
DATASETS = {
    "factcheck_bench": {"max_docs": 5,
                        "status": "exploratory under amendment 1 "
                                  "(preregistered as the confirmatory test)"},
    "selfcheckgpt_heldout": {"max_docs": 10,
                             "status": "exploratory under amendment 1 "
                                       "(preregistered as a replication)"},
    "factscore_test": {"max_docs": 10,
                       "status": "exploratory under amendment 1 "
                                  "(preregistered as a replication)"},
}
STATUS = ("amendment 1 (docs/multivariate_ddre_design.md §11): RQ1, RQ2 and the "
          "RQ3 chain exploratory on every test set")
FROZEN_FIRST_REPLAY = {
    "result": "927e0a2d2388efa37147f2d4b4da7f9e533db55f2571275ca4abb23862541544",
    "predictions": "c874d05f7bab41dfe0c033eb36ce2fb0f5648a452348e6ac5e30aee22526423a",
}


class MDDREReplayRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dtype", choices=SF.DTYPES, required=True)
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(SF.MDDRE))
    parser.add_argument("--n-resamples", type=int, default=R.N_RESAMPLES,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def output_paths(out_dir, dtype):
    return {"result": out_dir / f"mddre_replay_result_{dtype}_amendment1.json",
            "predictions": out_dir / f"mddre_replay_predictions_{dtype}_amendment1.csv",
            "frozen_result": out_dir / f"mddre_replay_result_{dtype}.json",
            "frozen_predictions": out_dir / f"mddre_replay_predictions_{dtype}.csv",
            "estimators": out_dir / f"mddre_estimators_{dtype}.json",
            **{f"features_{k}": v for k, v in SF.output_paths(out_dir, dtype).items()}}


def verify_inputs(p, n_resamples):
    problems = []
    if n_resamples != R.N_RESAMPLES:
        problems.append(f"n_resamples is fixed at {R.N_RESAMPLES}")
    problems.extend(M.design_problems(PROJECT_ROOT, require_amendment_1=True))
    fm = json.loads(p["features_manifest"].read_text())
    if fm.get("status") != "complete" or sha256_path(p["features_cache"]) != fm.get("cache_sha256"):
        problems.append("the feature cache is not the complete, manifest-certified one")
    est = json.loads(p["estimators"].read_text())
    if est.get("feature_cache_sha256") != fm.get("cache_sha256") or est.get("test_data_read"):
        problems.append("the estimators were not fit on this feature cache alone")
    name, sha = ABLATION_PREDICTIONS
    if sha256_path(SF.GATE1 / name) != sha:
        problems.append(f"{name} is not the frozen ablation predictions")
    for k, sha in FROZEN_FIRST_REPLAY.items():
        if not p[f"frozen_{k}"].exists() or sha256_path(p[f"frozen_{k}"]) != sha:
            problems.append(f"{p[f'frozen_{k}']} is not the frozen first replay")
    for k in ("result", "predictions"):
        if p[k].exists():
            problems.append(f"{p[k]} already exists; the amended replay runs once")
    if problems:
        raise MDDREReplayRefused("inputs:\n  " + "\n  ".join(problems))
    return fm, est


# ---------------------------------------------------------------- gate G3

G3_FIELDS = ("gold_label", "p_factual", "prediction", "documents", "nli_calls",
             "subclaim_documents")


def frozen_rows(path):
    """{(dataset, cell, id): row} from the frozen first-replay predictions."""
    with open(path, newline="") as handle:
        return {(r["dataset"], r["cell"], r["id"]): r for r in csv.DictReader(handle)}


def gate_frozen_operating_point(frozen, name, rows):
    """Rows of this dataset that differ from the frozen first replay (exact)."""
    want = {k: v for k, v in frozen.items() if k[0] == name}
    got = {(name, r["cell"], r["id"]): r for r in rows}
    if set(want) != set(got):
        return len(set(want) ^ set(got))
    return sum(any(str(got[k][f]) != want[k][f] for f in G3_FIELDS) for k in want)


# ---------------------------------------------------------------- gate G1

def gate_engine(wang_root):
    """This engine reproduces the canonical A and D held-out predictions exactly."""
    from src.baseline_core import discretize_document_score

    p = RM.paths(SF.GATE1)
    for name, (_, sha) in RM.PINNED.items():
        if sha256_path(p[name]) != sha:
            raise MDDREReplayRefused(f"gate G1 input {name} is not the pinned one")
    cells = AR.frozen_cells(json.loads(p["validation_freeze"].read_text()))
    detectors, scorer = RM.build(SimpleNamespace(data_root=wang_root), p, cells)
    canonical = {}
    with open(SF.GATE1 / ABLATION_PREDICTIONS[0], newline="") as handle:
        for row in csv.DictReader(handle):
            canonical[(row["method"], int(row["passage_index"]), int(row["sentence_index"]))] = row
    records, _ = RM.held_out(wang_root, json.loads(p["canonical_result"].read_text()))
    bse, d = detectors["A"], detectors["D"]
    mismatches = {"cell_A": 0, "cell_D": 0}
    try:
        for record in records:
            a, dd = [], []
            for sub in record.subclaims:
                scored = [scorer.score_document(sub.text, doc.page_content)
                          for doc in sub.documents[:10]]
                spans = [n for _, n in scored]
                lik = [bse._bucket_likelihoods(discretize_document_score(s)) for s, _ in scored]
                a.append(R.bse_subclaim(lik, spans, bse, 10))
                lr = [math.log(d.ratio_estimator.ratio(s)) for s, _ in scored]
                dd.append(R.band_subclaim(lr, spans, d.lower_threshold, d.upper_threshold, 10))
            for method, subs in (("cell_A", a), ("cell_D", dd)):
                got = R.sentence_result(subs)
                row = canonical[(method, record.passage_index, record.sentence_index)]
                same = (float(row["p_factual"]) == got["p_factual"]
                        and int(row["retrieved_documents"]) == got["documents"]
                        and int(row["nli_span_calls"]) == got["nli_calls"]
                        and json.loads(row["subclaim_documents_used"]) == got["subclaim_documents"])
                mismatches[method] += not same
    finally:
        scorer.close()
    if any(mismatches.values()):
        raise MDDREReplayRefused(f"gate G1 failed: {mismatches} of {len(records)} sentences")
    return {"sentences": len(records), "mismatches": mismatches, "passed": True}


# ---------------------------------------------------------------- datasets

def test_sets(wang_root, evidence):
    _, test, split = M.factscore_split(SF.FACTSCORE / "data")
    heldout, _ = CA.held_out_records(
        wang_root, json.loads((SF.GATE1 / CA.RESULT_NAME).read_text()))
    sets = {
        "selfcheckgpt_heldout": [
            {"cluster": r.passage_index, "label": int(r.label),
             "id": f"{r.passage_index}:{r.sentence_index}",
             "subclaims": [(s.text, [d.page_content for d in s.documents]) for s in r.subclaims]}
            for r in heldout],
        "factscore_test": [
            {"cluster": s.entity, "label": s.label,
             "id": f"{s.generator}:{s.response_index}:{s.sentence_index}",
             "subclaims": [(f.text, evidence[(s.generator, s.response_index,
                                                s.sentence_index, i)])
                           for i, f in enumerate(s.facts)]}
            for s in test],
        "factcheck_bench": [
            {"cluster": s.response_index, "label": s.label,
             "id": f"{s.response_index}:{s.sentence_key}",
             "subclaims": [(c.text, list(c.documents)) for c in s.claims]}
            for s in M.load_factcheck(SF.FACTCHECK)],
    }
    return sets, split


def nbc_histogram(reader, wang_root):
    from src.baseline_core import build_nbc_histograms
    from src.wang_data import load_nbc_pairs

    class Adapter:
        def score_pairs(self, pairs, **_):
            return [reader.pair(p, h)["entailment"] for p, h in pairs]

    positive, negative = load_nbc_pairs(wang_root, per_class=None)
    pos, neg, _, _ = build_nbc_histograms(positive, negative, Adapter())
    return E.HistogramEvidence(pos, neg)


def replay_dataset(sentences, estimators, bse, band, max_docs, reader):
    import numpy as np

    nb = len(R.BAND_SCALES)
    n = len(sentences)
    curves = {c: {"p_factual": np.zeros((nb, n)), "documents": np.zeros((nb, n)),
                  "nli": np.zeros((nb, n)), "op_p_factual": np.zeros(n),
                  "op_documents": np.zeros(n), "op_nli": np.zeros(n)} for c in CELLS}
    reference = {"p_factual": np.zeros(n), "documents": np.zeros(n), "nli": np.zeros(n)}
    rows, band_mismatch = [], 0
    bands = R.curve_bands(band)
    for j, s in enumerate(sentences):
        docs = [[reader.document(text, page, rank) for rank, page in
                 enumerate(pages[:max_docs], start=1)] for text, pages in s["subclaims"]]
        spans = [[d["spans"] for d in sub] for sub in docs]
        official = R.sentence_result([
            R.bse_subclaim([estimators["A"].likelihoods(d) for d in sub], sp, bse, max_docs)
            for sub, sp in zip(docs, spans)])
        reference["p_factual"][j] = official["p_factual"]
        reference["documents"][j] = official["documents"]
        reference["nli"][j] = official["nli_calls"]
        rows.append({"dataset": None, "cell": "A_official", "id": s["id"],
                     "cluster": s["cluster"], "gold_label": s["label"], **official})
        for c in CELLS:
            lrs = [[estimators[c].log_ratio(d) for d in sub] for sub in docs]
            for b, (lo, hi) in enumerate(bands):
                r = R.sentence_result([R.band_subclaim(lr, sp, lo, hi, max_docs)
                                       for lr, sp in zip(lrs, spans)])
                curves[c]["p_factual"][b, j] = r["p_factual"]
                curves[c]["documents"][b, j] = r["documents"]
                curves[c]["nli"][b, j] = r["nli_calls"]
            op = R.sentence_result([R.band_subclaim(lr, sp, band[0], band[1], max_docs)
                                    for lr, sp in zip(lrs, spans)])
            curves[c]["op_p_factual"][j] = op["p_factual"]
            curves[c]["op_documents"][j] = op["documents"]
            curves[c]["op_nli"][j] = op["nli_calls"]
            rows.append({"dataset": None, "cell": c, "id": s["id"], "cluster": s["cluster"],
                         "gold_label": s["label"], **op})
            if c == "A" and op["subclaim_documents"] != official["subclaim_documents"]:
                band_mismatch += 1
    return curves, reference, rows, band_mismatch


def main():
    from src.baseline_core import BSEDetector
    from src.evaluation import wang_pr_auc
    from src.mddre_features import FeatureReader

    args = parse_args()
    p = output_paths(Path(args.out_dir).expanduser(), args.dtype)
    fm, est = verify_inputs(p, args.n_resamples)
    g1 = gate_engine(args.wang_data_root)
    print(f"gate G1 (engine): {g1}")

    evidence = SF.evidence_rows(SF.FACTSCORE / SF.FACTSCORE_EVIDENCE[0])
    sets, split = test_sets(args.wang_data_root, evidence)
    reader = FeatureReader(p["features_cache"], frozen.OFFICIAL_MODEL, fm["dtype"])
    try:
        estimators = {name: E.from_json(est["estimators"][name])
                      for name in ("A_prime", "U", "M", "L")}
        estimators["A"] = nbc_histogram(reader, args.wang_data_root)
        results, all_rows = {}, []
        frozen_ops = frozen_rows(p["frozen_predictions"])
        for name, cfg in DATASETS.items():
            bse = BSEDetector(estimators["A"].pos_hist, estimators["A"].neg_hist,
                              mode="official", p0=R.P0, c_miss=R.C_MISS,
                              c_false_alarm=R.C_FALSE_ALARM, c_retrieve=1,
                              max_docs=cfg["max_docs"])
            band = R.bse_band(bse)
            curves, reference, rows, band_mismatch = replay_dataset(
                sets[name], estimators, bse, band, cfg["max_docs"], reader)
            if band_mismatch:
                raise MDDREReplayRefused(f"gate G2 failed on {name}: {band_mismatch} sentences")
            g3 = gate_frozen_operating_point(frozen_ops, name, rows)
            if g3:
                raise MDDREReplayRefused(f"gate G3 failed on {name}: {g3} rows differ "
                                         "from the frozen first replay")
            labels = [s["label"] for s in sets[name]]
            clusters = [s["cluster"] for s in sets[name]]
            boot = R.bootstrap(curves, reference, CELLS, CHAIN, labels, clusters, wang_pr_auc)
            results[name] = {"status": cfg["status"], "max_docs": cfg["max_docs"],
                             "sentences": len(labels), "band": band,
                             "gate_G2_band_mismatches": 0,
                             "gate_G3_frozen_row_mismatches": 0, "bootstrap": boot,
                             "decision": R.decide(boot["endpoints"])}
            for r in rows:
                r["dataset"] = name
            all_rows.extend(rows)
            print(f"{name}: done ({len(labels)} sentences, {boot['clusters']} clusters)")
    finally:
        reader.close()

    AR.check_language(json.dumps({k: v["decision"] for k, v in results.items()}))
    result = {
        "artifact": "mddre-replay",
        "status": STATUS,
        "protocol": M.DESIGN_PATH,
        "design_sha256": M.DESIGN_SHA256,
        "dtype": fm["dtype"],
        "feature_cache_sha256": fm["cache_sha256"],
        "estimators_sha256": sha256_path(p["estimators"]),
        "split": split,
        "gates": {"G1_engine": g1, "G2_band": "passed on every test set",
                  "G3_frozen_operating_point": "passed on every test set"},
        "band_scales": list(R.BAND_SCALES),
        "amendment": {"number": 1, "design_sha256": M.AMENDMENT_1_SHA256,
                      "test_results_seen_before": True,
                      "supersedes_curve_endpoints_of": FROZEN_FIRST_REPLAY},
        "cells": list(CELLS),
        "chain": [list(c) for c in CHAIN],
        "results": results,
        "code_commit": frozen._git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    p["result"].write_text(json.dumps(result, indent=1, default=float) + "\n")
    with p["predictions"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    for k in ("result", "predictions"):
        p[k].chmod(0o444)
        print(f"{k} {p[k]}\n  sha256 {sha256_path(p[k])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MDDREReplayRefused, M.MDDREDataRefused, AR.ReplayRefused) as exc:
        print(f"\nREFUSED: {exc}")
        sys.exit(1)
