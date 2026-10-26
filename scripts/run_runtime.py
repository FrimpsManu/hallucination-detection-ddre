#!/usr/bin/env python3
"""Running time on HELM, run once (docs/runtime_protocol.md).

Times BSE and DDRE-1D live, document by document, on 200 seeded HELM
sentences in eight cells (protocol §4), and times each method's offline fit
(§8). Checks, all before anything is reported (§7):

- **V1:** every input matches its pin; the outputs do not exist yet; the
  working tree is clean (not for ``--dry-run``).
- **V3, from the feature cache, before the model loads:** a replay of every
  cell through the HELM test's own code reproduces the frozen predictions
  (default cells) and the frozen curve means over all 3,736 sentences.
- **V2 and V3, live:** every scored span's entailment equals the HELM feature
  cache bit for bit, and every live trajectory equals the cached replay.

The first failed check stops the run and is written down as a failed result.
``--dry-run`` runs V1 (without the output and clean-tree checks) and the
cached V3, then stops: no model is loaded and nothing is written.
"""

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_frozen_heldout as frozen  # noqa: E402
import run_helm_confirmatory as HC  # noqa: E402
import run_v2_replay as RV  # noqa: E402
from src import helm_data as H  # noqa: E402
from src import runtime as T  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.factscore_data import sha256_path  # noqa: E402

PINS = {
    "estimators": RV.PINS["estimators"],
    "helm_evidence": HC.PINS["helm_evidence"],
    "helm_features": HC.PINS["helm_features"],
    "helm_result": (H.ROOT / "helm_confirmatory_result_float16.json",
                    "10bbd7306d3710fb4da076d94ebd98df20f624ff7d974a245c49e0639e448151"),
    "helm_predictions": (H.ROOT / "helm_confirmatory_predictions_float16.csv",
                         "994d85f223b45eaa0afe5740ea5cd98e72f8193d05c50266935dd79302d7a8e8"),
}
OUT_DIR = Path("~/ddre-artifacts/runtime").expanduser()
FIT_REPEATS = 5
CURVE_TOLERANCE = 1e-12


class RuntimeRefused(RuntimeError):
    pass


class CheckFailed(RuntimeError):
    def __init__(self, check, detail):
        super().__init__(f"{check}: {detail}")
        self.check, self.detail = check, detail


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_paths(out_dir):
    return {"result": out_dir / "runtime_result.json",
            "timings": out_dir / "runtime_timings.csv"}


def git_state():
    def git(*a):
        return subprocess.check_output(["git", *a], cwd=str(PROJECT_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def power_source():
    try:
        out = subprocess.check_output(["pmset", "-g", "ps"], text=True)
    except Exception:
        return None
    m = re.search(r"Now drawing from '([^']+)'", out)
    return m.group(1) if m else None


def verify_inputs(out, dry_run):
    problems = []
    if not (PROJECT_ROOT / T.PROTOCOL_PATH).exists():
        problems.append(f"{T.PROTOCOL_PATH} is missing")
    for name, (path, sha) in PINS.items():
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
        elif sha256_path(path) != sha:
            problems.append(f"{name}: {path.name} differs from its pinned digest")
    if not dry_run:
        for path in out.values():
            if path.exists():
                problems.append(f"{path} already exists; the measurement runs once")
        if git_state()["dirty"] is not False:
            problems.append("the working tree is not clean")
        if power_source() != "AC Power":
            problems.append(f"power source is {power_source()!r}; protocol §6 needs mains power")
    if problems:
        raise RuntimeRefused("V1:\n  " + "\n  ".join(problems))
    return json.loads(HC.HELM_FEATURES_MANIFEST.read_text())


# ---------------------------------------------------------------- cached replay (V3)


def frozen_default_rows(path):
    with open(path, newline="") as handle:
        return {(r["method"], r["setting"], r["id"]): r for r in csv.DictReader(handle)}


def cached_replay(prepared, est, pol):
    """Every cell over all sentences, from the feature cache: the HELM test's code."""
    lrs = {m: R.log_ratios(prepared, est[m]) for m in ("BSE", "DDRE-1D")}
    out = {}
    for cell, cm, cfa, method, k, cr in T.CELLS:
        out[cell] = R.run_backward_induction(prepared, lrs[method], pol[cell],
                                             S.Costs(cm, cfa, cr), k)
    return out


def check_cached(replay, units, idx, result, frozen_rows):
    """V3 before the model: curve means over all sentences, default cells per sentence."""
    for cell, _, _, method, _, _ in T.CELLS:
        pair, point = T.setting_key(cell)
        want = result["curves"][f"{method}|{pair}"][point]
        for field, key in (("nli_calls", "nli_calls_per_sentence"),
                           ("documents", "documents_per_sentence")):
            got = float(replay[cell][field].mean())
            if abs(got - want[key]) > CURVE_TOLERANCE:
                raise CheckFailed("V3", f"{cell} {field}: cached replay {got} vs frozen "
                                        f"curve {want[key]}")
        if cell.startswith("D"):
            for j in idx:
                row = frozen_rows[(method, pair, units[j].id)]
                got = (int(replay[cell]["documents"][j]), int(replay[cell]["nli_calls"][j]),
                       float(replay[cell]["p_factual"][j]))
                want_row = (int(row["documents"]), int(row["nli_calls"]),
                            float(row["p_factual"]))
                if got != want_row:
                    raise CheckFailed("V3", f"{cell} {units[j].id}: cached replay {got} vs "
                                            f"frozen predictions {want_row}")


# ---------------------------------------------------------------- live


def load_live(dtype):
    import score_mddre_features as SF

    start = time.perf_counter()
    tokenizer, model, device, loaded = SF.load_model(dtype)
    return tokenizer, model, device, loaded, time.perf_counter() - start


def live_scorer(tokenizer, model, loaded):
    """A fresh, empty in-memory cache per sentence and cell (protocol §5)."""
    import torch

    from src.feature_scorer import FeatureScorer, document_features
    from src.utils import split_text

    scorer = FeatureScorer(tokenizer, model, frozen.OFFICIAL_MODEL, dtype=loaded,
                           cache_path=":memory:", batch_size=1)

    def score(claim, text):
        torch.mps.synchronize()
        segments = split_text(text, segment_length=400, overlap_length=100)
        rows = scorer.score_pairs([(s, claim) for s in segments])
        torch.mps.synchronize()
        return document_features(rows), [(s, r["entailment"]) for s, r in zip(segments, rows)]

    return scorer, score


def check_live(live, j, replay, reader, claim, unit_id):
    """V2 (spans equal the cache bit for bit) and V3 (trajectory equals the replay)."""
    for rows in live["spans"]:
        for segment, e in rows:
            if reader.pair(segment, claim)["entailment"] != e:
                raise CheckFailed("V2", f"{live['cell']} {unit_id}: live entailment {e} vs "
                                        f"cache {reader.pair(segment, claim)['entailment']}")
    r = replay[live["cell"]]
    want = (int(r["documents"][j]), int(r["nli_calls"][j]), float(r["p_factual"][j]))
    got = (live["documents"], live["nli_calls"], live["p_factual"])
    if got != want:
        raise CheckFailed("V3", f"{live['cell']} {unit_id}: live {got} vs cached replay {want}")


def warm_up(score, wang_root):
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(wang_root, per_class=None)
    for x in (positive + negative)[:T.WARMUP_CALLS]:
        score(x["hypothesis"], x["premise"])


def timed_passes(units, idx, texts, est, pol, replay, reader, tokenizer, model, loaded,
                 wang_root):
    scorer, score = live_scorer(tokenizer, model, loaded)
    try:
        warm_up(score, wang_root)
    finally:
        scorer.close()
    rows = []
    for n, seed in enumerate(T.PASS_SEEDS, start=1):
        order, cell_orders = T.pass_order(len(idx), seed)
        started = time.perf_counter()
        for pos, cells in zip(order, cell_orders):
            j = idx[pos]
            u = units[j]
            for cell in cells:
                method = T.CELL[cell][3]
                scorer, score = live_scorer(tokenizer, model, loaded)
                try:
                    live = T.run_live(u.sentence, texts[j], cell, est[method], pol[cell],
                                      score, time.perf_counter)
                finally:
                    scorer.close()
                check_live(live, j, replay, reader, u.sentence, u.id)
                rows.append({"pass": n, "sentence": u.id, "cluster": u.cluster,
                             **{k: v for k, v in live.items() if k != "spans"}})
        print(f"pass {n}/{len(T.PASS_SEEDS)}: {time.perf_counter() - started:.0f} s",
              flush=True)
    return rows


# ---------------------------------------------------------------- offline fit (§8)


def fit_times(reader, wang_root, record, sup_stored):
    import numpy as np

    from src import mddre_estimators as E
    from src.wang_data import load_nbc_pairs

    docs = {}
    for name, pairs in zip(("factual", "hallucinated"),
                           load_nbc_pairs(wang_root, per_class=None)):
        rows = [reader.pair(x["premise"], x["hypothesis"]) for x in pairs]
        docs[name] = ([{"e": r["entailment"], "k": r["contradiction"], "rank": 1}
                       for r in rows], [x["hypothesis"] for x in pairs])
    (f, fg), (h, hg) = docs["factual"], docs["hallucinated"]

    from src.baseline_core import discretize_nbc_score

    def bse():
        # build_nbc_histograms' binning (Wang's NBC bins, +1 Laplace), on scores
        # already read, so neither fit is timed reading the cache.
        pos, neg = [1] * 10, [1] * 10
        for d in f:
            pos[discretize_nbc_score(d["e"])] += 1
        for d in h:
            neg[discretize_nbc_score(d["e"])] += 1
        est = E.HistogramEvidence(pos, neg)
        return est, S.Support.from_histogram(est.pos_hist, est.neg_hist)

    def ddre():
        est = E.MultiULSIF(("e",)).fit(f, h, sigma_multipliers=E.SIGMA_MULTIPLIERS_V2,
                                       factual_groups=fg, hallucinated_groups=hg)
        return est, S.build_support(est, h)

    out, fitted = {}, {}
    for name, fn in (("BSE", bse), ("DDRE-1D", ddre)):
        times = []
        for _ in range(FIT_REPEATS):
            a = time.perf_counter()
            fitted[name] = fn()
            times.append(time.perf_counter() - a)
        out[name] = {"estimator_and_support_seconds": float(np.median(times))}
    stored = record["tier1"]["estimators"]
    if [fitted["BSE"][0].pos_hist, fitted["BSE"][0].neg_hist] != \
            [stored["BSE"]["pos_hist"], stored["BSE"]["neg_hist"]]:
        raise CheckFailed("fit", "the refitted BSE histogram differs from the stored one")
    est_stored, _ = RV.load_estimators(record)
    grid = [i / 4 for i in range(401)]
    worst = max(abs(fitted["DDRE-1D"][0].log_ratio({"e": e, "k": 0.0, "rank": 1})
                    - est_stored["DDRE-1D"].log_ratio({"e": e, "k": 0.0, "rank": 1}))
                for e in grid)
    if worst > 1e-9:
        raise CheckFailed("fit", f"the refitted DDRE-1D differs from the stored one by {worst}")
    for cell, *_ in T.CELLS:
        method = T.CELL[cell][3]
        times = []
        for _ in range(FIT_REPEATS):
            a = time.perf_counter()
            T.policies({method: sup_stored[method]}, cells=(cell,))
            times.append(time.perf_counter() - a)
        out.setdefault(method, {}).setdefault("policy_seconds", {})[cell] = \
            float(np.median(times))
    out["ddre_refit_max_abs_log_ratio_difference"] = worst
    return out


# ---------------------------------------------------------------- output


def versions():
    import numpy
    import torch
    import transformers

    return {"power_source": power_source(), "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(), "numpy": numpy.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__}


def write_outputs(out, result, rows):
    out["result"].parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = ["pass", "sentence", "cluster", "cell", "documents", "nli_calls",
                  "p_factual", "prediction", "total", *T.TIMERS]
        with open(out["timings"], "w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        out["timings"].chmod(0o444)
    out["result"].write_text(json.dumps(result, indent=1) + "\n")
    out["result"].chmod(0o444)


def main():
    from src.mddre_features import FeatureReader

    args = parse_args()
    out = output_paths(Path(args.out_dir).expanduser())
    fm = verify_inputs(out, args.dry_run)
    units, titles = H.load_helm()
    H.check_expected(units, titles)
    if len(units) != T.N_UNITS:
        raise RuntimeRefused(f"{len(units)} HELM units, expected {T.N_UNITS}")
    idx = T.subset()
    record = json.loads(PINS["estimators"][0].read_text())
    est, sup = RV.load_estimators(record)
    pol = T.policies(sup)
    result_frozen = json.loads(PINS["helm_result"][0].read_text())
    evidence = [json.loads(line) for line in open(PINS["helm_evidence"][0], encoding="utf-8")]
    if [r["id"] for r in evidence] != [u.id for u in units]:
        raise RuntimeRefused("the evidence does not hold one list per unit in order")
    texts = {j: [d["text"] for d in evidence[j]["documents"][:HC.MAX_DOCS]] for j in idx}

    reader = FeatureReader(PINS["helm_features"][0], frozen.OFFICIAL_MODEL, fm["dtype"])
    base = {"artifact": "helm-runtime", "protocol": T.PROTOCOL_PATH,
            "pins": {k: v[1] for k, v in PINS.items()}, "code": git_state(),
            "subset": {"seed": T.SUBSET_SEED, "ids": [units[j].id for j in idx]},
            "cells": [dict(zip(("cell", "c_miss", "c_false_alarm", "method", "k",
                                "c_retrieve"), c)) for c in T.CELLS]}
    rows = []
    try:
        prepared = HC.prepare_helm(units, PINS["helm_evidence"][0], reader)
        replay = cached_replay(prepared, est, pol)
        check_cached(replay, units, idx, result_frozen,
                     frozen_default_rows(PINS["helm_predictions"][0]))
        expected = {c[0]: float(replay[c[0]]["nli_calls"][idx].mean()) for c in T.CELLS}
        print(f"V1 and cached V3 passed; {len(idx)} sentences, "
              f"{len({units[j].cluster for j in idx})} articles; expected NLI calls per "
              f"sentence on the subset: {expected}", flush=True)
        if args.dry_run:
            print("--dry-run: no model loaded, nothing written.")
            return 0

        started = datetime.now(timezone.utc).isoformat()
        tokenizer, model, device, loaded, load_seconds = load_live("float16")
        if loaded != fm["dtype"] or device != fm["device"]:
            raise CheckFailed("V2", f"model loaded as {loaded} on {device}; the cache was "
                                    f"scored as {fm['dtype']} on {fm['device']}")
        rows = timed_passes(units, idx, texts, est, pol, replay, reader, tokenizer, model,
                            loaded, args.wang_data_root)
        fit = fit_times(reader, args.wang_data_root, record, sup)
    except CheckFailed as failure:
        write_outputs(out, {**base, "status": "failed", "failed_check": failure.check,
                            "detail": failure.detail,
                            "created_utc": datetime.now(timezone.utc).isoformat()}, rows)
        print(f"FAILED {failure}; written to {out['result']}")
        return 1
    finally:
        reader.close()

    med = T.medians(rows)
    ids = [units[j].id for j in idx]
    clusters = [units[j].cluster for j in idx]
    result = {**base, "status": "complete", "started_utc": started,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "machine": versions(), "device": device, "dtype": loaded,
              "model_load_seconds": load_seconds,
              "checks": {"V1": "passed", "V2": "passed", "V3": "passed"},
              "cells_summary": T.cell_summary(med, ids),
              "contrasts": T.contrasts(med, ids, clusters),
              "fit": fit}
    write_outputs(out, result, rows)
    print(json.dumps({"contrasts": result["contrasts"]}, indent=1))
    print(f"wrote {out['result']} and {out['timings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
