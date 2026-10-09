#!/usr/bin/env python3
"""Descriptive, post-hoc analysis of the completed frozen held-out run.

Replays BSE official and the frozen DDRE from the read-only held-out NLI cache,
refuses to analyse anything unless the replay reproduces the canonical
predictions CSV exactly, then reports confusion/recall/stopping breakdowns and
per-document evidence trajectories.

What this script cannot do, by construction:

* load a model or run NLI inference -- every score is served by
  ``CachedDocumentScorer`` over SQLite opened ``mode=ro``, and a missing score
  raises instead of being computed;
* select, tune or change anything -- the detectors are rebuilt from the runner's
  frozen literals and the canonical freeze artifact;
* modify an input artifact -- every input is hashed before and after, and the
  outputs are new files that may not collide with an input.

Everything it reports is descriptive / post-hoc / exploratory: the held-out set
had already been evaluated when this analysis was designed.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

import run_frozen_heldout as frozen  # noqa: E402
from src import heldout_analysis as A  # noqa: E402

# Canonical frozen held-out artifacts (runner commit fae3eee). Pinned: the
# analysis describes these files and no others.
RESULT_SHA256 = "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67"
PREDICTIONS_SHA256 = "a78b09b7aa09803a21a761b3bede7600180014b97142a381b509d7a1267211c1"
D03_REPORT_SHA256 = "f2468e5bb1265e107f4060d36782f6f21ba9f5f55141dde4814d5ffd11aa7f6d"

RESULT_NAME = "frozen_heldout_result_fae3eee.json"
PREDICTIONS_NAME = "frozen_heldout_predictions_fae3eee.csv"
FREEZE_NAME = "ddre_validation_freeze.json"
HELD_OUT_CACHE_NAME = "frozen_heldout_nli_cache.sqlite"
D03_CACHE_NAME = "d03_diagnostic_cache_main_2661b04.sqlite"
D03_REPORT_NAME = "d03_density_ratio_diagnostic_main_2661b04.json"

METHODS = ("bse_official", "frozen_ddre")
MODAL_SCORES = (30.0, 33.0, 35.0, 37.0, 40.0)
EVIDENCE_GRID = np.round(np.arange(0.0, 100.0001, 0.05), 4)
EVIDENCE_BINS = (0, 1, 5, 10, 20, 40, 60, 80, 90, 95, 98, 100.0001)
NBC_BINS = (0, 20, 30, 40, 50, 60, 100.0001)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts-dir", default=str(frozen.ARTIFACTS))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write the analysis outputs (default: the artifacts dir).",
    )
    parser.add_argument(
        "--write-traces", action="store_true",
        help="Also write the full per-document trajectories of the flipped sentences.",
    )
    return parser.parse_args()


def input_paths(artifacts):
    return {
        "result": artifacts / RESULT_NAME,
        "predictions": artifacts / PREDICTIONS_NAME,
        "freeze": artifacts / FREEZE_NAME,
        "held_out_cache": artifacts / HELD_OUT_CACHE_NAME,
        "d03_cache": artifacts / D03_CACHE_NAME,
        "d03_report": artifacts / D03_REPORT_NAME,
    }


def pinned_digests():
    return {
        "result": RESULT_SHA256,
        "predictions": PREDICTIONS_SHA256,
        "freeze": frozen.FREEZE_SHA256,
        "d03_cache": frozen.D03_CACHE_SHA256,
        "d03_report": D03_REPORT_SHA256,
    }


def check_result_cross_references(result):
    """The result artifact must name the same freeze, cache and configuration."""
    problems = []
    expect = {
        "freeze_artifact_sha256": frozen.FREEZE_SHA256,
        "frozen_sigma": frozen.FROZEN_SIGMA,
        "frozen_lambda": frozen.FROZEN_LAMBDA,
        "frozen_lower": frozen.FROZEN_LOWER,
        "frozen_upper": frozen.FROZEN_UPPER,
    }
    for field, want in expect.items():
        if result.get(field) != want:
            problems.append(f"result.{field} = {result.get(field)!r}, expected {want!r}")
    if (result.get("d03_cache") or {}).get("d03_cache_sha256") != frozen.D03_CACHE_SHA256:
        problems.append("result.d03_cache.d03_cache_sha256 does not match the D-03 cache")
    if (result.get("checkpoint") or {}).get("d03_report_sha256") != D03_REPORT_SHA256:
        problems.append("result.checkpoint.d03_report_sha256 does not match the D-03 report")
    if problems:
        raise A.ArtifactDigestMismatch("\n  ".join(["result cross-references:"] + problems))


def build_detectors(args, paths):
    """Rebuild both detectors exactly as the runner did, from cached scores only."""
    from src.baseline_core import BSEDetector, build_nbc_histograms
    from src.cached_document_scores import CachedDocumentScorer
    from src.ddre_core import DDREDetector, ULSIFDensityRatio
    from src.hyperparameter_outcome_sensitivity import (
        candidate_estimator,
        verify_production_fit,
    )
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(args.data_root, per_class=None)
    nbc = CachedDocumentScorer(paths["d03_cache"], frozen.OFFICIAL_MODEL, SCORE_VERSION, split_text)
    try:
        factual = nbc.score_pairs([(x["premise"], x["hypothesis"]) for x in positive])
        hallucinated = nbc.score_pairs([(x["premise"], x["hypothesis"]) for x in negative])
        pos_hist, neg_hist, _, _ = build_nbc_histograms(positive, negative, nbc)
    finally:
        nbc.close()

    production = ULSIFDensityRatio(random_state=frozen.EXPECTED_SPLIT_SEED).fit(
        factual, hallucinated
    )
    verify_production_fit(production)
    fx = production._as_column(factual)
    hx = production._as_column(hallucinated)
    estimator = candidate_estimator(
        production, fx, hx, frozen.FROZEN_SIGMA, frozen.FROZEN_LAMBDA
    )
    bse = BSEDetector(
        pos_hist, neg_hist, mode="official", p0=frozen.P0, c_miss=frozen.C_MISS,
        c_false_alarm=frozen.C_FALSE_ALARM, c_retrieve=frozen.C_RETRIEVE,
        max_docs=frozen.MAX_DOCS,
    )
    ddre = DDREDetector(
        estimator, lower_threshold=frozen.FROZEN_LOWER,
        upper_threshold=frozen.FROZEN_UPPER, p0=frozen.P0, c_miss=frozen.C_MISS,
        c_false_alarm=frozen.C_FALSE_ALARM, max_docs=frozen.MAX_DOCS,
    )
    scorer = CachedDocumentScorer(
        paths["held_out_cache"], frozen.OFFICIAL_MODEL, SCORE_VERSION, split_text
    )
    return {
        "bse": bse, "ddre": ddre, "estimator": estimator, "production": production,
        "fx": fx, "hx": hx, "factual_scores": np.asarray(factual, float),
        "hallucinated_scores": np.asarray(hallucinated, float),
        "pos_hist": pos_hist, "neg_hist": neg_hist, "scorer": scorer,
        "candidate_estimator": candidate_estimator,
    }


def trace_subclaim(subclaim, bse, ddre, estimator, scorer, hist_llr, in_flip):
    """Step-by-step trajectories for one subclaim under both detectors.

    Mirrors ``BSEDetector.detect_subclaim`` and ``DDREDetector.detect_subclaim``
    step for step; ``replay_sentence`` checks the result against the detectors
    themselves, so a drift here cannot go unnoticed.
    """
    from src.baseline_core import bayes_update

    p, bse_steps = bse.p0, []
    for document in subclaim.documents[: bse.max_docs]:
        if not bse.should_continue(p):
            break
        score, _ = scorer.score_document(subclaim.text, document.page_content)
        llr, bucket = hist_llr(score)
        p1, p0 = bse._bucket_likelihoods(bucket)
        p = bayes_update(p, p1, p0)
        bse_steps.append({"score": score, "bucket": bucket, "llr": llr, "p": p})
    bse_p = p

    log_odds, p, ddre_steps, reason = ddre._logit(ddre.p0), ddre.p0, [], None
    documents = subclaim.documents[: ddre.max_docs]
    for document in documents:
        score, _ = scorer.score_document(subclaim.text, document.page_content)
        log_ratio = math.log(estimator.ratio(score))
        log_odds = float(np.clip(log_odds + log_ratio, -40.0, 40.0))
        p = ddre._sigmoid(log_odds)
        ddre_steps.append({
            "score": score, "log_ratio": log_ratio, "cum_log_odds": log_odds,
            "p": p, "flip_region": in_flip(score), "bse_histogram_llr": hist_llr(score)[0],
        })
        if p <= ddre.lower_threshold:
            reason = "lower"
            break
        if p >= ddre.upper_threshold:
            reason = "upper"
            break
    if reason is None:
        reason = "budget" if len(documents) == ddre.max_docs else "exhausted"
    return {
        "text": subclaim.text, "available": len(subclaim.documents),
        "bse": bse_steps, "bse_p": bse_p,
        "ddre": ddre_steps, "ddre_p": p, "ddre_stop": reason,
    }


def replay_sentence(record, saved_pair, built, hist_llr, in_flip):
    """Replay one sentence through both detectors and the trace; fail closed."""
    bse, ddre, scorer = built["bse"], built["ddre"], built["scorer"]
    traces = [
        trace_subclaim(s, bse, ddre, built["estimator"], scorer, hist_llr, in_flip)
        for s in record.subclaims
    ]
    problems = []
    for method, detector, side in (("bse_official", bse, "bse"), ("frozen_ddre", ddre, "ddre")):
        sentence, subclaim_results = detector.detect_sentence_with_trace(record, scorer)
        replayed = {
            "p_factual": sentence.p_factual,
            "prediction": sentence.prediction,
            "documents_used": sentence.documents_used,
            "nli_calls": sentence.nli_calls,
            "subclaim_documents_used": [r.documents_used for r in subclaim_results],
            "subclaim_nli_calls": [r.nli_calls for r in subclaim_results],
        }
        problems += [f"{method}: {p}" for p in A.verify_replay_row(saved_pair[method], replayed)]
        for trace, result in zip(traces, subclaim_results):
            if len(trace[side]) != result.documents_used or trace[f"{side}_p"] != result.p_factual:
                problems.append(f"{method}: step trace disagrees with the detector")
    if problems:
        raise A.ReplayMismatch(
            f"sentence {(record.passage_index, record.sentence_index)}:\n  "
            + "\n  ".join(problems)
        )
    return traces


def main():
    args = parse_args()
    artifacts = Path(args.artifacts_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else artifacts
    paths = input_paths(artifacts)

    before = {name: A.sha256_path(path) for name, path in paths.items()}
    A.verify_digests(before, pinned_digests())
    result = json.loads(paths["result"].read_text())
    check_result_cross_references(result)
    d03 = json.loads(paths["d03_report"].read_text())

    held_out, split = frozen.frozen_split(args.data_root)
    saved_rows = list(csv.DictReader(paths["predictions"].open()))
    saved = {}
    for row in saved_rows:
        saved.setdefault((int(row["passage_index"]), int(row["sentence_index"])), {})[
            row["method"]
        ] = row
    paired = A.pair_prediction_rows(saved_rows, METHODS)

    built = build_detectors(args, paths)
    bse, ddre, estimator = built["bse"], built["ddre"], built["estimator"]
    cost = {"c_miss": frozen.C_MISS, "c_false_alarm": frozen.C_FALSE_ALARM}
    cut = ddre.cost_decision_threshold

    from src.baseline_core import discretize_document_score

    def hist_llr(score):
        bucket = discretize_document_score(score)
        p1, p0 = bse._bucket_likelihoods(bucket)
        return math.log(p1 / p0), bucket

    # The D-03 (sigma, lambda) surface, exactly the production cv_table pairs.
    pairs = [(float(p["sigma"]), float(p["lambda"]))
             for p in d03["hyperparameter_sensitivity"]["pairs"]]
    cv_pairs = sorted({(float(r["sigma"]), float(r["lambda"]))
                       for r in built["production"].cv_table})
    if sorted(pairs) != cv_pairs:
        raise A.ArtifactDigestMismatch("D-03 surface pairs differ from the production cv_table")
    surface = [built["candidate_estimator"](built["production"], built["fx"], built["hx"], s, l)
               for s, l in pairs]
    signs = np.array([[np.sign(math.log(e.ratio(float(x)))) for x in EVIDENCE_GRID]
                      for e in surface])
    flip_mask = A.flip_region(signs)

    def in_flip(score):
        index = int(np.clip(np.searchsorted(EVIDENCE_GRID, score), 0, len(EVIDENCE_GRID) - 1))
        return bool(flip_mask[index])

    # ------------------------------------------------------------ replay
    sentences = []
    try:
        for record in held_out:
            key = (record.passage_index, record.sentence_index)
            traces = replay_sentence(record, saved[key], built, hist_llr, in_flip)
            sentences.append({"key": key, "gold": record.label, "subclaims": traces})
        misses = built["scorer"].misses
    finally:
        built["scorer"].close()
    if misses:
        raise A.ReplayMismatch(f"{misses} cache misses during replay")
    if len(sentences) != len(paired):
        raise A.ReplayMismatch("replayed sentence count differs from the CSV")

    # ------------------------------------------------------------ sentence level
    gold = [x["gold"] for x in paired]
    pred = {m: [x[m]["prediction"] for x in paired] for m in METHODS}
    docs = {m: [x[m]["documents"] for x in paired] for m in METHODS}
    nli = {m: [x[m]["nli_calls"] for x in paired] for m in METHODS}
    post = {m: [x[m]["p_factual"] for x in paired] for m in METHODS}
    depth = {m: [d for x in paired for d in x[m]["subclaim_depths"]] for m in METHODS}
    b, d = METHODS
    sentence_level = {
        "bse": A.class_metrics(gold, pred[b]),
        "ddre": A.class_metrics(gold, pred[d]),
        "paired": A.paired_correctness(gold, pred[b], pred[d]),
        "efficiency_by_label": A.efficiency_by_label(gold, docs[b], docs[d], nli[b], nli[d]),
        "saving_concentration": A.saving_concentration(docs[b], docs[d]),
        "saving_by_correctness": A.saving_by_correctness(gold, pred[b], pred[d], docs[b], docs[d]),
        "posterior_placement": {
            m: A.posterior_placement(gold, post[m], frozen.FROZEN_LOWER, cut, frozen.FROZEN_UPPER)
            for m in METHODS
        },
        "recall_by_subclaim_count": A.recall_by_subclaim_count(
            gold, pred[b], pred[d], [x["n_subclaims"] for x in paired]
        ),
    }
    depth_level = {
        "joint": A.depth_joint_table(depth[b], depth[d], frozen.MAX_DOCS),
        "long_chain_share": A.long_chain_share(depth[b], depth[d]),
    }

    # ------------------------------------------------------------ evidence
    production = built["production"]
    log_r = lambda e: (lambda x: math.log(e.ratio(float(x))))
    modal = []
    for score in MODAL_SCORES:
        modal.append({
            "score": score,
            "bse_histogram_llr": hist_llr(score)[0],
            "frozen_ddre_log_ratio": log_r(estimator)(score),
            "production_log_ratio": log_r(production)(score),
            "pairs_positive": int(sum(log_r(e)(score) > 0 for e in surface)),
            "pairs_total": len(surface),
        })
    f_scores, h_scores = built["factual_scores"], built["hallucinated_scores"]
    evidence = {
        "table": A.evidence_table(
            EVIDENCE_GRID, EVIDENCE_BINS,
            {"bse_histogram_llr": lambda x: hist_llr(x)[0],
             "frozen_ddre_log_ratio": log_r(estimator),
             "production_log_ratio": log_r(production)},
            flip_mask,
        ),
        "flip_region_boundaries": A.region_boundaries(EVIDENCE_GRID, flip_mask),
        "modal_score_weights": modal,
        "nbc_training_scores_by_bin": [
            {"bin": [lo, min(hi, 100.0)],
             "factual": int(np.sum((f_scores >= lo) & (f_scores < hi))),
             "hallucinated": int(np.sum((h_scores >= lo) & (h_scores < hi)))}
            for lo, hi in zip(NBC_BINS, NBC_BINS[1:])
        ],
        "held_out_consumed_scores_ddre": A.score_bin_fractions(
            [st["score"] for x in sentences for s in x["subclaims"] for st in s["ddre"]]
        ),
        "held_out_max_consumed_score": float(max(
            st["score"] for x in sentences for s in x["subclaims"]
            for side in ("bse", "ddre") for st in s[side]
        )),
        "nbc_pos_hist": built["pos_hist"],
        "nbc_neg_hist": built["neg_hist"],
    }

    # ------------------------------------------------------------ trajectories
    by_key = {x["key"]: x for x in paired}
    fixes = [x for x in sentences if by_key[x["key"]][b]["prediction"] == 1
             and by_key[x["key"]][d]["prediction"] == 0 and x["gold"] == 0]
    regressions = [x for x in sentences if by_key[x["key"]][b]["prediction"] == 1
                   and by_key[x["key"]][d]["prediction"] == 0 and x["gold"] == 1]
    trajectories = {
        name: A.group_trajectory_summary(
            group, lower=frozen.FROZEN_LOWER, log_ratio=log_r(estimator), **cost
        )
        for name, group in (("ddre_fixes_gold_hallucinated", fixes),
                            ("ddre_regressions_gold_factual", regressions))
    }
    long_chains = A.long_chain_summary(sentences, upper=frozen.FROZEN_UPPER, **cost)

    after = {name: A.sha256_path(path) for name, path in paths.items()}
    if after != before:
        raise A.ArtifactDigestMismatch(f"an input artifact changed during analysis: {after}")

    commit = (frozen._git_commit() or "unknown")[:7]
    report = {
        "analysis": A.ANALYSIS_NAME,
        "analysis_version": A.ANALYSIS_VERSION,
        "status": A.ANALYSIS_STATUS,
        "interpretation_rules": list(A.INTERPRETATION_RULES),
        "repository_commit": frozen._git_commit(),
        "inference_performed": False,
        "model_loaded": False,
        "inputs": {name: {"path": str(path), "sha256": before[name]}
                   for name, path in paths.items()},
        "inputs_unchanged": True,
        "replay": {
            "sentences": len(sentences),
            "methods": list(METHODS),
            "exact_match_with_canonical_csv": True,
            "cache_misses": 0,
        },
        "cost_decision_threshold": cut,
        "frozen_band": [frozen.FROZEN_LOWER, frozen.FROZEN_UPPER],
        "split": {k: split[k] for k in ("held_out_passages", "held_out_sentences",
                                         "held_out_subclaims", "held_out_passage_ids_sha256")},
        "sentence_level": sentence_level,
        "subclaim_depth": depth_level,
        "evidence": evidence,
        "trajectories": trajectories,
        "long_chains": long_chains,
    }
    summary = A.render_summary(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "json": output_dir / f"frozen_heldout_analysis_{commit}.json",
        "summary": output_dir / f"frozen_heldout_analysis_{commit}.md",
    }
    if args.write_traces:
        outputs["traces"] = output_dir / f"frozen_heldout_traces_{commit}.json"
    inputs = {p.resolve() for p in paths.values()}
    for path in outputs.values():
        if path.resolve() in inputs:
            raise A.ArtifactDigestMismatch(f"refusing to overwrite input artifact {path}")
    outputs["json"].write_text(json.dumps(report, indent=1, default=float) + "\n")
    outputs["summary"].write_text(summary)
    if args.write_traces:
        outputs["traces"].write_text(json.dumps(
            {"status": A.ANALYSIS_STATUS, "result_sha256": RESULT_SHA256,
             "fixes": fixes, "regressions": regressions}, default=float
        ))
    print(summary)
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (A.ArtifactDigestMismatch, A.ReplayMismatch, frozen.Aborted) as exc:
        print(f"\nREFUSED: {exc}\nNothing was written.")
        sys.exit(1)
