"""Endpoints and decision rule of the confirmatory test (preregistration §4-§6).

Pure functions over per-unit replay arrays; the replay engine itself is
``src.mddre_replay`` (BSE's band, the Amendment-1 curve, Wang's update rules).

- **E2, matched-cost gain:** cell c's balanced PR-AUC at BSE official's
  documents per unit, interpolated on c's curve, minus BSE official's.
- **E1, net matched-quality saving:** D_A - D_c, where D_c is the first
  crossing of BSE official's balanced PR-AUC on c's curve and D_A is the
  smaller of BSE official's documents and the first crossing on A's curve.
  "Not reached" (c never reaches the target, or the target cost lies outside
  c's curve) is the sentinel ``NOT_REACHED`` and counts against the hypothesis.
"""

import numpy as np

from src.mddre_replay import (
    NOT_REACHED,
    N_RESAMPLES,
    SEED,
    CI_LEVEL,
    balanced_pr_auc,
    blocks_of,
    holm,
    matched_quality_docs,
    quality_at_docs,
)

PRIMARY = ("E1", "E2")
ALPHA = 0.05


def _curves(cell, idx, labels, pr_auc):
    """{cost: (mean cost per band, balanced PR-AUC per band)}; quality computed once."""
    n_bands = len(cell["documents"])
    qual = [balanced_pr_auc(labels[idx], cell["p_factual"][b][idx], pr_auc)
            for b in range(n_bands)]
    return {k: ([float(np.mean(cell[k][b][idx])) for b in range(n_bands)], qual)
            for k in ("documents", "nli")}


def endpoints(curves, reference, cells, labels, pr_auc, idx, chain=()):
    """{name: value} for every cell's E1, E2, E1 in NLI calls and same-band
    differences, plus the chain differences, on the unit index set ``idx``."""
    lab = labels[idx]
    q_ref = balanced_pr_auc(lab, reference["p_factual"][idx], pr_auc)
    ref_cost = {k: float(np.mean(reference[k][idx])) for k in ("documents", "nli")}
    out = {"reference.balanced_pr_auc": q_ref, "reference.documents": ref_cost["documents"],
           "reference.nli": ref_cost["nli"]}
    curves_at = {c: _curves(curves[c], idx, labels, pr_auc) for c in cells}
    first = {k: {} for k in ("documents", "nli")}
    for c in cells:
        for k in ("documents", "nli"):
            first[k][c] = matched_quality_docs(*curves_at[c][k], q_ref)
    d_a = {k: min(ref_cost[k], first[k]["A"]) if first[k]["A"] is not None else ref_cost[k]
           for k in ("documents", "nli")}
    for c in cells:
        for k, name in (("documents", "E1"), ("nli", "E1_nli")):
            out[f"{c}.{name}"] = NOT_REACHED if first[k][c] is None else d_a[k] - first[k][c]
        q_star = quality_at_docs(*curves_at[c]["documents"], ref_cost["documents"])
        out[f"{c}.E2"] = NOT_REACHED if q_star is None else q_star - q_ref
        op_q = balanced_pr_auc(lab, curves[c]["op_p_factual"][idx], pr_auc)
        out[f"{c}.same_band_docs_saving"] = ref_cost["documents"] - float(
            np.mean(curves[c]["op_documents"][idx]))
        out[f"{c}.same_band_nli_saving"] = ref_cost["nli"] - float(
            np.mean(curves[c]["op_nli"][idx]))
        out[f"{c}.same_band_quality_gain"] = op_q - q_ref
    for a, b in chain:
        for k in ("E1", "E2"):
            x, y = out[f"{b}.{k}"], out[f"{a}.{k}"]
            out[f"{b}-{a}.{k}"] = NOT_REACHED if NOT_REACHED in (x, y) else x - y
    return out, curves_at


def bootstrap(curves, reference, cells, labels, clusters, pr_auc, *, chain=(),
              n_resamples=N_RESAMPLES, seed=SEED, ci_level=CI_LEVEL):
    """Paired cluster bootstrap over every endpoint, all cells on the same draws."""
    labels = np.asarray(labels)
    observed, curves_at = endpoints(curves, reference, cells, labels, pr_auc,
                                    np.arange(len(labels)), chain)
    blocks = blocks_of(clusters)
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_resamples) for k in observed}
    for r in range(n_resamples):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if len(np.unique(labels[idx])) < 2:
            raise ValueError("a replicate has one class; PR-AUC undefined (fail closed)")
        flat, _ = endpoints(curves, reference, cells, labels, pr_auc, idx, chain)
        for k, v in flat.items():
            draws[k][r] = v
    tail = 100 * (1 - ci_level) / 2
    summary = {k: {"observed": observed[k],
                   "ci_lower": float(np.percentile(v, tail)),
                   "ci_upper": float(np.percentile(v, 100 - tail)),
                   "not_reached_replicates": int(np.sum(v == NOT_REACHED)),
                   "p_two_sided": float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))}
               for k, v in draws.items()}
    return {"endpoints": summary, "curves": {c: {k: {"cost": d, "balanced_pr_auc": q}
                                                 for k, (d, q) in v.items()}
                                             for c, v in curves_at.items()},
            "clusters": len(blocks), "n_resamples": n_resamples, "seed": seed,
            "ci_level": ci_level}


def decide(endpoint_summaries, cell="M", alpha=ALPHA):
    """§5: Holm over the four primary tests, then the conclusion table.

    ``endpoint_summaries`` maps dataset name -> bootstrap ``endpoints``.
    """
    tests = {f"{d}.{e}": s[f"{cell}.{e}"] for d, s in endpoint_summaries.items()
             for e in PRIMARY}
    h = holm({k: v["p_two_sided"] for k, v in tests.items()}, alpha)
    per_test = {k: {"observed": v["observed"], "ci": [v["ci_lower"], v["ci_upper"]],
                    "holm": h[k],
                    "supported": bool(h[k]["reject"] and v["ci_lower"] > 0),
                    "m_worse": bool(v["ci_upper"] < 0)}
                for k, v in tests.items()}
    datasets = list(endpoint_summaries)
    statements = {}
    for e, claim in (("E2", "better detection at matched cost"),
                     ("E1", "lower cost at matched quality")):
        ok = [d for d in datasets if per_test[f"{d}.{e}"]["supported"]]
        if len(ok) == len(datasets):
            statements[e] = f"confirmed: {claim}"
        elif ok:
            statements[e] = f"partial support ({', '.join(ok)}): {claim}"
        else:
            statements[e] = f"not confirmed: {claim}"
    both = all(s.startswith("confirmed") for s in statements.values())
    return {"tests": per_test, "statements": statements,
            "lower_cost_and_better_detection_confirmed": both}
