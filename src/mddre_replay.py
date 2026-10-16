"""Replay engine, cost-quality curves and endpoints (design §5-§7).

Pure functions over per-document evidence that has already been computed. No
model, no cache, no file I/O.

Stopping semantics are Wang's: before each document the detector continues
only while P(factual) lies strictly inside the band (l, u), and stops after
``max_docs``. For a band containing P0 this equals the frozen DDRE rule
(stop once P <= l or P >= u). Cell A's official look-ahead is replayed with
``BSEDetector.should_continue`` and ``bayes_update`` exactly as BSE runs it.

The cost-quality curve follows design amendment 1: BSE's band scaled in
log-odds about P0 = 0.5, so s = 1 is the operating point and every band
contains P0. (The preregistered symmetric sweep around the decision threshold
included bands below P0, which read no document at all.)
"""

import math

import numpy as np

from src.baseline_core import bayes_update, cost_based_prediction

C_MISS, C_FALSE_ALARM, P0 = 28.0, 96.0, 0.5
LOG_ODDS_CLIP = 40.0
BAND_SCALES = tuple(0.25 * i for i in range(1, 17))  # 0.25 .. 4.0 (amendment 1)
N_RESAMPLES, SEED, CI_LEVEL = 10_000, 42, 0.95
NOT_REACHED = -1e9  # sentinel: a matched endpoint the curve cannot reach


def decision_threshold():
    """P at which the cost rule switches: (1 - P) CM = P CFA."""
    return C_MISS / (C_MISS + C_FALSE_ALARM)


def _logit(p):
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def scaled_band(band, scale):
    """BSE's band scaled in log-odds about P0 = 0.5 (amendment 1)."""
    lower, upper = band
    return _sigmoid(scale * _logit(lower)), _sigmoid(scale * _logit(upper))


def curve_bands(band):
    """The amended curve's bands. Refuses unless each strictly contains P0."""
    if P0 != 0.5:
        raise ValueError("scaling about log-odds 0 assumes P0 = 0.5")
    bands = [scaled_band(band, s) for s in BAND_SCALES]
    for s, (lower, upper) in zip(BAND_SCALES, bands):
        if not lower < P0 < upper:
            raise ValueError(f"curve band at scale {s} does not contain P0: {(lower, upper)}")
    return bands


def bse_band(bse, tol=1e-13):
    """The interval where ``bse.should_continue`` holds, by bisection.

    Verified on a dense grid: the continue region must be exactly (l, u).
    """
    def edge(inside, outside):
        while abs(outside - inside) > tol:
            mid = 0.5 * (inside + outside)
            inside, outside = (mid, outside) if bse.should_continue(mid) else (inside, mid)
        return outside

    if not bse.should_continue(P0):
        raise ValueError("BSE does not continue at P0")
    lower, upper = edge(P0, 0.0), edge(P0, 1.0)
    for p in np.linspace(1e-6, 1 - 1e-6, 20001):
        if bse.should_continue(p) != (lower < p < upper):
            if min(abs(p - lower), abs(p - upper)) > 1e-9:
                raise ValueError(f"BSE continue region is not the interval at P = {p}")
    return lower, upper


# ---------------------------------------------------------------- detection

def band_subclaim(log_ratios, spans, lower, upper, max_docs):
    """(P, documents, NLI calls) under the band rule with log-odds updates."""
    log_odds, p = _logit(P0), P0
    docs = nli = 0
    for lr, n in zip(log_ratios[:max_docs], spans[:max_docs]):
        if not lower < p < upper:
            break
        log_odds = float(np.clip(log_odds + lr, -LOG_ODDS_CLIP, LOG_ODDS_CLIP))
        p = _sigmoid(log_odds)
        docs += 1
        nli += n
    return p, docs, nli


def bse_subclaim(likelihoods, spans, bse, max_docs):
    """(P, documents, NLI calls) under BSE's official look-ahead, as BSE runs it."""
    p = bse.p0
    docs = nli = 0
    for (p1, p0), n in zip(likelihoods[:max_docs], spans[:max_docs]):
        if not bse.should_continue(p):
            break
        p = bayes_update(p, p1, p0)
        docs += 1
        nli += n
    return p, docs, nli


def sentence_result(subclaim_results):
    p = min(r[0] for r in subclaim_results)
    return {"p_factual": float(p),
            "prediction": cost_based_prediction(p, C_MISS, C_FALSE_ALARM),
            "documents": sum(r[1] for r in subclaim_results),
            "nli_calls": sum(r[2] for r in subclaim_results),
            "subclaim_documents": [r[1] for r in subclaim_results]}


# ---------------------------------------------------------------- curves

def balanced_pr_auc(labels, p, pr_auc):
    return 0.5 * (pr_auc(labels, p) + pr_auc(1 - labels, 1.0 - p))


def matched_quality_docs(docs, quality, target):
    """Fewest documents at which the curve reaches ``target`` (interpolated)."""
    for i, (d, q) in enumerate(zip(docs, quality)):
        if q >= target:
            if i == 0 or quality[i - 1] == q:
                return d
            d0, q0 = docs[i - 1], quality[i - 1]
            return d0 + (d - d0) * (target - q0) / (q - q0)
    return None


def quality_at_docs(docs, quality, target):
    """Curve quality at ``target`` documents (linear in documents)."""
    if not docs or target < docs[0] or target > docs[-1]:
        return None
    for i in range(1, len(docs)):
        if docs[i] >= target:
            d0, d1 = docs[i - 1], docs[i]
            if d1 == d0:
                return quality[i]
            return quality[i - 1] + (quality[i] - quality[i - 1]) * (target - d0) / (d1 - d0)
    return quality[0]


def endpoints(curves, reference, cells, pr_auc, labels, idx):
    """Matched-quality saving, matched-cost gain and same-band differences.

    ``curves[cell]`` holds per-band arrays (bands x sentences) of p_factual and
    documents, plus the operating-point arrays; ``reference`` is cell A's
    official run. ``idx`` is the (resampled) sentence index set.
    """
    lab = labels[idx]
    ref_q = balanced_pr_auc(lab, reference["p_factual"][idx], pr_auc)
    ref_d = float(np.mean(reference["documents"][idx]))
    out = {"reference": {"balanced_pr_auc": ref_q, "documents": ref_d}}
    for cell in cells:
        c = curves[cell]
        nb = len(c["documents"])
        docs = [float(np.mean(c["documents"][b][idx])) for b in range(nb)]
        qual = [balanced_pr_auc(lab, c["p_factual"][b][idx], pr_auc) for b in range(nb)]
        d_star = matched_quality_docs(docs, qual, ref_q)
        q_star = quality_at_docs(docs, qual, ref_d)
        op_q = balanced_pr_auc(lab, c["op_p_factual"][idx], pr_auc)
        op_d = float(np.mean(c["op_documents"][idx]))
        out[cell] = {
            "matched_quality_saving": NOT_REACHED if d_star is None else ref_d - d_star,
            "matched_cost_gain": NOT_REACHED if q_star is None else q_star - ref_q,
            "same_band_docs_saving": ref_d - op_d,
            "same_band_quality_gain": op_q - ref_q,
            "curve": {"documents": docs, "balanced_pr_auc": qual},
        }
    return out


def scalar_endpoints(e, cells, chain):
    """Flatten endpoints to named scalars, adding the RQ3 chain differences."""
    flat = {}
    for cell in cells:
        for k in ("matched_quality_saving", "matched_cost_gain",
                  "same_band_docs_saving", "same_band_quality_gain"):
            flat[f"{cell}.{k}"] = e[cell][k]
    for a, b in chain:
        for k in ("matched_quality_saving", "matched_cost_gain"):
            x, y = e[b][k], e[a][k]
            flat[f"{b}-{a}.{k}"] = (NOT_REACHED if NOT_REACHED in (x, y) else x - y)
    return flat


def blocks_of(clusters):
    order, blocks = [], {}
    for i, c in enumerate(clusters):
        if c not in blocks:
            blocks[c] = []
            order.append(c)
        blocks[c].append(i)
    return [np.asarray(blocks[c], dtype=int) for c in order]


def bootstrap(curves, reference, cells, chain, labels, clusters, pr_auc, *,
              n_resamples=N_RESAMPLES, seed=SEED, ci_level=CI_LEVEL):
    """Paired cluster bootstrap over every endpoint, all cells on the same draw."""
    labels = np.asarray(labels)
    full = np.arange(len(labels))
    observed = endpoints(curves, reference, cells, pr_auc, labels, full)
    obs_flat = scalar_endpoints(observed, cells, chain)
    blocks = blocks_of(clusters)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in obs_flat}
    for _ in range(n_resamples):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if len(np.unique(labels[idx])) < 2:
            raise ValueError("a replicate has one class; PR-AUC undefined (fail closed)")
        flat = scalar_endpoints(endpoints(curves, reference, cells, pr_auc, labels, idx),
                                cells, chain)
        for k, v in flat.items():
            draws[k].append(v)
    tail = 100 * (1 - ci_level) / 2
    summary = {}
    for k, values in draws.items():
        v = np.asarray(values, dtype=float)
        summary[k] = {
            "observed": obs_flat[k],
            "ci_lower": float(np.percentile(v, tail)),
            "ci_upper": float(np.percentile(v, 100 - tail)),
            "not_reached_replicates": int(np.sum(v == NOT_REACHED)),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)))),
        }
    return {"observed": observed, "endpoints": summary, "clusters": len(blocks),
            "n_resamples": n_resamples, "seed": seed, "ci_level": ci_level}


def holm(p_values, alpha=0.05):
    """Holm step-down decisions for a {name: p} dict."""
    order = sorted(p_values, key=lambda k: p_values[k])
    decisions, stopped = {}, False
    for i, k in enumerate(order):
        threshold = alpha / (len(order) - i)
        ok = (not stopped) and p_values[k] < threshold
        stopped = stopped or not ok
        decisions[k] = {"p": p_values[k], "threshold": threshold, "reject": ok}
    return decisions


def decide(endpoint_summary, cell="M"):
    """Design §7: RQ1 and RQ2 rules for one dataset."""
    rq1 = endpoint_summary[f"{cell}.matched_quality_saving"]
    rq2 = endpoint_summary[f"{cell}.matched_cost_gain"]
    h = holm({"RQ1": rq1["p_two_sided"], "RQ2": rq2["p_two_sided"]})
    return {
        "RQ1": {"ci_above_zero": rq1["ci_lower"] > 0,
                "holm": h["RQ1"], "supported": rq1["ci_lower"] > 0 and h["RQ1"]["reject"]},
        "RQ2": {"ci_above_zero": rq2["ci_lower"] > 0,
                "holm": h["RQ2"], "supported": rq2["ci_lower"] > 0 and h["RQ2"]["reject"]},
        "both": rq1["ci_lower"] > 0 and rq2["ci_lower"] > 0
                and h["RQ1"]["reject"] and h["RQ2"]["reject"],
    }
