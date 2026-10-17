"""Plan-v2 replay: every method under one stopping rule, swept over C_retrieve.

Research plan v2 §3-§4 and amendment 1, Phase 5. Pure functions over
per-document features that have already been computed; no model, cache or
file I/O.

A *prepared* sentence is ``{"id", "cluster", "label", "subclaims"}`` where
each subclaim is a list of documents ``{"e", "k", "rank", "spans"}`` in rank
order. Every method shares the documents, the prior, the log-odds update, the
min-over-subclaims aggregation and Wang's decision rule; methods differ only
in the likelihood ratio (and, for the reference rows, the stopping rule).
"""

import numpy as np

from src import metrics_v2 as V
from src import sequential as S

COST_PAIRS = ((28.0, 96.0), (14.0, 24.0))
C_RETRIEVE_GRID = tuple(2.0 ** (k / 2) for k in range(-6, 9))  # 0.125 .. 16
OPERATING_C_RETRIEVE = 1.0
TIER1 = ("BSE", "DDRE-1D")
TIER2 = ("Hist-1D", "DDRE-1D'", "NB-M", "DDRE-M", "Logit-M")
# (reference, method, role)
CONTRASTS = (
    ("BSE", "DDRE-1D", "primary: ratio estimator in Wang's setting (Tier 1)"),
    ("NB-M", "DDRE-M", "primary: ratio estimator with three features (Tier 2)"),
    ("Hist-1D", "DDRE-1D'", "ratio estimator, one feature, larger data"),
    ("Hist-1D", "NB-M", "features, histogram estimator"),
    ("DDRE-1D'", "DDRE-M", "features, uLSIF estimator"),
    ("BSE", "Hist-1D", "training data, histogram estimator"),
    ("Logit-M", "DDRE-M", "direct-ratio family: uLSIF vs logistic"),
    ("BSE-official", "BSE", "stopping rule: Wang's code vs backward induction"),
)
PRIMARY = CONTRASTS[:2]

assert OPERATING_C_RETRIEVE in C_RETRIEVE_GRID


class ReplayError(RuntimeError):
    pass


def setting_key(cm, cfa, cr):
    return f"{cm:g}/{cfa:g}/{cr:g}"


def settings():
    return [(cm, cfa, cr) for cm, cfa in COST_PAIRS for cr in C_RETRIEVE_GRID]


# ---------------------------------------------------------------- running


def log_ratios(prepared, estimator):
    """Per sentence, per subclaim: the estimator's log ratio of each document."""
    return [[[estimator.log_ratio(d) for d in sub] for sub in s["subclaims"]]
            for s in prepared]


def _collect(prepared, per_subclaim, costs):
    n = len(prepared)
    out = {"p_factual": np.zeros(n), "documents": np.zeros(n), "nli_calls": np.zeros(n),
           "prediction": np.zeros(n, dtype=int), "subclaim_documents": []}
    for j, s in enumerate(prepared):
        if not s["subclaims"]:
            raise ReplayError(f"sentence {s['id']} has no subclaims")
        r = S.sentence([per_subclaim(j, i) for i in range(len(s["subclaims"]))], costs)
        out["p_factual"][j] = r["p_factual"]
        out["documents"][j] = r["documents"]
        out["nli_calls"][j] = r["nli_calls"]
        out["prediction"][j] = r["prediction"]
        out["subclaim_documents"].append(r["subclaim_documents"])
    return out


def run_backward_induction(prepared, lrs, policies, costs, max_docs):
    """``policies``: an ``S.PolicyByHorizon`` for this method and ``costs``."""

    def one(j, i):
        sub = prepared[j]["subclaims"][i]
        return S.run_subclaim_bi(lrs[j][i], [d["spans"] for d in sub], policies, max_docs)

    return _collect(prepared, one, costs)


def run_eq8(prepared, lrs, policy, costs, max_docs):
    """``policy``: an ``S.Eq8Policy`` for this method and ``costs``."""

    def one(j, i):
        sub = prepared[j]["subclaims"][i]
        return S.run_subclaim(lrs[j][i], [d["spans"] for d in sub], policy, max_docs)

    return _collect(prepared, one, costs)


def run_bse_official(prepared, histogram, costs, max_docs):
    """Wang's released rule with Wang's histogram, replayed as BSE runs it."""
    from src.baseline_core import BSEDetector
    from src.mddre_replay import bse_subclaim

    bse = BSEDetector(histogram.pos_hist, histogram.neg_hist, mode="official", p0=0.5,
                      c_miss=costs.c_miss, c_false_alarm=costs.c_false_alarm,
                      c_retrieve=costs.c_retrieve, max_docs=max_docs)

    def one(j, i):
        sub = prepared[j]["subclaims"][i]
        return bse_subclaim([histogram.likelihoods(d) for d in sub],
                            [d["spans"] for d in sub], bse, max_docs)

    return _collect(prepared, one, costs)


def run_fixed_k(prepared, lrs, k, how, max_docs):
    """Wang's Figure 2 baseline; the sentence score is the min over subclaims."""
    n = len(prepared)
    out = {"score": np.zeros(n), "documents": np.zeros(n), "nli_calls": np.zeros(n)}
    kk = min(k, max_docs)
    for j, s in enumerate(prepared):
        results = [S.fixed_k_subclaim([d["e"] for d in sub], lrs[j][i] if lrs else [],
                                      [d["spans"] for d in sub], kk, how)
                   for i, sub in enumerate(s["subclaims"])]
        out["score"][j] = min(r[0] for r in results)
        out["documents"][j] = sum(r[1] for r in results)
        out["nli_calls"][j] = sum(r[2] for r in results)
    return out


# ---------------------------------------------------------------- endpoints


def _curve(results, method, cm, cfa, y, idx, cost, quality_cache):
    """(costs, qualities) of one method's C_retrieve sweep on the units ``idx``."""
    qkey = (method, cm, cfa)
    if qkey not in quality_cache:
        quality_cache[qkey] = [
            V.balanced_ap_fast(y[idx], results[method][setting_key(cm, cfa, cr)]["p_rounded"][idx])
            for cr in C_RETRIEVE_GRID]
    costs = [float(np.mean(results[method][setting_key(cm, cfa, cr)][cost][idx]))
             for cr in C_RETRIEVE_GRID]
    return costs, quality_cache[qkey]


def contrast_endpoints(results, labels, idx, contrasts=CONTRASTS, cost_pairs=COST_PAIRS):
    """Every contrast's endpoints on the units ``idx`` (one bootstrap replicate).

    Costs are NLI calls per sentence (primary) and documents per sentence.
    The reference's operating point is C_retrieve = 1 on its own sweep.
    """
    y = labels
    out = {}
    quality_cache = {}

    def curve(method, cm, cfa, cost):
        return _curve(results, method, cm, cfa, y, idx, cost, quality_cache)

    op = C_RETRIEVE_GRID.index(OPERATING_C_RETRIEVE)
    for ref, m, _ in contrasts:
        for cm, cfa in cost_pairs:
            tag = f"{m}|{ref}|{cm:g}/{cfa:g}"
            for cost, unit in (("nli_calls", "nli"), ("documents", "docs")):
                rc, rq = curve(ref, cm, cfa, cost)
                mc, mq = curve(m, cm, cfa, cost)
                env_r, env_m = V.Envelope(rc, rq), V.Envelope(mc, mq)
                out[f"{tag}|e_cost_{unit}"] = V.e_cost(env_m, env_r, rq[op])
                out[f"{tag}|op_{unit}_saved"] = rc[op] - mc[op]
                if unit == "nli":
                    out[f"{tag}|e_quality"] = V.e_quality(env_m, env_r, rc[op])
                    out[f"{tag}|auc_gap"] = V.auc_gap(env_m, env_r)
                    out[f"{tag}|op_quality_gain"] = mq[op] - rq[op]
            if (ref, m) in {(c[0], c[1]) for c in PRIMARY}:
                key = setting_key(cm, cfa, OPERATING_C_RETRIEVE)
                cal_r = V.calibration(y[idx], results[ref][key]["p_rounded"][idx])
                cal_m = V.calibration(y[idx], results[m][key]["p_rounded"][idx])
                for name in ("brier", "ece"):
                    out[f"{tag}|{name}_reduction"] = cal_r[name] - cal_m[name]
    return out


def with_rounded(results):
    """Add the rounded posteriors the fast AP needs (in place); return results."""
    for by_setting in results.values():
        for r in by_setting.values():
            r["p_rounded"] = np.round(r["p_factual"], V.RANK_DECIMALS)
    return results


def observed_tables(results, labels, passages=None):
    """Wang's table and calibration for every method at every setting."""
    out = {}
    for method, by_setting in results.items():
        out[method] = {}
        for key, r in by_setting.items():
            row = V.wang_table(labels, r["p_factual"], r["prediction"], r["documents"],
                               r["nli_calls"], passages=passages)
            row["calibration"] = V.calibration(labels, r["p_factual"])
            out[method][key] = row
    return out

