"""Budget-knob curves for plan v2 (amendment 2). Pure functions; no I/O.

Phase 5 found that under backward induction the C_retrieve sweep is nearly
all-or-nothing: methods read almost every document or none. Amendment 2 adds a
second knob, K, the maximum number of documents per subclaim. A method's
operating points are the (K, C_retrieve) grid; its curve is the step
envelope over all of them (``src.metrics_v2.Envelope``). With K = the
dataset's maximum and C_retrieve = 1 the point is Phase 5's operating point.

Wang's fixed-k "Maximum" baseline is a method with one point per K. It joins
the bootstrap as a contrast, anchored at the sequential method's operating
point: "at this method's own cost, is the best fixed-k within that cost
better or worse, and how much would fixed-k need to reach its quality?"
"""

import numpy as np

from src import metrics_v2 as V

BUDGET_C_RETRIEVE = (0.125, 0.25, 0.5, 1.0)  # below Phase 5's cliff, plus the operating point
METHODS = ("BSE", "DDRE-1D", "NB-M", "DDRE-M")
FIXED = "fixed-max"
# (reference, method, anchor): anchor "reference" = the reference's operating
# point, as in Phase 5; "method" = the method's own operating point.
CONTRASTS = (
    ("BSE", "DDRE-1D", "reference"),
    ("NB-M", "DDRE-M", "reference"),
    (FIXED, "BSE", "method"),
    (FIXED, "DDRE-1D", "method"),
    (FIXED, "NB-M", "method"),
    (FIXED, "DDRE-M", "method"),
)


class BudgetError(RuntimeError):
    pass


def point_key(k, cr):
    return f"K{k}/cr{cr:g}"


def grid(max_docs):
    return [(k, cr) for k in range(1, max_docs + 1) for cr in BUDGET_C_RETRIEVE]


def operating_key(max_docs):
    return point_key(max_docs, 1.0)


def _points(points, idx, y, cost):
    """(costs, qualities, keys) of one method's points on the units ``idx``."""
    keys = sorted(points)
    costs = [float(np.mean(points[k][cost][idx])) for k in keys]
    quals = [V.balanced_ap_fast(y[idx], points[k]["p_rounded"][idx]) for k in keys]
    return costs, quals, keys


def contrast_endpoints(curves, labels, idx, max_docs, cost_pairs, contrasts=CONTRASTS):
    """Endpoints of every contrast on the units ``idx`` (one replicate).

    ``curves[(method, cost_pair)]`` maps a point key to arrays ``p_rounded``,
    ``nli_calls`` and ``documents``; the fixed-k baseline is stored under
    ``(FIXED, None)`` because it has no costs to set.
    """
    y = labels
    cache = {}

    def pts(method, pair, cost):
        key = (method, pair, cost)
        if key not in cache:
            cache[key] = _points(curves[(method, None if method == FIXED else pair)],
                                 idx, y, cost)
        return cache[key]

    op = operating_key(max_docs)
    out = {}
    for ref, m, anchor in contrasts:
        for pair in cost_pairs:
            tag = f"{m}|{ref}|{pair[0]:g}/{pair[1]:g}"
            for cost, unit in (("nli_calls", "nli"), ("documents", "docs")):
                rc, rq, rk = pts(ref, pair, cost)
                mc, mq, mk = pts(m, pair, cost)
                env_r, env_m = V.Envelope(rc, rq), V.Envelope(mc, mq)
                if anchor == "reference":
                    i = rk.index(op)
                    c_a, q_a = rc[i], rq[i]
                else:
                    i = mk.index(op)
                    c_a, q_a = mc[i], mq[i]
                qm, qr = env_m.quality_at(c_a), env_r.quality_at(c_a)
                cm_, cr_ = env_m.cost_for(q_a), env_r.cost_for(q_a)
                out[f"{tag}|e_cost_{unit}"] = None if None in (cm_, cr_) else cr_ - cm_
                if anchor == "method":
                    out[f"{tag}|reference_never_reaches_{unit}"] = float(cr_ is None)
                if unit == "nli":
                    out[f"{tag}|e_quality"] = None if None in (qm, qr) else qm - qr
                    out[f"{tag}|auc_gap"] = V.auc_gap(env_m, env_r)
    return out


def observed_curves(curves, labels):
    """Every point's cost and balanced AP on all units (for tables and figures)."""
    out = {}
    for (method, pair), points in curves.items():
        name = method if pair is None else f"{method}|{pair[0]:g}/{pair[1]:g}"
        out[name] = {k: {"nli_calls_per_sentence": float(np.mean(v["nli_calls"])),
                         "documents_per_sentence": float(np.mean(v["documents"])),
                         "balanced_ap": V.balanced_ap(labels, v["p_rounded"])}
                     for k, v in sorted(points.items())}
    return out
