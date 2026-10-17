"""Plan-v2 metrics, curve endpoints and bootstrap summaries (plan §4).

Pure functions over per-sentence arrays; no model, cache or file I/O.

- **Quality.** The primary measure is balanced average precision: the mean of
  AP for the factual class (score P) and the hallucinated class (score 1 - P).
  Wang's trapezoidal PR-AUC (``src.evaluation.wang_pr_auc``) is reported
  alongside for comparability. Posteriors are rounded to ``RANK_DECIMALS``
  before ranking, so floating-point noise cannot reorder ties.
- **Wang's Table 1** for any method: nonfactual and factual PR-AUC, accuracy,
  documents and NLI calls per sentence, and passage-level Pearson and Spearman
  where passages exist. Also macro-F1 and ROC-AUC.
- **Calibration (G3).** Brier score, ECE over 15 equal-mass bins, and the
  reliability table.
- **Cost-quality envelopes.** A method's operating points (one per cost
  setting) give the step function Q(c) = max{q_i : c_i <= c}: the best quality
  reachable within cost c. Nothing is interpolated between points, so every
  value is achieved by a real operating point.
- **Endpoints**, against a reference method with operating point (c_ref, q_ref):
  - E-cost (G1): C_ref(q_ref) - C_m(q_ref), where C(q) = min{c_i : q_i >= q}
    on each envelope. Net of the reference's own best showing; "not reached"
    (``None``) if method m never reaches q_ref.
  - E-quality (G2): Q_m(c_ref) - Q_ref(c_ref); ``None`` if method m has no
    point at or below c_ref.
  - AUC-gap: mean of Q_m - Q_ref over the shared cost range of the two
    envelopes; ``None`` if the ranges do not overlap.
- **Bootstrap summaries.** "Not reached" replicates are counted and excluded
  from the interval, never entered as a sentinel value. One-sided p-values
  count them against the hypothesis.
"""

import math

import numpy as np

RANK_DECIMALS = 9
ECE_BINS = 15


class MetricsError(RuntimeError):
    pass


def _arrays(labels, p):
    y = np.asarray(labels, dtype=int)
    s = np.round(np.asarray(p, dtype=float), RANK_DECIMALS)
    if y.shape != s.shape or y.ndim != 1 or y.size == 0:
        raise MetricsError("labels and scores must be equal-length 1-D arrays")
    if not set(np.unique(y)) <= {0, 1}:
        raise MetricsError("labels must be 0 (hallucinated) or 1 (factual)")
    if not np.all(np.isfinite(s)):
        raise MetricsError("non-finite scores")
    return y, s


def _both_classes(y):
    if len(np.unique(y)) < 2:
        raise MetricsError("both classes are needed")


# ---------------------------------------------------------------- quality


def balanced_ap(labels, p):
    """Mean of factual AP (score P) and hallucinated AP (score 1 - P)."""
    from sklearn.metrics import average_precision_score

    y, s = _arrays(labels, p)
    _both_classes(y)
    return 0.5 * (float(average_precision_score(y, s))
                  + float(average_precision_score(1 - y, 1.0 - s)))


def balanced_wang_pr_auc(labels, p):
    """Wang's trapezoidal PR-AUC, averaged over the two classes."""
    from src.evaluation import wang_pr_auc

    y, s = _arrays(labels, p)
    _both_classes(y)
    return 0.5 * (wang_pr_auc(y, s) + wang_pr_auc(1 - y, 1.0 - s))


def wang_table(labels, p, predictions, documents, nli_calls, passages=None):
    """Wang's Table 1 columns (plus macro-F1, ROC-AUC and balanced AP)."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, roc_auc_score

    from src.evaluation import wang_pr_auc

    y, s = _arrays(labels, p)
    _both_classes(y)
    pred = np.asarray(predictions, dtype=int)
    out = {
        "sentences": int(y.size),
        "nonfactual_pr_auc": wang_pr_auc(1 - y, 1.0 - s),
        "factual_pr_auc": wang_pr_auc(y, s),
        "balanced_pr_auc": balanced_wang_pr_auc(y, s),
        "balanced_ap": balanced_ap(y, s),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, s)),
        "documents_per_sentence": float(np.mean(documents)),
        "nli_calls_per_sentence": float(np.mean(nli_calls)),
    }
    if passages is not None:
        groups = {}
        for g, label, score in zip(passages, y, s):
            groups.setdefault(g, ([], []))
            groups[g][0].append(label)
            groups[g][1].append(score)
        gold = [float(np.mean(v[0])) for v in groups.values()]
        pred_p = [float(np.mean(v[1])) for v in groups.values()]
        ok = len(gold) > 1 and np.std(gold) > 0 and np.std(pred_p) > 0
        out["passages"] = len(gold)
        out["pearson"] = float(pearsonr(gold, pred_p)[0]) if ok else None
        out["spearman"] = float(spearmanr(gold, pred_p)[0]) if ok else None
    return out


# ---------------------------------------------------------------- calibration


def calibration(labels, p, bins=ECE_BINS):
    """Brier score, ECE over equal-mass bins, and the reliability table.

    Bins are consecutive blocks of the sentences sorted by P (ties kept in
    input order), sizes differing by at most one.
    """
    y, s = _arrays(labels, p)
    order = np.argsort(s, kind="stable")
    table, ece = [], 0.0
    for block in np.array_split(order, min(bins, y.size)):
        mean_p, frac = float(np.mean(s[block])), float(np.mean(y[block]))
        ece += len(block) / y.size * abs(mean_p - frac)
        table.append({"n": int(len(block)), "mean_p": mean_p, "fraction_factual": frac})
    return {"brier": float(np.mean((s - y) ** 2)), "ece": float(ece), "bins": len(table),
            "reliability": table, "prevalence_factual": float(np.mean(y))}


# ---------------------------------------------------------------- envelopes


class Envelope:
    """Q(c) = max{q_i : c_i <= c} over a method's operating points."""

    def __init__(self, costs, qualities):
        c = np.asarray(costs, dtype=float)
        q = np.asarray(qualities, dtype=float)
        if c.shape != q.shape or c.size == 0:
            raise MetricsError("one quality per cost, at least one point")
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(q))):
            raise MetricsError("non-finite operating point")
        order = np.lexsort((-q, c))
        self.costs, self.points_q = c[order], q[order]
        self.best = np.maximum.accumulate(self.points_q)

    def quality_at(self, cost):
        """Best quality within ``cost``; None below the cheapest point."""
        i = np.searchsorted(self.costs, cost, side="right") - 1
        return None if i < 0 else float(self.best[i])

    def cost_for(self, quality):
        """Least cost reaching ``quality``; None if never reached."""
        hit = np.flatnonzero(self.best >= quality)
        return None if hit.size == 0 else float(self.costs[hit[0]])

    @property
    def cost_range(self):
        return float(self.costs[0]), float(self.costs[-1])

    def breakpoints(self):
        return self.costs


def e_cost(method, reference, ref_quality):
    """Net cost saved at the reference's quality (G1); None if not reached."""
    cm, cr = method.cost_for(ref_quality), reference.cost_for(ref_quality)
    if cm is None:
        return None
    if cr is None:
        raise MetricsError("the reference envelope must reach its own quality")
    return cr - cm


def e_quality(method, reference, ref_cost):
    """Net quality gained at the reference's cost (G2); None if not reachable."""
    qm, qr = method.quality_at(ref_cost), reference.quality_at(ref_cost)
    if qm is None:
        return None
    if qr is None:
        raise MetricsError("the reference envelope must contain its own cost")
    return qm - qr


def auc_gap(method, reference):
    """Mean Q_m - Q_ref over the shared cost range (exact for step functions)."""
    lo = max(method.cost_range[0], reference.cost_range[0])
    hi = min(method.cost_range[1], reference.cost_range[1])
    if not hi > lo:
        return None
    cuts = np.unique(np.concatenate([[lo, hi], method.costs, reference.costs]))
    cuts = cuts[(cuts >= lo) & (cuts <= hi)]
    area = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        area += (b - a) * (method.quality_at(a) - reference.quality_at(a))
    return area / (hi - lo)


# ---------------------------------------------------------------- bootstrap


def cluster_blocks(clusters):
    order, blocks = [], {}
    for i, c in enumerate(clusters):
        if c not in blocks:
            blocks[c] = []
            order.append(c)
        blocks[c].append(i)
    return [np.asarray(blocks[c], dtype=int) for c in order]


def cluster_bootstrap(statistic, clusters, n_resamples, seed):
    """Replicates of ``statistic(idx) -> {name: float or None}``, paired.

    Every statistic sees the same resampled index set in each replicate. The
    observed values use all units.
    """
    blocks = cluster_blocks(clusters)
    full = np.arange(sum(len(b) for b in blocks))
    observed = statistic(full)
    draws = {k: [] for k in observed}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        values = statistic(np.concatenate([blocks[i] for i in pick]))
        if set(values) != set(observed):
            raise MetricsError("a replicate returned different statistics")
        for k, v in values.items():
            draws[k].append(v)
    return observed, draws, len(blocks)


def summarise(observed, draws, ci_level=0.95):
    """Interval over reached replicates; "not reached" counted, not imputed.

    ``p_greater`` is the one-sided bootstrap p-value for "the value is > 0",
    counting not-reached replicates as failures; ``p_less`` likewise for < 0.
    """
    n = len(draws)
    reached = np.asarray([v for v in draws if v is not None], dtype=float)
    tail = 100 * (1 - ci_level) / 2
    missing = n - reached.size
    out = {"observed": observed, "replicates": n, "not_reached": int(missing),
           "not_reached_fraction": missing / n if n else None}
    if reached.size:
        out["ci_lower"] = float(np.percentile(reached, tail))
        out["ci_upper"] = float(np.percentile(reached, 100 - tail))
    else:
        out["ci_lower"] = out["ci_upper"] = None
    out["p_greater"] = (float(np.sum(reached <= 0)) + missing) / n if n else None
    out["p_less"] = (float(np.sum(reached >= 0)) + missing) / n if n else None
    return out


def format_p(p, replicates):
    """A bootstrap p-value as text: never 0, at most its resolution."""
    if p is None:
        return "n/a"
    floor = 1.0 / replicates
    return f"< {floor:.0e}" if p < floor else f"{p:.4f}"


def _ap_sorted(y_sorted, distinct_end):
    """Average precision for labels already sorted by descending score."""
    tps = np.cumsum(y_sorted)[distinct_end]
    total = tps[-1]
    if total == 0:
        raise MetricsError("both classes are needed")
    fps = (np.flatnonzero(distinct_end) + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / total
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def balanced_ap_fast(y, s):
    """``balanced_ap`` for the bootstrap: same value, no sklearn overhead.

    ``s`` must already be rounded to ``RANK_DECIMALS`` (``_arrays`` does it).
    Thresholds sit at distinct scores, as in sklearn.
    """
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    distinct_end = np.r_[ss[1:] != ss[:-1], True]
    factual = _ap_sorted(ys, distinct_end)
    # hallucinated class, score 1 - s: the reversed order, ties grouped alike
    yr, sr = 1 - ys[::-1], ss[::-1]
    distinct_end_r = np.r_[sr[1:] != sr[:-1], True]
    return 0.5 * (factual + _ap_sorted(yr, distinct_end_r))
