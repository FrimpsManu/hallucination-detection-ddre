"""Wang's Bayes-risk look-ahead (Eq. 8) for any likelihood-ratio estimator.

Research plan v2 §3.2. Pure functions; no model, cache or file I/O.

BSE continues before a document iff

    R_stop(pi) > C_retrieve + E_next[R_stop(pi')],
    R_stop(pi) = min((1 - pi) C_M, pi C_FA),

with the expectation over the next document's features under the predictive
mixture pi p_f + (1 - pi) p_h. Because p_f = r p_h, any estimator of the ratio
r together with a sample from p_h gives that expectation:

    E_next[g] = E_h[(pi r + 1 - pi) g(pi')],   pi' = pi r / (pi r + 1 - pi).

For g = R_stop the weight cancels the posterior's denominator:

    (pi r + 1 - pi) R_stop(pi') = min((1 - pi) C_M, pi r C_FA).

So a look-ahead needs only a ``Support``: atoms (log r_i, w_i) describing the
distribution of the next document's log ratio under p_h. For BSE's histogram
the atoms are the ten bins with w_b = P(b | hallucinated), which reproduces
``BSEDetector(mode="eq8")``. For a direct ratio estimator they are its own
hallucinated training documents, w_i = 1/N. As in ``BSEDetector`` the
expectation is divided by the predictive mass sum_i w_i (pi r_i + 1 - pi),
which is exactly 1 for a histogram and for a self-normalised ratio.

The continue decision depends only on pi (and on the next rank, for an
estimator that reads rank). ``ContinueRegion`` finds the set where it holds
once, as a union of open intervals, and verifies it against the exact rule.
"""

import math

import numpy as np

LOG_ODDS_CLIP = 40.0
GRID_POINTS = 20001  # logit-spaced, plus the exact rule at every probe
EDGE_TOL = 1e-13


class SequentialError(RuntimeError):
    pass


def stop_risk(p, c_miss, c_false_alarm):
    return min((1.0 - p) * c_miss, p * c_false_alarm)


def decision(p, c_miss, c_false_alarm):
    """Wang's final rule: factual (1) iff (1 - P) C_M < P C_FA."""
    return 1 if (1.0 - p) * c_miss < p * c_false_alarm else 0


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p):
    p = min(max(float(p), 1e-300), 1.0 - 1e-16)
    return math.log(p) - math.log1p(-p)


# ---------------------------------------------------------------- support


class Support:
    """Atoms (log r_i, w_i) of the next document's log ratio under p_h."""

    def __init__(self, log_ratios, weights):
        lr = np.asarray(log_ratios, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        if lr.shape != w.shape or lr.size == 0:
            raise SequentialError("support needs one weight per atom and at least one atom")
        if not (np.all(np.isfinite(lr)) and np.all(np.isfinite(w)) and np.all(w >= 0)):
            raise SequentialError("support atoms must be finite with non-negative weights")
        if w.sum() <= 0:
            raise SequentialError("support weights sum to zero")
        self.log_ratios = lr
        self.ratios = np.exp(lr)
        self.weights = w / w.sum()

    @property
    def mean_ratio(self):
        """E_h[r]: 1 for a histogram and for a self-normalised ratio."""
        return float(np.dot(self.weights, self.ratios))

    @classmethod
    def from_histogram(cls, pos_hist, neg_hist):
        pos = np.asarray(pos_hist, dtype=float)
        neg = np.asarray(neg_hist, dtype=float)
        if pos.shape != (10,) or neg.shape != (10,) or pos.min() <= 0 or neg.min() <= 0:
            raise SequentialError("histograms must be 10 positive (smoothed) bins")
        p1, p0 = pos / pos.sum(), neg / neg.sum()
        return cls(np.log(p1 / p0), p0)

    @classmethod
    def from_sample(cls, log_ratios):
        """Equal-weight atoms: an estimator's log ratios on its p_h training sample."""
        lr = np.asarray(log_ratios, dtype=float)
        return cls(lr, np.ones_like(lr))

    def expected_next_stop_risk(self, p, c_miss, c_false_alarm):
        """E_next[R_stop(pi')] under the predictive mixture (Eq. 8)."""
        risks = np.minimum((1.0 - p) * c_miss, p * self.ratios * c_false_alarm)
        mass = p * self.mean_ratio + (1.0 - p)
        return float(np.dot(self.weights, risks) / mass)

    def expected_next_stop_risk_grid(self, ps, c_miss, c_false_alarm, chunk=512):
        ps = np.asarray(ps, dtype=float)
        out = np.empty_like(ps)
        for start in range(0, len(ps), chunk):
            p = ps[start:start + chunk, None]
            risks = np.minimum((1.0 - p) * c_miss, p * self.ratios[None, :] * c_false_alarm)
            mass = p[:, 0] * self.mean_ratio + (1.0 - p[:, 0])
            out[start:start + chunk] = (risks @ self.weights) / mass
        return out


class RankedSupport:
    """One ``Support`` per next-document rank, for estimators that read rank."""

    def __init__(self, by_rank):
        if not by_rank:
            raise SequentialError("ranked support is empty")
        self.by_rank = dict(by_rank)

    def at(self, rank):
        if rank not in self.by_rank:
            raise SequentialError(f"no support atoms at rank {rank}")
        return self.by_rank[rank]


# ---------------------------------------------------------------- policy


class Costs:
    def __init__(self, c_miss=28.0, c_false_alarm=96.0, c_retrieve=1.0):
        self.c_miss, self.c_false_alarm = float(c_miss), float(c_false_alarm)
        self.c_retrieve = float(c_retrieve)
        if min(self.c_miss, self.c_false_alarm) <= 0 or self.c_retrieve < 0:
            raise SequentialError("costs must be positive (C_retrieve non-negative)")

    def as_dict(self):
        return {"c_miss": self.c_miss, "c_false_alarm": self.c_false_alarm,
                "c_retrieve": self.c_retrieve}


def continue_margin(p, support, costs):
    """R_stop(pi) - [C_retrieve + E_next R_stop]: continue iff positive (exact)."""
    return (stop_risk(p, costs.c_miss, costs.c_false_alarm) - costs.c_retrieve
            - support.expected_next_stop_risk(p, costs.c_miss, costs.c_false_alarm))


class ContinueRegion:
    """{pi in (0, 1): continue_margin > 0} as a union of open intervals."""

    def __init__(self, support, costs, grid_points=GRID_POINTS):
        self.support, self.costs = support, costs
        x = np.linspace(-LOG_ODDS_CLIP, LOG_ODDS_CLIP, grid_points)
        ps = 1.0 / (1.0 + np.exp(-x))
        stop = np.minimum((1.0 - ps) * costs.c_miss, ps * costs.c_false_alarm)
        g = stop - costs.c_retrieve - support.expected_next_stop_risk_grid(
            ps, costs.c_miss, costs.c_false_alarm)
        inside = g > 0
        self.intervals = []
        start = None
        for i in range(len(ps)):
            if inside[i] and start is None:
                start = self._edge(ps[i - 1], ps[i]) if i > 0 else 0.0
            if not inside[i] and start is not None:
                self.intervals.append((start, self._edge(ps[i], ps[i - 1])))
                start = None
        if start is not None:
            self.intervals.append((start, 1.0))
        self._verify(ps)

    def _margin(self, p):
        return continue_margin(p, self.support, self.costs)

    def _edge(self, outside, inside):
        """The boundary between an outside and an inside probe, by bisection."""
        while abs(inside - outside) > EDGE_TOL:
            mid = 0.5 * (inside + outside)
            if self._margin(mid) > 0:
                inside = mid
            else:
                outside = mid
        return 0.5 * (inside + outside)

    def contains(self, p):
        return any(lo < p < hi for lo, hi in self.intervals)

    def _verify(self, ps):
        rng = np.random.default_rng(0)
        probes = np.concatenate([ps, rng.uniform(0, 1, 2000)])
        for p in probes:
            if not 0.0 < p < 1.0:
                continue
            if self.contains(p) != (self._margin(p) > 0):
                if min((abs(p - e) for iv in self.intervals for e in iv), default=1.0) > 1e-9:
                    raise SequentialError(f"continue region disagrees with the exact rule at {p}")

    def to_json(self):
        return {"intervals": [list(iv) for iv in self.intervals], **self.costs.as_dict(),
                "support_atoms": int(self.support.log_ratios.size),
                "support_mean_ratio": self.support.mean_ratio}


class Eq8Policy:
    """Stop-or-continue by Eq. 8 for one estimator's support and one cost setting."""

    def __init__(self, support, costs):
        self.costs = costs
        if isinstance(support, RankedSupport):
            self.regions = {r: ContinueRegion(s, costs) for r, s in support.by_rank.items()}
            self.ranked = True
        else:
            self.regions = {None: ContinueRegion(support, costs)}
            self.ranked = False

    def should_continue(self, p, next_rank):
        key = next_rank if self.ranked else None
        if key not in self.regions:
            raise SequentialError(f"no continue region for rank {next_rank}")
        return self.regions[key].contains(p)

    def to_json(self):
        return {str(k): v.to_json() for k, v in self.regions.items()}


# ---------------------------------------------------------------- detection


def run_subclaim(log_ratios, spans, policy, max_docs, p0=0.5):
    """(P, documents, NLI calls) under ``policy``. The rule is checked before
    every document, the first included, as in Wang's code; documents are read
    in rank order and the log-odds update is clipped at +-LOG_ODDS_CLIP."""
    p, log_odds = float(p0), _logit(p0)
    docs = nli = 0
    for rank, (lr, n) in enumerate(zip(log_ratios[:max_docs], spans[:max_docs]), start=1):
        if not policy.should_continue(p, rank):
            break
        log_odds = min(max(log_odds + float(lr), -LOG_ODDS_CLIP), LOG_ODDS_CLIP)
        p = _sigmoid(log_odds)
        docs += 1
        nli += int(n)
    return p, docs, nli


def sentence(subclaim_results, costs):
    """Wang's aggregation: P = min over subclaims; costs add up."""
    if not subclaim_results:
        raise SequentialError("a sentence needs at least one subclaim")
    p = min(r[0] for r in subclaim_results)
    return {"p_factual": float(p),
            "prediction": decision(p, costs.c_miss, costs.c_false_alarm),
            "documents": sum(r[1] for r in subclaim_results),
            "nli_calls": sum(r[2] for r in subclaim_results),
            "subclaim_documents": [r[1] for r in subclaim_results]}


# ---------------------------------------------------------------- fixed k

FIXED_K_AGGREGATES = ("max", "avg", "nbc")


def fixed_k_subclaim(entailments, log_ratios, spans, k, how, p0=0.5):
    """Wang's Figure 2 baselines: read the first k documents, then aggregate.

    ``max``/``avg``: the maximum or mean entailment (0-100) mapped to [0, 1];
    ``nbc``: the posterior from P0 after all k log-ratio updates. With no
    document the score is P0. Returns (score, documents, NLI calls).
    """
    if how not in FIXED_K_AGGREGATES:
        raise SequentialError(f"unknown aggregate {how!r}")
    if k < 1:
        raise SequentialError("k must be at least 1")
    e = [float(x) for x in entailments[:k]]
    used = len(e)
    nli = int(sum(spans[:used]))
    if used == 0:
        return float(p0), 0, 0
    if how == "max":
        return max(e) / 100.0, used, nli
    if how == "avg":
        return sum(e) / used / 100.0, used, nli
    log_odds = _logit(p0)
    for lr in log_ratios[:used]:
        log_odds = min(max(log_odds + float(lr), -LOG_ODDS_CLIP), LOG_ODDS_CLIP)
    return _sigmoid(log_odds), used, nli


# ---------------------------------------------------------------- building


def reads_rank(estimator):
    """Whether an estimator's ratio depends on the document's rank."""
    names = getattr(estimator, "names", ())
    return "rank" in names or "log_rank" in names


def build_support(estimator, hallucinated_docs):
    """An estimator's look-ahead support from its own p_h training documents.

    Rank-conditional (one ``Support`` per rank) iff the estimator reads rank.
    Histogram estimators on e alone should use ``Support.from_histogram``.
    """
    if not hallucinated_docs:
        raise SequentialError("no hallucinated training documents")
    if not reads_rank(estimator):
        return Support.from_sample([estimator.log_ratio(d) for d in hallucinated_docs])
    by_rank = {}
    for d in hallucinated_docs:
        by_rank.setdefault(int(d["rank"]), []).append(estimator.log_ratio(d))
    return RankedSupport({r: Support.from_sample(v) for r, v in sorted(by_rank.items())})


def support_to_json(support):
    if isinstance(support, RankedSupport):
        return {"kind": "ranked", "by_rank": {str(r): support_to_json(s)
                                              for r, s in support.by_rank.items()}}
    return {"kind": "atoms", "log_ratios": support.log_ratios.tolist(),
            "weights": support.weights.tolist()}


def support_from_json(d):
    if d["kind"] == "ranked":
        return RankedSupport({int(r): support_from_json(s) for r, s in d["by_rank"].items()})
    if d["kind"] == "atoms":
        return Support(d["log_ratios"], d["weights"])
    raise SequentialError(f"unknown support kind {d['kind']!r}")


# ---------------------------------------------------------------- backward induction

BI_GRID_POINTS = 8001        # value function on a logit grid over +-LOG_ODDS_CLIP
BI_LOG_RATIO_BIN = 0.005     # support compression: atoms pooled within 0.005 nats


def compress(support, bin_width=BI_LOG_RATIO_BIN):
    """Pool atoms whose log ratios fall in the same ``bin_width`` bin.

    Each pooled atom carries the summed weight and the weight-averaged ratio,
    so E_h[r] is preserved exactly; log ratios move by at most ``bin_width``.
    """
    keys = np.floor(support.log_ratios / bin_width).astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys, w, r = keys[order], support.weights[order], support.ratios[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    w_sum = np.add.reduceat(w, starts)
    r_mean = np.add.reduceat(w * r, starts) / w_sum
    return Support(np.log(r_mean), w_sum)


class BackwardInductionPolicy:
    """Bayes-optimal stopping over a finite horizon (plan v2 amendment 1).

    With n documents read and at most ``horizon`` available, the optimal
    expected cost is V_n(pi) = min(R_stop(pi), C_retrieve + E_next V_{n+1}(pi'))
    with V_horizon = R_stop. The expectation uses the same predictive mixture
    as Eq. 8, over the support at the next document's rank n + 1. With
    horizon 1 this is exactly Eq. 8. V is held on a logit grid and read by
    linear interpolation; the continue set of each stage is extracted as open
    intervals with edges refined by bisection, as ``ContinueRegion`` does.
    """

    def __init__(self, support, costs, horizon, grid_points=BI_GRID_POINTS):
        if horizon < 1:
            raise SequentialError("horizon must be at least 1")
        self.costs, self.horizon = costs, int(horizon)
        self.ranked = isinstance(support, RankedSupport)
        self._support = support
        self.x = np.linspace(-LOG_ODDS_CLIP, LOG_ODDS_CLIP, grid_points)
        self.p = 1.0 / (1.0 + np.exp(-self.x))
        self.stop = np.minimum((1.0 - self.p) * costs.c_miss, self.p * costs.c_false_alarm)
        self._compressed = {}
        values = self.stop.copy()
        self.values = {self.horizon: values}
        self.intervals = {}
        for n in range(self.horizon - 1, -1, -1):
            nxt = values
            margin = self.stop - costs.c_retrieve - self._expected(n, self.p, self.x, nxt)
            values = np.where(margin > 0, self.stop - margin, self.stop)
            self.values[n] = values
            self.intervals[n] = self._intervals(n, margin > 0, nxt)

    def _atoms(self, n):
        if n not in self._compressed:
            s = self._support.at(n + 1) if self.ranked else self._support
            self._compressed[n] = compress(s)
        return self._compressed[n]

    def _expected(self, n, p, x, nxt):
        """E_next V_{n+1} at posteriors p (logits x), for the document of rank n + 1."""
        s = self._atoms(n)
        p = np.atleast_1d(p)[:, None]
        x = np.atleast_1d(x)[:, None]
        pred = p * s.ratios[None, :] + (1.0 - p)
        if n + 1 == self.horizon:
            # V_horizon = R_stop exactly; the weight cancels the posterior's
            # denominator, as in Support.expected_next_stop_risk.
            risks = np.minimum((1.0 - p) * self.costs.c_miss,
                               p * s.ratios[None, :] * self.costs.c_false_alarm)
            return risks @ s.weights / (pred @ s.weights)
        nx = np.clip(x + s.log_ratios[None, :], -LOG_ODDS_CLIP, LOG_ODDS_CLIP)
        v = np.interp(nx, self.x, nxt)
        return (v * pred) @ s.weights / (pred @ s.weights)

    def _margin(self, n, p, nxt):
        x = _logit(p)
        return (stop_risk(p, self.costs.c_miss, self.costs.c_false_alarm)
                - self.costs.c_retrieve - float(self._expected(n, p, x, nxt)[0]))

    def _intervals(self, n, inside, nxt):
        def edge(outside, inner):
            while abs(inner - outside) > EDGE_TOL:
                mid = 0.5 * (inner + outside)
                if self._margin(n, mid, nxt) > 0:
                    inner = mid
                else:
                    outside = mid
            return 0.5 * (inner + outside)

        out, start, ps = [], None, self.p
        for i in range(len(ps)):
            if inside[i] and start is None:
                start = edge(ps[i - 1], ps[i]) if i > 0 else 0.0
            if not inside[i] and start is not None:
                out.append((start, edge(ps[i], ps[i - 1])))
                start = None
        if start is not None:
            out.append((start, 1.0))
        return out

    def should_continue(self, p, next_rank):
        n = next_rank - 1
        if not 0 <= n < self.horizon:
            return False
        return any(lo < p < hi for lo, hi in self.intervals[n])

    def to_json(self):
        return {"horizon": self.horizon, **self.costs.as_dict(),
                "intervals_by_stage": {str(n): [list(iv) for iv in self.intervals[n]]
                                       for n in sorted(self.intervals)}}


class PolicyByHorizon:
    """One backward-induction policy per horizon, built on first use.

    A subclaim with fewer available documents than ``max_docs`` is planned
    over the documents it actually has.
    """

    def __init__(self, support, costs, max_docs):
        self.support, self.costs, self.max_docs = support, costs, int(max_docs)
        self._cache = {}

    def for_documents(self, available):
        h = min(int(available), self.max_docs)
        if h < 1:
            return None
        if h not in self._cache:
            self._cache[h] = BackwardInductionPolicy(self.support, self.costs, h)
        return self._cache[h]


def run_subclaim_bi(log_ratios, spans, policies, max_docs, p0=0.5):
    """``run_subclaim`` under the backward-induction policy for this subclaim's
    number of available documents. No document available: (P0, 0, 0)."""
    policy = policies.for_documents(min(len(log_ratios), max_docs))
    if policy is None:
        return float(p0), 0, 0
    return run_subclaim(log_ratios, spans, policy, max_docs, p0)
