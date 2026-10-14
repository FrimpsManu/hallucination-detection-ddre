"""FActScore frozen-transfer analysis (transfer preregistration §6-§8).

Pure functions over replay observations:

- the paired entity-cluster bootstrap over A, B, C and D (10,000 resamples,
  seed 42, 95% percentile intervals), computing the SelfCheckGPT ablation
  quantities plus the D - A quality differences this preregistration adds;
- the RQ1 decision rules (efficiency CI above 0; balanced PR-AUC D - A lower
  bound >= -0.03);
- the score-support diagnostic (shares above 76.27, 83.35 and at or above 95;
  "outside the frozen estimator's support" if more than 10% of D's consumed
  scores exceed 83.35).

The SelfCheckGPT replay module is reused, not modified: its resampling, metric
and interval functions and its interpretation rules are imported as they are.
"""

import numpy as np

from src import ablation_replay as AR
from src.paired_bootstrap import (
    _metrics_on,
    _wang_pr_auc,
    passage_blocks,
    percentile_interval,
    replicate_indices,
)

STATUS = "preregistered frozen-transfer test (FActScore); RQ2 rules exploratory"
QUALITY_MARGIN = 0.03
SUPPORT_THRESHOLDS = (76.26953125, 83.35, 95.0)
SUPPORT_LIMIT_SHARE = 0.10
SUPPORT_EDGE = 83.35
N_RESAMPLES, SEED, CI_LEVEL = AR.N_RESAMPLES, AR.SEED, AR.CI_LEVEL
RATIO_GUARD = AR.RATIO_DENOMINATOR_GUARD


def quantities(m):
    """SelfCheckGPT ablation quantities plus the D - A quality differences."""
    q = AR._quantities(m)
    for name in ("nonfactual_auc_pr", "factual_auc_pr"):
        q[f"{name}_D_minus_A"] = m["D"][name] - m["A"][name]
    bal = {c: 0.5 * (m[c]["nonfactual_auc_pr"] + m[c]["factual_auc_pr"]) for c in "AD"}
    q["balanced_D_minus_A"] = bal["D"] - bal["A"]
    return q


def bootstrap(observations, *, n_resamples=N_RESAMPLES, seed=SEED, ci_level=CI_LEVEL,
              pr_auc=None):
    """Paired cluster bootstrap across A-D. Clusters = ``passage_index`` (the entity).

    The same procedure as the SelfCheckGPT replay: one cluster draw per
    replicate applied to every cell, single-class replicates fail closed, and
    the ratio intervals are marked unstable rather than reported if any replicate
    has S_D <= 0.05.
    """
    shape = AR.validate_cells(observations)
    pr_auc = _wang_pr_auc() if pr_auc is None else pr_auc
    arrays = {c: {
        "label": np.asarray([int(o.gold_label) for o in observations[c]], dtype=int),
        "p_factual": np.asarray([float(o.result.p_factual) for o in observations[c]]),
        "documents": np.asarray([float(o.result.documents_used) for o in observations[c]]),
        "nli_calls": np.asarray([float(o.result.nli_calls) for o in observations[c]]),
    } for c in AR.CELLS}
    blocks = passage_blocks(observations["A"])
    full = np.arange(len(observations["A"]), dtype=int)
    observed = quantities({c: _metrics_on(arrays[c], full, pr_auc) for c in AR.CELLS})
    for name, numerator in AR.RATIOS.items():
        sd = observed["docs_S_D"]
        observed[name] = observed[numerator] / sd if sd > RATIO_GUARD else None

    rng = np.random.default_rng(seed)
    draws = {k: [] for k in observed if k not in AR.RATIOS}
    ratio_draws = {k: [] for k in AR.RATIOS}
    guarded = 0
    for _ in range(n_resamples):
        indices, _ = replicate_indices(blocks, rng)
        q = quantities({c: _metrics_on(arrays[c], indices, pr_auc) for c in AR.CELLS})
        for k in draws:
            draws[k].append(q[k])
        if q["docs_S_D"] <= RATIO_GUARD:
            guarded += 1
            continue
        for name, numerator in AR.RATIOS.items():
            ratio_draws[name].append(q[numerator] / q["docs_S_D"])

    endpoints = {}
    for k, values in draws.items():
        lo, hi, lp, hp = percentile_interval(values, ci_level)
        endpoints[k] = {"observed": observed[k], "ci_lower": lo, "ci_upper": hi,
                        "ci_percentiles": [lp, hp]}
    for name in AR.RATIOS:
        if guarded:
            endpoints[name] = {"observed": observed[name], "ci_lower": None, "ci_upper": None,
                               "ci_status": "unstable", "replicates_at_or_below_guard": guarded}
        else:
            lo, hi, lp, hp = percentile_interval(ratio_draws[name], ci_level)
            endpoints[name] = {"observed": observed[name], "ci_lower": lo, "ci_upper": hi,
                               "ci_percentiles": [lp, hp], "ci_status": "reported",
                               "replicates_at_or_below_guard": 0}
    return {"unit": "entity", "n_resamples": n_resamples, "seed": seed, "ci_level": ci_level,
            "ci_method": "percentile", "paired_across": list(AR.CELLS),
            "ratio_denominator_guard_docs_per_sentence": RATIO_GUARD,
            "shape": shape, "clusters": len(blocks), "endpoints": endpoints}


def rq1(endpoints, margin=QUALITY_MARGIN):
    """The preregistered RQ1 rules (§7). Mechanical; no other threshold."""
    sd = endpoints["docs_S_D"]
    bal = endpoints["balanced_D_minus_A"]
    efficiency = sd["ci_lower"] > 0
    comparable = bal["ci_lower"] >= -margin
    return {
        "efficiency_transfers": bool(efficiency),
        "quality_comparable": bool(comparable),
        "rq1_supported_on_factscore": bool(efficiency and comparable),
        "quality_improvement_shown": bool(bal["ci_lower"] > 0),
        "rules": {
            "efficiency": "95% CI of S(D) = docs(A) - docs(D) lies entirely above 0",
            "quality": f"lower 95% CI of balanced PR-AUC(D) - (A) >= -{margin}",
            "improvement": "lower 95% CI of balanced PR-AUC(D) - (A) > 0 (reported, not required)",
        },
        "S_D": sd,
        "balanced_D_minus_A": bal,
    }


def rq2(endpoints, balanced):
    """SelfCheckGPT's A->B, B->C, C->D rules applied unchanged (§7, secondary)."""
    out = AR.interpret(endpoints, balanced)
    return {k: out[k] for k in ("A_to_B", "B_to_C", "C_to_D", "tuning_budgets",
                                 "tuning_budget_note", "language_rule")}


def support_shares(scores):
    scores = np.asarray(scores, dtype=float)
    n = int(scores.size)
    t1, t2, t3 = SUPPORT_THRESHOLDS
    return {
        "documents": n,
        "share_above_76.27": float(np.mean(scores > t1)) if n else None,
        "share_above_83.35": float(np.mean(scores > t2)) if n else None,
        "share_at_or_above_95": float(np.mean(scores >= t3)) if n else None,
        "max": float(scores.max()) if n else None,
    }


def support_diagnostic(consumed_by_cell, all_retrieved):
    """§8: shares per cell and over all retrieved documents, plus the D label."""
    cells = {c: support_shares(v) for c, v in consumed_by_cell.items()}
    d_share = cells["D"]["share_above_83.35"]
    outside = d_share is not None and d_share > SUPPORT_LIMIT_SHARE
    return {
        "consumed": cells,
        "all_retrieved": support_shares(all_retrieved),
        "d_share_above_83.35": d_share,
        "limit_share": SUPPORT_LIMIT_SHARE,
        "outside_frozen_estimator_support": bool(outside),
        "statement": (
            "More than 10% of the documents D consumed score above 83.35, so the "
            "RQ1 result is outside the frozen estimator's support and is not read "
            "as evidence for or against the method in its supported range."
            if outside else
            "At most 10% of the documents D consumed score above 83.35; the RQ1 "
            "result is read within the frozen estimator's support, with these "
            "shares as context."),
    }


def subset(observations, keep):
    """Observations of every cell restricted to sentences where ``keep(obs)``."""
    return {c: [o for o in obs if keep(o)] for c, obs in observations.items()}
