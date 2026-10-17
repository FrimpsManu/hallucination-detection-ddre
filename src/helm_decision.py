"""The HELM preregistration's decision rule (docs/helm_preregistration.md §4-§5).

Pure functions over bootstrap draws of the four primary AUC-gaps:

- H1a/H1b: DDRE-1D vs BSE at C_M/C_FA 28/96 and 14/24;
- H2a/H2b: DDRE-1D vs fixed-k max entailment at 28/96 and 14/24.

Each test's one-sided p-value counts replicates with AUC-gap <= 0 or
undefined (None). Holm's step-down runs over the four at alpha = 0.05. A
hypothesis is supported if Holm rejects and the 5th percentile of the
replicates is above 0; if the 95th percentile is below 0 the method is
reported as below its comparator. Undefined replicates enter the
percentiles as -inf (they count against the hypothesis).
"""

import numpy as np

ALPHA = 0.05
TESTS = {  # name: (method, reference, cost pair) as keyed in src.v2_budget endpoints
    "H1a": ("DDRE-1D", "BSE", "28/96"),
    "H1b": ("DDRE-1D", "BSE", "14/24"),
    "H2a": ("DDRE-1D", "fixed-max", "28/96"),
    "H2b": ("DDRE-1D", "fixed-max", "14/24"),
}
FAMILIES = {"H1": ("H1a", "H1b", "BSE"), "H2": ("H2a", "H2b", "fixed-k max entailment")}


class DecisionError(RuntimeError):
    pass


def endpoint_key(test):
    method, reference, pair = TESTS[test]
    return f"{method}|{reference}|{pair}|auc_gap"


def test_summary(observed, draws):
    """One-sided p-value and the 5th/95th percentiles (undefined = -inf)."""
    if not draws:
        raise DecisionError("no bootstrap replicates")
    values = np.array([-np.inf if v is None else float(v) for v in draws])
    n = len(values)
    return {
        "observed": observed,
        "replicates": n,
        "undefined": int(np.sum(np.isneginf(values))),
        "p_one_sided": float(np.sum(values <= 0) / n),
        "q05": float(np.percentile(values, 5)),
        "q95": float(np.percentile(values, 95)),
    }


def holm(p_values, alpha=ALPHA):
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    out, stopped = {}, False
    for i, k in enumerate(order):
        threshold = alpha / (len(order) - i)
        reject = (not stopped) and p_values[k] < threshold
        stopped = stopped or not reject
        out[k] = {"p": p_values[k], "threshold": threshold, "reject": reject}
    return out


def decide(observed, draws):
    """The preregistered decision from the four primary AUC-gap draws."""
    tests = {t: test_summary(observed[endpoint_key(t)], draws[endpoint_key(t)])
             for t in TESTS}
    h = holm({t: s["p_one_sided"] for t, s in tests.items()})
    for t, s in tests.items():
        s["holm"] = h[t]
        s["supported"] = bool(h[t]["reject"] and s["q05"] > 0)
        s["below"] = bool(s["q95"] < 0)
    statements = {}
    for fam, (a, b, comparator) in FAMILIES.items():
        sa, sb = tests[a]["supported"], tests[b]["supported"]
        if sa and sb:
            text = f"confirmed: DDRE-1D's cost-quality curve lies above {comparator}'s"
        elif sa or sb:
            which = "28/96" if sa else "14/24"
            text = (f"partial support at C_M/C_FA {which}: DDRE-1D's curve lies above "
                    f"{comparator}'s")
        else:
            text = f"not confirmed: DDRE-1D's curve above {comparator}'s"
        below = [TESTS[t][2] for t in (a, b) if tests[t]["below"]]
        if below:
            text += f"; DDRE-1D lies below {comparator} at {', '.join(below)}"
        statements[fam] = text
    return {"tests": tests, "statements": statements}
