"""Held-out ablation replay: frozen cells, consistency gate, bootstrap, rules.

Implements preregistration §6-§8 for the cells frozen on validation:

  A  published BSE official (CM 28, CFA 96, C_retrieve 1)
  B  histogram evidence, band [0.20, 0.85]        (validation freeze)
  C  2.5 x histogram evidence, band [0.05, 0.95]  (validation freeze)
  D  frozen uLSIF sigma 0.11346435546875, lambda 1.0, band [0.20, 0.80]
  D0 not evaluated: no eligible validation configuration

EXPLORATORY / POST-HOC held-out ablation, not confirmatory. Nothing here
selects, tunes or searches. Every configuration is a constant checked
against the frozen validation artifact; there is no parameter to override.

Pure: numpy only. The script does the I/O and the replay.
"""

import json
import math

import numpy as np

from src.paired_bootstrap import (
    BootstrapUnavailable,
    _metrics_on,
    _wang_pr_auc,
    passage_blocks,
    percentile_interval,
    replicate_indices,
    validate_paired_inputs,
)

STATUS = "exploratory / post-hoc held-out ablation"
NOT_CONFIRMATORY = (
    "This replay is an exploratory / post-hoc held-out ablation, not a "
    "confirmatory analysis. The held-out split was already used once for the "
    "frozen BSE-official vs frozen-DDRE comparison."
)

# ---------------------------------------------------------------- frozen cells
EXPECTED_B = {"kappa": 1.0, "lower": 0.20, "upper": 0.85}
EXPECTED_C = {"kappa": 2.5, "lower": 0.05, "upper": 0.95}
EXPECTED_D = {"sigma": 0.11346435546875, "lambda": 1.0, "lower": 0.20, "upper": 0.80}
COSTS = {"c_miss": 28.0, "c_false_alarm": 96.0, "c_retrieve": 1.0, "p0": 0.5, "max_docs": 10}
CELLS = ("A", "B", "C", "D")
TUNING_BUDGETS = {"A": 0, "B": 32, "C": 224, "D": 640}
D0_NOT_EVALUATED = (
    "D0 (original CV-selected uLSIF, sigma 0.2269287109375, lambda 1.0) had "
    "no_eligible_configuration on validation, so per preregistration §4 it is "
    "not evaluated on held-out. This re-derives the result the sensitivity "
    "artifact had already recorded; it is a selection-history result, not a "
    "new finding."
)

# ---------------------------------------------------------------- bootstrap
N_RESAMPLES = 10_000
SEED = 42
CI_LEVEL = 0.95
RATIO_DENOMINATOR_GUARD = 0.05  # documents per sentence, preregistration §6

# ---------------------------------------------------------------- rules (§7, §8)
A_TO_B_SHARE = 0.5
B_TO_C_SHARE = 0.8
BALANCED_MARGIN = 0.01
PREDICTION_SHARE = 0.25

# Machine-precision allowance used ONLY to implement the inclusive "within
# 0.01" boundary in binary floating point. Not a scientific tolerance.
ROUNDOFF_ULPS = 4

FORBIDDEN_CAUSAL_PHRASES = ("proves", "causes", "is caused by", "comes from", "explains why")


class ReplayRefused(RuntimeError):
    """A precondition failed. No ablation result is reported."""


class ConsistencyGateFailed(ReplayRefused):
    """A or D did not reproduce the canonical frozen predictions exactly."""


# ---------------------------------------------------------------- freeze

def frozen_cells(freeze):
    """B and C exactly as frozen on validation; D0's recorded outcome.

    Everything is read from the artifact and checked against the constants.
    Any disagreement refuses: the replay cannot run a configuration that is
    not the frozen one, and has no way to be told another.
    """
    problems = []
    cells = {}
    for cell, expected in (("B", EXPECTED_B), ("C", EXPECTED_C)):
        outcome = freeze.get(cell) or {}
        chosen = outcome.get("selected") or {}
        if outcome.get("outcome") != "selected":
            problems.append(f"{cell}: frozen outcome is {outcome.get('outcome')!r}")
            continue
        got = {"kappa": float(chosen.get("kappa", 1.0)),
               "lower": chosen.get("lower"), "upper": chosen.get("upper")}
        if got != expected:
            problems.append(f"{cell}: frozen {got} != expected {expected}")
        cells[cell] = got
    d0 = freeze.get("D0") or {}
    if d0.get("outcome") != "no_eligible_configuration" or d0.get("selected") is not None:
        problems.append(f"D0: frozen outcome {d0.get('outcome')!r}; expected "
                        "no_eligible_configuration")
    counts = freeze.get("candidate_counts")
    if counts != {"B": 32, "C": 224, "D0": 32}:
        problems.append(f"candidate counts {counts}")
    if freeze.get("held_out_records_evaluated") != 0 or freeze.get("held_out_cache_opened"):
        problems.append("the validation freeze does not record a held-out-free selection")
    if problems:
        raise ReplayRefused("validation freeze:\n  " + "\n  ".join(problems))
    return {"A": dict(COSTS), "B": cells["B"], "C": cells["C"], "D": dict(EXPECTED_D),
            "D0": {"evaluated": False, "reason": D0_NOT_EVALUATED,
                   "validation_eligible": d0.get("eligible"),
                   "validation_candidates": d0.get("candidates")}}


# ---------------------------------------------------------------- gate

def consistency_gate(cell, canonical_rows, replayed_rows):
    """Exact reproduction of the canonical frozen predictions for one method.

    Compares final posterior, prediction, retrieved documents and NLI span
    calls, plus the per-subclaim vectors, sentence by sentence. Raises on any
    difference; never reports a partial match as a pass.
    """
    canon = {(int(r["passage_index"]), int(r["sentence_index"])): r for r in canonical_rows}
    mine = {(int(r["passage_index"]), int(r["sentence_index"])): r for r in replayed_rows}
    problems = []
    if set(canon) != set(mine):
        problems.append(f"sentence sets differ: {len(set(canon) ^ set(mine))} keys")
    for key in sorted(set(canon) & set(mine)):
        a, b = canon[key], mine[key]
        checks = (
            ("p_factual", float(a["p_factual"]), float(b["p_factual"])),
            ("prediction", int(a["prediction"]), int(b["prediction"])),
            ("retrieved_documents", int(a["retrieved_documents"]), int(b["retrieved_documents"])),
            ("nli_span_calls", int(a["nli_span_calls"]), int(b["nli_span_calls"])),
            ("subclaim_documents_used", json.loads(a["subclaim_documents_used"]),
             json.loads(b["subclaim_documents_used"])),
            ("subclaim_nli_calls", json.loads(a["subclaim_nli_calls"]),
             json.loads(b["subclaim_nli_calls"])),
        )
        problems += [f"{key} {f}: canonical {x!r}, replayed {y!r}" for f, x, y in checks if x != y]
    if problems:
        raise ConsistencyGateFailed(
            f"cell {cell} does not reproduce the canonical frozen predictions "
            f"({len(problems)} differences):\n  " + "\n  ".join(problems[:20])
        )
    return {"cell": cell, "sentences": len(canon), "exact": True}


# ---------------------------------------------------------------- bootstrap

def _quantities(m):
    """The preregistered quantities from per-cell metrics on one index set."""
    bal = {c: 0.5 * (m[c]["nonfactual_auc_pr"] + m[c]["factual_auc_pr"]) for c in CELLS}
    docs = {c: m[c]["documents_per_sentence"] for c in CELLS}
    nli = {c: m[c]["nli_calls_per_sentence"] for c in CELLS}
    return {
        "docs_S_B": docs["A"] - docs["B"],
        "docs_S_C": docs["A"] - docs["C"],
        "docs_S_D": docs["A"] - docs["D"],
        "docs_B_minus_C": docs["B"] - docs["C"],
        "docs_C_minus_D": docs["C"] - docs["D"],
        "nli_A_minus_B": nli["A"] - nli["B"],
        "nli_A_minus_C": nli["A"] - nli["C"],
        "nli_A_minus_D": nli["A"] - nli["D"],
        "nli_B_minus_C": nli["B"] - nli["C"],
        "nli_C_minus_D": nli["C"] - nli["D"],
        "balanced_B_minus_A": bal["B"] - bal["A"],
        "balanced_C_minus_B": bal["C"] - bal["B"],
        "balanced_D_minus_C": bal["D"] - bal["C"],
    }


RATIOS = {"ratio_S_B_over_S_D": "docs_S_B", "ratio_S_C_over_S_D": "docs_S_C"}


def validate_cells(observations):
    """Every cell scored on the identical sentence sequence as A."""
    if set(observations) != set(CELLS):
        raise ReplayRefused(f"bootstrap needs cells {CELLS}, got {sorted(observations)}")
    shape = None
    for cell in CELLS[1:]:
        shape = validate_paired_inputs(observations[cell], observations["A"])
    return shape


def multi_cell_bootstrap(observations, *, n_resamples=N_RESAMPLES, seed=SEED,
                         ci_level=CI_LEVEL, pr_auc=None):
    """Paired passage-level cluster bootstrap across A, B, C and D.

    Each replicate draws passages with replacement ONCE and applies the same
    sentence indices to every cell, so every difference is paired. A single-class
    replicate fails closed (BootstrapUnavailable), as in the frozen protocol.

    Ratio guard (§6): if any replicate has S_D <= 0.05 docs/sentence, no
    replicate is discarded; the ratio intervals are marked unstable and not
    reported, and the count is recorded.
    """
    shape = validate_cells(observations)
    pr_auc = _wang_pr_auc() if pr_auc is None else pr_auc
    arrays = {}
    for cell in CELLS:
        obs = observations[cell]
        arrays[cell] = {
            "label": np.asarray([int(o.gold_label) for o in obs], dtype=int),
            "p_factual": np.asarray([float(o.result.p_factual) for o in obs], dtype=float),
            "documents": np.asarray([float(o.result.documents_used) for o in obs], dtype=float),
            "nli_calls": np.asarray([float(o.result.nli_calls) for o in obs], dtype=float),
        }
    blocks = passage_blocks(observations["A"])

    full = np.arange(len(observations["A"]), dtype=int)
    observed = _quantities({c: _metrics_on(arrays[c], full, pr_auc) for c in CELLS})
    for name, numerator in RATIOS.items():
        sd = observed["docs_S_D"]
        observed[name] = observed[numerator] / sd if sd > RATIO_DENOMINATOR_GUARD else None

    rng = np.random.default_rng(seed)
    draws = {k: [] for k in _quantities_keys()}
    ratio_draws = {k: [] for k in RATIOS}
    guarded = 0
    for _ in range(n_resamples):
        indices, _draw = replicate_indices(blocks, rng)
        q = _quantities({c: _metrics_on(arrays[c], indices, pr_auc) for c in CELLS})
        for k, v in q.items():
            draws[k].append(v)
        if q["docs_S_D"] <= RATIO_DENOMINATOR_GUARD:
            guarded += 1
            continue
        for name, numerator in RATIOS.items():
            ratio_draws[name].append(q[numerator] / q["docs_S_D"])

    endpoints = {}
    for k, values in draws.items():
        lo, hi, lo_pct, hi_pct = percentile_interval(values, ci_level)
        endpoints[k] = {"observed": observed[k], "ci_lower": lo, "ci_upper": hi,
                        "ci_percentiles": [lo_pct, hi_pct]}
    for name in RATIOS:
        if guarded:
            endpoints[name] = {"observed": observed[name], "ci_lower": None, "ci_upper": None,
                               "ci_status": "unstable",
                               "replicates_at_or_below_guard": guarded}
        else:
            lo, hi, lo_pct, hi_pct = percentile_interval(ratio_draws[name], ci_level)
            endpoints[name] = {"observed": observed[name], "ci_lower": lo, "ci_upper": hi,
                               "ci_percentiles": [lo_pct, hi_pct], "ci_status": "reported",
                               "replicates_at_or_below_guard": 0}
    return {
        "unit": "passage",
        "n_resamples": n_resamples,
        "seed": seed,
        "ci_level": ci_level,
        "ci_method": "percentile",
        "paired_across": list(CELLS),
        "ratio_denominator_guard_docs_per_sentence": RATIO_DENOMINATOR_GUARD,
        "shape": shape,
        "passages": len(blocks),
        "endpoints": endpoints,
    }


def _quantities_keys():
    dummy = {c: {"nonfactual_auc_pr": 0.0, "factual_auc_pr": 0.0,
                 "documents_per_sentence": 0.0, "nli_calls_per_sentence": 0.0} for c in CELLS}
    return list(_quantities(dummy))


# ---------------------------------------------------------------- interpretation

def within_closed_margin(difference, *operands, margin=BALANCED_MARGIN):
    """``difference <= margin``, with the preregistered boundary inclusive.

    Numerical implementation of the closed boundary, not a tolerance change.
    In binary floating point 0.74 - 0.73 evaluates to 0.010000000000000009, so
    a difference that is exactly 0.01 in decimal would fail a plain ``<=``.
    The allowance is a few units in the last place of the floats involved
    (about 1e-16 at PR-AUC scale), i.e. pure roundoff; any difference that
    exceeds 0.01 by more than roundoff still fails.
    """
    allowance = ROUNDOFF_ULPS * max(math.ulp(abs(x)) for x in (margin, *operands))
    return difference <= margin + allowance


def interpret(endpoints, balanced):
    """The preregistered rules (§7) and prediction (§8), evaluated mechanically.

    ``balanced`` maps cell -> observed held-out balanced PR-AUC. Statements are
    the preregistered wording when a condition holds; otherwise the record
    says only that the condition was not met.
    """
    s_b = endpoints["docs_S_B"]["observed"]
    s_c = endpoints["docs_S_C"]["observed"]
    s_d = endpoints["docs_S_D"]["observed"]
    cd = endpoints["docs_C_minus_D"]
    a_b = s_b >= A_TO_B_SHARE * s_d
    b_c = (s_c >= B_TO_C_SHARE * s_d) and within_closed_margin(
        abs(balanced["C"] - balanced["D"]), balanced["C"], balanced["D"])
    ci_excludes_zero = cd["ci_lower"] > 0 or cd["ci_upper"] < 0
    # "D's balanced PR-AUC is not lower than C's by more than 0.01".
    c_d = ci_excludes_zero and within_closed_margin(
        balanced["C"] - balanced["D"], balanced["C"], balanced["D"])
    ratio_b = endpoints["ratio_S_B_over_S_D"]["observed"]
    return {
        "status": STATUS,
        "A_to_B": {
            "rule": "S(B) >= 0.5 x S(D)",
            "S_B": s_b, "S_D": s_d, "ratio_S_B_over_S_D": ratio_b,
            "condition_met": bool(a_b),
            "statement": (
                "The stopping-rule change accounts for a substantial share of "
                "DDRE's observed saving, and most of the saving cannot be "
                "attributed solely to the DDRE evidence model."
                if a_b else "The preregistered A -> B condition was not met."),
        },
        "B_to_C": {
            "rule": "S(C) >= 0.8 x S(D) and |balanced(C) - balanced(D)| <= 0.01",
            "direct_effect_docs_B_minus_C": endpoints["docs_B_minus_C"],
            "S_C": s_c, "S_D": s_d,
            "ratio_S_C_over_S_D": endpoints["ratio_S_C_over_S_D"]["observed"],
            "balanced_C": balanced["C"], "balanced_D": balanced["D"],
            "condition_met": bool(b_c),
            "statement": (
                "Evidence that the magnitude assigned to weak evidence accounts "
                "for most of the observed efficiency behaviour, making the "
                "detailed uLSIF shape secondary."
                if b_c else "The preregistered B -> C condition was not met."),
        },
        "C_to_D": {
            "rule": ("95% CI of docs(C) - docs(D) excludes 0 and "
                     "balanced(D) >= balanced(C) - 0.01"),
            "docs_C_minus_D": cd,
            "ci_excludes_zero": bool(ci_excludes_zero),
            "balanced_C": balanced["C"], "balanced_D": balanced["D"],
            "condition_met": bool(c_d),
            "statement": (
                "In this exploratory analysis, the learned continuous "
                "density-ratio shape contributes beyond a global rescaling."
                if c_d else "The preregistered C -> D condition was not met."),
        },
        "D0_to_D": {"rule": "no binary threshold; validation outcome reported directly",
                    "statement": D0_NOT_EVALUATED},
        "prediction": {
            "text": ("I expect B to recover well under one quarter of D's document "
                     "saving, because the modal BSE histogram bucket contributes only "
                     "about -0.14 log-odds per weak-support document, whereas frozen "
                     "DDRE contributes roughly -0.44."),
            "rule": "S(B) / S(D) < 0.25",
            "ratio_S_B_over_S_D": ratio_b,
            "held": None if ratio_b is None else bool(ratio_b < PREDICTION_SHARE),
        },
        "tuning_budgets": dict(TUNING_BUDGETS),
        "tuning_budget_note": ("Search budgets differ (A 0, B 32, C 224, D 640). This "
                               "favours the cells further down and is disclosed with "
                               "every comparison."),
        "language_rule": "No proof or causal language is used for any outcome.",
    }


def check_language(text):
    lowered = text.lower()
    found = [p for p in FORBIDDEN_CAUSAL_PHRASES if p in lowered]
    if found:
        raise ValueError(f"causal or proof language in the result: {found}")
    return text
