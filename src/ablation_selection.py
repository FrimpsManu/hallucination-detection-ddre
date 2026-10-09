"""Validation selection for ablation cells B, C and D0 (preregistration §4).

``docs/ablation_preregistration.md`` freezes three validation selections:

* B  -- histogram log-likelihood evidence, posterior band, 32 bands;
* C  -- kappa x histogram log-likelihood evidence, band, 7 x 32 = 224;
* D0 -- the original CV-selected uLSIF (sigma 0.2269287109375, lambda 1.0),
        band, 32 bands.

All three use the rule that froze D: eligible iff nonfactual, factual and
balanced validation PR-AUC are each >= the stored BSE reference - 0.005;
among eligible, fewest documents per sentence, then higher balanced,
nonfactual and factual PR-AUC, then fewer NLI spans per sentence, then the
parameters in ascending order (kappa for C, then lower, then upper). A cell
with no eligible configuration records exactly that; there is no fallback.

This module is pure: no torch, no cache, no records. The script supplies
validation metrics; this decides eligibility and selection and builds the
freeze artifact. Nothing here can see a held-out record.
"""

import math

from src.baseline_core import discretize_document_score

SELECTION_RULE_VERSION = "ablation-validation-selection-v1"
PREREGISTRATION_PATH = "docs/ablation_preregistration.md"
PREREGISTRATION_SHA256 = "07fa666e3a5550123e06fefeb905d2057506bb0e7421ea23534485cfd31b587d"
PREREGISTRATION_MERGE_COMMIT = "d9238d682c51c39df4540ca1769960cd6bf2653f"

QUALITY_TOLERANCE = 0.005
LOWER_GRID = (0.05, 0.10, 0.15, 0.20)
UPPER_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
BAND_GRID = tuple((lower, upper) for lower in LOWER_GRID for upper in UPPER_GRID)
KAPPA_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
D0_SIGMA = 0.2269287109375
D0_LAMBDA = 1.0
EXPECTED_CANDIDATES = {"B": 32, "C": 224, "D0": 32}

NO_ELIGIBLE = "no_eligible_configuration"
SELECTED = "selected"

SELECTION_RULE = (
    "Eligible iff validation nonfactual, factual and balanced PR-AUC are each "
    ">= the stored BSE validation reference - 0.005. Among eligible: "
    "(1) minimum documents per sentence; (2) higher balanced PR-AUC; "
    "(3) higher nonfactual PR-AUC; (4) higher factual PR-AUC; (5) fewer NLI "
    "span calls per sentence; (6) ascending kappa (C only), lower, upper. "
    "No eligible configuration -> no_eligible_configuration; no fallback."
)

# Where the stored BSE validation reference lives in the pinned sensitivity
# artifact. The gate compares against these values as read, never literals.
REFERENCE_FIELDS = {
    "nonfactual_auc_pr": ("nonfactual", "auc_pr"),
    "factual_auc_pr": ("factual", "auc_pr"),
    "balanced_pr_auc": ("balanced_pr_auc",),
}


class SelectionRefused(RuntimeError):
    """A precondition of the preregistered selection failed. Nothing frozen."""


# ------------------------------------------------------------------ evidence

class HistogramRatio:
    """BSE's per-document evidence as a ratio estimator for ``DDREDetector``.

    ratio(s) = exp(kappa * LLR(s)), LLR(s) = log(P(b|factual) / P(b|halluc)),
    b = discretize_document_score(s), with the Laplace-smoothed NBC histograms
    exactly as BSE uses them. kappa = 1 is cell B.
    """

    def __init__(self, pos_hist, neg_hist, kappa=1.0):
        if len(pos_hist) != 10 or len(neg_hist) != 10:
            raise ValueError("histograms must have 10 buckets")
        if min(pos_hist) <= 0 or min(neg_hist) <= 0:
            raise ValueError("histograms must be Laplace-smoothed (strictly positive)")
        if not (math.isfinite(kappa) and kappa > 0):
            raise ValueError(f"kappa must be finite and positive, got {kappa!r}")
        total_pos, total_neg = float(sum(pos_hist)), float(sum(neg_hist))
        self.kappa = float(kappa)
        self.llr = tuple(
            math.log((p / total_pos) / (n / total_neg))
            for p, n in zip(pos_hist, neg_hist)
        )

    def log_ratio(self, score):
        if not math.isfinite(score):
            raise ValueError(f"non-finite document score {score!r}")
        return self.kappa * self.llr[discretize_document_score(score)]

    def ratio(self, score):
        return math.exp(self.log_ratio(score))


# ------------------------------------------------------------------ grid

def verify_band_grid(cost_consistent_space):
    """The 32 bands must be exactly the cost-consistent pairs the repo computes."""
    computed = tuple(
        (float(lower), float(upper))
        for lower in cost_consistent_space["effective_lower_grid"]
        for upper in cost_consistent_space["effective_upper_grid"]
        if lower < upper
    )
    if computed != BAND_GRID:
        raise SelectionRefused(f"cost-consistent bands {computed} != preregistered {BAND_GRID}")
    return BAND_GRID


def cell_parameters(cell):
    if cell in ("B", "D0"):
        return [{"lower": lo, "upper": up} for lo, up in BAND_GRID]
    if cell == "C":
        return [{"kappa": k, "lower": lo, "upper": up} for k in KAPPA_GRID for lo, up in BAND_GRID]
    raise ValueError(f"unknown cell {cell!r}")


# ------------------------------------------------------------------ reference

def stored_reference(sensitivity_artifact):
    """The stored BSE validation reference, at full precision, from the artifact."""
    bse = sensitivity_artifact.get("bse_official_validation") or {}
    out = {}
    for name, path in REFERENCE_FIELDS.items():
        node = bse
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, float):
            raise SelectionRefused(f"stored reference {name} is not recorded as a float: {node!r}")
        out[name] = node
    return out


def verify_reference(replayed_metrics, reference):
    """Exact float equality between the replayed BSE and the stored reference."""
    replayed = {
        "nonfactual_auc_pr": replayed_metrics["nonfactual"]["auc_pr"],
        "factual_auc_pr": replayed_metrics["factual"]["auc_pr"],
        "balanced_pr_auc": replayed_metrics["balanced_pr_auc"],
    }
    problems = [
        f"{k}: replayed {replayed[k]!r}, stored {reference[k]!r}"
        for k in REFERENCE_FIELDS if replayed[k] != reference[k]
    ]
    if problems:
        raise SelectionRefused("BSE validation reference not reproduced exactly:\n  "
                               + "\n  ".join(problems))
    return replayed


# ------------------------------------------------------------------ candidates

def candidate_row(cell, params, metrics, reference, tolerance=QUALITY_TOLERANCE):
    eff = metrics["efficiency"]
    row = {
        "cell": cell,
        **{k: float(v) for k, v in params.items()},
        "nonfactual_auc_pr": metrics["nonfactual"]["auc_pr"],
        "factual_auc_pr": metrics["factual"]["auc_pr"],
        "balanced_pr_auc": metrics["balanced_pr_auc"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "factual_precision": metrics["factual"]["precision"],
        "factual_recall": metrics["factual"]["recall"],
        "nonfactual_precision": metrics["nonfactual"]["precision"],
        "nonfactual_recall": metrics["nonfactual"]["recall"],
        "avg_retrieved_documents_per_sentence": eff["avg_retrieved_documents_per_sentence"],
        "avg_retrieved_documents_per_subclaim": eff["avg_retrieved_documents_per_subclaim"],
        "avg_nli_span_calls_per_sentence": eff["avg_nli_span_calls_per_sentence"],
    }
    for name in REFERENCE_FIELDS:
        row[f"{name}_delta_vs_bse"] = row[name] - reference[name]
        row[f"preserves_{name}"] = bool(row[name] >= reference[name] - tolerance)
    row["eligible"] = all(row[f"preserves_{n}"] for n in REFERENCE_FIELDS)
    return row


def order_key(row):
    params = (row["kappa"],) if row["cell"] == "C" else ()
    return (
        row["avg_retrieved_documents_per_sentence"],
        -row["balanced_pr_auc"],
        -row["nonfactual_auc_pr"],
        -row["factual_auc_pr"],
        row["avg_nli_span_calls_per_sentence"],
        *params,
        row["lower"],
        row["upper"],
    )


def select(cell, rows):
    expected = EXPECTED_CANDIDATES[cell]
    if len(rows) != expected:
        raise SelectionRefused(f"cell {cell}: {len(rows)} candidates, preregistered {expected}")
    if any(r["cell"] != cell for r in rows):
        raise SelectionRefused(f"cell {cell}: rows from another cell")
    eligible = sorted((r for r in rows if r["eligible"]), key=order_key)
    if not eligible:
        return {"cell": cell, "candidates": len(rows), "eligible": 0,
                "outcome": NO_ELIGIBLE, "selected": None}
    return {"cell": cell, "candidates": len(rows), "eligible": len(eligible),
            "outcome": SELECTED, "selected": eligible[0]}


# ------------------------------------------------------------------ consistency

COMPARED_FIELDS = (
    ("nonfactual_auc_pr", "nonfactual_auc_pr"),
    ("factual_auc_pr", "factual_auc_pr"),
    ("balanced_pr_auc", "balanced_pr_auc"),
    ("avg_retrieved_documents_per_sentence", "avg_retrieved_documents_per_sentence"),
    ("avg_nli_span_calls_per_sentence", "avg_nli_span_calls_per_sentence"),
)


def recorded_production_rows(sensitivity_artifact):
    rows = [r for r in sensitivity_artifact.get("hyperparameter_rows") or []
            if r.get("is_production_pair")]
    if len(rows) != 1:
        raise SelectionRefused(f"{len(rows)} production rows recorded; expected 1")
    row = rows[0]
    if (float(row["sigma"]), float(row["lambda"])) != (D0_SIGMA, D0_LAMBDA):
        raise SelectionRefused("recorded production pair is not the preregistered D0 pair")
    return row["threshold_candidates"]


def d0_matches_record(d0_rows, recorded):
    """D0 must re-derive the recorded validation rows exactly (prereg §4)."""
    by_band = {(float(c["lower"]), float(c["upper"])): c for c in recorded}
    problems = []
    if len(by_band) != len(d0_rows):
        problems.append(f"{len(by_band)} recorded bands vs {len(d0_rows)} re-derived")
    for row in d0_rows:
        rec = by_band.get((row["lower"], row["upper"]))
        if rec is None:
            problems.append(f"band {(row['lower'], row['upper'])} not recorded")
            continue
        for mine, theirs in COMPARED_FIELDS:
            if row[mine] != rec[theirs]:
                problems.append(f"band {(row['lower'], row['upper'])} {mine}: "
                                f"{row[mine]!r} vs recorded {rec[theirs]!r}")
        recorded_eligible = all(rec[f] for f in
                                ("preserves_nonfactual", "preserves_factual", "preserves_balanced"))
        if row["eligible"] != recorded_eligible:
            problems.append(f"band {(row['lower'], row['upper'])} eligibility differs")
    return problems


def c_kappa_one_matches_b(b_rows, c_rows):
    """C at kappa = 1 is cell B; the two must agree exactly."""
    key = lambda r: (r["lower"], r["upper"])
    fields = [m for m, _ in COMPARED_FIELDS] + ["eligible"]
    b = {key(r): r for r in b_rows}
    return [
        f"kappa=1 band {key(r)} {f}: {r[f]!r} vs B {b[key(r)][f]!r}"
        for r in c_rows if r["kappa"] == 1.0 for f in fields if r[f] != b[key(r)][f]
    ]
