"""D-03: is DDRE's evidence stable and supported? A frozen diagnostic protocol.

**Written before the real measurement.** Nothing here selects a cap, a
calibration or any method change; the diagnostic reports, and a separate
reviewed decision follows.

The finding is often misread as "large density ratios are wrong". It is not. A
large ratio is exactly what a density-ratio estimator *should* produce where
factual support is strong and hallucinated support is weak, and treating
magnitude alone as an error would discard the method's signal. The question is
narrower and answerable:

    Are the evidence values DDRE actually consumes stable under the production
    uLSIF hyperparameter surface, and supported by score regions the training
    data represents?

That separates **large-but-stable** evidence from **weak-support / tail-driven**
evidence. Three things are measured, on validation evidence only:

1. **Support.** For each evaluation score, how many factual and hallucinated
   training scores lie within one selected bandwidth, in the same normalized
   space the kernel uses. Reported in pre-registered strata, not against one
   cutoff invented after seeing the numbers.
2. **Hyperparameter sensitivity.** The same evidence re-derived at every
   ``(sigma, lambda)`` pair the production CV already scored, holding the final
   kernel centres fixed so the measurement isolates the hyperparameters rather
   than confounding them with a new centre sample.
3. **One-document stopping.** Whether a single document's evidence would end
   retrieval, under the general log-odds rule and the actual cost-consistent
   threshold grid -- and whether that decision survives the hyperparameter
   surface.

Escalation is pre-registered and conservative (see ``escalation_triggers``). It
can conclude only "a further sensitivity study is required", never "apply a
cap". And ``additional_sensitivity_required = False`` means the predeclared
criteria did not fire -- **not** that uLSIF is proven correct.

numpy + stdlib. No torch, so the methodology is testable without a GPU; the
runner that scores documents lives in ``scripts/diagnose_ddre_ratio_support.py``.
"""

import math

import numpy as np

DIAGNOSTIC_NAME = "D-03 density-ratio support/stability"
PROTOCOL_VERSION = "d03-support-stability-v1"

# The production clip, restated here for reporting. NOT changed by this module.
RATIO_CLIP_LOWER = 1e-6
RATIO_CLIP_UPPER = 1e6

# The frozen validation split this diagnostic runs on. The held-out test
# evidence is never touched.
DIAGNOSTIC_VALIDATION_FRACTION = 0.20
DIAGNOSTIC_SPLIT_SEED = 42
DIAGNOSTIC_MAX_DOCS = 10

# The production NBC training sizes. Anything else is a different estimator.
EXPECTED_FACTUAL_TRAINING = 199
EXPECTED_HALLUCINATED_TRAINING = 199

# The formal primary protocol's prior. Stated rather than assumed, because the
# one-document rule below is only symmetric in log-ratio at P0 = 0.5.
DIAGNOSTIC_P0 = 0.5

QUANTILE_LABELS = (
    ("min", 0.0),
    ("p01", 1.0),
    ("p05", 5.0),
    ("p25", 25.0),
    ("p50", 50.0),
    ("p75", 75.0),
    ("p95", 95.0),
    ("p99", 99.0),
    ("max", 100.0),
)

# Pre-registered support strata. Bands, not one cutoff: choosing a single
# "weak support" threshold after seeing the distribution would be exactly the
# post-hoc move this protocol exists to avoid.
SUPPORT_STRATA = (
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    (">=10", 10, None),
)

# D-13 descriptive histogram: fixed 10-point bins over the 0-100 score range.
D13_BIN_EDGES = tuple(float(x) for x in range(0, 101, 10))

DECISION_NONE = "NONE"
DECISION_LOW = "LOW"
DECISION_HIGH = "HIGH"

DIRECTION_ABOVE = "above_one"
DIRECTION_BELOW = "below_one"
DIRECTION_UNIT = "exactly_one"
DIRECTION_MIXED = "crosses_one"

MAGNITUDE_NOTE = (
    "None of the following, on its own, is evidence of a defect and none of "
    "them triggers a method change: a ratio above 10 or 100, a large "
    "|log ratio|, one-document stopping, low hallucinated support alone, or a "
    "score outside one class's observed range. A large ratio with coherent "
    "local support that is stable across the production hyperparameter surface "
    "is a legitimate estimate, and is what a density-ratio method is supposed "
    "to produce where the two classes genuinely separate."
)


class DiagnosticIncomplete(RuntimeError):
    """The diagnostic cannot be reported as complete. Never downgraded."""


# --------------------------------------------------------------------------
# Quantiles and clip accounting
# --------------------------------------------------------------------------


def quantile_summary(values):
    """Fixed quantiles of one population. No adaptive levels, ever."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise DiagnosticIncomplete(
            "quantile_summary received an empty population; an empty summary "
            "would read as a measurement that was taken"
        )
    if not np.all(np.isfinite(array)):
        raise DiagnosticIncomplete(
            f"{int(np.sum(~np.isfinite(array)))} of {array.size} values are "
            "non-finite. A quantile over non-finite values is not a "
            "measurement; the offending rows are NOT dropped."
        )
    summary = {
        label: float(np.percentile(array, level)) for label, level in QUANTILE_LABELS
    }
    summary["count"] = int(array.size)
    summary["mean"] = float(np.mean(array))
    return summary


def returned_ratio(raw):
    """The value DDRE actually consumes: the raw fit, clipped as in production."""
    return float(np.clip(float(raw), RATIO_CLIP_LOWER, RATIO_CLIP_UPPER))


def clip_activity(raw_ratios):
    """How often the production clip is the thing determining the evidence.

    Counted on RAW ratios. Once clipped, a value that hit a bound is
    indistinguishable from one that landed there honestly, so the two bounds
    are counted separately and never summed into one "clipped" total.
    """
    array = np.asarray(list(raw_ratios), dtype=float)
    return {
        "count": int(array.size),
        "raw_non_positive": int(np.sum(array <= 0.0)),
        "raw_below_lower_clip": int(np.sum(array < RATIO_CLIP_LOWER)),
        "raw_above_upper_clip": int(np.sum(array > RATIO_CLIP_UPPER)),
        "lower_clip_hits": int(np.sum(array < RATIO_CLIP_LOWER)),
        "upper_clip_hits": int(np.sum(array > RATIO_CLIP_UPPER)),
        "any_clip_hits": int(
            np.sum(array < RATIO_CLIP_LOWER) + np.sum(array > RATIO_CLIP_UPPER)
        ),
        "clip_lower": RATIO_CLIP_LOWER,
        "clip_upper": RATIO_CLIP_UPPER,
    }


def selected_fit_ratios(estimator, scores):
    """Raw, returned and log-returned ratios under the SELECTED production fit.

    Uses the estimator's own kernel and coefficients, so this measures the
    deployed implementation rather than a second copy of the equations.
    """
    x = estimator._as_column(scores)
    raw = np.asarray(
        estimator._kernel(x, estimator.centers, estimator.sigma) @ estimator.alpha,
        dtype=float,
    ).reshape(-1)
    if not np.all(np.isfinite(raw)):
        raise DiagnosticIncomplete(
            "the selected production fit produced a non-finite raw ratio on the "
            "diagnostic population; the row is not dropped and the diagnostic "
            "is not complete"
        )
    returned = np.clip(raw, RATIO_CLIP_LOWER, RATIO_CLIP_UPPER)
    return {
        "raw": raw,
        "returned": returned,
        "log_returned": np.log(returned),
    }


def population_ratio_report(estimator, scores, *, name):
    """One score population under the selected fit: quantiles and clip activity.

    Raw and returned quantiles are reported side by side and never conflated:
    the raw values say what the fit produced, the returned values say what DDRE
    consumed, and the difference between them is the clip's contribution.
    """
    ratios = selected_fit_ratios(estimator, scores)
    return {
        "population": name,
        "n_scores": int(len(list(scores))),
        "score_quantiles": quantile_summary(scores),
        "raw_ratio_quantiles": quantile_summary(ratios["raw"]),
        "returned_ratio_quantiles": quantile_summary(ratios["returned"]),
        "log_returned_ratio_quantiles": quantile_summary(ratios["log_returned"]),
        "clip_activity": clip_activity(ratios["raw"]),
        "alpha_sum": float(np.sum(estimator.alpha)),
        "alpha_sum_bound_note": (
            "The fitted ratio is a finite sum of non-negative coefficients "
            "times Gaussian kernels bounded by 1, so on ordinary fitted "
            "evaluation raw_ratio <= sum(alpha). The estimator is BOUNDED; it "
            "is not mathematically unbounded."
        ),
    }


# --------------------------------------------------------------------------
# Local support
# --------------------------------------------------------------------------


def normalized(scores):
    """The kernel's own coordinates: score/100 clipped to [0, 1].

    Support must be measured here, not on the 0-100 scale, because the
    bandwidth sigma is a distance in this space. A radius of sigma applied to
    raw scores would be off by a factor of 100.
    """
    return np.clip(np.asarray(list(scores), dtype=float).reshape(-1) / 100.0, 0.0, 1.0)


def support_counts(scores, factual_training, hallucinated_training, sigma):
    """Training examples of each class within ONE bandwidth of each score."""
    if not (math.isfinite(float(sigma)) and float(sigma) > 0.0):
        raise DiagnosticIncomplete(
            f"support radius must be a finite positive bandwidth, got {sigma!r}"
        )
    x = normalized(scores)
    factual = normalized(factual_training)
    hallucinated = normalized(hallucinated_training)
    radius = float(sigma)
    factual_within = np.sum(
        np.abs(x[:, None] - factual[None, :]) <= radius, axis=1
    ).astype(int)
    hallucinated_within = np.sum(
        np.abs(x[:, None] - hallucinated[None, :]) <= radius, axis=1
    ).astype(int)
    return {
        "radius_sigma": radius,
        "space": "normalized [0, 1] (score / 100), the kernel's own coordinates",
        "factual_support_within_sigma": factual_within,
        "hallucinated_support_within_sigma": hallucinated_within,
        "min_class_support": np.minimum(factual_within, hallucinated_within),
    }


def support_stratum(min_class_support):
    """The pre-registered band a support count falls in."""
    value = int(min_class_support)
    for label, low, high in SUPPORT_STRATA:
        if value >= low and (high is None or value <= high):
            return label
    raise DiagnosticIncomplete(f"support count {value!r} matched no stratum")


def range_status(scores, factual_training, hallucinated_training):
    """Where each score sits relative to the observed training ranges.

    Being outside one class's range is recorded, not judged. It is ordinary for
    a strongly factual score to sit above every hallucinated training score,
    and calling that an error would make the method's own signal look like a
    fault.
    """
    x = normalized(scores)
    factual = normalized(factual_training)
    hallucinated = normalized(hallucinated_training)
    combined_low = float(min(factual.min(), hallucinated.min()))
    combined_high = float(max(factual.max(), hallucinated.max()))

    statuses = []
    for value in x:
        if value < combined_low:
            statuses.append("below_combined_range")
        elif value > combined_high:
            statuses.append("above_combined_range")
        else:
            statuses.append("inside_combined_range")
    return {
        "combined_range_normalized": [combined_low, combined_high],
        "factual_range_normalized": [float(factual.min()), float(factual.max())],
        "hallucinated_range_normalized": [
            float(hallucinated.min()),
            float(hallucinated.max()),
        ],
        "status": statuses,
        "inside_combined_range": int(
            sum(1 for s in statuses if s == "inside_combined_range")
        ),
        "below_combined_range": int(
            sum(1 for s in statuses if s == "below_combined_range")
        ),
        "above_combined_range": int(
            sum(1 for s in statuses if s == "above_combined_range")
        ),
        "outside_factual_range": int(
            np.sum((x < factual.min()) | (x > factual.max()))
        ),
        "outside_hallucinated_range": int(
            np.sum((x < hallucinated.min()) | (x > hallucinated.max()))
        ),
        "note": (
            "Descriptive. A score outside one class's observed range is NOT an "
            "error and does not by itself indicate an unsupported estimate."
        ),
    }


# --------------------------------------------------------------------------
# Hyperparameter surface
# --------------------------------------------------------------------------


def production_hyperparameter_pairs(estimator):
    """Exactly the (sigma, lambda) pairs the production CV already scored.

    Taken from the estimator's own ``cv_table`` rather than regenerated, so the
    sensitivity surface cannot drift from the grid the production fit searched.
    Each pair appears once.
    """
    if not estimator.cv_table:
        raise DiagnosticIncomplete(
            "the estimator has no cv_table; the production hyperparameter "
            "surface is unknown and cannot be re-derived"
        )
    pairs = []
    seen = set()
    for row in estimator.cv_table:
        key = (float(row["sigma"]), float(row["lambda"]))
        if key in seen:
            raise DiagnosticIncomplete(
                f"the production cv_table lists {key!r} more than once; the "
                "sensitivity surface would weight it twice"
            )
        seen.add(key)
        pairs.append(key)
    return tuple(pairs)


def hyperparameter_surface(
    estimator, factual_training, hallucinated_training, evaluation_scores
):
    """Re-derive the evidence at every production (sigma, lambda) pair.

    The final kernel **centres are held fixed** at the selected fit's. Letting
    each refit draw its own centre sample would confound hyperparameter
    sensitivity with sampling noise in the centres, and the question here is
    specifically the former.

    The production estimator's own ``_solve`` and ``_kernel`` are used --
    deliberately reaching for private helpers -- so this measures the deployed
    mathematics rather than a second implementation that could drift from it.
    Production fitting is untouched.
    """
    factual_x = estimator._as_column(factual_training)
    hallucinated_x = estimator._as_column(hallucinated_training)
    evaluation_x = estimator._as_column(evaluation_scores)
    centers = estimator.centers
    pairs = production_hyperparameter_pairs(estimator)

    raw_by_pair = []
    for sigma, lam in pairs:
        alpha = estimator._solve(
            factual_x, hallucinated_x, centers, sigma, lam
        )
        if not np.all(np.isfinite(alpha)):
            raise DiagnosticIncomplete(
                f"the diagnostic refit at sigma={sigma!r}, lambda={lam!r} "
                "produced non-finite coefficients. The pair is NOT skipped: a "
                "surface with a hole in it is not the production surface."
            )
        raw = np.asarray(
            estimator._kernel(evaluation_x, centers, sigma) @ alpha, dtype=float
        ).reshape(-1)
        if not np.all(np.isfinite(raw)):
            raise DiagnosticIncomplete(
                f"the diagnostic refit at sigma={sigma!r}, lambda={lam!r} "
                "produced a non-finite ratio on the evaluation population"
            )
        raw_by_pair.append(raw)

    raw_matrix = np.vstack(raw_by_pair) if raw_by_pair else np.empty((0, 0))
    returned_matrix = np.clip(raw_matrix, RATIO_CLIP_LOWER, RATIO_CLIP_UPPER)
    log_matrix = np.log(returned_matrix)

    directions = []
    for column in range(returned_matrix.shape[1]):
        values = returned_matrix[:, column]
        above = bool(np.all(values > 1.0))
        below = bool(np.all(values < 1.0))
        if above:
            directions.append(DIRECTION_ABOVE)
        elif below:
            directions.append(DIRECTION_BELOW)
        elif bool(np.all(values == 1.0)):
            directions.append(DIRECTION_UNIT)
        else:
            directions.append(DIRECTION_MIXED)

    return {
        "n_pairs": len(pairs),
        "pairs": [{"sigma": s, "lambda": l} for s, l in pairs],
        "centers_held_fixed": True,
        "n_centers": int(np.size(centers)),
        "centers_note": (
            "Every refit uses the SELECTED fit's final centre set, so the "
            "surface isolates sigma/lambda instead of also resampling centres."
        ),
        "returned_ratio_min": returned_matrix.min(axis=0),
        "returned_ratio_max": returned_matrix.max(axis=0),
        "returned_ratio_median": np.median(returned_matrix, axis=0),
        "log_ratio_min": log_matrix.min(axis=0),
        "log_ratio_max": log_matrix.max(axis=0),
        "log_ratio_median": np.median(log_matrix, axis=0),
        "log_ratio_span": log_matrix.max(axis=0) - log_matrix.min(axis=0),
        "direction": directions,
        "returned_matrix": returned_matrix,
    }


# --------------------------------------------------------------------------
# One-document stopping
# --------------------------------------------------------------------------


def logit(p):
    value = float(p)
    if not 0.0 < value < 1.0:
        raise DiagnosticIncomplete(f"logit requires 0 < p < 1, got {p!r}")
    return math.log(value / (1.0 - value))


def state_after_one_document(ratio, *, p0=DIAGNOSTIC_P0):
    """The general log-odds state after a single evidence update.

    ``logit(P0) + log(r)``. This is the actual rule; ``|log r|`` is only its
    special case at a symmetric band with P0 = 0.5, and using the shorthand for
    an asymmetric band gives the wrong answer.
    """
    return logit(p0) + math.log(returned_ratio(ratio))


def stop_decision(state, lower, upper):
    """NONE / LOW / HIGH under one threshold pair, in log-odds space."""
    if state <= logit(lower):
        return DECISION_LOW
    if state >= logit(upper):
        return DECISION_HIGH
    return DECISION_NONE


def one_document_stopping(ratios, threshold_pairs, *, p0=DIAGNOSTIC_P0):
    """Low / high / either stopping rates, reported SEPARATELY.

    A single "stops" rate hides the asymmetry that matters: stopping low means
    concluding hallucination on one document, stopping high means concluding
    factual on one. They are different scientific events.
    """
    if not threshold_pairs:
        raise DiagnosticIncomplete("no cost-consistent threshold pairs supplied")
    states = [state_after_one_document(r, p0=p0) for r in ratios]
    if not states:
        raise DiagnosticIncomplete("no first-document ratios supplied")

    per_pair = []
    for lower, upper in threshold_pairs:
        decisions = [stop_decision(s, lower, upper) for s in states]
        low = sum(1 for d in decisions if d == DECISION_LOW)
        high = sum(1 for d in decisions if d == DECISION_HIGH)
        per_pair.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n_scores": len(states),
                "low_stop_count": low,
                "high_stop_count": high,
                "either_stop_count": low + high,
                "low_stop_fraction": low / len(states),
                "high_stop_fraction": high / len(states),
                "either_stop_fraction": (low + high) / len(states),
            }
        )
    return {
        "p0": float(p0),
        "rule": "state_after_one = logit(P0) + log(returned_ratio)",
        "rule_note": (
            "The general log-odds rule. |log r| is only the special case of a "
            "band symmetric about P0 = 0.5 and is not used as the criterion."
        ),
        "n_first_documents": len(states),
        "n_threshold_pairs": len(threshold_pairs),
        "per_threshold_pair": per_pair,
        "states": states,
    }


def stops_under_selected_fit(ratios, threshold_pairs, *, p0=DIAGNOSTIC_P0):
    """Per score: does ANY cost-consistent pair stop after one document?"""
    flags = []
    for ratio in ratios:
        state = state_after_one_document(ratio, p0=p0)
        flags.append(
            any(
                stop_decision(state, lower, upper) != DECISION_NONE
                for lower, upper in threshold_pairs
            )
        )
    return flags


def stop_decision_stability(
    selected_ratios, surface_returned_matrix, threshold_pairs, *, p0=DIAGNOSTIC_P0
):
    """Does the one-document decision survive the hyperparameter surface?

    Flip kinds are kept apart. A LOW-to-HIGH reversal -- the same evidence
    concluding hallucination under one hyperparameter setting and factual under
    another -- is a different and far more serious event than a stop softening
    to no decision, and averaging them into one number would hide it.
    """
    matrix = np.asarray(surface_returned_matrix, dtype=float)
    if matrix.shape[1] != len(selected_ratios):
        raise DiagnosticIncomplete(
            f"the surface covers {matrix.shape[1]} scores but "
            f"{len(selected_ratios)} were evaluated under the selected fit"
        )

    counters = {
        "unanimous_across_hyperparameters": 0,
        "any_decision_flip": 0,
        "low_to_none": 0,
        "high_to_none": 0,
        "none_to_low": 0,
        "none_to_high": 0,
        "direction_flip_low_to_high": 0,
        "direction_flip_high_to_low": 0,
    }
    unanimous_per_score = []
    for index, ratio in enumerate(selected_ratios):
        selected_state = state_after_one_document(ratio, p0=p0)
        score_unanimous = True
        for lower, upper in threshold_pairs:
            selected = stop_decision(selected_state, lower, upper)
            for value in matrix[:, index]:
                alternative = stop_decision(
                    state_after_one_document(value, p0=p0), lower, upper
                )
                if alternative == selected:
                    continue
                score_unanimous = False
                counters["any_decision_flip"] += 1
                pair = (selected, alternative)
                if pair == (DECISION_LOW, DECISION_NONE):
                    counters["low_to_none"] += 1
                elif pair == (DECISION_HIGH, DECISION_NONE):
                    counters["high_to_none"] += 1
                elif pair == (DECISION_NONE, DECISION_LOW):
                    counters["none_to_low"] += 1
                elif pair == (DECISION_NONE, DECISION_HIGH):
                    counters["none_to_high"] += 1
                elif pair == (DECISION_LOW, DECISION_HIGH):
                    counters["direction_flip_low_to_high"] += 1
                elif pair == (DECISION_HIGH, DECISION_LOW):
                    counters["direction_flip_high_to_low"] += 1
        unanimous_per_score.append(score_unanimous)
        if score_unanimous:
            counters["unanimous_across_hyperparameters"] += 1

    counters["n_scores"] = len(selected_ratios)
    counters["unanimous_per_score"] = unanimous_per_score
    counters["note"] = (
        "Flip kinds are counted separately. A LOW-to-HIGH reversal is a "
        "direction change in the conclusion, not a small numerical difference."
    )
    return counters


# --------------------------------------------------------------------------
# Escalation triggers -- pre-registered, conservative
# --------------------------------------------------------------------------


def escalation_triggers(
    *,
    selected_document_clip_activity,
    first_document_stopped,
    first_document_direction,
    first_document_min_class_support,
    first_document_unanimous,
):
    """The three predeclared triggers. Frozen before the real measurement.

    Each can conclude only that a **separate cap / calibration / evidence-scale
    sensitivity study is required**. None of them selects a cap, a calibration
    or any method change -- that decision belongs to a reviewed follow-up, not
    to a diagnostic that has just seen the data.

    Magnitude alone is deliberately absent. A ratio above 10 or 100, a large
    |log ratio|, one-document stopping, low hallucinated support on its own, or
    a score outside one class's range: none of these fires anything. A large
    ratio with coherent local support that is stable across the surface is a
    legitimate estimate.
    """
    n = len(first_document_stopped)
    for name, values in (
        ("first_document_direction", first_document_direction),
        ("first_document_min_class_support", first_document_min_class_support),
        ("first_document_unanimous", first_document_unanimous),
    ):
        if len(values) != n:
            raise DiagnosticIncomplete(
                f"{name} covers {len(values)} first documents but "
                f"{n} were evaluated; the triggers would be misaligned"
            )

    clip_hits = int(selected_document_clip_activity.get("any_clip_hits", 0))
    trigger_one = clip_hits > 0

    direction_flips = [
        index
        for index in range(n)
        if first_document_stopped[index]
        and first_document_direction[index] == DIRECTION_MIXED
    ]
    trigger_two = bool(direction_flips)

    weak_and_unstable = [
        index
        for index in range(n)
        if first_document_stopped[index]
        and int(first_document_min_class_support[index]) == 0
        and not first_document_unanimous[index]
    ]
    trigger_three = bool(weak_and_unstable)

    triggers = {
        "trigger_1_selected_fit_hits_ratio_clip": {
            "fired": trigger_one,
            "count": clip_hits,
            "lower_clip_hits": int(
                selected_document_clip_activity.get("lower_clip_hits", 0)
            ),
            "upper_clip_hits": int(
                selected_document_clip_activity.get("upper_clip_hits", 0)
            ),
            "criterion": (
                "The selected production fit hits either existing ratio clip "
                f"[{RATIO_CLIP_LOWER}, {RATIO_CLIP_UPPER}] on validation "
                "document evidence."
            ),
        },
        "trigger_2_stopping_direction_flips_over_hyperparameters": {
            "fired": trigger_two,
            "count": len(direction_flips),
            "criterion": (
                "A validation FIRST-document score that produces a one-document "
                "stop under the selected fit has a ratio direction (>1 vs <1) "
                "that flips somewhere over the production sigma/lambda surface."
            ),
        },
        "trigger_3_zero_support_and_unstable_stop": {
            "fired": trigger_three,
            "count": len(weak_and_unstable),
            "criterion": (
                "A validation FIRST-document score that produces a one-document "
                "stop under the selected fit has min_class_support == 0 AND its "
                "stop decision is not unanimous across the production "
                "sigma/lambda surface."
            ),
        },
    }
    required = trigger_one or trigger_two or trigger_three
    return {
        "triggers": triggers,
        "additional_sensitivity_required": bool(required),
        "meaning": (
            "additional_sensitivity_required = true means a SEPARATE cap / "
            "calibration / evidence-scale sensitivity study is required. It "
            "does NOT select a cap or authorise any method change."
            if required
            else "additional_sensitivity_required = false means the predeclared "
            "D-03 escalation criteria did not fire. It does NOT mean uLSIF is "
            "proven correct."
        ),
        "magnitude_note": MAGNITUDE_NOTE,
        # Frozen outputs. The diagnostic never fills these in.
        "selected_cap": None,
        "selected_calibration": None,
        "method_change_made": False,
    }


def support_strata_report(
    *,
    min_class_support,
    returned_ratios,
    log_ratio_span,
    directions,
    stopped_flags,
    unanimous_flags,
):
    """Everything, broken down by the pre-registered support band.

    This is where "large-but-stable" separates from "tail-driven": a stratum
    with high ratios, a narrow log-ratio span and unanimous decisions is
    behaving; one with high ratios, a wide span and flipping decisions is the
    case D-03 exists to find. Reporting them together, per band, is what makes
    that distinction visible instead of asserted.
    """
    n = len(min_class_support)
    for name, values in (
        ("returned_ratios", returned_ratios),
        ("log_ratio_span", log_ratio_span),
        ("directions", directions),
        ("stopped_flags", stopped_flags),
        ("unanimous_flags", unanimous_flags),
    ):
        if len(values) != n:
            raise DiagnosticIncomplete(
                f"{name} covers {len(values)} scores but support covers {n}"
            )

    labels = [support_stratum(value) for value in min_class_support]
    strata = {}
    for label, _, _ in SUPPORT_STRATA:
        indices = [i for i, value in enumerate(labels) if value == label]
        if not indices:
            strata[label] = {
                "count": 0,
                "fraction": 0.0,
                "note": "no evaluation score fell in this stratum",
            }
            continue
        ratios = [float(returned_ratios[i]) for i in indices]
        strata[label] = {
            "count": len(indices),
            "fraction": len(indices) / n,
            "returned_ratio_quantiles": quantile_summary(ratios),
            "log_returned_ratio_quantiles": quantile_summary(
                [math.log(r) for r in ratios]
            ),
            "log_ratio_span_quantiles": quantile_summary(
                [float(log_ratio_span[i]) for i in indices]
            ),
            "direction_crosses_one": int(
                sum(1 for i in indices if directions[i] == DIRECTION_MIXED)
            ),
            "one_document_stop_count": int(
                sum(1 for i in indices if stopped_flags[i])
            ),
            "one_document_stop_fraction": (
                sum(1 for i in indices if stopped_flags[i]) / len(indices)
            ),
            "unanimous_across_hyperparameters": int(
                sum(1 for i in indices if unanimous_flags[i])
            ),
            "non_unanimous": int(
                sum(1 for i in indices if not unanimous_flags[i])
            ),
        }
    return {
        "strata_definition": [
            {"label": label, "min": low, "max": high}
            for label, low, high in SUPPORT_STRATA
        ],
        "n_scores": n,
        "strata": strata,
        "note": (
            "Bands, not a cutoff. No single support threshold is declared "
            "'weak'; choosing one after seeing the distribution is the "
            "post-hoc move this protocol avoids."
        ),
    }


# --------------------------------------------------------------------------
# D-13 descriptive distribution shift
# --------------------------------------------------------------------------


def fixed_bin_histogram(scores, edges=D13_BIN_EDGES):
    """Counts over the fixed 0-10, 10-20, ..., 90-100 bins. Never adaptive."""
    array = np.asarray(list(scores), dtype=float)
    counts = {}
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        last = index == len(edges) - 2
        inside = (array >= low) & ((array <= high) if last else (array < high))
        counts[f"{int(low)}-{int(high)}"] = int(np.sum(inside))
    return counts


def distribution_shift(nbc_scores, document_scores):
    """D-13, DESCRIPTIVELY. The shift is measured, not corrected.

    The estimator is trained on Wang's NBC *pair* scores and applied to
    *document-max* scores. That is a real application shift, and it is recorded
    here because this diagnostic already holds both populations. Nothing is
    recalibrated, nothing is retrained on document-max scores, and D-13 is not
    marked resolved.
    """
    return {
        "finding": "D-13",
        "status": "described only; not corrected and not resolved",
        "nbc_pair_scores": {
            "quantiles": quantile_summary(nbc_scores),
            "histogram": fixed_bin_histogram(nbc_scores),
        },
        "validation_document_max_scores": {
            "quantiles": quantile_summary(document_scores),
            "histogram": fixed_bin_histogram(document_scores),
        },
        "bin_edges": list(D13_BIN_EDGES),
        "note": (
            "Both BSE and DDRE inherit this pair-score -> document-max-score "
            "application shift, so it does not by itself break the controlled "
            "comparison. No correction, recalibration or retraining on "
            "document-max scores is performed here."
        ),
    }


# --------------------------------------------------------------------------
# Completion accounting -- a partial run is never labelled complete
# --------------------------------------------------------------------------


def completion_accounting(*, expected_occurrences, scored_occurrences, first_documents,
                          expected_first_documents):
    """Every required document occurrence must be accounted for.

    A resumable cache is fine; a JSON verdict claiming COMPLETE on a partial
    measurement is not.
    """
    complete = (
        expected_occurrences == scored_occurrences
        and first_documents == expected_first_documents
        and expected_occurrences > 0
    )
    return {
        "expected_document_occurrences": int(expected_occurrences),
        "scored_document_occurrences": int(scored_occurrences),
        "expected_first_documents": int(expected_first_documents),
        "first_documents": int(first_documents),
        "complete": bool(complete),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "note": (
            "Every required validation document occurrence is accounted for."
            if complete
            else "The measurement is PARTIAL. Problematic rows are never "
            "dropped to reach completion, and the verdict must not claim "
            "COMPLETE until every required occurrence is scored."
        ),
    }


def require_complete(accounting):
    if not accounting.get("complete"):
        raise DiagnosticIncomplete(
            "refusing to emit a D-03 verdict from a partial measurement: "
            f"{accounting['scored_document_occurrences']} of "
            f"{accounting['expected_document_occurrences']} validation document "
            "occurrences were scored"
        )
    return accounting
