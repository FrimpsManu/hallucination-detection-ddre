"""Step 2: isolate which forward-path argument explains the A1/A3 divergence.

Step 1 (``scripts/diagnose_scorer_paths.py``) established that the literal Wang
scorer and this repository's scorer tokenize identically -- 589/589 input_ids
matched -- yet disagreed on exactly one BSE decision: positive NBC pair 169 fell
in bucket 1 under the literal path and bucket 2 under ours, moving the positive
Laplace-smoothed histogram from ``[1, 78, 19, 13, 6, 5, 28, 57, 1, 1]`` to
``[1, 77, 20, 13, 6, 5, 28, 57, 1, 1]``.

Tokenization is therefore excluded, and ``token_type_ids`` are excluded too
(they are emitted with values 0 and 1, but ``type_vocab_size`` is 0, so no
token-type embedding exists to consume them). Two candidate differences remain
between Wang's released call and ours:

1. ``torch.inference_mode()``, which this repository wraps the forward pass in
   and the released ``utils.py:57`` does not; and
2. passing ``attention_mask``, which ``model(**inputs)`` does and Wang's
   ``model(inputs["input_ids"])`` does not.

This module holds the comparison and interpretation logic for a four-arm
factorial over those two factors. It is deliberately standard-library only
(beyond :mod:`src.scoring_diagnostics`, which is itself stdlib-only), so the
reporting logic is unit-testable without downloading DeBERTa.

Reporting discipline
--------------------
A single word like "MATERIAL" conflates two different findings, so this module
never emits one. Every comparison reports three separate fields:

``numerical_difference``
    YES when the raw entailment scores are not bit-identical. This is a
    statement about floating point and nothing else.
``bse_decision_impact``
    YES only when at least one NBC bucket changes. The BSE update consumes the
    bucket, so this is the only field that licenses a claim about baseline
    behaviour. **A raw score change alone is not evidence that BSE behaviour
    changed**, and a comparison can and often will be
    ``numerical_difference=YES, bse_decision_impact=NO``.
``causal_candidate``
    Which factor the comparison toggles, and whether it showed an effect.

The overall verdict adds ``explains_reference_observation``, because the point
of the experiment is not to find *a* difference but to find the one that
produced the Step 1 result.
"""

from src.scoring_diagnostics import (
    bucket_disagreements,
    delta_stats,
    laplace_histogram,
    nbc_bucket,
    one_decimal_disagreements,
    round_one_decimal,
)


# Wang's released utils.py:60 hardcodes label_names[0] == "entailment".
ENTAILMENT_INDEX = 0

# Wang's released utils.py:59 divides logits by 5 before the softmax.
SOFTMAX_TEMPERATURE = 5.0

# Raw scores are on a 0-100 scale. Below this, a difference is recorded but
# treated as negligible when deciding which factor is implicated.
#
# The anchor is the effect being explained: the Step 1 divergence at positive
# pair 169 was 0.0068359375, about 68x this bound. A threshold has to sit well
# under the effect it is used to attribute, or it would classify the very
# difference under investigation as noise.
NEGLIGIBLE_MAX_ABS_DELTA = 1e-4

DIFFERENCE_IDENTICAL = "IDENTICAL"
DIFFERENCE_NEGLIGIBLE = "NEGLIGIBLE"
DIFFERENCE_SUBSTANTIAL = "SUBSTANTIAL"

YES = "YES"
NO = "NO"


# --------------------------------------------------------------------------
# Arm definitions
# --------------------------------------------------------------------------
#
# Every arm consumes the SAME pre-tokenized tensors and the SAME score
# extraction. Only the forward call differs, so any divergence is attributable
# to the forward path alone.

ARM_SPECS = (
    {
        "name": "C0",
        "label": "literal Wang reference",
        "attention_mask": False,
        "token_type_ids": False,
        "inference_mode": False,
        "call": 'model(input_ids)',
        "role": "reference",
        "note": "Transcribes released utils.py:57 exactly: input_ids only, grad enabled.",
    },
    {
        "name": "C1",
        "label": "inference-mode only",
        "attention_mask": False,
        "token_type_ids": False,
        "inference_mode": True,
        "call": 'with torch.inference_mode(): model(input_ids)',
        "role": "factor",
        "isolates": "torch.inference_mode()",
        "note": "C0 vs C1 isolates torch.inference_mode() with the arguments held fixed.",
    },
    {
        "name": "C2",
        "label": "attention-mask only",
        "attention_mask": True,
        "token_type_ids": False,
        "inference_mode": False,
        "call": 'model(input_ids, attention_mask=attention_mask)',
        "role": "factor",
        "isolates": "attention_mask",
        "note": "C0 vs C2 isolates attention_mask with the grad context held fixed.",
    },
    {
        "name": "C3",
        "label": "attention-mask + inference-mode",
        "attention_mask": True,
        "token_type_ids": False,
        "inference_mode": True,
        "call": (
            "with torch.inference_mode(): "
            "model(input_ids, attention_mask=attention_mask)"
        ),
        "role": "factor",
        "isolates": "attention_mask + torch.inference_mode()",
        "note": "Both factors on. This is what src/utils.py effectively does at batch 1.",
    },
    {
        "name": "C4",
        "label": "confirmation only: + token_type_ids",
        "attention_mask": True,
        "token_type_ids": True,
        "inference_mode": True,
        "call": (
            "with torch.inference_mode(): model(input_ids, "
            "attention_mask=attention_mask, token_type_ids=token_type_ids)"
        ),
        "role": "confirmation",
        "isolates": "token_type_ids",
        "note": (
            "CONFIRMATION ONLY. type_vocab_size is 0, so no token-type embedding "
            "is instantiated and this argument is expected to be inert. C4 exists "
            "to demonstrate that, not to test a live hypothesis."
        ),
    },
)

# A second execution of the C0 arm, run last. Without it, a C0-vs-C1 difference
# could not be distinguished from run-to-run nondeterminism, and every
# attribution below would be unfounded. This control is not optional.
DETERMINISM_CONTROL = {
    "name": "C0R",
    "label": "C0 repeated (determinism control)",
    "attention_mask": False,
    "token_type_ids": False,
    "inference_mode": False,
    "call": "model(input_ids)  # identical to C0, executed again at the end",
    "role": "control",
    "note": (
        "Re-runs C0 unchanged. If C0 and C0R are not bit-identical, the forward "
        "pass is nondeterministic on this device and no factor attribution in "
        "this report is sound."
    ),
}


def arm_by_name(name):
    for spec in ARM_SPECS:
        if spec["name"] == name:
            return spec
    if DETERMINISM_CONTROL["name"] == name:
        return DETERMINISM_CONTROL
    raise KeyError(f"unknown arm {name!r}")


# --------------------------------------------------------------------------
# Reference observation from Step 1
# --------------------------------------------------------------------------
#
# This is a recorded measurement from the previous diagnostic, carried here so
# the new run can state whether it reproduced it. It is an observation, not a
# tolerance and not a reference value of the kind Gate 1 compares against.

REFERENCE_OBSERVATION = {
    "source": "Step 1 A/B/C diagnostic (scripts/diagnose_scorer_paths.py)",
    "polarity": "positive",
    "index": 169,
    "literal_wang": {"raw": 19.9462890625, "rounded": 19.9, "nbc_bucket": 1},
    "repository": {"raw": 19.953125, "rounded": 20.0, "nbc_bucket": 2},
    "raw_delta": 19.953125 - 19.9462890625,
    "positive_histogram_literal_wang": [1, 78, 19, 13, 6, 5, 28, 57, 1, 1],
    "positive_histogram_repository": [1, 77, 20, 13, 6, 5, 28, 57, 1, 1],
    "note": (
        "The single BSE decision-level disagreement Step 1 found between the "
        "literal Wang scorer and this repository's scorer at batch size 1, with "
        "identical tokenization on all 589 pairs."
    ),
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def entailment_score_from_probabilities(probabilities):
    """Wang's released utils.py:59-65 score extraction.

    Takes the already-softmaxed probability list, selects the entailment class
    by hardcoded index, and scales to 0-100 in float64 -- exactly as
    ``round(float(pred) * 100, 1)`` does before rounding.

    Held identical across every arm on purpose. If the arms extracted scores
    differently, a divergence could not be attributed to the forward call.
    """
    return float(probabilities[ENTAILMENT_INDEX]) * 100.0


def score_report(raw):
    """The three views of one score that a BSE decision passes through."""
    return {
        "raw": float(raw),
        "rounded_one_decimal": round_one_decimal(raw),
        "nbc_bucket": nbc_bucket(raw),
    }


def histograms_by_polarity(scores, polarities):
    """Laplace-smoothed positive and negative NBC histograms for one arm."""
    if len(scores) != len(polarities):
        raise ValueError("scores and polarities must be the same length")
    positive = [s for s, p in zip(scores, polarities) if p == "positive"]
    negative = [s for s, p in zip(scores, polarities) if p == "negative"]
    return {
        "positive": laplace_histogram(positive),
        "negative": laplace_histogram(negative),
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def classify_difference(deltas):
    """Bucket the magnitude of a raw-score difference."""
    max_absolute = deltas.get("max_absolute")
    if max_absolute is None or max_absolute == 0.0:
        return DIFFERENCE_IDENTICAL
    if max_absolute <= NEGLIGIBLE_MAX_ABS_DELTA:
        return DIFFERENCE_NEGLIGIBLE
    return DIFFERENCE_SUBSTANTIAL


def isolated_factor(left_name, right_name):
    """Which factor actually differs between two arms.

    Derived from the arms' own flags rather than read off the right-hand arm's
    static label. The difference matters: C1 vs C3 toggles ``attention_mask``
    alone (both arms already run under inference_mode) and C2 vs C3 toggles
    ``torch.inference_mode()`` alone (both already pass a mask). Those are
    precisely the two comparisons :func:`interpret_factor_isolation` uses to
    separate a dominant factor from an interaction, so labelling either of them
    with the right arm's "both factors" description would misdescribe the
    experiment.

    Returns ``None`` when nothing is toggled, which is the determinism control.
    """
    left = arm_by_name(left_name)
    right = arm_by_name(right_name)
    toggled = []
    if left["attention_mask"] != right["attention_mask"]:
        toggled.append("attention_mask")
    if left["inference_mode"] != right["inference_mode"]:
        toggled.append("torch.inference_mode()")
    if left["token_type_ids"] != right["token_type_ids"]:
        toggled.append("token_type_ids")
    return " + ".join(toggled) if toggled else None


def compare_forward_arms(left_name, right_name, left_scores, right_scores, polarities):
    """Compare two forward-path arms over the same ordered NBC pairs.

    Returns the three separately-reported findings described in the module
    docstring, never a single conflated verdict.
    """
    deltas = delta_stats(left_scores, right_scores)
    one_decimal = one_decimal_disagreements(left_scores, right_scores)
    buckets = bucket_disagreements(left_scores, right_scores, nbc_bucket)
    difference_class = classify_difference(deltas)

    numerical = NO if difference_class == DIFFERENCE_IDENTICAL else YES
    decision_impact = YES if buckets["count"] > 0 else NO

    try:
        factor = isolated_factor(left_name, right_name)
    except KeyError:
        factor = None

    if factor is None:
        candidate = (
            f"{left_name} vs {right_name} toggles no factor; this is a "
            "determinism control, not a factor comparison"
        )
    elif decision_impact == YES:
        candidate = f"toggling {factor} changes at least one NBC bucket"
    elif numerical == YES:
        candidate = (
            f"toggling {factor} perturbs the raw score "
            f"({difference_class.lower()}) but changes no NBC bucket"
        )
    else:
        candidate = (
            f"toggling {factor} is bit-identical here and cannot be the cause"
        )

    return {
        "comparison": f"{left_name}_vs_{right_name}",
        "left": left_name,
        "right": right_name,
        "isolated_factor": factor,
        "pairs": deltas["count"],
        "deltas": deltas,
        "difference_class": difference_class,
        "one_decimal_disagreements": one_decimal,
        "nbc_bucket_disagreements": buckets,
        "numerical_difference": numerical,
        "bse_decision_impact": decision_impact,
        "causal_candidate": candidate,
        "histograms": {
            "left": histograms_by_polarity(left_scores, polarities),
            "right": histograms_by_polarity(right_scores, polarities),
        },
        "note": (
            "numerical_difference describes floating point only. Only "
            "bse_decision_impact licenses a claim about baseline behaviour."
        ),
    }


def probe_report(probe_polarity, probe_index, scores_by_arm, polarities):
    """Per-arm raw / rounded / bucket for one named NBC pair.

    Defaults to the pair Step 1 flagged, so the new run states directly whether
    each forward-path variant lands on the literal Wang side or ours.
    """
    flat = flat_index(probe_polarity, probe_index, polarities)
    rows = {}
    for arm, scores in scores_by_arm.items():
        rows[arm] = score_report(scores[flat])
    return {
        "polarity": probe_polarity,
        "index": probe_index,
        "flat_index": flat,
        "arms": rows,
        "reference": REFERENCE_OBSERVATION,
    }


def flat_index(polarity, index, polarities):
    """Position of the nth pair of a polarity within the flat ordered list."""
    seen = -1
    for position, value in enumerate(polarities):
        if value == polarity:
            seen += 1
            if seen == index:
                return position
    raise IndexError(f"no {polarity} pair at index {index}")


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------

def _substantial(block):
    return block["difference_class"] == DIFFERENCE_SUBSTANTIAL


def _similar(block):
    return not _substantial(block)


def interpret_factor_isolation(matrix):
    """Apply the predeclared isolation rules to the comparison matrix.

    ``matrix`` maps ``"C0_vs_C1"``-style keys to comparison blocks. The rules
    are fixed in advance:

    * C0 ~ C1 and C2 ~ C3, with C0 vs C2 substantial -> attention_mask;
    * C0 ~ C2 and C1 ~ C3, with C0 vs C1 substantial -> inference_mode;
    * C0 ~ C1 and C0 ~ C2 but C0 vs C3 substantial   -> interaction;
    * nothing substantial anywhere                    -> inconclusive;
    * anything else                                   -> both contribute.
    """
    c0_c1 = matrix["C0_vs_C1"]
    c0_c2 = matrix["C0_vs_C2"]
    c0_c3 = matrix["C0_vs_C3"]
    c1_c3 = matrix["C1_vs_C3"]
    c2_c3 = matrix["C2_vs_C3"]

    inference_mode_contributes = _substantial(c0_c1)
    attention_mask_contributes = _substantial(c0_c2)

    if not any(_substantial(b) for b in (c0_c1, c0_c2, c0_c3)):
        headline = "NO_FORWARD_PATH_EFFECT"
        candidate = (
            "Neither torch.inference_mode() nor attention_mask perturbs the "
            "score beyond the negligible bound. Forward-path isolation is "
            "inconclusive: another difference remains unaccounted for."
        )
    elif _similar(c0_c1) and _similar(c2_c3) and _substantial(c0_c2):
        headline = "ATTENTION_MASK_PRIMARY"
        candidate = (
            "attention_mask is the primary candidate: toggling it moves the "
            "score in both grad contexts, while torch.inference_mode() does not "
            "move it in either."
        )
    elif _similar(c0_c2) and _similar(c1_c3) and _substantial(c0_c1):
        headline = "INFERENCE_MODE_PRIMARY"
        candidate = (
            "torch.inference_mode() is the primary candidate: toggling it moves "
            "the score with and without attention_mask, while attention_mask "
            "does not move it in either grad context."
        )
    elif _similar(c0_c1) and _similar(c0_c2) and _substantial(c0_c3):
        headline = "FACTOR_INTERACTION"
        candidate = (
            "Neither factor moves the score alone, but both together do: an "
            "interaction between attention-mask handling and "
            "torch.inference_mode(), most likely a different attention kernel "
            "being selected when both are set."
        )
    else:
        headline = "BOTH_FACTORS_CONTRIBUTE"
        candidate = (
            "Both torch.inference_mode() and attention_mask perturb the score, "
            "and neither is cleanly dominant on this sample."
        )

    return {
        "headline": headline,
        "causal_candidate": candidate,
        "inference_mode_contributes": YES if inference_mode_contributes else NO,
        "attention_mask_contributes": YES if attention_mask_contributes else NO,
        "difference_classes": {
            key: matrix[key]["difference_class"] for key in sorted(matrix)
        },
    }


def assess_reference_reproduction(probe, control_block):
    """Did any single factor reproduce the Step 1 bucket flip?

    The experiment succeeds only if some arm moves positive pair 169 from the
    literal Wang bucket to the repository bucket. Finding merely *a* numerical
    difference does not explain the Step 1 result.
    """
    reference = REFERENCE_OBSERVATION
    wang_bucket = reference["literal_wang"]["nbc_bucket"]
    repository_bucket = reference["repository"]["nbc_bucket"]

    arms = probe["arms"]
    c0 = arms.get("C0")
    reproduced_by = [
        name
        for name, row in arms.items()
        if name not in ("C0", "C0R") and row["nbc_bucket"] == repository_bucket
    ]

    c0_matches_wang = c0 is not None and c0["nbc_bucket"] == wang_bucket
    deterministic = control_block is None or control_block["numerical_difference"] == NO

    if not deterministic:
        explains = "UNDETERMINED"
        message = (
            "The determinism control failed: C0 repeated is not bit-identical to "
            "C0, so the forward pass is nondeterministic on this device and no "
            "attribution in this report is sound. Rerun with deterministic "
            "kernels before interpreting anything else."
        )
    elif not c0_matches_wang:
        explains = "UNDETERMINED"
        message = (
            "C0 did not reproduce the literal Wang bucket for the probe pair, so "
            "this run is not comparable to Step 1. Check that the same model "
            "revision and device are in use before interpreting the arms."
        )
    elif reproduced_by:
        explains = YES
        message = (
            "Reproduced. "
            + ", ".join(sorted(reproduced_by))
            + f" move the probe pair from bucket {wang_bucket} to bucket "
            f"{repository_bucket}, matching the Step 1 divergence."
        )
    else:
        explains = NO
        message = (
            "Not reproduced. Every arm keeps the probe pair in bucket "
            f"{wang_bucket}, so neither torch.inference_mode() nor "
            "attention_mask explains the Step 1 result on this run. Forward-path "
            "isolation is inconclusive and another difference remains."
        )

    return {
        "explains_reference_observation": explains,
        "reproduced_by_arms": sorted(reproduced_by),
        "c0_matches_literal_wang": c0_matches_wang,
        "determinism_control_passed": deterministic,
        "message": message,
    }


def build_verdict(matrix, probe, control_block):
    """Assemble the interpretation, kept free of any single conflated word."""
    isolation = interpret_factor_isolation(matrix)
    reproduction = assess_reference_reproduction(probe, control_block)

    decision_impacting = sorted(
        key for key, block in matrix.items() if block["bse_decision_impact"] == YES
    )
    numerically_differing = sorted(
        key for key, block in matrix.items() if block["numerical_difference"] == YES
    )

    return {
        "numerical_difference": YES if numerically_differing else NO,
        "bse_decision_impact": YES if decision_impacting else NO,
        "causal_candidate": isolation["causal_candidate"],
        "headline": isolation["headline"],
        "comparisons_with_numerical_difference": numerically_differing,
        "comparisons_with_bse_decision_impact": decision_impacting,
        "factor_isolation": isolation,
        "reference_reproduction": reproduction,
        "reporting_note": (
            "numerical_difference and bse_decision_impact are reported "
            "separately and are not interchangeable. A raw floating-point score "
            "change alone is NOT evidence that BSE behaviour changed; only an "
            "NBC bucket change is, because the Bayesian update consumes the "
            "bucket."
        ),
    }
