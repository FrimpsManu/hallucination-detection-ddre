"""Step 2: isolate which forward-path argument explains the A1/A3 divergence.

Step 1 (``scripts/diagnose_scorer_paths.py``) established that the literal Wang
scorer and this repository's scorer tokenize identically -- 589/589 input_ids
matched -- yet disagreed on exactly one BSE decision: positive NBC pair 169 fell
in bucket 1 under the literal path and bucket 2 under ours, moving the positive
Laplace-smoothed histogram from ``[1, 78, 19, 13, 6, 5, 28, 57, 1, 1]`` to
``[1, 77, 20, 13, 6, 5, 28, 57, 1, 1]``.

Tokenization is therefore excluded. ``token_type_ids`` are *predicted* inert
because ``type_vocab_size`` is 0, but that prediction is measured rather than
assumed: C4 carries the argument and the C3-vs-C4 check is a required link.

This module covers the SECONDARY decomposition, which splits the forward call
into its individual arguments. It is secondary because the two scorers also
differ in their extraction/scaling path, and on a half-precision tensor that
difference alone reproduces the Step 1 observation exactly -- see
``src/extraction_path_diagnostics``, which must be run first. Every arm here
holds extraction fixed at Wang's form, so this module cannot see an
extraction-path effect at all.

Within the forward call, two candidate differences remain between Wang's
released call and ours:

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
UNDETERMINED = "UNDETERMINED"

# Guard reasons that force the overall verdict to withhold causal attribution.
GUARD_NONDETERMINISTIC = "nondeterministic_device"
GUARD_REFERENCE_NOT_REPRODUCED = "reference_not_reproduced"
GUARD_BRIDGE_NOT_EVALUATED = "bridge_not_evaluated"
GUARD_TOKEN_TYPE_EFFECT = "token_type_ids_not_inert"

HEADLINE_UNDETERMINED_NONDETERMINISTIC = "UNDETERMINED_NONDETERMINISTIC"
HEADLINE_UNDETERMINED_REFERENCE = "UNDETERMINED_REFERENCE_NOT_REPRODUCED"
HEADLINE_UNDETERMINED_BRIDGE = "UNDETERMINED_BRIDGE_NOT_EVALUATED"
HEADLINE_UNDETERMINED_TOKEN_TYPE = "UNDETERMINED_TOKEN_TYPE_EFFECT"


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
        "label": "repository-side bridge: + token_type_ids",
        "attention_mask": True,
        "token_type_ids": True,
        "inference_mode": True,
        "call": (
            "with torch.inference_mode(): model(input_ids, "
            "attention_mask=attention_mask, token_type_ids=token_type_ids)"
        ),
        "role": "bridge",
        "isolates": "token_type_ids",
        "note": (
            "The repository-side bridge: the only arm carrying attention_mask, "
            "inference_mode and token_type_ids together, so the only one that "
            "reconstructs Step 1 A3. type_vocab_size is 0 and therefore PREDICTS "
            "this argument is inert, but a prediction is not a measurement: C3 "
            "vs C4 measures it, and that check is a REQUIRED link in the causal "
            "chain rather than a confirmation. C3 is never accepted as a "
            "substitute for C4."
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


def build_forward_kwargs(encoded, spec):
    """Keyword arguments for one arm's model call.

    ``input_ids`` is always positional, matching Wang's released
    ``model(inputs["input_ids"])``. Only the extra arguments vary by arm.
    """
    kwargs = {}
    if spec["attention_mask"]:
        kwargs["attention_mask"] = encoded["attention_mask"]
    if spec["token_type_ids"]:
        if "token_type_ids" not in encoded:
            raise KeyError(
                f"arm {spec['name']} requires token_type_ids but the tokenizer "
                "emitted none"
            )
        kwargs["token_type_ids"] = encoded["token_type_ids"]
    return kwargs


def forward_once(model, torch_module, input_ids, kwargs, inference_mode):
    """Run the model call, and ONLY the model call, under the chosen context.

    This mirrors ``src/utils.py::EntailmentScorer._infer_batch`` exactly, where
    the ``with torch.inference_mode():`` block contains the model call and
    nothing else::

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits / 5.0, dim=-1)

    Putting the softmax inside the context too would add a second uncontrolled
    variable and stop the arm from reproducing the repository's forward path.
    The boundary is therefore load-bearing, which is why the call lives in its
    own function that a test can wrap.
    """
    if inference_mode:
        with torch_module.inference_mode():
            return model(input_ids, **kwargs)
    return model(input_ids, **kwargs)


def score_one_pair(model, encoded, spec, torch_module):
    """Score one tokenized pair under one arm.

    ``torch`` is injected rather than imported so this stays testable without a
    real torch install, and so the control-flow boundary above can be asserted
    directly.

    Everything after :func:`forward_once` is common to every arm and always
    runs OUTSIDE any inference-mode context. The extraction is Wang's released
    ``utils.py:59-65`` float64 form (``.tolist()`` then ``* 100``) for all arms
    alike, so score extraction is not a variable in this experiment. Note that
    the production scorer instead scales by 100 while still inside the tensor.
    That is a SEPARATE FACTOR, not a rounding curiosity: on a half-precision
    tensor it reproduces the whole Step 1 divergence on its own (0.199462890625
    gives 19.9462890625 one way and 19.953125 the other). It is deliberately
    held fixed here so the forward arguments can be isolated, and it is tested
    by the primary decomposition in ``src/extraction_path_diagnostics``.
    """
    kwargs = build_forward_kwargs(encoded, spec)
    output = forward_once(
        model, torch_module, encoded["input_ids"], kwargs, spec["inference_mode"]
    )
    probabilities = torch_module.softmax(
        output["logits"][0] / SOFTMAX_TEMPERATURE, -1
    ).tolist()
    return entailment_score_from_probabilities(probabilities)


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


def score_matches_reference(raw, expected, bound=NEGLIGIBLE_MAX_ABS_DELTA):
    """Does a measured score reproduce a recorded Step 1 score?

    All three views must agree, not just the bucket. A bucket match alone is far
    too weak for a causal experiment whose whole subject is a 0.0068 divergence:
    two scores several buckets' worth of noise apart can share a bucket by luck.
    The raw score must also land within ``bound`` of the recorded value.

    The comparison is bounded rather than exact to absorb kernel-level and
    library-level perturbation, which sits far below the bound. It is NOT bounded
    to absorb the extraction/scaling difference: that is a separate factor,
    capable of moving a score by 0.0068 on a half-precision tensor, and it is
    measured by the primary decomposition rather than tolerated here.
    """
    report = score_report(raw)
    raw_delta = abs(float(raw) - float(expected["raw"]))
    raw_ok = raw_delta <= bound
    rounded_ok = report["rounded_one_decimal"] == expected["rounded"]
    bucket_ok = report["nbc_bucket"] == expected["nbc_bucket"]
    return {
        "raw": report["raw"],
        "expected_raw": float(expected["raw"]),
        "raw_delta": raw_delta,
        "bound": bound,
        "raw_within_bound": raw_ok,
        "rounded_one_decimal": report["rounded_one_decimal"],
        "expected_rounded": expected["rounded"],
        "rounded_matches": rounded_ok,
        "nbc_bucket": report["nbc_bucket"],
        "expected_nbc_bucket": expected["nbc_bucket"],
        "bucket_matches": bucket_ok,
        "reproduces": bool(raw_ok and rounded_ok and bucket_ok),
    }


def bridge_arm_name(available_arms):
    """The arm that reconstructs the Step 1 repository scorer: C4, or nothing.

    C4 carries attention_mask, inference_mode and token_type_ids together, so it
    is the only arm that reconstructs A3. C3 is deliberately NOT accepted as a
    stand-in. If C3 could substitute, a run where C3 lands on A1 and C4 lands on
    A3 would be scored as a success for attention_mask and inference_mode, when
    in fact the only argument that moved the result was token_type_ids.
    """
    return "C4" if "C4" in available_arms else None


def assess_reference_reproduction(probe, control_block, matrix=None, c3_vs_c4=None):
    """Evaluate the formal causal chain against the recorded Step 1 divergence.

    Four links are required, and all four must pass before the C0/C1/C2/C3
    factorial may be promoted as an explanation of the A1 -> A3 divergence:

    1. C0 reproduces Step 1 A1 -- raw within bound, rounded value, and bucket;
    2. C0R is bit-identical to C0, so the device is deterministic;
    3. C4 reproduces Step 1 A3 -- raw within bound, rounded value, and bucket;
    4. C3 and C4 are bit-identical.

    C4 is therefore part of the formal chain, not a confirmation arm.

    Link 4 is required, not a confirmation. ``type_vocab_size`` being 0 predicts
    that token_type_ids are inert, but a prediction is not a measurement: an
    empirical C3 != C4 overrides it. Consider the run where C0 ~ A1, C3 ~ A1 and
    C4 ~ A3. Every other link passes, yet the only argument that moved the
    result is token_type_ids, and attention_mask and inference_mode have
    reconstructed nothing. Treating link 4 as a warning would report that run as
    a success for the target factors, which would be wrong.

    When link 4 does pass alongside link 3, C3 is bit-identical to an arm that
    reproduces A3, so C3 reconstructs the repository-side behaviour *without*
    token_type_ids. That is the empirical validation of inertness, and only then
    can the factorial isolate attention_mask, inference_mode, their interaction,
    or no effect.

    A fifth link -- some isolated factor moving the probe pair off the A1 bucket
    -- is informational and never gates.

    Landing in the repository bucket is explicitly NOT sufficient on its own.
    """
    reference = REFERENCE_OBSERVATION
    arms = probe["arms"]
    wang_bucket = reference["literal_wang"]["nbc_bucket"]
    repository_bucket = reference["repository"]["nbc_bucket"]

    deterministic = control_block is None or control_block["numerical_difference"] == NO

    c0_check = (
        score_matches_reference(arms["C0"]["raw"], reference["literal_wang"])
        if "C0" in arms
        else None
    )
    bridge = bridge_arm_name(set(arms) - {"C0", "C0R"})
    bridge_check = (
        score_matches_reference(arms[bridge]["raw"], reference["repository"])
        if bridge
        else None
    )

    # Informational: where C3 lands relative to A3. This is what exposes the
    # C3 ~ A1 / C4 ~ A3 pattern that link 4 exists to catch.
    c3_check = (
        score_matches_reference(arms["C3"]["raw"], reference["repository"])
        if "C3" in arms
        else None
    )

    arms_reaching_repository_bucket = sorted(
        name
        for name, row in arms.items()
        if name not in ("C0", "C0R") and row["nbc_bucket"] == repository_bucket
    )

    c0_reproduces = bool(c0_check and c0_check["reproduces"])
    bridge_reproduces = bool(bridge_check and bridge_check["reproduces"])
    bridge_evaluated = bridge is not None and c3_vs_c4 is not None
    token_type_ids_inert = (
        None if c3_vs_c4 is None else c3_vs_c4["numerical_difference"] == NO
    )

    chain = [
        {
            "link": 1,
            "requirement": "C0 reproduces Step 1 A1 raw, rounded and bucket",
            "required": True,
            "passed": c0_reproduces,
            "detail": c0_check,
        },
        {
            "link": 2,
            "requirement": "C0R is bit-identical to C0 (device is deterministic)",
            "required": True,
            "passed": bool(deterministic),
            "detail": None
            if control_block is None
            else {"max_absolute_delta": control_block["deltas"]["max_absolute"]},
        },
        {
            "link": 3,
            "requirement": "C4 reproduces Step 1 A3 raw, rounded and bucket",
            "required": True,
            "passed": bridge_reproduces,
            "detail": bridge_check,
        },
        {
            "link": 4,
            "requirement": (
                "C3 and C4 are bit-identical (token_type_ids empirically inert)"
            ),
            "required": True,
            "passed": token_type_ids_inert,
            "detail": None
            if c3_vs_c4 is None
            else {
                "numerical_difference": c3_vs_c4["numerical_difference"],
                "max_absolute_delta": c3_vs_c4["deltas"]["max_absolute"],
            },
        },
        {
            "link": 5,
            "requirement": (
                f"an isolated factor moves the probe pair off bucket {wang_bucket}"
            ),
            "required": False,
            "passed": bool(arms_reaching_repository_bucket),
            "detail": {
                "arms_reaching_repository_bucket": arms_reaching_repository_bucket
            },
        },
    ]

    if not deterministic:
        guard = GUARD_NONDETERMINISTIC
        explains = UNDETERMINED
        message = (
            "The determinism control failed: C0 repeated is not bit-identical to "
            "C0, so the forward pass is nondeterministic on this device and no "
            "attribution in this report is sound. Causal attribution is withheld. "
            "Rerun with deterministic kernels before interpreting anything else."
        )
    elif not c0_reproduces:
        guard = GUARD_REFERENCE_NOT_REPRODUCED
        explains = UNDETERMINED
        message = (
            "C0 did not reproduce Step 1 A1 within the predeclared bound "
            f"({NEGLIGIBLE_MAX_ABS_DELTA:.0e} on the raw score, plus the recorded "
            "rounded value and bucket), so this run is not comparable to Step 1 "
            "and causal attribution is withheld. Check that the same model "
            "revision, device and dependency versions are in use."
        )
    elif not bridge_evaluated:
        guard = GUARD_BRIDGE_NOT_EVALUATED
        explains = UNDETERMINED
        message = (
            "C4 was not run, so the bridge arm could not be evaluated and C3 "
            "cannot stand in for it. Without C4 there is no way to show that "
            "token_type_ids are inert, and a run where C3 lands on A1 while C4 "
            "lands on A3 would be indistinguishable from one where "
            "attention_mask and inference_mode did the work. Causal attribution "
            "is withheld. --skip-c4 remains useful for debugging but cannot "
            "establish the formal causal chain."
        )
    elif token_type_ids_inert is not True:
        guard = GUARD_TOKEN_TYPE_EFFECT
        explains = UNDETERMINED
        message = (
            "C3 and C4 are not bit-identical, so the token_type_ids argument "
            "empirically changed the result even though type_vocab_size is 0 and "
            "predicts it should be inert. A measurement overrides that "
            "prediction. attention_mask and torch.inference_mode() therefore "
            "cannot yet be isolated as the explanation of the Step 1 A1 -> A3 "
            "divergence, and causal attribution is withheld. Investigate the "
            "token_type_ids path before interpreting the factorial."
        )
    elif bridge_reproduces:
        guard = None
        explains = YES
        message = (
            "Reproduced. C0 matches Step 1 A1, C4 matches Step 1 A3 on raw score, "
            "rounded value and NBC bucket, and C3 is bit-identical to C4 -- so C3 "
            "reconstructs the repository-side behaviour without token_type_ids, "
            "empirically confirming they are inert. The C0/C1/C2/C3 factorial may "
            "be read as an explanation of the forward-path divergence."
        )
    else:
        guard = None
        explains = NO
        extra = ""
        if arms_reaching_repository_bucket:
            extra = (
                ", although "
                + ", ".join(arms_reaching_repository_bucket)
                + " land in the repository bucket, which is not on its own "
                "evidence that the divergence was reproduced"
            )
        message = (
            "Not reproduced. C0 matches Step 1 A1 and token_type_ids are "
            "confirmed inert, but C4 does not reconstruct Step 1 A3 to within "
            f"{NEGLIGIBLE_MAX_ABS_DELTA:.0e} on the raw score with the recorded "
            "rounded value and bucket" + extra + ". Neither torch.inference_mode() "
            "nor attention_mask explains the Step 1 result; another difference "
            "remains."
        )

    warnings = []
    if token_type_ids_inert is False:
        warnings.append(
            "C3 and C4 differ even though type_vocab_size is 0. This is a "
            "finding in its own right about the token_type_ids path, and it "
            "blocks causal attribution for the target factors."
        )
    if bridge is None:
        warnings.append(
            "C4 was not run. C3 is NOT accepted as a substitute bridge arm, "
            "because that substitution is exactly what would hide a "
            "token_type_ids-driven result."
        )
    if c3_check is not None and bridge_check is not None:
        if bridge_check["reproduces"] and not c3_check["reproduces"]:
            warnings.append(
                "C4 reconstructs Step 1 A3 but C3 does not. The difference "
                "between them is token_type_ids alone, so the target factors "
                "have not reconstructed A3."
            )

    return {
        "explains_reference_observation": explains,
        "guard": guard,
        "causal_chain": chain,
        "bridge_arm": bridge,
        "bridge_evaluated": bridge_evaluated,
        "c0_vs_step1_a1": c0_check,
        "bridge_vs_step1_a3": bridge_check,
        "c3_vs_step1_a3_informational": c3_check,
        "arms_reaching_repository_bucket": arms_reaching_repository_bucket,
        "c0_reproduces_literal_wang": c0_reproduces,
        "bridge_reproduces_repository": bridge_reproduces,
        "determinism_control_passed": bool(deterministic),
        "token_type_ids_inert": token_type_ids_inert,
        "required_links_passed": all(
            link["passed"] is True for link in chain if link["required"]
        ),
        "warnings": warnings,
        "message": message,
    }


def build_verdict(matrix, probe, control_block, c3_vs_c4=None):
    """Assemble the interpretation, withholding attribution when guarded.

    The factor-isolation headline is promoted to the overall verdict only when
    every guard passes: the device reproduced itself, C0 reproduced the Step 1
    literal-Wang score, C4 was actually run, and C3 was bit-identical to C4.
    Reporting a causal headline while simultaneously saying
    attribution is withheld would be a contradiction, so on a guard failure the
    overall ``headline`` and ``causal_candidate`` say attribution is withheld
    and nothing else.

    The pairwise numerical diagnostics and the unpromoted isolation block are
    still carried in full under ``factor_isolation``, for inspection.
    """
    isolation = interpret_factor_isolation(matrix)
    reproduction = assess_reference_reproduction(
        probe, control_block, matrix=matrix, c3_vs_c4=c3_vs_c4
    )

    decision_impacting = sorted(
        key for key, block in matrix.items() if block["bse_decision_impact"] == YES
    )
    numerically_differing = sorted(
        key for key, block in matrix.items() if block["numerical_difference"] == YES
    )

    guard = reproduction["guard"]
    if guard == GUARD_NONDETERMINISTIC:
        headline = HEADLINE_UNDETERMINED_NONDETERMINISTIC
        candidate = (
            "Causal attribution withheld: the determinism control failed, so a "
            "difference between arms cannot be distinguished from run-to-run "
            "nondeterminism on this device."
        )
        withheld = True
    elif guard == GUARD_REFERENCE_NOT_REPRODUCED:
        headline = HEADLINE_UNDETERMINED_REFERENCE
        candidate = (
            "Causal attribution withheld: C0 did not reproduce the Step 1 "
            "literal-Wang score, so this run is not comparable to the "
            "observation it is meant to explain."
        )
        withheld = True
    elif guard == GUARD_BRIDGE_NOT_EVALUATED:
        headline = HEADLINE_UNDETERMINED_BRIDGE
        candidate = (
            "Causal attribution withheld: C4 was not run, so the repository-side "
            "bridge could not be evaluated and token_type_ids could not be shown "
            "to be inert."
        )
        withheld = True
    elif guard == GUARD_TOKEN_TYPE_EFFECT:
        headline = HEADLINE_UNDETERMINED_TOKEN_TYPE
        candidate = (
            "Causal attribution withheld: the supposedly inert token_type_ids "
            "argument empirically changed the result (C3 != C4), so the target "
            "factors cannot yet be isolated as the explanation."
        )
        withheld = True
    else:
        headline = isolation["headline"]
        candidate = isolation["causal_candidate"]
        withheld = False

    return {
        "numerical_difference": YES if numerically_differing else NO,
        "bse_decision_impact": YES if decision_impacting else NO,
        "causal_candidate": candidate,
        "headline": headline,
        "causal_attribution_withheld": withheld,
        "guard": guard,
        "comparisons_with_numerical_difference": numerically_differing,
        "comparisons_with_bse_decision_impact": decision_impacting,
        "factor_isolation": isolation,
        "factor_isolation_promoted": not withheld,
        "reference_reproduction": reproduction,
        "reporting_note": (
            "numerical_difference and bse_decision_impact are reported separately "
            "and are not interchangeable. A raw floating-point score change alone "
            "is NOT evidence that BSE behaviour changed; only an NBC bucket "
            "change is, because the Bayesian update consumes the bucket. When "
            "causal_attribution_withheld is true the pairwise numbers remain "
            "valid measurements, but no factor may be named as the cause."
        ),
    }
