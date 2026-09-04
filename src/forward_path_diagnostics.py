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
UNDETERMINED = "UNDETERMINED"

# Guard reasons that force the overall verdict to withhold causal attribution.
GUARD_NONDETERMINISTIC = "nondeterministic_device"
GUARD_REFERENCE_NOT_REPRODUCED = "reference_not_reproduced"

HEADLINE_UNDETERMINED_NONDETERMINISTIC = "UNDETERMINED_NONDETERMINISTIC"
HEADLINE_UNDETERMINED_REFERENCE = "UNDETERMINED_REFERENCE_NOT_REPRODUCED"


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
    the production scorer instead scales in float32 before widening; that is a
    separate, already-catalogued difference of order 1e-6, not under test here,
    and it is why the reference-reproduction checks compare within a bound
    rather than demanding bit-equality.
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

    The comparison is bounded rather than exact because the production scorer
    scales to 0-100 in float32 while this diagnostic uses Wang's float64 form
    for every arm. That difference is of order 1e-6, far inside the bound.
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
    """The arm that most closely reconstructs the Step 1 repository scorer.

    C4 carries attention_mask, inference_mode and token_type_ids together, so it
    is the closest available reconstruction of A3. When C4 was not run, C3
    stands in: at ``type_vocab_size`` 0 the two are expected to be identical,
    and the C3-vs-C4 check is what tests that expectation.
    """
    if "C4" in available_arms:
        return "C4"
    if "C3" in available_arms:
        return "C3"
    return None


def assess_reference_reproduction(probe, control_block, matrix=None, c3_vs_c4=None):
    """Evaluate the full causal chain against the recorded Step 1 divergence.

    A causal claim needs the whole chain, not one coincidence:

    1. C0 reproduces Step 1 A1 -- raw within bound, rounded value, and bucket;
    2. C0R is bit-identical to C0, so the device is deterministic;
    3. some isolated factor moves the probe pair off the A1 bucket;
    4. the bridge arm reproduces Step 1 A3 -- raw within bound, rounded, bucket;
    5. C3 and C4 agree, confirming token_type_ids are inert at
       ``type_vocab_size`` 0.

    Links 1, 2 and 4 are required. Link 3 is informational. Link 5 is a
    confirmation whose failure surfaces as a warning rather than blocking the
    reproduction claim, since C3 != C4 would be a separate finding about
    token_type_ids rather than a fault in the A1 -> A3 reconstruction.

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

    arms_reaching_repository_bucket = sorted(
        name
        for name, row in arms.items()
        if name not in ("C0", "C0R") and row["nbc_bucket"] == repository_bucket
    )

    c0_reproduces = bool(c0_check and c0_check["reproduces"])
    bridge_reproduces = bool(bridge_check and bridge_check["reproduces"])
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
            "requirement": (
                f"an isolated factor moves the probe pair off bucket {wang_bucket}"
            ),
            "required": False,
            "passed": bool(arms_reaching_repository_bucket),
            "detail": {
                "arms_reaching_repository_bucket": arms_reaching_repository_bucket
            },
        },
        {
            "link": 4,
            "requirement": (
                f"the bridge arm ({bridge or 'none available'}) reproduces Step 1 "
                "A3 raw, rounded and bucket"
            ),
            "required": True,
            "passed": bridge_reproduces,
            "detail": bridge_check,
        },
        {
            "link": 5,
            "requirement": "C3 == C4, confirming token_type_ids are inert",
            "required": False,
            "passed": token_type_ids_inert,
            "detail": None
            if c3_vs_c4 is None
            else {
                "numerical_difference": c3_vs_c4["numerical_difference"],
                "max_absolute_delta": c3_vs_c4["deltas"]["max_absolute"],
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
    elif bridge_reproduces:
        guard = None
        explains = YES
        message = (
            f"Reproduced. C0 matches Step 1 A1 and the bridge arm {bridge} matches "
            "Step 1 A3 on raw score, rounded value and NBC bucket, so the "
            "forward-path factors account for the Step 1 divergence."
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
            "Not reproduced. C0 matches Step 1 A1, but no arm reconstructs Step 1 "
            f"A3 to within {NEGLIGIBLE_MAX_ABS_DELTA:.0e} on the raw score with "
            "the recorded rounded value and bucket" + extra + ". Neither "
            "torch.inference_mode() nor attention_mask explains the Step 1 "
            "result; another difference remains."
        )

    warnings = []
    if token_type_ids_inert is False:
        warnings.append(
            "C3 and C4 differ even though type_vocab_size is 0. token_type_ids "
            "were expected to be inert; this is a separate finding and does not "
            "by itself invalidate the A1 -> A3 reconstruction."
        )
    if bridge == "C3":
        warnings.append(
            "C4 was not run, so C3 stands in as the bridge arm. Link 5 cannot be "
            "evaluated and token_type_ids remain unconfirmed as inert."
        )

    return {
        "explains_reference_observation": explains,
        "guard": guard,
        "causal_chain": chain,
        "bridge_arm": bridge,
        "c0_vs_step1_a1": c0_check,
        "bridge_vs_step1_a3": bridge_check,
        "arms_reaching_repository_bucket": arms_reaching_repository_bucket,
        "c0_reproduces_literal_wang": c0_reproduces,
        "bridge_reproduces_repository": bridge_reproduces,
        "determinism_control_passed": bool(deterministic),
        "token_type_ids_inert": token_type_ids_inert,
        "required_links_passed": all(
            link["passed"] for link in chain if link["required"]
        ),
        "warnings": warnings,
        "message": message,
    }


def build_verdict(matrix, probe, control_block, c3_vs_c4=None):
    """Assemble the interpretation, withholding attribution when guarded.

    The factor-isolation headline is promoted to the overall verdict only when
    both guards pass: the device reproduced itself, and C0 reproduced the Step 1
    literal-Wang score. Reporting a causal headline while simultaneously saying
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
