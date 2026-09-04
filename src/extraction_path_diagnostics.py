"""Step 2 primary: is the divergence in the forward path or the extraction path?

The Step 1 A1/A3 divergence was originally attributed to a remaining difference
in the *forward call*. That premise was incomplete. The two scorers also differ
in how they turn logits into a 0-100 score, and on a half-precision tensor that
difference alone reproduces the observation exactly.

Released ``utils.py:59-65`` converts out of the tensor and *then* scales::

    probabilities = torch.softmax(output["logits"][0] / 5, -1).tolist()
    raw = float(probabilities[0]) * 100.0          # float64 multiply

``src/utils.py:122-124`` scales *inside* the tensor and then converts::

    probs = torch.softmax(outputs.logits / 5.0, dim=-1)
    scores = probs[:, entailment_index] * 100.0    # multiply in tensor dtype
    return [float(x) for x in scores.detach().cpu().tolist()]

For the Step 1 probe pair the underlying probability is 0.199462890625, which is
exactly representable in float16. Then:

===========================  ==================  ================
path                         value               Step 1 record
===========================  ==================  ================
``float(p) * 100``           19.9462890625       A1 = 19.9462890625
``float16(p * 100)``         19.953125           A3 = 19.953125
===========================  ==================  ================

The gap is 0.0068359375, exactly the recorded divergence. float16 spacing in
[16, 32) is 0.015625, and 19.9462890625 sits at 1276.5625 steps, so it rounds up
to 19.953125. The same arithmetic in float32 does **not** reproduce it, so this
mechanism requires the half-precision tensor the T4 run appears to have used.

That makes the extraction path a candidate that must be tested *before* the
forward-path factorial, and it is why the forward-path arms cannot settle the
question on their own: they deliberately hold extraction fixed at Wang's form
for every arm, which removes the very factor under suspicion.

This module holds the 2x2 decomposition over two genuinely independent factors:

    F0 = literal Wang forward      F1 = repository batch-1 forward
    X0 = literal Wang extraction   X1 = repository extraction

    D00 = F0 + X0   exact A1 reconstruction
    D01 = F0 + X1   extraction changed only
    D10 = F1 + X0   forward changed only
    D11 = F1 + X1   exact A3 reconstruction

Both extractions are pure functions of the logits, so one forward pass per cell
row serves both columns. D00 and D01 therefore share bit-identical logits, and
so do D10 and D11 -- the extraction comparison is exact by construction rather
than by assumption.

Standard library only, with ``torch`` injected, so every function here is
testable without a model download.
"""

from src.forward_path_diagnostics import (
    DIFFERENCE_SUBSTANTIAL,
    ENTAILMENT_INDEX,
    NEGLIGIBLE_MAX_ABS_DELTA,
    NO,
    REFERENCE_OBSERVATION,
    UNDETERMINED,
    YES,
    build_forward_kwargs,
    classify_difference,
    forward_once,
    histograms_by_polarity,
    score_matches_reference,
    score_report,
)
from src.scoring_diagnostics import (
    bucket_disagreements,
    delta_stats,
    nbc_bucket,
    one_decimal_disagreements,
)


FACTOR_FORWARD = "forward_path"
FACTOR_EXTRACTION = "extraction_path"

HEADLINE_EXTRACTION_EXPLAINS = "EXTRACTION_PATH_EXPLAINS"
HEADLINE_EXTRACTION_IMPLICATED = "EXTRACTION_PATH_IMPLICATED"
HEADLINE_FORWARD_EXPLAINS = "FORWARD_PATH_EXPLAINS"
HEADLINE_BOTH_CONTRIBUTE = "BOTH_PATHS_CONTRIBUTE"
HEADLINE_NEITHER = "NEITHER_PATH_EXPLAINS"
HEADLINE_UNDETERMINED_NONDETERMINISTIC = "UNDETERMINED_NONDETERMINISTIC"
HEADLINE_UNDETERMINED_ENDPOINTS = "UNDETERMINED_ENDPOINTS_NOT_RECONSTRUCTED"

GUARD_NONDETERMINISTIC = "nondeterministic_device"
GUARD_ENDPOINTS_NOT_RECONSTRUCTED = "endpoints_not_reconstructed"


# --------------------------------------------------------------------------
# Factor levels
# --------------------------------------------------------------------------

FORWARD_SPECS = {
    "F0": {
        "name": "F0",
        "label": "literal Wang forward",
        "attention_mask": False,
        "token_type_ids": False,
        "inference_mode": False,
        "call": "model(input_ids)",
        "source": "released utils.py:57",
    },
    "F1": {
        "name": "F1",
        "label": "repository batch-1 forward",
        "attention_mask": True,
        "token_type_ids": True,
        "inference_mode": True,
        "call": (
            "with torch.inference_mode(): model(input_ids, "
            "attention_mask=..., token_type_ids=...)"
        ),
        "source": "src/utils.py:119-120, at batch size 1",
    },
}

EXTRACTION_SPECS = {
    "X0": {
        "name": "X0",
        "label": "literal Wang extraction",
        "code": 'float(softmax(logits[0] / 5, -1).tolist()[0]) * 100.0',
        "scales_in": "float64, after leaving the tensor",
        "source": "released utils.py:59-65",
    },
    "X1": {
        "name": "X1",
        "label": "repository extraction",
        "code": 'float((softmax(logits / 5.0, dim=-1)[:, 0] * 100.0).tolist()[0])',
        "scales_in": "the tensor dtype, before leaving the tensor",
        "source": "src/utils.py:122-124",
    },
}

CELL_SPECS = (
    {"name": "D00", "forward": "F0", "extraction": "X0", "reconstructs": "Step 1 A1"},
    {"name": "D01", "forward": "F0", "extraction": "X1", "reconstructs": None},
    {"name": "D10", "forward": "F1", "extraction": "X0", "reconstructs": None},
    {"name": "D11", "forward": "F1", "extraction": "X1", "reconstructs": "Step 1 A3"},
)

CELL_BY_NAME = {cell["name"]: cell for cell in CELL_SPECS}

# The four comparisons the review asked for, each isolating exactly one factor.
CELL_COMPARISONS = (
    ("D00", "D01"),  # extraction, under the Wang forward
    ("D00", "D10"),  # forward, under Wang extraction
    ("D01", "D11"),  # forward, under repository extraction
    ("D10", "D11"),  # extraction, under the repository forward
)


def cell_isolated_factor(left_name, right_name):
    """Which of the two factors differs between two cells."""
    left = CELL_BY_NAME[left_name]
    right = CELL_BY_NAME[right_name]
    toggled = []
    if left["forward"] != right["forward"]:
        toggled.append(FACTOR_FORWARD)
    if left["extraction"] != right["extraction"]:
        toggled.append(FACTOR_EXTRACTION)
    return " + ".join(toggled) if toggled else None


# --------------------------------------------------------------------------
# The two extraction paths, transcribed literally
# --------------------------------------------------------------------------

def wang_extraction(logits, torch_module, entailment_index=ENTAILMENT_INDEX):
    """Released ``utils.py:59-65``.

    Leaves the tensor via ``.tolist()`` first, so the ``* 100`` happens in
    Python float64 and cannot be affected by the tensor dtype.
    """
    probabilities = torch_module.softmax(logits[0] / 5, -1).tolist()
    return float(probabilities[entailment_index]) * 100.0


def repository_extraction(logits, torch_module, entailment_index=ENTAILMENT_INDEX):
    """``src/utils.py:122-124``.

    Multiplies by 100 while still in the tensor, so the product is rounded to
    the tensor dtype. On a half-precision tensor that rounding is coarse enough
    to move a score by up to half of 0.015625 in the 16-32 range, which is what
    makes this a candidate cause rather than a rounding curiosity.
    """
    probs = torch_module.softmax(logits / 5.0, dim=-1)
    scores = probs[:, entailment_index] * 100.0
    return float(scores.detach().cpu().tolist()[0])


EXTRACTORS = {"X0": wang_extraction, "X1": repository_extraction}


# --------------------------------------------------------------------------
# Sub-diagnostic: softmax shape vs scaling order
# --------------------------------------------------------------------------
#
# X0 and X1 differ in TWO ways at once: the softmax is applied to a 1-D row in
# one and to the 2-D batch in the other, and the * 100 happens outside the
# tensor in one and inside it in the other. The 2x2 can therefore establish
# that the extraction path as a whole explains the discrepancy, but not that
# the scaling order specifically is the cause.
#
# This separates them using the SAME logits, so it adds no forward calls.

def shape_and_scaling_probe(logits, torch_module, entailment_index=ENTAILMENT_INDEX):
    """Split the extraction difference into softmax shape and scaling order.

    Two questions, answered independently:

    1. Does softmaxing the 1-D row give a different entailment *probability*
       than softmaxing the 2-D batch and taking row 0? This is measured before
       any scaling, so it is purely about the softmax call.
    2. Holding one probability tensor fixed, does it matter whether the ``* 100``
       happens after leaving the tensor or inside it? This is the scaling-order
       question in isolation.

    The repository-shaped tensor is used for (2) so that the scaling comparison
    is exact: both sides read the same tensor object.
    """
    p_wang_shape_tensor = torch_module.softmax(logits[0] / 5.0, -1)
    p_repo_shape_tensor = torch_module.softmax(logits / 5.0, dim=-1)

    p_wang_shape = float(p_wang_shape_tensor.tolist()[entailment_index])
    p_repo_shape = float(p_repo_shape_tensor[0].tolist()[entailment_index])
    probability_delta = p_repo_shape - p_wang_shape

    # Scaling order only, on one shared probability tensor.
    p = p_repo_shape_tensor
    scale_after = float(p[0, entailment_index].tolist()) * 100.0
    scale_inside = float((p[:, entailment_index] * 100.0).detach().cpu().tolist()[0])

    return {
        "p_wang_shape": p_wang_shape,
        "p_repo_shape": p_repo_shape,
        "probability_delta": probability_delta,
        "probabilities_bit_identical": p_wang_shape == p_repo_shape,
        "scale_after": scale_after,
        "scale_inside": scale_inside,
        "scaling_only_delta": scale_inside - scale_after,
        "scaling_order_bit_identical": scale_after == scale_inside,
    }


def summarize_shape_and_scaling(probes):
    """Aggregate the per-pair sub-diagnostic across a forward row."""
    probability_deltas = [abs(p["probability_delta"]) for p in probes]
    scaling_deltas = [abs(p["scaling_only_delta"]) for p in probes]
    identical_probabilities = sum(1 for p in probes if p["probabilities_bit_identical"])
    identical_scaling = sum(1 for p in probes if p["scaling_order_bit_identical"])
    return {
        "pairs": len(probes),
        "softmax_shape": {
            "max_absolute_probability_delta": max(probability_deltas) if probes else None,
            "bit_identical_pairs": identical_probabilities,
            "all_bit_identical": identical_probabilities == len(probes),
        },
        "scaling_order": {
            "max_absolute_delta": max(scaling_deltas) if probes else None,
            "bit_identical_pairs": identical_scaling,
            "all_bit_identical": identical_scaling == len(probes),
        },
    }


def interpret_shape_and_scaling(probe_row, summary):
    """Say whether scaling order alone is sufficient, or the shape matters too.

    ``probe_row`` is the sub-diagnostic for the Step 1 probe pair.
    """
    reference = REFERENCE_OBSERVATION
    shape_negligible = (
        summary["softmax_shape"]["all_bit_identical"]
        or (summary["softmax_shape"]["max_absolute_probability_delta"] or 0.0)
        <= NEGLIGIBLE_MAX_ABS_DELTA
    )
    scaling_spans_endpoints = (
        abs(probe_row["scale_after"] - reference["literal_wang"]["raw"])
        <= NEGLIGIBLE_MAX_ABS_DELTA
        and abs(probe_row["scale_inside"] - reference["repository"]["raw"])
        <= NEGLIGIBLE_MAX_ABS_DELTA
    )

    if shape_negligible and scaling_spans_endpoints:
        return {
            "scaling_order_specifically_sufficient": True,
            "softmax_shape_material": False,
            "message": (
                "The softmax shape makes no material difference to the "
                "probability, and holding one probability tensor fixed, moving "
                "the * 100 from outside the tensor to inside it reproduces "
                f"{reference['literal_wang']['raw']} -> "
                f"{reference['repository']['raw']}. The scaling ORDER is "
                "specifically sufficient to explain the Step 1 discrepancy."
            ),
        }
    if not shape_negligible:
        return {
            "scaling_order_specifically_sufficient": False,
            "softmax_shape_material": True,
            "message": (
                "The softmax shape also changes the probability materially, so "
                "the combined extraction path is implicated. Do not attribute "
                "the discrepancy to the scaling order alone."
            ),
        }
    return {
        "scaling_order_specifically_sufficient": False,
        "softmax_shape_material": False,
        "message": (
            "The softmax shape is immaterial, but the scaling-order comparison "
            "on its own does not span the two recorded endpoints, so scaling "
            "order alone is not established as sufficient."
        ),
    }


def score_pair_both_extractions(model, encoded, forward_spec, torch_module):
    """One forward pass, both extractions.

    Returning both from a single set of logits is what makes the extraction
    comparison exact: D00 and D01 cannot differ in the forward pass because
    there was only one.
    """
    kwargs = build_forward_kwargs(encoded, forward_spec)
    output = forward_once(
        model,
        torch_module,
        encoded["input_ids"],
        kwargs,
        forward_spec["inference_mode"],
    )
    logits = output["logits"]
    return {
        "X0": wang_extraction(logits, torch_module),
        "X1": repository_extraction(logits, torch_module),
        # Computed from the same logits, so this costs no extra forward call.
        "shape_scaling": shape_and_scaling_probe(logits, torch_module),
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def compare_cells(left_name, right_name, left_scores, right_scores, polarities):
    """Compare two cells of the 2x2 over the same ordered NBC pairs."""
    deltas = delta_stats(left_scores, right_scores)
    one_decimal = one_decimal_disagreements(left_scores, right_scores)
    buckets = bucket_disagreements(left_scores, right_scores, nbc_bucket)
    difference_class = classify_difference(deltas)
    factor = cell_isolated_factor(left_name, right_name)

    numerical = NO if deltas.get("max_absolute") in (None, 0.0) else YES
    decision_impact = YES if buckets["count"] > 0 else NO

    if factor is None:
        candidate = f"{left_name} vs {right_name} toggles no factor"
    elif decision_impact == YES:
        candidate = f"changing the {factor} changes at least one NBC bucket"
    elif numerical == YES:
        candidate = (
            f"changing the {factor} perturbs the raw score "
            f"({difference_class.lower()}) but changes no NBC bucket"
        )
    else:
        candidate = f"changing the {factor} is bit-identical here"

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
    }


def histogram_movement_matches_step1(left_histogram, right_histogram):
    """Does a cell pair reproduce the recorded Step 1 positive-histogram move?

    Step 1 recorded the positive Laplace-smoothed histogram moving from
    ``[1, 78, 19, ...]`` to ``[1, 77, 20, ...]``. Reproducing that exact
    movement is much stronger evidence than reproducing a single pair, because
    it constrains all 199 positive pairs at once.
    """
    expected_left = REFERENCE_OBSERVATION["positive_histogram_literal_wang"]
    expected_right = REFERENCE_OBSERVATION["positive_histogram_repository"]
    return {
        "left_matches_step1_a1": list(left_histogram) == list(expected_left),
        "right_matches_step1_a3": list(right_histogram) == list(expected_right),
        "movement_reproduced": (
            list(left_histogram) == list(expected_left)
            and list(right_histogram) == list(expected_right)
        ),
        "expected_left": list(expected_left),
        "expected_right": list(expected_right),
        "observed_left": list(left_histogram),
        "observed_right": list(right_histogram),
    }


def probe_cell_report(probe_polarity, probe_index, scores_by_cell, polarities):
    """Raw / rounded / bucket for the Step 1 probe pair, per cell."""
    from src.forward_path_diagnostics import flat_index

    flat = flat_index(probe_polarity, probe_index, polarities)
    return {
        "polarity": probe_polarity,
        "index": probe_index,
        "flat_index": flat,
        "cells": {
            name: score_report(scores[flat]) for name, scores in scores_by_cell.items()
        },
        "reference": REFERENCE_OBSERVATION,
    }


def probe_bucket_flip_reproduced(probe, left_name, right_name):
    """Does one cell pair reproduce the recorded pair-169 bucket flip?"""
    cells = probe["cells"]
    if left_name not in cells or right_name not in cells:
        return False
    return (
        cells[left_name]["nbc_bucket"]
        == REFERENCE_OBSERVATION["literal_wang"]["nbc_bucket"]
        and cells[right_name]["nbc_bucket"]
        == REFERENCE_OBSERVATION["repository"]["nbc_bucket"]
    )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def _substantial(block):
    return block["difference_class"] == DIFFERENCE_SUBSTANTIAL


def build_primary_verdict(
    matrix, probe, scores_by_cell, polarities, controls=None, shape_scaling=None
):
    """Decide, with guards, whether extraction or forward explains Step 1.

    Guards (block any attribution): D00 must reconstruct Step 1 A1, D11 must
    reconstruct Step 1 A3, and every determinism control must be bit-identical.
    Without the endpoints the 2x2 does not span the observed divergence, so no
    comparison inside it can explain it.

    The strongest verdict, EXTRACTION_PATH_EXPLAINS, additionally requires the
    whole conjunction:

    * D01 reproduces A3 -- changing extraction alone reaches the repository
      endpoint;
    * D10 reproduces A1 -- changing the forward alone stays at the Wang
      endpoint;
    * D00 -> D01 reproduces the pair-169 bucket flip;
    * neither forward comparison has ANY NBC bucket disagreement.

    That last condition is not the same as "the forward raw delta is small". A
    forward perturbation below the negligible bound can still cross a bucket
    edge, and the BSE update consumes the bucket, so bucket disagreement is
    checked directly. A forward raw perturbation with zero bucket impact is
    permitted and reported, but the causal text then says so rather than
    claiming the forward path is bit-identical.
    """
    controls = controls or {}
    nondeterministic = [
        name for name, block in controls.items() if block["numerical_difference"] == YES
    ]

    reference = REFERENCE_OBSERVATION
    d00_check = score_matches_reference(
        probe["cells"]["D00"]["raw"], reference["literal_wang"]
    )
    d11_check = score_matches_reference(
        probe["cells"]["D11"]["raw"], reference["repository"]
    )
    # Direct endpoint checks for the two off-diagonal cells: the extraction-only
    # cell should land on the repository endpoint, and the forward-only cell
    # should stay on the Wang endpoint.
    d01_check = score_matches_reference(
        probe["cells"]["D01"]["raw"], reference["repository"]
    )
    d10_check = score_matches_reference(
        probe["cells"]["D10"]["raw"], reference["literal_wang"]
    )

    extraction_blocks = [matrix["D00_vs_D01"], matrix["D10_vs_D11"]]
    forward_blocks = [matrix["D00_vs_D10"], matrix["D01_vs_D11"]]

    extraction_raw_moves = any(_substantial(b) for b in extraction_blocks)
    forward_raw_moves = any(_substantial(b) for b in forward_blocks)
    extraction_decides = any(b["bse_decision_impact"] == YES for b in extraction_blocks)
    forward_decides = any(b["bse_decision_impact"] == YES for b in forward_blocks)
    forward_bucket_disagreements = sum(
        b["nbc_bucket_disagreements"]["count"] for b in forward_blocks
    )
    forward_bit_identical = all(
        b["numerical_difference"] == NO for b in forward_blocks
    )
    forward_max_delta = max(
        (b["deltas"]["max_absolute"] or 0.0) for b in forward_blocks
    )

    # An effect is a substantial raw move OR any bucket change. A sub-threshold
    # delta that crosses a bucket edge is an effect.
    extraction_effect = extraction_raw_moves or extraction_decides
    forward_effect = forward_raw_moves or forward_decides

    histogram_move = histogram_movement_matches_step1(
        matrix["D00_vs_D01"]["histograms"]["left"]["positive"],
        matrix["D00_vs_D01"]["histograms"]["right"]["positive"],
    )
    probe_flip = probe_bucket_flip_reproduced(probe, "D00", "D01")

    strong_extraction = (
        extraction_effect
        and d01_check["reproduces"]
        and d10_check["reproduces"]
        and probe_flip
        and forward_bucket_disagreements == 0
    )

    chain = [
        {
            "link": 1,
            "requirement": "D00 reproduces Step 1 A1 raw, rounded and bucket",
            "required": True,
            "strong_verdict": True,
            "passed": d00_check["reproduces"],
            "detail": d00_check,
        },
        {
            "link": 2,
            "requirement": "D11 reproduces Step 1 A3 raw, rounded and bucket",
            "required": True,
            "strong_verdict": True,
            "passed": d11_check["reproduces"],
            "detail": d11_check,
        },
        {
            "link": 3,
            "requirement": "every determinism control is bit-identical",
            "required": True,
            "strong_verdict": True,
            "passed": not nondeterministic,
            "detail": {"nondeterministic_controls": nondeterministic},
        },
        {
            "link": 4,
            "requirement": (
                "D01 reproduces Step 1 A3 -- extraction alone reaches the "
                "repository endpoint"
            ),
            "required": False,
            "strong_verdict": True,
            "passed": d01_check["reproduces"],
            "detail": d01_check,
        },
        {
            "link": 5,
            "requirement": (
                "D10 reproduces Step 1 A1 -- the forward alone stays at the Wang "
                "endpoint"
            ),
            "required": False,
            "strong_verdict": True,
            "passed": d10_check["reproduces"],
            "detail": d10_check,
        },
        {
            "link": 6,
            "requirement": "neither forward comparison changes any NBC bucket",
            "required": False,
            "strong_verdict": True,
            "passed": forward_bucket_disagreements == 0,
            "detail": {
                "forward_bucket_disagreements": forward_bucket_disagreements,
                "forward_bit_identical": forward_bit_identical,
                "forward_max_absolute_delta": forward_max_delta,
            },
        },
        {
            "link": 7,
            "requirement": "D00 -> D01 reproduces the recorded pair-169 bucket flip",
            "required": False,
            "strong_verdict": True,
            "passed": probe_flip,
            "detail": {"probe_bucket_flip_reproduced": probe_flip},
        },
        {
            "link": 8,
            "requirement": (
                "D00 -> D01 reproduces the recorded positive histogram movement "
                "(preferred, not required)"
            ),
            "required": False,
            "strong_verdict": False,
            "passed": histogram_move["movement_reproduced"],
            "detail": histogram_move,
        },
    ]

    if nondeterministic:
        guard = GUARD_NONDETERMINISTIC
        headline = HEADLINE_UNDETERMINED_NONDETERMINISTIC
        candidate = (
            "Causal attribution withheld: the determinism control(s) "
            + ", ".join(nondeterministic)
            + " did not reproduce themselves, so a difference between cells "
            "cannot be distinguished from run-to-run nondeterminism."
        )
        withheld = True
    elif not (d00_check["reproduces"] and d11_check["reproduces"]):
        guard = GUARD_ENDPOINTS_NOT_RECONSTRUCTED
        headline = HEADLINE_UNDETERMINED_ENDPOINTS
        missing = []
        if not d00_check["reproduces"]:
            missing.append("D00 does not reconstruct Step 1 A1")
        if not d11_check["reproduces"]:
            missing.append("D11 does not reconstruct Step 1 A3")
        candidate = (
            "Causal attribution withheld: "
            + "; ".join(missing)
            + ". The 2x2 does not span the observed divergence, so no comparison "
            "inside it can explain it. Check the model revision, dtype and "
            "device against the recorded Gate 1 run."
        )
        withheld = True
    else:
        guard = None
        withheld = False
        if strong_extraction:
            headline = HEADLINE_EXTRACTION_EXPLAINS
            if forward_bit_identical:
                forward_clause = (
                    "changing the forward pass under either extraction is "
                    "bit-identical"
                )
            else:
                forward_clause = (
                    "changing the forward pass under either extraction perturbs "
                    f"the raw score by at most {forward_max_delta:.3e} and "
                    "changes no NBC bucket"
                )
            candidate = (
                "The extraction/scaling path is sufficient to explain the Step 1 "
                "discrepancy. Changing extraction alone moves D00 to the "
                "recorded A3 endpoint and reproduces the pair-169 bucket flip, "
                "while " + forward_clause + ". No forward-path change is "
                "required to account for the Step 1 result."
            )
        elif forward_decides and extraction_decides:
            headline = HEADLINE_BOTH_CONTRIBUTE
            candidate = (
                "Both paths change NBC buckets. They are reported separately; "
                "neither alone accounts for the Step 1 divergence."
            )
        elif forward_effect and extraction_effect:
            headline = HEADLINE_BOTH_CONTRIBUTE
            candidate = (
                "Both the forward path and the extraction path have an effect "
                "(a substantial raw move, a bucket change, or both). They are "
                "reported separately and never conflated."
            )
        elif extraction_effect:
            headline = HEADLINE_EXTRACTION_IMPLICATED
            unmet = [
                link["requirement"]
                for link in chain
                if link["strong_verdict"] and link["passed"] is not True
            ]
            candidate = (
                "The extraction path has an effect and the forward path does "
                "not, but the strongest verdict is not established because: "
                + "; ".join(unmet)
                + "."
            )
        elif forward_effect:
            headline = HEADLINE_FORWARD_EXPLAINS
            candidate = (
                "The forward path has an effect and the extraction path does "
                "not, so the divergence is attributable to the forward call."
            )
        else:
            headline = HEADLINE_NEITHER
            candidate = (
                "Neither path has an effect, yet D00 and D11 reconstruct the two "
                "recorded endpoints. That combination is contradictory and "
                "indicates a problem with the run rather than a finding."
            )

    return {
        "numerical_difference": YES
        if any(b["numerical_difference"] == YES for b in matrix.values())
        else NO,
        "bse_decision_impact": YES
        if any(b["bse_decision_impact"] == YES for b in matrix.values())
        else NO,
        "causal_candidate": candidate,
        "headline": headline,
        "causal_attribution_withheld": withheld,
        "guard": guard,
        "causal_chain": chain,
        "required_links_passed": all(
            link["passed"] is True for link in chain if link["required"]
        ),
        "strong_verdict_links_passed": all(
            link["passed"] is True for link in chain if link["strong_verdict"]
        ),
        "endpoints": {
            "D00_vs_step1_a1": d00_check,
            "D11_vs_step1_a3": d11_check,
            "D01_vs_step1_a3": d01_check,
            "D10_vs_step1_a1": d10_check,
        },
        "extraction_path": {
            "moves_raw_score": YES if extraction_raw_moves else NO,
            "changes_nbc_bucket": YES if extraction_decides else NO,
            "has_effect": YES if extraction_effect else NO,
            "comparisons": ["D00_vs_D01", "D10_vs_D11"],
        },
        "forward_path": {
            "moves_raw_score": YES if forward_raw_moves else NO,
            "changes_nbc_bucket": YES if forward_decides else NO,
            "has_effect": YES if forward_effect else NO,
            "bit_identical": forward_bit_identical,
            "bucket_disagreements": forward_bucket_disagreements,
            "max_absolute_delta": forward_max_delta,
            "comparisons": ["D00_vs_D10", "D01_vs_D11"],
        },
        "probe_bucket_flip_reproduced": probe_flip,
        "histogram_movement": histogram_move,
        "shape_and_scaling": shape_scaling,
        "determinism_controls": {
            name: block["numerical_difference"] == NO for name, block in controls.items()
        },
        "reporting_note": (
            "numerical_difference and bse_decision_impact are reported "
            "separately. A raw floating-point score change alone is NOT evidence "
            "that BSE behaviour changed; only an NBC bucket change is. A raw "
            "change BELOW the negligible bound that still crosses a bucket edge "
            "IS an effect and is counted as one. The forward-path and "
            "extraction-path effects are reported separately and never "
            "conflated."
        ),
    }


UNDETERMINED_HEADLINES = (
    HEADLINE_UNDETERMINED_NONDETERMINISTIC,
    HEADLINE_UNDETERMINED_ENDPOINTS,
)


def secondary_expectation(primary_verdict):
    """What the forward-path factorial should be expected to show.

    The secondary C0-C4 factorial holds extraction fixed at Wang's form for
    every arm. If the primary shows that extraction explains the divergence,
    then no secondary arm can reconstruct Step 1 A3, and the secondary chain
    will correctly report that it did not. Saying so in advance stops that
    expected outcome from being misread as a failure of the secondary
    experiment.
    """
    if primary_verdict["headline"] == HEADLINE_EXTRACTION_EXPLAINS:
        return (
            "The primary decomposition attributes the divergence to the "
            "extraction path. Every secondary arm uses Wang extraction, so no "
            "secondary arm can reconstruct Step 1 A3 and the secondary causal "
            "chain is EXPECTED to report that it did not. That is consistent "
            "with the primary result, not a contradiction of it."
        )
    if primary_verdict["headline"] in UNDETERMINED_HEADLINES:
        return (
            "The primary decomposition withheld attribution, so the secondary "
            "factorial cannot be interpreted either."
        )
    return (
        "The primary decomposition did not attribute the divergence to the "
        "extraction path alone, so the secondary forward-path factorial carries "
        "information about which forward argument is responsible."
    )
