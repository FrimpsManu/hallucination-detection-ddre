"""Confirmatory statistical protocol: paired passage-level cluster bootstrap.

**This protocol is frozen prospectively, before any held-out DDRE result is
inspected.** The constants below are the pre-registration. Changing one after
seeing test results turns a confirmatory analysis into an exploratory one, so
the code treats any deviation as ``NOT_CONFIRMATORY`` rather than silently
adjusting the claim rule.

Why a *cluster* bootstrap over passages. The held-out split has 190 passages
and 1,525 sentences, and sentences from one passage share a topic, a source
document set and an annotator pass. They are not independent, so resampling
sentences would understate the variance and narrow every interval. Resampling
whole passages -- carrying each one's complete sentence block, duplicated if the
passage is drawn twice -- respects the nested passage -> sentence structure.

Why *paired*. DDRE and BSE are evaluated on the identical sentence set in the
identical order, so each replicate applies the **same** passage draw to both
methods and the difference is taken within the replicate. That removes the
between-passage variance the two methods share, which is exactly the variance
that is irrelevant to which method is better.

Sign conventions, stated once and asserted in the output:

    performance delta = DDRE - BSE      (positive favours DDRE)
    efficiency saving = BSE - DDRE      (positive favours DDRE)

Efficiency is a *difference*, not a fraction: a fractional reduction divides by
a bootstrap denominator that can be very small, which makes the estimand behave
badly at exactly the replicates that matter. Fractions remain useful as
descriptive point estimates and are reported as such elsewhere.

The claim rule is **conjunctive** (intersection-union): every primary gate must
pass. Because passing requires all of them simultaneously rather than any one of
them, there is no cherry-picking to correct for, and no ad-hoc multiplicity
adjustment is applied. No p-values are produced.

Standard library, numpy, and the repository's single Wang PR-AUC helper.
"""

import math

import numpy as np

# --------------------------------------------------------------------------
# FROZEN PROTOCOL. Pre-registered before held-out evaluation. Do not retune.
# --------------------------------------------------------------------------

# Absolute PR-AUC difference. "Non-inferior" means the lower confidence bound on
# (DDRE - BSE) sits at or above -0.005, i.e. any loss is smaller than half a
# PR-AUC point. Chosen to match the validation quality tolerance so that a
# configuration selected on validation is judged on test by the same yardstick.
CONFIRMATORY_PR_AUC_MARGIN = 0.005

CONFIRMATORY_BOOTSTRAP_RESAMPLES = 10_000
CONFIRMATORY_BOOTSTRAP_SEED = 42
CONFIRMATORY_CI_LEVEL = 0.95
CONFIRMATORY_BOOTSTRAP_UNIT = "passage"
CONFIRMATORY_CI_METHOD = "percentile"

CLAIM_SUPPORTED = "SUPPORTED"
CLAIM_NOT_SUPPORTED = "NOT_SUPPORTED"
CLAIM_NOT_CONFIRMATORY = "NOT_CONFIRMATORY"

PERFORMANCE_ENDPOINTS = (
    "nonfactual_auc_pr_delta",
    "factual_auc_pr_delta",
    "balanced_pr_auc_delta",
)
PRIMARY_EFFICIENCY_ENDPOINT = "retrieved_documents_per_sentence_savings"
SECONDARY_EFFICIENCY_ENDPOINT = "nli_span_calls_per_sentence_savings"
BOOTSTRAP_ENDPOINTS = (
    PERFORMANCE_ENDPOINTS + (PRIMARY_EFFICIENCY_ENDPOINT, SECONDARY_EFFICIENCY_ENDPOINT)
)


class BootstrapUnavailable(RuntimeError):
    """The confirmatory bootstrap could not be computed. Fail closed.

    Never caught and downgraded to a weaker analysis: a replicate that cannot
    support the estimand makes the confirmatory result unavailable, not
    negative.
    """


class PairedInputMismatch(ValueError):
    """DDRE and BSE were not evaluated on the same sentences in the same order."""


def _wang_pr_auc():
    """The repository's single Wang PR-AUC definition, imported lazily.

    Lazy so this module can be imported for its frozen constants without
    pulling in scikit-learn.
    """
    from src.evaluation import wang_pr_auc

    return wang_pr_auc


def validate_paired_inputs(records, ddre_results, baseline_results):
    """Confirm the two methods were scored on the identical sentence sequence.

    A paired analysis is only paired if position *i* means the same sentence in
    both result lists. Nothing downstream can detect a misalignment, so it is
    checked here.
    """
    if not records:
        raise PairedInputMismatch("no records to bootstrap over")
    if not (len(records) == len(ddre_results) == len(baseline_results)):
        raise PairedInputMismatch(
            "paired bootstrap requires one result per record for each method: "
            f"{len(records)} records, {len(ddre_results)} DDRE results, "
            f"{len(baseline_results)} baseline results"
        )
    labels = [int(record.label) for record in records]
    if len(set(labels)) < 2:
        raise PairedInputMismatch(
            "the evaluated sentences contain only one class, so PR-AUC is "
            "undefined for the observed sample"
        )
    return {
        "sentences": len(records),
        "passages": len({record.passage_index for record in records}),
        "factual_sentences": int(sum(labels)),
        "nonfactual_sentences": int(len(labels) - sum(labels)),
    }


def passage_blocks(records):
    """Sentence indices grouped by passage, in first-appearance order.

    The blocks are the resampling unit. Order is deterministic so a seed fully
    determines a replicate.
    """
    order = []
    blocks = {}
    for index, record in enumerate(records):
        key = record.passage_index
        if key not in blocks:
            blocks[key] = []
            order.append(key)
        blocks[key].append(index)
    return [(key, tuple(blocks[key])) for key in order]


def replicate_indices(blocks, rng):
    """One replicate: N passages drawn with replacement, blocks concatenated.

    A passage drawn twice contributes its whole sentence block twice. Collapsing
    the duplicate back into one passage would silently shrink the resample and
    destroy the variance the cluster bootstrap exists to capture.
    """
    draw = rng.integers(0, len(blocks), size=len(blocks))
    indices = []
    for position in draw:
        indices.extend(blocks[int(position)][1])
    return np.asarray(indices, dtype=int), draw


def _method_arrays(records, results):
    return {
        "label": np.asarray([int(r.label) for r in records], dtype=int),
        "p_factual": np.asarray([float(r.p_factual) for r in results], dtype=float),
        "documents": np.asarray(
            [float(r.documents_used) for r in results], dtype=float
        ),
        "nli_calls": np.asarray([float(r.nli_calls) for r in results], dtype=float),
    }


def _metrics_on(arrays, indices, pr_auc):
    """Wang PR-AUC and the compute counts on one resampled index set."""
    labels = arrays["label"][indices]
    p_factual = arrays["p_factual"][indices]
    if len(np.unique(labels)) < 2:
        raise BootstrapUnavailable(
            "a bootstrap replicate contains only one class, so Wang PR-AUC is "
            "mathematically undefined for it. Failing closed: the replicate is "
            "NOT dropped, NOT redrawn, and NO substitute metric is used, because "
            "each of those silently changes the estimand. The confirmatory "
            "analysis is unavailable for this sample."
        )
    return {
        "nonfactual_auc_pr": pr_auc(1 - labels, 1.0 - p_factual),
        "factual_auc_pr": pr_auc(labels, p_factual),
        "documents_per_sentence": float(np.mean(arrays["documents"][indices])),
        "nli_calls_per_sentence": float(np.mean(arrays["nli_calls"][indices])),
    }


def _paired_deltas(ddre, baseline):
    """Performance = DDRE - BSE. Efficiency = BSE - DDRE. Both positive-favours-DDRE."""
    ddre_balanced = 0.5 * (ddre["nonfactual_auc_pr"] + ddre["factual_auc_pr"])
    baseline_balanced = 0.5 * (
        baseline["nonfactual_auc_pr"] + baseline["factual_auc_pr"]
    )
    return {
        "nonfactual_auc_pr_delta": ddre["nonfactual_auc_pr"]
        - baseline["nonfactual_auc_pr"],
        "factual_auc_pr_delta": ddre["factual_auc_pr"] - baseline["factual_auc_pr"],
        "balanced_pr_auc_delta": ddre_balanced - baseline_balanced,
        PRIMARY_EFFICIENCY_ENDPOINT: baseline["documents_per_sentence"]
        - ddre["documents_per_sentence"],
        SECONDARY_EFFICIENCY_ENDPOINT: baseline["nli_calls_per_sentence"]
        - ddre["nli_calls_per_sentence"],
    }


def percentile_interval(values, ci_level=CONFIRMATORY_CI_LEVEL):
    """Fixed percentile interval. At 95%: exactly the 2.5th and 97.5th."""
    # Rounded so the recorded percentiles read exactly 2.5 / 97.5 at the frozen
    # 95% level rather than 2.500000000000002; the provenance is the point.
    tail = (1.0 - ci_level) / 2.0
    lower_pct = round(100.0 * tail, 10)
    upper_pct = round(100.0 * (1.0 - tail), 10)
    array = np.asarray(values, dtype=float)
    return (
        float(np.percentile(array, lower_pct)),
        float(np.percentile(array, upper_pct)),
        lower_pct,
        upper_pct,
    )


def paired_passage_bootstrap(
    records,
    ddre_results,
    baseline_results,
    *,
    n_resamples=CONFIRMATORY_BOOTSTRAP_RESAMPLES,
    seed=CONFIRMATORY_BOOTSTRAP_SEED,
    ci_level=CONFIRMATORY_CI_LEVEL,
    pr_auc=None,
):
    """Paired cluster bootstrap over passages. Returns observed values and CIs."""
    shape = validate_paired_inputs(records, ddre_results, baseline_results)
    pr_auc = _wang_pr_auc() if pr_auc is None else pr_auc

    ddre_arrays = _method_arrays(records, ddre_results)
    baseline_arrays = _method_arrays(records, baseline_results)
    blocks = passage_blocks(records)

    observed_indices = np.arange(len(records), dtype=int)
    observed = _paired_deltas(
        _metrics_on(ddre_arrays, observed_indices, pr_auc),
        _metrics_on(baseline_arrays, observed_indices, pr_auc),
    )

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_resamples, dtype=float) for name in BOOTSTRAP_ENDPOINTS}
    for replicate in range(n_resamples):
        indices, _ = replicate_indices(blocks, rng)
        # The SAME resampled indices for both methods: that is what makes it
        # paired, and it is why the shared between-passage variance cancels.
        deltas = _paired_deltas(
            _metrics_on(ddre_arrays, indices, pr_auc),
            _metrics_on(baseline_arrays, indices, pr_auc),
        )
        for name in BOOTSTRAP_ENDPOINTS:
            draws[name][replicate] = deltas[name]

    endpoints = {}
    for name in BOOTSTRAP_ENDPOINTS:
        lower, upper, lower_pct, upper_pct = percentile_interval(draws[name], ci_level)
        endpoints[name] = {
            "observed": float(observed[name]),
            "bootstrap_mean": float(np.mean(draws[name])),
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_level": float(ci_level),
            "ci_method": CONFIRMATORY_CI_METHOD,
            "ci_lower_percentile": lower_pct,
            "ci_upper_percentile": upper_pct,
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "bootstrap_unit": CONFIRMATORY_BOOTSTRAP_UNIT,
            "sign_convention": (
                "DDRE - BSE; positive favours DDRE"
                if name in PERFORMANCE_ENDPOINTS
                else "BSE - DDRE; positive favours DDRE"
            ),
        }

    return {
        "bootstrap_unit": CONFIRMATORY_BOOTSTRAP_UNIT,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "ci_method": CONFIRMATORY_CI_METHOD,
        "unique_passages": shape["passages"],
        "sentences": shape["sentences"],
        "factual_sentences": shape["factual_sentences"],
        "nonfactual_sentences": shape["nonfactual_sentences"],
        "pr_auc_definition": (
            "src.evaluation.wang_pr_auc -- precision_recall_curve followed by "
            "auc(recall, precision), identical to the normal evaluation path"
        ),
        "endpoints": endpoints,
    }


def confirmatory_protocol():
    """The frozen protocol, for the experiment summary's provenance section."""
    return {
        "frozen_before_held_out_evaluation": True,
        "primary_comparator": "bse_official (CM=28, CFA=96, c_retrieve=1)",
        "bootstrap_unit": CONFIRMATORY_BOOTSTRAP_UNIT,
        "n_resamples": CONFIRMATORY_BOOTSTRAP_RESAMPLES,
        "seed": CONFIRMATORY_BOOTSTRAP_SEED,
        "ci_method": CONFIRMATORY_CI_METHOD,
        "ci_level": CONFIRMATORY_CI_LEVEL,
        "ci_percentiles": [2.5, 97.5],
        "pr_auc_noninferiority_margin": CONFIRMATORY_PR_AUC_MARGIN,
        "pr_auc_definition": "src.evaluation.wang_pr_auc (Wang-compatible)",
        "sign_conventions": {
            "performance": "DDRE - BSE; positive favours DDRE",
            "efficiency": "BSE - DDRE; positive favours DDRE",
        },
        "primary_performance_endpoints": list(PERFORMANCE_ENDPOINTS),
        "primary_efficiency_endpoint": PRIMARY_EFFICIENCY_ENDPOINT,
        "secondary_efficiency_endpoint": SECONDARY_EFFICIENCY_ENDPOINT,
        "primary_claim_rule": (
            "CONJUNCTIVE (intersection-union). The primary claim is supported "
            "only if the validation threshold selection was confirmatory, the "
            "validation quality tolerance equals the frozen "
            f"{CONFIRMATORY_PR_AUC_MARGIN} margin, the lower 95% bound of each "
            "of the three PR-AUC deltas is at or above "
            f"-{CONFIRMATORY_PR_AUC_MARGIN}, AND the lower 95% bound of the "
            "retrieved-documents-per-sentence saving is strictly above 0."
        ),
        "multiplicity": (
            "No correction is applied. The rule requires EVERY primary gate to "
            "pass simultaneously, so there is no selection among endpoints to "
            "correct for. Fixed 95% intervals are used throughout."
        ),
        "p_values": "none produced",
        "invalid_replicate_policy": (
            "Fail closed. A replicate with only one class raises "
            "BootstrapUnavailable; replicates are never dropped, redrawn, or "
            "replaced by a substitute metric."
        ),
        "validation_fallback_policy": (
            "A configuration selected by the validation fallback can never "
            "support the confirmatory claim, however large the held-out effect."
        ),
    }


def assess_claim(
    bootstrap,
    *,
    validation_selection_confirmatory,
    quality_tolerance,
    smoke_test=False,
    bootstrap_error=None,
):
    """Structured claim assessment. Every gate is recorded separately.

    ``NOT_CONFIRMATORY`` and ``NOT_SUPPORTED`` are different scientific
    statements -- "this run cannot address the claim" versus "this run addressed
    it and the evidence did not support it" -- and are never collapsed.
    """
    margin = CONFIRMATORY_PR_AUC_MARGIN
    tolerance_matches = quality_tolerance is not None and math.isclose(
        float(quality_tolerance), margin, rel_tol=0.0, abs_tol=1e-12
    )
    bootstrap_available = bootstrap is not None and bootstrap_error is None

    disqualifiers = []
    if smoke_test:
        disqualifiers.append("run_mode is a smoke/debug run")
    if not validation_selection_confirmatory:
        disqualifiers.append(
            "validation threshold selection used the fallback objective, so no "
            "configuration preserved BSE-official nonfactual, factual and "
            "balanced PR-AUC within tolerance"
        )
    if not tolerance_matches:
        disqualifiers.append(
            f"validation quality tolerance {quality_tolerance!r} differs from the "
            f"frozen non-inferiority margin {margin}"
        )
    if not bootstrap_available:
        disqualifiers.append(
            "the confirmatory bootstrap is unavailable"
            + (f": {bootstrap_error}" if bootstrap_error else "")
        )
    confirmatory_eligible = not disqualifiers

    def lower_bound(name):
        return bootstrap["endpoints"][name]["ci_lower"] if bootstrap_available else None

    performance = {}
    for name, key in (
        ("nonfactual_pass", "nonfactual_auc_pr_delta"),
        ("factual_pass", "factual_auc_pr_delta"),
        ("balanced_pass", "balanced_pr_auc_delta"),
    ):
        bound = lower_bound(key)
        performance[name] = bound is not None and bound >= -margin
        performance[f"{name}_ci_lower"] = bound
    performance["all_pass"] = bool(
        performance["nonfactual_pass"]
        and performance["factual_pass"]
        and performance["balanced_pass"]
    )
    performance["margin"] = margin
    performance["rule"] = f"lower 95% CI of (DDRE - BSE) >= -{margin}"

    retrieval_bound = lower_bound(PRIMARY_EFFICIENCY_ENDPOINT)
    retrieval_pass = retrieval_bound is not None and retrieval_bound > 0.0
    nli_bound = lower_bound(SECONDARY_EFFICIENCY_ENDPOINT)
    nli_pass = nli_bound is not None and nli_bound > 0.0

    primary_claim_supported = bool(
        confirmatory_eligible and performance["all_pass"] and retrieval_pass
    )
    if not confirmatory_eligible:
        claim_status = CLAIM_NOT_CONFIRMATORY
    elif primary_claim_supported:
        claim_status = CLAIM_SUPPORTED
    else:
        claim_status = CLAIM_NOT_SUPPORTED

    if claim_status == CLAIM_SUPPORTED and nli_pass:
        interpretation = (
            "The evidence supports lower retrieval cost AND lower NLI-evaluation "
            "cost with preserved detection performance."
        )
    elif claim_status == CLAIM_SUPPORTED:
        interpretation = (
            "The evidence supports lower retrieval cost with preserved detection "
            "performance, but NOT a statistically supported reduction in NLI "
            "span calls."
        )
    elif claim_status == CLAIM_NOT_SUPPORTED:
        interpretation = (
            "This run addressed the confirmatory claim and the evidence did not "
            "support it. This is a negative result, not an inconclusive one."
        )
    else:
        interpretation = (
            "This run cannot address the confirmatory claim: "
            + "; ".join(disqualifiers)
            + ". This is NOT a negative result, and must never be reported as one."
        )

    return {
        "confirmatory_eligible": confirmatory_eligible,
        "confirmatory_disqualifiers": disqualifiers,
        "validation_selection_confirmatory": bool(validation_selection_confirmatory),
        "validation_quality_tolerance": (
            None if quality_tolerance is None else float(quality_tolerance)
        ),
        "validation_tolerance_matches_frozen_margin": tolerance_matches,
        "bootstrap_available": bootstrap_available,
        "performance_noninferiority": performance,
        "retrieval_efficiency_superiority_pass": retrieval_pass,
        "retrieval_efficiency_ci_lower": retrieval_bound,
        "nli_efficiency_superiority_pass": nli_pass,
        "nli_efficiency_ci_lower": nli_bound,
        "primary_claim_supported": primary_claim_supported,
        "claim_status": claim_status,
        "interpretation": interpretation,
        "claim_rule": confirmatory_protocol()["primary_claim_rule"],
    }
