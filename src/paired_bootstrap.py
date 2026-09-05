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

import hashlib
import json
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

# The estimand itself. A bootstrap can carry every other frozen setting --
# 10,000 resamples, seed 42, a 95% percentile interval over passages -- and
# still be measuring something else entirely, because the metric is injectable.
# So the implementation is pinned by a single canonical identifier that only the
# default production path may record.
CONFIRMATORY_PR_AUC_DEFINITION = (
    "src.evaluation.wang_pr_auc:precision_recall_curve+auc(recall,precision)"
)

# The frozen primary comparator and split. The CLI can still change any of
# these -- alternative configurations remain valid for exploratory work -- but a
# run that differs from these values is NOT the run this protocol pre-registered
# and cannot produce a confirmatory claim, however good its numbers look.
CONFIRMATORY_C_MISS = 28.0
CONFIRMATORY_C_FALSE_ALARM = 96.0
CONFIRMATORY_C_RETRIEVE = 1.0
CONFIRMATORY_P0 = 0.5
CONFIRMATORY_MAX_DOCS = 10
CONFIRMATORY_VALIDATION_FRACTION = 0.20
CONFIRMATORY_SPLIT_SEED = 42

FROZEN_RUN_CONFIGURATION = {
    "c_miss": CONFIRMATORY_C_MISS,
    "c_false_alarm": CONFIRMATORY_C_FALSE_ALARM,
    "c_retrieve": CONFIRMATORY_C_RETRIEVE,
    "p0": CONFIRMATORY_P0,
    "max_docs": CONFIRMATORY_MAX_DOCS,
    "validation_fraction": CONFIRMATORY_VALIDATION_FRACTION,
    "split_seed": CONFIRMATORY_SPLIT_SEED,
}

PERFORMANCE_SIGN_CONVENTION = "DDRE - BSE; positive favours DDRE"
EFFICIENCY_SIGN_CONVENTION = "BSE - DDRE; positive favours DDRE"

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


def _pr_auc_identity(injected):
    """What metric was ACTUALLY used, recorded honestly.

    Only ``pr_auc=None`` -- the default production path, which resolves to
    ``src.evaluation.wang_pr_auc`` -- may record the canonical identifier.
    Anything injected is labelled ``custom:<module>.<qualname>`` and can never
    be confirmatory, whatever it computes.

    No attempt is made to decide whether an injected function is *equivalent*
    to Wang PR-AUC. Equivalence cannot be read off a name, and a wrong guess
    here would let a different estimand inherit the frozen protocol's
    authority. A custom metric makes the run exploratory, which is not a
    negative result.
    """
    if injected is None:
        return CONFIRMATORY_PR_AUC_DEFINITION
    module = getattr(injected, "__module__", None) or "<unknown>"
    qualname = (
        getattr(injected, "__qualname__", None)
        or getattr(injected, "__name__", None)
        or repr(injected)
    )
    return f"custom:{module}.{qualname}"


def validate_paired_inputs(ddre_observations, baseline_observations):
    """Confirm the two methods were scored on the identical sentence sequence.

    This compares **identities**, not just lengths. Two result lists of equal
    length look paired even when one is permuted, and nothing downstream can
    detect that -- the analysis would silently difference method A on sentence
    *i* against method B on some other sentence. So every position must agree on
    ``(passage_index, sentence_index, gold_label)``, and the first disagreement
    raises with the position and both identities.

    The identities come from ``EvaluatedSentence`` observations built inside the
    evaluation loop, so they record what was actually scored rather than what a
    later caller assumed.
    """
    if not ddre_observations:
        raise PairedInputMismatch("no observations to bootstrap over")
    if len(ddre_observations) != len(baseline_observations):
        raise PairedInputMismatch(
            "paired bootstrap requires one observation per sentence for each "
            f"method: {len(ddre_observations)} DDRE, "
            f"{len(baseline_observations)} baseline"
        )

    seen = {}
    for position, (ddre, baseline) in enumerate(
        zip(ddre_observations, baseline_observations)
    ):
        if ddre.identity != baseline.identity:
            raise PairedInputMismatch(
                "DDRE and BSE were not evaluated on the same sentence at "
                f"position {position}: DDRE identity "
                f"(passage={ddre.passage_index}, sentence={ddre.sentence_index}, "
                f"label={ddre.gold_label}) vs baseline identity "
                f"(passage={baseline.passage_index}, "
                f"sentence={baseline.sentence_index}, "
                f"label={baseline.gold_label}). The analysis is not paired."
            )
        key = (ddre.passage_index, ddre.sentence_index)
        if key in seen:
            raise PairedInputMismatch(
                f"sentence identity {key} appears at positions {seen[key]} and "
                f"{position}. The released Wang data has one row per "
                "(passage, sentence), so a duplicate means the evaluated set was "
                "built wrongly; resampling it would double-count that sentence."
            )
        seen[key] = position

    # Implied by the positional equality above, but asserted explicitly because
    # the passage grouping is what the cluster bootstrap resamples.
    ddre_passages = {}
    baseline_passages = {}
    for ddre, baseline in zip(ddre_observations, baseline_observations):
        ddre_passages.setdefault(ddre.passage_index, []).append(ddre.sentence_index)
        baseline_passages.setdefault(baseline.passage_index, []).append(
            baseline.sentence_index
        )
    if ddre_passages != baseline_passages:
        raise PairedInputMismatch(
            "the passage membership implied by the two methods' identities "
            "differs, so the cluster bootstrap would resample different blocks "
            "for each method"
        )

    labels = [int(o.gold_label) for o in ddre_observations]
    if len(set(labels)) < 2:
        raise PairedInputMismatch(
            "the evaluated sentences contain only one class, so PR-AUC is "
            "undefined for the observed sample"
        )
    return {
        "sentences": len(ddre_observations),
        "passages": len(ddre_passages),
        "factual_sentences": int(sum(labels)),
        "nonfactual_sentences": int(len(labels) - sum(labels)),
    }


def passage_blocks(observations):
    """Sentence positions grouped by passage, in first-appearance order.

    The blocks are the resampling unit. Order is deterministic so a seed fully
    determines a replicate.
    """
    order = []
    blocks = {}
    for index, observation in enumerate(observations):
        key = observation.passage_index
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


def _method_arrays(observations):
    return {
        "label": np.asarray([int(o.gold_label) for o in observations], dtype=int),
        "p_factual": np.asarray(
            [float(o.result.p_factual) for o in observations], dtype=float
        ),
        "documents": np.asarray(
            [float(o.result.documents_used) for o in observations], dtype=float
        ),
        "nli_calls": np.asarray(
            [float(o.result.nli_calls) for o in observations], dtype=float
        ),
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
    ddre_observations,
    baseline_observations,
    *,
    n_resamples=CONFIRMATORY_BOOTSTRAP_RESAMPLES,
    seed=CONFIRMATORY_BOOTSTRAP_SEED,
    ci_level=CONFIRMATORY_CI_LEVEL,
    pr_auc=None,
):
    """Paired cluster bootstrap over passages. Returns observed values and CIs."""
    shape = validate_paired_inputs(ddre_observations, baseline_observations)
    # Resolved before substitution, so the record describes the function the
    # replicates were actually computed with.
    pr_auc_definition = _pr_auc_identity(pr_auc)
    pr_auc = _wang_pr_auc() if pr_auc is None else pr_auc

    ddre_arrays = _method_arrays(ddre_observations)
    baseline_arrays = _method_arrays(baseline_observations)
    blocks = passage_blocks(ddre_observations)

    observed_indices = np.arange(len(ddre_observations), dtype=int)
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
                PERFORMANCE_SIGN_CONVENTION
                if name in PERFORMANCE_ENDPOINTS
                else EFFICIENCY_SIGN_CONVENTION
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
        "pr_auc_definition": pr_auc_definition,
        "endpoints": endpoints,
    }


def validate_bootstrap_provenance(bootstrap):
    """Every way the recorded bootstrap can fail to be the frozen one.

    Without this the pre-registration is only a comment: a 200-resample,
    seed-7, 80%-interval bootstrap could be handed to ``assess_claim`` and
    receive SUPPORTED if its bounds happened to pass. Exploratory bootstraps
    remain perfectly legal -- they simply cannot produce a confirmatory claim.

    Nothing is repaired. Every mismatch is listed, so a reader sees exactly
    which part of the protocol the run departed from.
    """
    if not bootstrap:
        return ["no bootstrap record was produced"]

    mismatches = []
    for field, expected in (
        ("bootstrap_unit", CONFIRMATORY_BOOTSTRAP_UNIT),
        ("n_resamples", CONFIRMATORY_BOOTSTRAP_RESAMPLES),
        ("seed", CONFIRMATORY_BOOTSTRAP_SEED),
        ("ci_level", CONFIRMATORY_CI_LEVEL),
        ("ci_method", CONFIRMATORY_CI_METHOD),
        # The estimand. Every other setting can be right while the metric is
        # something else, and then the frozen protocol is measuring the wrong
        # thing at high precision.
        ("pr_auc_definition", CONFIRMATORY_PR_AUC_DEFINITION),
    ):
        actual = bootstrap.get(field)
        if actual != expected:
            mismatches.append(
                f"bootstrap {field} is {actual!r}, frozen protocol requires "
                f"{expected!r}"
            )

    endpoints = bootstrap.get("endpoints") or {}
    for name in BOOTSTRAP_ENDPOINTS:
        if name not in endpoints:
            mismatches.append(f"bootstrap is missing required endpoint {name!r}")
            continue
        endpoint = endpoints[name]
        # An endpoint that disagrees with its own parent record describes a
        # different analysis from the one the header claims.
        for field in ("n_resamples", "seed", "ci_level", "ci_method", "bootstrap_unit"):
            if endpoint.get(field) != bootstrap.get(field):
                mismatches.append(
                    f"endpoint {name!r} records {field}="
                    f"{endpoint.get(field)!r} but the bootstrap header records "
                    f"{bootstrap.get(field)!r}"
                )
        expected_sign = (
            PERFORMANCE_SIGN_CONVENTION
            if name in PERFORMANCE_ENDPOINTS
            else EFFICIENCY_SIGN_CONVENTION
        )
        if endpoint.get("sign_convention") != expected_sign:
            mismatches.append(
                f"endpoint {name!r} records sign convention "
                f"{endpoint.get('sign_convention')!r}, frozen protocol requires "
                f"{expected_sign!r}"
            )
        mismatches.extend(_interval_mismatches(name, endpoint))
    return mismatches


def _interval_mismatches(name, endpoint):
    """A confidence interval must be present, numeric, finite and ordered.

    NaN matters more than it looks. ``float("nan") >= -0.005`` is False, so a
    corrupted endpoint would quietly fail its non-inferiority gate and be
    reported as ``NOT_SUPPORTED`` -- a negative scientific result manufactured
    out of a broken computation. An interval that cannot be trusted makes the
    analysis unavailable, not negative.

    Nothing is repaired: a reversed interval is reported, never reordered, and
    a non-finite bound is reported, never clipped.
    """
    mismatches = []
    bounds = {}
    for field in ("ci_lower", "ci_upper"):
        value = endpoint.get(field)
        if value is None:
            mismatches.append(f"endpoint {name!r} has no {field}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating,
                                                            np.integer)):
            mismatches.append(
                f"endpoint {name!r} has a non-numeric {field} ({value!r})"
            )
            continue
        if not math.isfinite(float(value)):
            mismatches.append(
                f"endpoint {name!r} has a non-finite {field} ({value!r}); the "
                "analysis is unavailable, not negative"
            )
            continue
        bounds[field] = float(value)
    if len(bounds) == 2 and bounds["ci_lower"] > bounds["ci_upper"]:
        mismatches.append(
            f"endpoint {name!r} has ci_lower {bounds['ci_lower']!r} above "
            f"ci_upper {bounds['ci_upper']!r}; the bounds are not reordered, "
            "the analysis is unavailable"
        )
    return mismatches


def validate_run_configuration(run_configuration):
    """The run must BE the pre-registered run, not merely resemble it.

    ``confirmatory_protocol()`` states the comparator is BSE official at
    C_M=28, C_FA=96, c_retrieve=1. The CLI can change every one of those, so a
    ``--c-miss 14`` run could otherwise report SUPPORTED against a protocol
    document describing a different comparator.
    """
    if not run_configuration:
        return ["the run configuration was not recorded"]
    mismatches = []
    for field, expected in FROZEN_RUN_CONFIGURATION.items():
        if field not in run_configuration:
            mismatches.append(f"run configuration does not record {field!r}")
            continue
        actual = run_configuration[field]
        if actual is None or float(actual) != float(expected):
            mismatches.append(
                f"run {field} is {actual!r}, frozen confirmatory configuration "
                f"requires {expected!r}"
            )
    return mismatches


def split_identity(validation_passage_ids, test_passage_ids):
    """A stable fingerprint of a split, for recording and comparison."""
    payload = json.dumps(
        {
            "validation": sorted(int(x) for x in validation_passage_ids),
            "test": sorted(int(x) for x in test_passage_ids),
        },
        sort_keys=True,
    ).encode("utf-8")
    return {
        "validation_passages": len(set(int(x) for x in validation_passage_ids)),
        "test_passages": len(set(int(x) for x in test_passage_ids)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_split_identity(split_metadata, expected_validation_ids, expected_test_ids):
    """The held-out set must be the canonical split, by IDENTITY not by size.

    "190 passages" is not the frozen test set -- a *different* 190 passages is a
    different test set, and would silently make the held-out evaluation a
    different experiment. So the actual passage IDs are compared against the
    split re-derived from the released records with the frozen fraction and
    seed, and no result is inspected to do it.
    """
    if not split_metadata:
        return ["the split metadata was not recorded"], None

    mismatches = []
    for field, expected in (
        ("validation_fraction", CONFIRMATORY_VALIDATION_FRACTION),
        ("random_state", CONFIRMATORY_SPLIT_SEED),
    ):
        actual = split_metadata.get(field)
        if actual is None or float(actual) != float(expected):
            mismatches.append(
                f"split {field} is {actual!r}, frozen confirmatory split "
                f"requires {expected!r}"
            )

    actual_validation = [int(x) for x in split_metadata.get("validation_passage_ids", [])]
    actual_test = [int(x) for x in split_metadata.get("test_passage_ids", [])]
    expected_validation = sorted(int(x) for x in expected_validation_ids)
    expected_test = sorted(int(x) for x in expected_test_ids)

    identity = split_identity(actual_validation, actual_test)
    expected_identity = split_identity(expected_validation, expected_test)
    identity["expected_sha256"] = expected_identity["sha256"]
    identity["matches_frozen_split"] = identity["sha256"] == expected_identity["sha256"]

    if sorted(actual_validation) != expected_validation:
        mismatches.append(
            "the validation passage IDs are not the frozen split "
            f"({len(actual_validation)} recorded vs "
            f"{len(expected_validation)} expected; "
            f"{len(set(actual_validation) ^ set(expected_validation))} differ). "
            "A different set of passages is a different experiment, even at the "
            "same size."
        )
    if sorted(actual_test) != expected_test:
        mismatches.append(
            "the held-out passage IDs are not the frozen split "
            f"({len(actual_test)} recorded vs {len(expected_test)} expected; "
            f"{len(set(actual_test) ^ set(expected_test))} differ). A different "
            "set of passages is a different held-out set, even at the same size."
        )
    return mismatches, identity


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
        "pr_auc_definition": CONFIRMATORY_PR_AUC_DEFINITION,
        "pr_auc_definition_requirement": (
            "A confirmatory run must use the canonical Wang PR-AUC "
            "implementation. The bootstrap accepts an injected metric for unit "
            "tests and exploratory analyses; such a run records "
            "'custom:<module>.<qualname>' and is NOT_CONFIRMATORY. No attempt "
            "is made to judge an injected function equivalent from its name."
        ),
        "interval_validity_requirement": (
            "Every required endpoint must carry a present, numeric, finite and "
            "correctly ordered interval. A NaN, infinite or reversed bound "
            "makes the analysis unavailable (NOT_CONFIRMATORY), never a "
            "negative result, and bounds are never repaired or reordered."
        ),
        "sign_conventions": {
            "performance": "DDRE - BSE; positive favours DDRE",
            "efficiency": "BSE - DDRE; positive favours DDRE",
        },
        "primary_performance_endpoints": list(PERFORMANCE_ENDPOINTS),
        "primary_efficiency_endpoint": PRIMARY_EFFICIENCY_ENDPOINT,
        "secondary_efficiency_endpoint": SECONDARY_EFFICIENCY_ENDPOINT,
        "frozen_run_configuration": dict(FROZEN_RUN_CONFIGURATION),
        "frozen_split_requirement": (
            "The held-out set must be the canonical split: validation_fraction "
            f"{CONFIRMATORY_VALIDATION_FRACTION}, random_state "
            f"{CONFIRMATORY_SPLIT_SEED}, and the ACTUAL validation/test passage "
            "IDs must equal the split re-derived from the released records. A "
            "different set of passages of the same size is a different held-out "
            "set."
        ),
        "bootstrap_provenance_requirement": (
            "The bootstrap record must carry the frozen bootstrap_unit, "
            "n_resamples, seed, ci_level, ci_method and pr_auc_definition, and "
            "every endpoint must carry the same provenance, the correct sign "
            "convention and a present, numeric, finite, correctly ordered "
            "interval. Exploratory bootstraps with other settings -- including "
            "an injected metric -- are allowed but can never produce a "
            "confirmatory claim."
        ),
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
    run_configuration=None,
    split_metadata=None,
    expected_validation_passage_ids=None,
    expected_test_passage_ids=None,
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

    # Every way this run can fail to BE the pre-registered run. Each is a
    # disqualifier, never a negative result.
    provenance_mismatches = (
        validate_bootstrap_provenance(bootstrap) if bootstrap_available else []
    )
    configuration_mismatches = validate_run_configuration(run_configuration)
    split_mismatches, split_identity_record = ([], None)
    if expected_validation_passage_ids is not None:
        split_mismatches, split_identity_record = validate_split_identity(
            split_metadata,
            expected_validation_passage_ids,
            expected_test_passage_ids or [],
        )
    else:
        split_mismatches = [
            "the frozen split was not verified: no expected passage IDs were "
            "supplied for comparison"
        ]

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
    disqualifiers.extend(provenance_mismatches)
    disqualifiers.extend(configuration_mismatches)
    disqualifiers.extend(split_mismatches)
    confirmatory_eligible = not disqualifiers

    def lower_bound(name):
        """The recorded lower bound, whatever it is. Reported verbatim."""
        if not bootstrap_available:
            return None
        endpoint = (bootstrap.get("endpoints") or {}).get(name)
        return None if endpoint is None else endpoint.get("ci_lower")

    def usable_bound(name):
        """The bound as a number, or None if it cannot be compared.

        A missing, non-numeric or non-finite bound fails its gate rather than
        raising or comparing. It is already a provenance disqualifier, so the
        run is NOT_CONFIRMATORY; this only stops a corrupted value from being
        silently treated as a passing or failing measurement.
        """
        value = lower_bound(name)
        if value is None or isinstance(value, bool):
            return None
        if not isinstance(value, (int, float, np.floating, np.integer)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    performance = {}
    for name, key in (
        ("nonfactual_pass", "nonfactual_auc_pr_delta"),
        ("factual_pass", "factual_auc_pr_delta"),
        ("balanced_pass", "balanced_pr_auc_delta"),
    ):
        bound = usable_bound(key)
        performance[name] = bound is not None and bound >= -margin
        performance[f"{name}_ci_lower"] = lower_bound(key)
    performance["all_pass"] = bool(
        performance["nonfactual_pass"]
        and performance["factual_pass"]
        and performance["balanced_pass"]
    )
    performance["margin"] = margin
    performance["rule"] = f"lower 95% CI of (DDRE - BSE) >= -{margin}"

    retrieval_bound = usable_bound(PRIMARY_EFFICIENCY_ENDPOINT)
    retrieval_pass = retrieval_bound is not None and retrieval_bound > 0.0
    nli_bound = usable_bound(SECONDARY_EFFICIENCY_ENDPOINT)
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
        "bootstrap_provenance_matches_frozen_protocol": not provenance_mismatches,
        "bootstrap_provenance_mismatches": provenance_mismatches,
        "run_configuration": run_configuration,
        "run_configuration_matches_frozen": not configuration_mismatches,
        "run_configuration_mismatches": configuration_mismatches,
        "frozen_run_configuration": dict(FROZEN_RUN_CONFIGURATION),
        "split_identity": split_identity_record,
        "split_matches_frozen": not split_mismatches,
        "split_mismatches": split_mismatches,
        "performance_noninferiority": performance,
        "retrieval_efficiency_superiority_pass": retrieval_pass,
        "retrieval_efficiency_ci_lower": lower_bound(PRIMARY_EFFICIENCY_ENDPOINT),
        "nli_efficiency_superiority_pass": nli_pass,
        "nli_efficiency_ci_lower": lower_bound(SECONDARY_EFFICIENCY_ENDPOINT),
        "primary_claim_supported": primary_claim_supported,
        "claim_status": claim_status,
        "interpretation": interpretation,
        "claim_rule": confirmatory_protocol()["primary_claim_rule"],
    }
