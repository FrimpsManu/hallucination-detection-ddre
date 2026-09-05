"""Validation threshold selection for DDRE, with explicit quality safeguards.

Kept out of ``main.py`` for the same reason as the confirmatory bootstrap: a
scientific decision rule that decides what the paper may claim should be
readable, unit-testable and reviewable on its own, not buried in an experiment
driver that needs a GPU and a model to import.

The rule this module encodes (audit finding D-07): a threshold configuration is
**confirmatorily feasible** only if it preserves BSE-official **nonfactual AND
factual AND balanced** PR-AUC, each within the validation tolerance.

Nonfactual PR-AUC is the one that must not be omitted. It is the
hallucination-detection metric and Wang's headline, and because balanced PR-AUC
is the *mean* of the two class metrics, a large factual gain can offset a
nonfactual loss and clear a balanced-only floor while the method has become
worse at the task it exists for. All three safeguards are recorded separately so
that trade-off is visible in the saved validation table rather than implied.

One honest note about the balanced clause. Given the same tolerance ``t``, it is
**mathematically implied** by the other two: if ``n >= bn - t`` and
``f >= bf - t`` then ``(n+f)/2 >= (bn+bf)/2 - t``. So it can never change the
feasibility verdict, and no behavioural test can detect its removal. It is kept
anyway because the requirement is *explicit auditability*: a reader of the
validation table should see each safeguard evaluated by name rather than having
to reconstruct the implication. ``test_the_balanced_safeguard_is_implied_but_
still_recorded`` states the implication so the redundancy is documented rather
than accidental.

A second rule lives here (audit finding D-11): the **fallback** objective's
retrieval penalty must be dimensionless. It previously divided a *per-sentence*
document count by a *per-subclaim* budget, which is not a fraction of anything.

Every threshold candidate in one tuning run is evaluated on the *same*
validation records, so the sentence and subclaim counts are fixed and

    avg_docs_per_sentence = avg_docs_per_subclaim * subclaims_per_sentence

where ``subclaims_per_sentence`` is a constant of the split (~1.57 here). The
old quantity was therefore the correct cost multiplied by that constant, so the
fallback traded balanced PR-AUC against retrieval cost at the wrong **exchange
rate**: effectively ``retrieval_penalty * subclaims_per_sentence`` rather than
``retrieval_penalty``. Balanced PR-AUC is not scaled alongside it, so the two
objectives are not order-equivalent and can select different configurations.
The old value could also exceed 1.0, which is how the unit error is visible
even without comparing candidates.

The corrected cost is
``avg_retrieved_documents_per_subclaim / max_documents_per_subclaim``, so
numerator and denominator share a unit and the value is a genuine fraction of
the per-subclaim budget.

This correction touches the **fallback objective only**. Selection among
confirmatorily feasible configurations is unchanged and still minimises
documents *per sentence*.

Standard library only.
"""

FALLBACK_COST_NORMALIZATION = (
    "avg_retrieved_documents_per_subclaim / max_documents_per_subclaim"
)
FALLBACK_COST_UNITS = (
    "dimensionless in [0, 1]; numerator documents/subclaim, denominator "
    "maximum documents/subclaim"
)


def candidate_record(
    lower,
    upper,
    metrics,
    baseline_metrics,
    *,
    quality_tolerance,
    retrieval_penalty,
    max_docs,
):
    """Score one threshold pair against the baseline, recording every safeguard."""
    baseline_nonfactual = baseline_metrics["nonfactual"]["auc_pr"]
    baseline_factual = baseline_metrics["factual"]["auc_pr"]
    baseline_balanced = baseline_metrics["balanced_pr_auc"]

    nonfactual = metrics["nonfactual"]["auc_pr"]
    factual = metrics["factual"]["auc_pr"]
    balanced = metrics["balanced_pr_auc"]
    avg_docs = metrics["efficiency"]["avg_retrieved_documents_per_sentence"]
    avg_docs_per_subclaim = metrics["efficiency"][
        "avg_retrieved_documents_per_subclaim"
    ]
    normalized_document_cost = _normalized_document_cost(
        avg_docs_per_subclaim, max_docs
    )

    preserves_nonfactual = nonfactual >= baseline_nonfactual - quality_tolerance
    preserves_factual = factual >= baseline_factual - quality_tolerance
    preserves_balanced = balanced >= baseline_balanced - quality_tolerance

    return {
        "lower": float(lower),
        "upper": float(upper),
        "factual_auc_pr": factual,
        "nonfactual_auc_pr": nonfactual,
        "balanced_pr_auc": balanced,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        # The PRIMARY feasible-selection objective. Per sentence, unchanged.
        "avg_documents": avg_docs,
        # The FALLBACK cost basis. Per subclaim, matching the budget's unit.
        "avg_documents_per_subclaim": float(avg_docs_per_subclaim),
        "normalized_document_cost": normalized_document_cost,
        "max_documents_per_subclaim": float(max_docs),
        "fallback_retrieval_penalty": float(retrieval_penalty),
        "fallback_cost_normalization": FALLBACK_COST_NORMALIZATION,
        "fallback_cost_units": FALLBACK_COST_UNITS,
        "avg_nli_span_calls": metrics["efficiency"]["avg_nli_span_calls_per_sentence"],
        "nonfactual_auc_pr_delta_vs_bse": float(nonfactual - baseline_nonfactual),
        "factual_auc_pr_delta_vs_bse": float(factual - baseline_factual),
        "balanced_pr_auc_delta_vs_bse": float(balanced - baseline_balanced),
        "preserves_nonfactual": bool(preserves_nonfactual),
        "preserves_factual": bool(preserves_factual),
        "preserves_balanced": bool(preserves_balanced),
        # All three, conjunctively. Omitting nonfactual was audit finding D-07.
        "preserves_baseline_quality": bool(
            preserves_nonfactual and preserves_factual and preserves_balanced
        ),
        "fallback_objective": float(
            balanced - retrieval_penalty * normalized_document_cost
        ),
    }


def _normalized_document_cost(avg_documents_per_subclaim, max_docs):
    """The fallback cost as a genuine fraction of the per-subclaim budget.

    Both quantities are documents per subclaim, so the ratio is dimensionless:
    a configuration retrieving 5 of a possible 10 documents per subclaim scores
    0.5, and the penalty's exchange rate against balanced PR-AUC is the declared
    ``retrieval_penalty`` rather than that value inflated by the split's
    subclaims-per-sentence constant.

    Nothing is clamped or rescaled. A value outside ``[0, 1]`` means the metric
    is impossible for the configured budget -- a subclaim cannot consume more
    than ``max_docs`` documents -- and silently pulling it back into range would
    hide exactly the unit error this check exists to catch.
    """
    budget = float(max_docs)
    if not budget > 0.0:
        raise ValueError(
            f"max_documents_per_subclaim must be positive, got {max_docs!r}. "
            "The fallback retrieval penalty divides by it, so a zero or "
            "negative budget has no meaning."
        )
    cost = float(avg_documents_per_subclaim) / budget
    if cost < 0.0:
        raise ValueError(
            f"avg_retrieved_documents_per_subclaim is {avg_documents_per_subclaim!r}, "
            "which is negative. A retrieval count cannot be negative; this "
            "indicates a corrupted efficiency metric, not a cheap configuration."
        )
    if cost > 1.0:
        raise ValueError(
            f"avg_retrieved_documents_per_subclaim is {avg_documents_per_subclaim!r} "
            f"against a budget of {budget!r} documents per subclaim, giving a "
            f"normalized cost of {cost!r}. A subclaim cannot consume more than "
            "the budget, so this is either a unit inconsistency (a per-sentence "
            "count supplied where a per-subclaim count is required) or an "
            "impossible retrieval count. The value is NOT clamped."
        )
    return cost


FEASIBLE_SELECTION_RULE = (
    "minimum retrieval cost among validation configurations preserving "
    "BSE-official nonfactual, factual, and balanced PR-AUC within the "
    "validation tolerance"
)

FALLBACK_SELECTION_RULE = (
    "fallback penalized balanced-PR-AUC/retrieval objective, with retrieval "
    f"cost normalized as {FALLBACK_COST_NORMALIZATION} (dimensionless); no "
    "DDRE threshold pair preserved BSE-official nonfactual, factual, and "
    "balanced validation quality. NOT ELIGIBLE for the confirmatory claim: a "
    "fallback configuration cannot support it however large the held-out "
    "effect."
)


def select_threshold_configuration(candidates):
    """Choose a configuration and say whether the choice can support the claim.

    The primary objective stays **retrieval efficiency** — that is the
    hypothesis under test. Quality only breaks EXACT document-count ties, and a
    final ordering on the thresholds themselves makes the choice deterministic
    even then.

    Returns ``(selected, rule, confirmatory)``. ``confirmatory`` is False when
    no configuration preserved all three metrics: the fallback may still be
    evaluated for exploratory purposes, but a spectacular held-out result must
    never turn a fallback selection into a confirmatory success.
    """
    if not candidates:
        raise ValueError("no threshold candidates were scored")

    feasible = [c for c in candidates if c["preserves_baseline_quality"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda c: (
                c["avg_documents"],
                -c["balanced_pr_auc"],
                -c["nonfactual_auc_pr"],
                -c["factual_auc_pr"],
                c["lower"],
                c["upper"],
            ),
        )
        return selected, FEASIBLE_SELECTION_RULE, True

    selected = max(
        candidates,
        key=lambda c: (
            c["fallback_objective"],
            c["balanced_pr_auc"],
            -c["avg_documents"],
        ),
    )
    return selected, FALLBACK_SELECTION_RULE, False


SAFEGUARD_NOTE = (
    "A candidate is confirmatorily feasible only if it preserves BSE-official "
    "nonfactual AND factual AND balanced PR-AUC, each within the validation "
    "tolerance. All three are recorded per candidate. Feasible selection "
    "minimises documents PER SENTENCE; the fallback objective's retrieval cost "
    f"is {FALLBACK_COST_NORMALIZATION}, dimensionless in [0, 1] (D-11)."
)
