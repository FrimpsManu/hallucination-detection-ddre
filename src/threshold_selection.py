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

Standard library only.
"""


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
    normalized_docs = avg_docs / max(1.0, float(max_docs))

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
        "avg_documents": avg_docs,
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
        "fallback_objective": float(balanced - retrieval_penalty * normalized_docs),
    }


FEASIBLE_SELECTION_RULE = (
    "minimum retrieval cost among validation configurations preserving "
    "BSE-official nonfactual, factual, and balanced PR-AUC within the "
    "validation tolerance"
)

FALLBACK_SELECTION_RULE = (
    "fallback penalized balanced-PR-AUC/retrieval objective; no DDRE threshold "
    "pair preserved BSE-official nonfactual, factual, and balanced validation "
    "quality. NOT ELIGIBLE for the confirmatory claim: a fallback configuration "
    "cannot support it however large the held-out effect."
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
    "tolerance. All three are recorded per candidate."
)
