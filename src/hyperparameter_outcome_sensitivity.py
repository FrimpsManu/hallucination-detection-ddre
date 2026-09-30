"""D-03 follow-up: is the DDRE conclusion sensitive to the uLSIF hyperparameters?

D-03 trigger 2 fired: the direction of the evidence is unstable across the 20
``(sigma, lambda)`` pairs the production cross-validation already scored (310
affected scores). That is a statement about the *evidence*. It does not by
itself say whether the paper-level quality/efficiency conclusion moves.

This module answers only that narrower question, and only on the frozen
validation half. It is a SENSITIVITY ANALYSIS:

* it selects nothing -- the production pair remains the uLSIF-CV-selected
  ``sigma=0.2269287109375, lambda=1.0``;
* it chooses no cap, no calibration, no evidence transform;
* it never touches the held-out passages;
* it reuses the repository's existing validation rule
  (:mod:`src.threshold_selection`) rather than inventing a quality criterion.

Everything here is pure: no model, no NLI inference, no cache writes. The
runner is ``scripts/d03_hyperparameter_outcome_sensitivity.py``.
"""

import hashlib
import json
import statistics

ANALYSIS_NAME = "d03-hyperparameter-outcome-sensitivity"
PROTOCOL_VERSION = "d03-outcome-sensitivity-v1"

# The frozen D-03 facts this follow-up is bound to. Disagreement is fatal:
# a different population is a different study.
D03_BASE_COMMIT = "2661b047d15592d45ad706bbe34e4ddcd852aa55"
EXPECTED_VALIDATION_FRACTION = 0.20
EXPECTED_SPLIT_SEED = 42
EXPECTED_VALIDATION_PASSAGES = 48
EXPECTED_VALIDATION_SENTENCES = 383
EXPECTED_VALIDATION_SUBCLAIMS = 603
EXPECTED_HELD_OUT_PASSAGES = 190
EXPECTED_FACTUAL_TRAINING = 199
EXPECTED_HALLUCINATED_TRAINING = 199
EXPECTED_HYPERPARAMETER_PAIRS = 20
EXPECTED_THRESHOLD_PAIRS = 32
PRODUCTION_SIGMA = 0.2269287109375
PRODUCTION_LAMBDA = 1.0

# Production cost configuration. Not tunable here.
C_MISS = 28.0
C_FALSE_ALARM = 96.0
C_RETRIEVE = 1.0
P0 = 0.5
MAX_DOCS = 10


class ArtifactVerificationFailed(RuntimeError):
    """An input artifact is not the frozen D-03 artifact this study requires."""


class ProductionFitChanged(RuntimeError):
    """The refit did not reproduce the recorded production hyperparameters."""


class HeldOutLeak(RuntimeError):
    """A held-out record reached something that scores or measures."""


def _dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _first(payload, *paths):
    for path in paths:
        value = _dig(payload, path)
        if value is not None:
            return value
    return None


def passage_ids_sha256(passage_ids):
    """The D-03 split digest: sha256 over the sorted passage IDs as JSON.

    Transcribed from ``frozen_validation_split`` in
    ``scripts/diagnose_ddre_ratio_support.py``. It is recomputed rather than
    imported so this module stays importable on its own; the recomputed value
    is then checked against the one the D-03 artifact recorded, so any drift
    between the two implementations fails the run instead of passing silently.
    """
    ordered = sorted(int(x) for x in passage_ids)
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True).encode("utf-8")
    ).hexdigest()


def verify_d03_artifact(report, *, report_sha256, expected_report_sha256,
                        cache_sha256, expected_cache_sha256):
    """Every precondition, checked before any analysis. Fails closed.

    Returns the verification record. Raises
    :class:`ArtifactVerificationFailed` listing every disagreement rather than
    the first, so a mismatched input is diagnosed in one pass.
    """
    checks = []

    def check(name, expected, observed, ok=None):
        passed = (expected == observed) if ok is None else bool(ok)
        checks.append(
            {"name": name, "expected": expected, "observed": observed,
             "passed": bool(passed)}
        )

    check("d03_report_sha256", expected_report_sha256, report_sha256)
    check("derived_cache_sha256", expected_cache_sha256, cache_sha256)
    check("d03_git_commit", D03_BASE_COMMIT, _first(report, "git_commit"))
    check("completion_complete", True, _first(report, "completion.complete"))
    check(
        "score_compatibility_established",
        True,
        _first(report, "score_compatibility_established"),
    )
    # held_out_scored must be present AND false. A missing key is not a pass.
    held_out = _first(report, "held_out_scored")
    if held_out is None and isinstance(report, dict):
        held_out = report.get("held_out_scored", None)
    check("d03_held_out_scored", False, held_out, ok=(held_out is False))

    split = _first(report, "split", "split_identity") or {}
    check("split_identity_matches", True, split.get("identity_matches"))
    check(
        "validation_fraction",
        EXPECTED_VALIDATION_FRACTION,
        split.get("validation_fraction"),
    )
    check(
        "validation_passages",
        EXPECTED_VALIDATION_PASSAGES,
        split.get("validation_passages"),
    )

    failures = [c for c in checks if not c["passed"]]
    record = {
        "checks": checks,
        "failures": [c["name"] for c in failures],
        "passed": not failures,
        "d03_split_sha256": split.get("sha256"),
    }
    if failures:
        detail = "\n".join(
            f"  {c['name']}: expected {c['expected']!r}, observed {c['observed']!r}"
            for c in failures
        )
        raise ArtifactVerificationFailed(
            "the D-03 inputs are not the frozen artifacts this follow-up "
            f"requires:\n{detail}\nNothing was analysed."
        )
    return record


def verify_production_fit(estimator):
    """The refit must reproduce the recorded production selection exactly."""
    sigma = float(estimator.sigma)
    lam = float(estimator.lam)
    pairs = {(float(r["sigma"]), float(r["lambda"])) for r in estimator.cv_table}
    problems = []
    if sigma != PRODUCTION_SIGMA:
        problems.append(f"sigma is {sigma!r}, expected {PRODUCTION_SIGMA!r}")
    if lam != PRODUCTION_LAMBDA:
        problems.append(f"lambda is {lam!r}, expected {PRODUCTION_LAMBDA!r}")
    if len(pairs) != EXPECTED_HYPERPARAMETER_PAIRS:
        problems.append(
            f"the cv_table holds {len(pairs)} unique (sigma, lambda) pairs, "
            f"expected {EXPECTED_HYPERPARAMETER_PAIRS}"
        )
    if problems:
        raise ProductionFitChanged(
            "the production uLSIF fit did not reproduce: " + "; ".join(problems)
        )
    return {
        "sigma": sigma,
        "lambda": lam,
        "unique_hyperparameter_pairs": len(pairs),
        "n_centers": int(estimator.centers.size),
        "reproduced_production_selection": True,
    }


def candidate_estimator(production, factual_x, hallucinated_x, sigma, lam):
    """A ratio estimator at one candidate ``(sigma, lambda)``.

    The FINAL CENTER SET IS HELD FIXED at the production estimator's own
    centers, exactly as the D-03 hyperparameter surface does, so the only thing
    varying across the 20 pairs is the bandwidth and the regulariser.

    The production ``_solve`` is reused, and the object returned is a real
    ``ULSIFDensityRatio`` whose ``ratio()`` applies the untouched production
    ``[1e-6, 1e6]`` clip and non-finite guards. No cap, no calibration, no
    rescaling, and no change to the ratio definition.
    """
    from src.ddre_core import ULSIFDensityRatio

    candidate = ULSIFDensityRatio(random_state=production.random_state)
    candidate.centers = production.centers
    candidate.sigma = float(sigma)
    candidate.lam = float(lam)
    candidate.alpha = production._solve(
        factual_x, hallucinated_x, production.centers, float(sigma), float(lam)
    )
    return candidate


def assert_validation_only(records, held_out_passage_ids, *, where):
    """Refuse to proceed if a held-out passage reached ``records``."""
    forbidden = set(int(x) for x in held_out_passage_ids)
    seen = {int(r.passage_index) for r in records}
    leaked = sorted(seen & forbidden)
    if leaked:
        raise HeldOutLeak(
            f"{len(leaked)} held-out passage(s) reached {where}: {leaked[:10]}. "
            "The held-out split must never be scored by this follow-up."
        )
    return True


def _spread(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(clean),
        "min": min(clean),
        "median": float(statistics.median(clean)),
        "max": max(clean),
    }


SPREAD_FIELDS = (
    "nonfactual_auc_pr_delta_vs_bse",
    "factual_auc_pr_delta_vs_bse",
    "balanced_pr_auc_delta_vs_bse",
    "retrieved_documents_reduction_fraction",
    "nli_span_calls_reduction_fraction",
)


def robustness_summary(rows):
    """Descriptive spread across the hyperparameter pairs. No verdict.

    Deliberately reports ranges and counts and stops there. Collapsing this
    into a new PASS/FAIL threshold would be inventing a decision rule, which is
    exactly what this study must not do.
    """
    with_feasible = [r for r in rows if r["feasible_threshold_count"] > 0]
    confirmatory = [r for r in rows if r["confirmatory_threshold_selection"]]
    production = next((r for r in rows if r["is_production_pair"]), None)

    def selected(row, key):
        return (row.get("selected") or {}).get(key)

    return {
        "hyperparameter_pairs": len(rows),
        "pairs_with_a_confirmatorily_feasible_threshold": len(with_feasible),
        "pairs_without_a_confirmatorily_feasible_threshold": len(rows) - len(with_feasible),
        "pairs_with_confirmatory_threshold_selection": len(confirmatory),
        "production_pair": production,
        "spread_all_pairs": {
            field: _spread([selected(r, field) for r in rows])
            for field in SPREAD_FIELDS
        },
        "spread_confirmatory_pairs_only": {
            field: _spread([selected(r, field) for r in confirmatory])
            for field in SPREAD_FIELDS
        },
        "pairs_whose_selection_reduces_retrieval": sum(
            1 for r in rows
            if (selected(r, "retrieved_documents_reduction_fraction") or 0.0) > 0.0
        ),
        "pairs_whose_selection_reduces_nli_calls": sum(
            1 for r in rows
            if (selected(r, "nli_span_calls_reduction_fraction") or 0.0) > 0.0
        ),
        "pairs_preserving_quality_and_reducing_retrieval": sum(
            1 for r in rows
            if selected(r, "preserves_baseline_quality")
            and (selected(r, "retrieved_documents_reduction_fraction") or 0.0) > 0.0
        ),
        "interpretation": None,
        "interpretation_note": (
            "Descriptive only. No pass/fail threshold is applied and no "
            "hyperparameter is selected: the production pair remains the "
            "uLSIF-CV-selected one. Reviewing what this spread means is a "
            "separate human decision."
        ),
    }
