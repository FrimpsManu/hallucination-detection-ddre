"""uLSIF normalisation diagnostic (training data only).

A density ratio r(s) = p(s | factual) / p(s | hallucinated) satisfies
E_hallucinated[r(s)] = 1. uLSIF does not impose this: with a large ridge
penalty lambda the coefficients shrink and r is biased toward 0, which turns
every document's evidence toward "hallucinated". This module measures that on
the NBC training samples the estimators are fit on, for any fitted estimator,
and shows what the log-ratio curve becomes after dividing by the empirical
normaliser.

Pure functions over scores. Nothing here reads held-out or test data, selects a
configuration, or changes an estimator.
"""

import math

import numpy as np

GRID = tuple(range(0, 101, 5))
MODE_SCORE = 35.0


def normaliser(estimator, hallucinated_scores):
    """Empirical E_hallucinated[r(s)] over the training hallucinated samples."""
    return float(np.mean([estimator.ratio(s) for s in hallucinated_scores]))


def normalisation_row(estimator, factual_scores, hallucinated_scores, *,
                      grid=GRID, mode_score=MODE_SCORE):
    """Normalisation and log-ratio curves for one fitted estimator.

    ``log_r_normalised(s) = log r(s) - log E_h[r]`` is what the curve would be
    if the estimator satisfied the density-ratio constraint on its own
    training data. It is reported, not used.
    """
    z = normaliser(estimator, hallucinated_scores)
    log_z = math.log(z)
    raw = {s: math.log(estimator.ratio(s)) for s in grid}
    mode_raw = math.log(estimator.ratio(mode_score))
    return {
        "mean_r_hallucinated": z,
        "mean_r_factual": float(np.mean([estimator.ratio(s) for s in factual_scores])),
        "mean_log_r_hallucinated": float(np.mean(
            [math.log(estimator.ratio(s)) for s in hallucinated_scores])),
        "mean_log_r_factual": float(np.mean(
            [math.log(estimator.ratio(s)) for s in factual_scores])),
        "log_normaliser": log_z,
        "log_r": {str(s): v for s, v in raw.items()},
        "log_r_normalised": {str(s): v - log_z for s, v in raw.items()},
        "mode_score": mode_score,
        "log_r_at_mode": mode_raw,
        "log_r_normalised_at_mode": mode_raw - log_z,
        "sign_at_mode": int(np.sign(mode_raw)),
        "sign_at_mode_normalised": int(np.sign(mode_raw - log_z)),
    }


def sign_agreement(rows, reference_key, key="sign_at_mode"):
    """How many rows share the reference row's sign at the mode score."""
    ref = rows[reference_key][key]
    same = sum(1 for r in rows.values() if r[key] == ref)
    return {"reference": reference_key, "reference_sign": ref, "same": same,
            "total": len(rows)}
