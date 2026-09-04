import math

import numpy as np

from src.baseline_core import DetectionResult, cost_based_prediction


class ULSIFDensityRatio:
    """Direct density-ratio estimator using unconstrained LSIF (uLSIF).

    Estimates r(s) = p(s | factual) / p(s | hallucinated) directly from
    continuous DeBERTa entailment scores. Hyperparameters are selected with the
    standard held-out uLSIF objective, using identical kernel centers for every
    candidate within a fold so hyperparameter comparisons are reproducible.
    """

    def __init__(self, max_centers=100, random_state=42):
        self.max_centers = int(max_centers)
        self.random_state = int(random_state)
        self.centers = None
        self.alpha = None
        self.sigma = None
        self.lam = None
        self.cv_table = []

    @staticmethod
    def _as_column(scores):
        x = np.asarray(scores, dtype=float).reshape(-1, 1) / 100.0
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def _kernel(x, centers, sigma):
        squared = (x - centers.T) ** 2
        return np.exp(-squared / (2.0 * sigma * sigma))

    def _choose_centers(self, factual_x, seed):
        rng = np.random.default_rng(seed)
        n = min(self.max_centers, len(factual_x))
        indices = rng.choice(len(factual_x), size=n, replace=False)
        return factual_x[indices].copy()

    def _solve(self, factual_x, hallucinated_x, centers, sigma, lam):
        phi_h = self._kernel(hallucinated_x, centers, sigma)
        phi_f = self._kernel(factual_x, centers, sigma)
        h_matrix = (phi_h.T @ phi_h) / max(1, len(hallucinated_x))
        h_vector = np.mean(phi_f, axis=0)
        regularized = h_matrix + float(lam) * np.eye(h_matrix.shape[0])
        try:
            alpha = np.linalg.solve(regularized, h_vector)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(regularized) @ h_vector
        return np.maximum(alpha, 0.0)

    def _objective(self, factual_x, hallucinated_x, centers, alpha, sigma):
        ratio_h = self._kernel(hallucinated_x, centers, sigma) @ alpha
        ratio_f = self._kernel(factual_x, centers, sigma) @ alpha
        return float(0.5 * np.mean(ratio_h ** 2) - np.mean(ratio_f))

    def fit(self, factual_scores, hallucinated_scores, folds=5):
        factual_x = self._as_column(factual_scores)
        hallucinated_x = self._as_column(hallucinated_scores)
        if len(factual_x) < 2 or len(hallucinated_x) < 2:
            raise ValueError("uLSIF requires at least two samples from each class")

        rng = np.random.default_rng(self.random_state)
        all_x = np.vstack([factual_x, hallucinated_x]).ravel()
        pairwise = np.abs(all_x[:, None] - all_x[None, :])
        positive_distances = pairwise[pairwise > 1e-8]
        median_distance = (
            float(np.median(positive_distances))
            if positive_distances.size
            else 0.10
        )
        base_sigma = max(0.02, median_distance)
        sigma_grid = sorted(
            {max(0.01, base_sigma * factor) for factor in (0.5, 1.0, 2.0, 4.0)}
        )
        lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

        f_indices = np.arange(len(factual_x))
        h_indices = np.arange(len(hallucinated_x))
        rng.shuffle(f_indices)
        rng.shuffle(h_indices)
        n_folds = min(folds, len(factual_x), len(hallucinated_x))
        if n_folds < 2:
            raise ValueError("uLSIF cross-validation requires at least two folds")
        f_folds = np.array_split(f_indices, n_folds)
        h_folds = np.array_split(h_indices, n_folds)

        fold_data = []
        for fold in range(n_folds):
            f_val_idx = f_folds[fold]
            h_val_idx = h_folds[fold]
            f_train_idx = np.concatenate(
                [f_folds[i] for i in range(n_folds) if i != fold]
            )
            h_train_idx = np.concatenate(
                [h_folds[i] for i in range(n_folds) if i != fold]
            )
            f_train = factual_x[f_train_idx]
            h_train = hallucinated_x[h_train_idx]
            centers = self._choose_centers(
                f_train, self.random_state + 1000 + fold
            )
            fold_data.append(
                (
                    f_train,
                    h_train,
                    factual_x[f_val_idx],
                    hallucinated_x[h_val_idx],
                    centers,
                )
            )

        best = None
        self.cv_table = []
        for sigma in sigma_grid:
            for lam in lambda_grid:
                fold_scores = []
                for f_train, h_train, f_val, h_val, centers in fold_data:
                    alpha = self._solve(f_train, h_train, centers, sigma, lam)
                    fold_scores.append(
                        self._objective(f_val, h_val, centers, alpha, sigma)
                    )
                cv_score = float(np.mean(fold_scores))
                row = {"sigma": sigma, "lambda": lam, "cv_objective": cv_score}
                self.cv_table.append(row)
                if best is None or cv_score < best["cv_objective"]:
                    best = row

        self.sigma = float(best["sigma"])
        self.lam = float(best["lambda"])
        self.centers = self._choose_centers(factual_x, self.random_state)
        self.alpha = self._solve(
            factual_x,
            hallucinated_x,
            self.centers,
            self.sigma,
            self.lam,
        )
        return self

    def ratio(self, score):
        if self.alpha is None:
            raise RuntimeError("ULSIFDensityRatio must be fit before use")
        x = self._as_column([score])
        value = float((self._kernel(x, self.centers, self.sigma) @ self.alpha)[0])
        return float(np.clip(value, 1e-6, 1e6))


# The stopping-threshold search space the tuner starts from. Filtering it
# against the configured costs is what keeps stopping consistent with the final
# classification rule; see cost_consistent_thresholds.
CANDIDATE_LOWER_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
CANDIDATE_UPPER_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def _positive_cost(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number


def cost_decision_threshold(c_miss, c_false_alarm):
    """The posterior at which Wang's cost rule switches to "factual".

    ``cost_based_prediction`` declares factual iff ``(1-P)*C_M < P*C_FA``, which
    rearranges to ``P > C_M / (C_M + C_FA)``. This returns that threshold, so
    the arithmetic exists in exactly one place: the detector's stopping guard,
    the tuner's search space and the recorded provenance all read it from here.

    It lives in this module rather than beside ``cost_based_prediction`` because
    ``src/baseline_core.py`` is the audited Wang reproduction and is kept
    byte-identical; nothing about the returned value depends on that choice.

    Costs are validated defensively — they parameterise a decision rule, and a
    silently coerced NaN or a negative cost would produce a threshold that looks
    ordinary and means nothing.
    """
    miss = _positive_cost("c_miss", c_miss)
    false_alarm = _positive_cost("c_false_alarm", c_false_alarm)
    total = miss + false_alarm
    if total <= 0.0:
        raise ValueError(
            "c_miss + c_false_alarm must be positive to define a decision "
            f"threshold, got c_miss={c_miss!r}, c_false_alarm={c_false_alarm!r}"
        )
    return miss / total


def threshold_consistency(lower_threshold, upper_threshold, c_miss, c_false_alarm):
    """Do the stopping thresholds agree with the cost rule they will be judged by?

    Two clauses, and both are required.

    **Analytic.** ``lower <= t < upper`` with ``t = C_M/(C_M+C_FA)``. Stopping
    LOW asserts the posterior will classify NONFACTUAL and stopping HIGH asserts
    FACTUAL; Wang's rule is strict, so in exact arithmetic ``P == t`` classifies
    nonfactual and the bound is inclusive on the low side, strict on the high
    side.

    **Operational.** The analytic rule reasons about real numbers, but claims
    are classified by ``cost_based_prediction``, which compares two rounded
    floats. The two disagree at the boundary for some cost pairs: at
    C_M=3/C_FA=7 a 1-ULP rounding in ``(1-t)*C_M`` makes ``P == t`` classify
    FACTUAL, so a low stop there would assert the opposite of the verdict, and
    at C_M=2/C_FA=7 the first float above ``t`` still classifies NONFACTUAL, so
    a high stop there would do the same in the other direction. The boundaries
    are therefore checked against the classifier itself::

        cost_based_prediction(lower, ...) == 0
        cost_based_prediction(upper, ...) == 1

    ``cost_based_prediction`` is monotone non-decreasing in ``P`` -- as ``P``
    rises ``(1-P)*C_M`` cannot increase and ``P*C_FA`` cannot decrease, and
    correctly-rounded arithmetic preserves that ordering -- so the two boundary
    checks are enough: **every** posterior at or below ``lower`` classifies
    nonfactual and **every** posterior at or above ``upper`` classifies factual.
    Those are exactly the two claims the detector's stopping rule makes.

    The classifier is the source of operational truth; the cost formula is not
    restated here. Returns the full verdict so callers can report which clause
    failed.
    """
    threshold = cost_decision_threshold(c_miss, c_false_alarm)
    analytic = lower_threshold <= threshold < upper_threshold
    lower_prediction = cost_based_prediction(lower_threshold, c_miss, c_false_alarm)
    upper_prediction = cost_based_prediction(upper_threshold, c_miss, c_false_alarm)
    operational = lower_prediction == 0 and upper_prediction == 1
    return {
        "consistent": analytic and operational,
        "analytic": analytic,
        "operational": operational,
        "cost_decision_threshold": threshold,
        "lower_threshold": float(lower_threshold),
        "upper_threshold": float(upper_threshold),
        "lower_classifies_as": lower_prediction,
        "upper_classifies_as": upper_prediction,
        "c_miss": float(c_miss),
        "c_false_alarm": float(c_false_alarm),
    }


def thresholds_are_cost_consistent(
    lower_threshold, upper_threshold, c_miss, c_false_alarm
):
    """``threshold_consistency(...)["consistent"]``, for use as a predicate."""
    return threshold_consistency(
        lower_threshold, upper_threshold, c_miss, c_false_alarm
    )["consistent"]


def cost_consistent_thresholds(
    c_miss,
    c_false_alarm,
    lower_grid=CANDIDATE_LOWER_GRID,
    upper_grid=CANDIDATE_UPPER_GRID,
):
    """Restrict a candidate threshold grid to the cost-consistent pairs.

    Derived from the *configured* costs rather than hardcoded, so a different
    cost pair produces a different search space: at C_M=28/C_FA=96 the
    threshold is 0.2258 and the lower grid keeps 0.05-0.20, while at
    C_M=14/C_FA=24 it is 0.3684 and 0.35 survives but 0.40 does not.

    Returns the full record the experiment summary should carry, so that the
    search space is auditable prospectively rather than inferred from a log
    line after the fact.
    """
    threshold = cost_decision_threshold(c_miss, c_false_alarm)
    lower_grid = [float(x) for x in lower_grid]
    upper_grid = [float(x) for x in upper_grid]

    # Analytic filtering, then the operational boundary check against the
    # classifier itself. Pairs are then confirmed with the SAME predicate
    # DDREDetector.__init__ applies, so the search-space provenance, the tuner
    # and the constructor cannot disagree about what is admissible.
    effective_lower = [
        x
        for x in lower_grid
        if x <= threshold and cost_based_prediction(x, c_miss, c_false_alarm) == 0
    ]
    effective_upper = [
        x
        for x in upper_grid
        if x > threshold and cost_based_prediction(x, c_miss, c_false_alarm) == 1
    ]
    pairs = [
        (lower, upper)
        for lower in effective_lower
        for upper in effective_upper
        if 0.0 < lower < upper < 1.0
        and thresholds_are_cost_consistent(lower, upper, c_miss, c_false_alarm)
    ]

    return {
        "c_miss": float(c_miss),
        "c_false_alarm": float(c_false_alarm),
        "cost_decision_threshold": threshold,
        "candidate_lower_grid": lower_grid,
        "candidate_upper_grid": upper_grid,
        "effective_lower_grid": effective_lower,
        "effective_upper_grid": effective_upper,
        "excluded_lower_grid": [x for x in lower_grid if x not in effective_lower],
        "excluded_upper_grid": [x for x in upper_grid if x not in effective_upper],
        "threshold_pairs": [list(pair) for pair in pairs],
        "threshold_pairs_evaluated": len(pairs),
        "candidate_pairs_before_filtering": sum(
            1
            for lower in lower_grid
            for upper in upper_grid
            if 0.0 < lower < upper < 1.0
        ),
        "rule": (
            "Stopping must agree with the final cost rule "
            "(factual iff (1-P)*C_M < P*C_FA) both analytically and "
            "operationally. Analytic: lower <= t < upper with "
            "t = C_M/(C_M+C_FA); stopping low asserts a nonfactual "
            "classification and P == t classifies nonfactual in exact "
            "arithmetic, so the low bound is inclusive, while stopping high "
            "asserts factual, which P == t does not give, so the high bound is "
            "strict. Operational: cost_based_prediction(lower) == 0 and "
            "cost_based_prediction(upper) == 1, checked against the classifier "
            "itself because floating-point rounding breaks the analytic "
            "boundary for some cost pairs. The classifier is monotone in P, so "
            "the two boundary checks cover every posterior beyond them."
        ),
    }


class DDREDetector:
    """Sequential retrieval using directly estimated evidence density ratios."""

    def __init__(
        self,
        ratio_estimator,
        *,
        lower_threshold=0.20,
        upper_threshold=0.80,
        p0=0.5,
        c_miss=28,
        c_false_alarm=96,
        max_docs=10,
    ):
        if not 0.0 < lower_threshold < upper_threshold < 1.0:
            raise ValueError("Require 0 < lower_threshold < upper_threshold < 1")

        # The stopping rule and the classification rule must agree about what
        # a stop means -- analytically AND against the classifier that will
        # actually be applied. Enforced at construction, not only in the tuner,
        # so an inconsistent detector cannot be built by hand either. The tuner
        # filters with this same predicate, so the two cannot disagree.
        verdict = threshold_consistency(
            lower_threshold, upper_threshold, c_miss, c_false_alarm
        )
        threshold = verdict["cost_decision_threshold"]
        if not verdict["consistent"]:
            failed = []
            if not verdict["analytic"]:
                failed.append("analytic (require lower <= t < upper)")
            if not verdict["operational"]:
                failed.append(
                    "operational (require cost_based_prediction(lower) == 0 and "
                    "cost_based_prediction(upper) == 1)"
                )
            raise ValueError(
                "DDRE stopping thresholds contradict the cost-based "
                "classification rule.\n"
                f"  failed clause(s)       = {'; '.join(failed)}\n"
                f"  lower_threshold        = {float(lower_threshold)!r}\n"
                f"  upper_threshold        = {float(upper_threshold)!r}\n"
                f"  cost decision threshold= {threshold!r}\n"
                f"  c_miss                 = {float(c_miss)!r}\n"
                f"  c_false_alarm          = {float(c_false_alarm)!r}\n"
                f"  cost_based_prediction(lower) = "
                f"{verdict['lower_classifies_as']} (must be 0, nonfactual)\n"
                f"  cost_based_prediction(upper) = "
                f"{verdict['upper_classifies_as']} (must be 1, factual)\n"
                "Stopping low asserts the posterior classifies nonfactual and "
                "stopping high asserts it classifies factual; outside this "
                "range a stop would assert the opposite of what "
                "cost_based_prediction returns."
            )

        self.cost_decision_threshold = threshold
        self.ratio_estimator = ratio_estimator
        self.lower_threshold = float(lower_threshold)
        self.upper_threshold = float(upper_threshold)
        self.p0 = float(p0)
        self.c_miss = float(c_miss)
        self.c_false_alarm = float(c_false_alarm)
        self.max_docs = int(max_docs)

    @staticmethod
    def _logit(p):
        p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def detect_subclaim(self, subclaim, scorer, *, use_cache=True):
        log_odds = self._logit(self.p0)
        p_factual = self.p0
        documents_used = 0
        nli_calls = 0

        for document in subclaim.documents[: self.max_docs]:
            score, segment_calls = scorer.score_document(
                subclaim.text,
                document.page_content,
                use_cache=use_cache,
                write_cache=use_cache,
            )
            documents_used += 1
            nli_calls += segment_calls

            ratio = self.ratio_estimator.ratio(score)
            log_odds += math.log(ratio)
            log_odds = float(np.clip(log_odds, -40.0, 40.0))
            p_factual = self._sigmoid(log_odds)

            if (
                p_factual <= self.lower_threshold
                or p_factual >= self.upper_threshold
            ):
                break

        return DetectionResult(
            p_factual=float(p_factual),
            prediction=cost_based_prediction(
                p_factual, self.c_miss, self.c_false_alarm
            ),
            documents_used=documents_used,
            nli_calls=nli_calls,
        )

    def detect_sentence(self, record, scorer, *, use_cache=True):
        subclaim_results = [
            self.detect_subclaim(subclaim, scorer, use_cache=use_cache)
            for subclaim in record.subclaims
        ]
        p_factual = min(result.p_factual for result in subclaim_results)
        return DetectionResult(
            p_factual=float(p_factual),
            prediction=cost_based_prediction(
                p_factual, self.c_miss, self.c_false_alarm
            ),
            documents_used=sum(r.documents_used for r in subclaim_results),
            nli_calls=sum(r.nli_calls for r in subclaim_results),
        )
