import math

import numpy as np

from src.baseline_core import DetectionResult, cost_based_prediction


class ULSIFDensityRatio:
    """Direct density-ratio estimator using unconstrained LSIF (uLSIF).

    We estimate r(s) = p(s | factual) / p(s | hallucinated) directly from
    continuous DeBERTa entailment scores, rather than estimating both class
    densities separately or discretizing scores into ten bins.
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

    def _choose_centers(self, factual_x, rng):
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
        f_folds = np.array_split(f_indices, min(folds, len(f_indices)))
        h_folds = np.array_split(h_indices, min(folds, len(h_indices)))
        n_folds = min(len(f_folds), len(h_folds))

        best = None
        self.cv_table = []

        for sigma in sigma_grid:
            for lam in lambda_grid:
                fold_scores = []
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
                    centers = self._choose_centers(f_train, rng)
                    alpha = self._solve(f_train, h_train, centers, sigma, lam)
                    fold_scores.append(
                        self._objective(
                            factual_x[f_val_idx],
                            hallucinated_x[h_val_idx],
                            centers,
                            alpha,
                            sigma,
                        )
                    )

                cv_score = float(np.mean(fold_scores))
                row = {"sigma": sigma, "lambda": lam, "cv_objective": cv_score}
                self.cv_table.append(row)
                if best is None or cv_score < best["cv_objective"]:
                    best = row

        self.sigma = float(best["sigma"])
        self.lam = float(best["lambda"])
        final_rng = np.random.default_rng(self.random_state)
        self.centers = self._choose_centers(factual_x, final_rng)
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


class DDREDetector:
    """Sequential retrieval using directly estimated evidence density ratios.

    Each retrieved document contributes a directly estimated likelihood ratio.
    Under the same conditional-independence assumption used by the Bayesian NBC
    baseline, log density ratios add across evidence. We stop once the posterior
    exits a validation-selected uncertainty interval [lower, upper].
    """

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
