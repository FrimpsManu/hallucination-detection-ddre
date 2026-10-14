"""Evidence estimators for the multivariate DDRE study (design §3).

Every estimator maps one document's features to a density ratio
r(x) = p(x | claim factual) / p(x | claim hallucinated), and exposes
``log_ratio(x)`` for the sequential update.

- ``HistogramEvidence`` (cells A and A'): Wang's 10-bin histogram of e with
  Laplace +1, binned with the document discretizer exactly as BSE does.
- ``MultiULSIF`` (cells U and M): Gaussian-kernel uLSIF on standardised
  features. sigma and lambda are chosen only by the uLSIF cross-validation
  objective on a grid fixed below, then the ratio is self-normalised so that
  its mean over the hallucinated training samples is exactly 1.
- ``LogisticRatio`` (cell L): L2 logistic regression with C chosen by 5-fold
  log-loss; r = p / (1 - p) * n_h / n_f.

The grids, fold count, centre count and seed are fixed here before any feature
exists. No detector cost or detection metric is ever used to choose anything.
"""

import math

import numpy as np

from src.baseline_core import discretize_document_score

SEED = 42
N_CENTERS = 100
CV_FOLDS = 5
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)  # x median pairwise distance, as the 1-D code
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MEDIAN_SAMPLE = 2000
RATIO_CLIP = (1e-6, 1e6)
LOGISTIC_CS = 10

FEATURE_SETS = {"U": ("e",), "M": ("e", "k", "log_rank")}


class EstimatorError(RuntimeError):
    pass


def features(doc, names):
    """One document's feature vector. ``doc`` has e, k (0-100) and rank (1..K)."""
    values = {"e": doc["e"], "k": doc["k"], "log_rank": math.log(doc["rank"])}
    out = np.array([float(values[n]) for n in names])
    if not np.all(np.isfinite(out)):
        raise EstimatorError(f"non-finite features {out} for {doc}")
    return out


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        if np.any(self.std <= 0):
            raise EstimatorError("a feature is constant on the training set")
        return self

    def __call__(self, x):
        return (np.atleast_2d(x) - self.mean) / self.std

    def to_json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, d):
        out = cls()
        out.mean, out.std = np.asarray(d["mean"]), np.asarray(d["std"])
        return out


class HistogramEvidence:
    """BSE's histogram evidence on e (cells A and A')."""

    def __init__(self, pos_hist, neg_hist):
        if len(pos_hist) != 10 or len(neg_hist) != 10 or min(pos_hist + neg_hist) <= 0:
            raise EstimatorError("histograms must be 10 Laplace-smoothed buckets")
        self.pos_hist, self.neg_hist = list(pos_hist), list(neg_hist)
        tp, tn = float(sum(pos_hist)), float(sum(neg_hist))
        self.llr = [math.log((p / tp) / (n / tn)) for p, n in zip(pos_hist, neg_hist)]

    @classmethod
    def fit(cls, factual_e, hallucinated_e):
        pos, neg = [1] * 10, [1] * 10
        for s in factual_e:
            pos[discretize_document_score(s)] += 1
        for s in hallucinated_e:
            neg[discretize_document_score(s)] += 1
        return cls(pos, neg)

    def log_ratio(self, doc):
        return self.llr[discretize_document_score(doc["e"])]

    def to_json(self):
        return {"kind": "histogram", "pos_hist": self.pos_hist, "neg_hist": self.neg_hist}

    def likelihoods(self, doc):
        """(P(bucket | factual), P(bucket | hallucinated)) as BSE uses them."""
        b = discretize_document_score(doc["e"])
        return (self.pos_hist[b] / float(sum(self.pos_hist)),
                self.neg_hist[b] / float(sum(self.neg_hist)))


def _kernel(x, centers, sigma):
    sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-sq / (2.0 * sigma * sigma))


def _solve(f, h, centers, sigma, lam):
    phi_h = _kernel(h, centers, sigma)
    phi_f = _kernel(f, centers, sigma)
    H = phi_h.T @ phi_h / len(h)
    hv = phi_f.mean(axis=0)
    try:
        alpha = np.linalg.solve(H + lam * np.eye(len(centers)), hv)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(H + lam * np.eye(len(centers))) @ hv
    return np.maximum(alpha, 0.0)


def _objective(f, h, centers, alpha, sigma):
    rh = _kernel(h, centers, sigma) @ alpha
    rf = _kernel(f, centers, sigma) @ alpha
    return float(0.5 * np.mean(rh ** 2) - np.mean(rf))


class MultiULSIF:
    """Self-normalised Gaussian-kernel uLSIF on standardised features."""

    def __init__(self, names):
        self.names = tuple(names)

    def fit(self, factual_docs, hallucinated_docs):
        F = np.array([features(d, self.names) for d in factual_docs])
        Hx = np.array([features(d, self.names) for d in hallucinated_docs])
        self.scale = Standardizer().fit(np.vstack([F, Hx]))
        F, Hx = self.scale(F), self.scale(Hx)
        rng = np.random.default_rng(SEED)
        pooled = np.vstack([F, Hx])
        sample = pooled[rng.choice(len(pooled), size=min(MEDIAN_SAMPLE, len(pooled)),
                                   replace=False)]
        d = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2))
        median = float(np.median(d[d > 1e-8]))
        self.sigma_grid = [median * m for m in SIGMA_MULTIPLIERS]

        fi, hi = rng.permutation(len(F)), rng.permutation(len(Hx))
        f_folds, h_folds = np.array_split(fi, CV_FOLDS), np.array_split(hi, CV_FOLDS)
        folds = []
        for k in range(CV_FOLDS):
            f_tr = F[np.concatenate([f_folds[i] for i in range(CV_FOLDS) if i != k])]
            h_tr = Hx[np.concatenate([h_folds[i] for i in range(CV_FOLDS) if i != k])]
            c_rng = np.random.default_rng(SEED + 1000 + k)
            centers = f_tr[c_rng.choice(len(f_tr), size=min(N_CENTERS, len(f_tr)),
                                        replace=False)]
            folds.append((f_tr, h_tr, F[f_folds[k]], Hx[h_folds[k]], centers))
        self.cv_table, best = [], None
        for sigma in self.sigma_grid:
            for lam in LAMBDA_GRID:
                score = float(np.mean([
                    _objective(fv, hv, c, _solve(ft, ht, c, sigma, lam), sigma)
                    for ft, ht, fv, hv, c in folds]))
                if not math.isfinite(score):
                    raise EstimatorError(f"non-finite CV objective at {sigma}, {lam}")
                row = {"sigma": sigma, "lambda": lam, "cv_objective": score}
                self.cv_table.append(row)
                if best is None or score < best["cv_objective"]:
                    best = row
        self.sigma, self.lam = best["sigma"], best["lambda"]
        c_rng = np.random.default_rng(SEED)
        self.centers = F[c_rng.choice(len(F), size=min(N_CENTERS, len(F)), replace=False)]
        self.alpha = _solve(F, Hx, self.centers, self.sigma, self.lam)
        if not np.any(self.alpha > 0):
            raise EstimatorError("uLSIF fit is identically zero")
        self.normaliser = 1.0
        raw_h = self._raw(Hx)
        self.normaliser = float(np.mean(np.clip(raw_h, *RATIO_CLIP)))
        self.diagnostics = {
            "n_factual": len(F), "n_hallucinated": len(Hx),
            "raw_mean_r_hallucinated": self.normaliser,
            "mean_r_hallucinated": float(np.mean(self._ratio_std(Hx))),
            "mean_r_factual": float(np.mean(self._ratio_std(F))),
            "sigma_at_grid_edge": self.sigma in (self.sigma_grid[0], self.sigma_grid[-1]),
            "lambda_at_grid_edge": self.lam in (LAMBDA_GRID[0], LAMBDA_GRID[-1]),
        }
        return self

    def _raw(self, x_std):
        return _kernel(x_std, self.centers, self.sigma) @ self.alpha

    def _ratio_std(self, x_std):
        return np.clip(self._raw(x_std) / self.normaliser, *RATIO_CLIP)

    def log_ratio(self, doc):
        return float(np.log(self._ratio_std(self.scale(features(doc, self.names)))[0]))

    def to_json(self):
        return {"kind": "ulsif", "features": list(self.names), "scale": self.scale.to_json(),
                "sigma": self.sigma, "lambda": self.lam, "normaliser": self.normaliser,
                "centers": self.centers.tolist(), "alpha": self.alpha.tolist(),
                "cv_table": self.cv_table, "diagnostics": self.diagnostics}

    @classmethod
    def from_json(cls, d):
        out = cls(d["features"])
        out.scale = Standardizer.from_json(d["scale"])
        out.sigma, out.lam, out.normaliser = d["sigma"], d["lambda"], d["normaliser"]
        out.centers, out.alpha = np.asarray(d["centers"]), np.asarray(d["alpha"])
        out.cv_table, out.diagnostics = d["cv_table"], d["diagnostics"]
        return out


class LogisticRatio:
    """Classifier-based density ratio (cell L)."""

    def __init__(self, names):
        self.names = tuple(names)

    def fit(self, factual_docs, hallucinated_docs):
        from sklearn.linear_model import LogisticRegressionCV

        F = np.array([features(d, self.names) for d in factual_docs])
        Hx = np.array([features(d, self.names) for d in hallucinated_docs])
        self.scale = Standardizer().fit(np.vstack([F, Hx]))
        X = self.scale(np.vstack([F, Hx]))
        y = np.r_[np.ones(len(F)), np.zeros(len(Hx))]
        self.model = LogisticRegressionCV(Cs=LOGISTIC_CS, cv=CV_FOLDS, scoring="neg_log_loss",
                                          max_iter=2000).fit(X, y)
        self.coef, self.intercept = self.model.coef_[0], float(self.model.intercept_[0])
        self.prior_correction = math.log(len(Hx) / len(F))
        r_h = np.exp([self._log_ratio_std(x) for x in self.scale(Hx)])
        r_f = np.exp([self._log_ratio_std(x) for x in self.scale(F)])
        self.diagnostics = {"n_factual": len(F), "n_hallucinated": len(Hx),
                            "C": float(self.model.C_[0]),
                            "mean_r_hallucinated": float(np.mean(r_h)),
                            "mean_r_factual": float(np.mean(r_f))}
        return self

    def _log_ratio_std(self, x_std):
        logit = float(np.dot(self.coef, np.ravel(x_std)) + self.intercept)
        return float(np.clip(logit + self.prior_correction, *np.log(RATIO_CLIP)))

    def log_ratio(self, doc):
        return self._log_ratio_std(self.scale(features(doc, self.names))[0])

    def to_json(self):
        return {"kind": "logistic", "features": list(self.names),
                "scale": self.scale.to_json(), "coef": self.coef.tolist(),
                "intercept": self.intercept,
                "prior_correction": self.prior_correction, "diagnostics": self.diagnostics}

    @classmethod
    def from_json(cls, d):
        out = cls(d["features"])
        out.scale = Standardizer.from_json(d["scale"])
        out.coef, out.intercept = np.asarray(d["coef"]), float(d["intercept"])
        out.prior_correction, out.diagnostics = d["prior_correction"], d["diagnostics"]
        return out


def from_json(d):
    """Any fitted estimator from its ``to_json`` record."""
    kind = d["kind"]
    if kind == "histogram":
        return HistogramEvidence(d["pos_hist"], d["neg_hist"])
    if kind == "ulsif":
        return MultiULSIF.from_json(d)
    if kind == "logistic":
        return LogisticRatio.from_json(d)
    raise EstimatorError(f"unknown estimator kind {kind!r}")
