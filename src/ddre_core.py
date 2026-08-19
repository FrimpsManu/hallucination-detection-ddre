import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import split_text, get_entailment_score


class DDREModel:
    """Classifier-based density-ratio estimator for hallucination detection.

    The logistic classifier estimates posterior class probabilities. We then
    convert posterior odds to an estimate of the class-conditional density
    ratio p(x|factual) / p(x|hallucinated) using the empirical training priors:

        r(x) = [P(F|x) / P(H|x)] * [P(H) / P(F)].

    This makes the density-ratio interpretation explicit rather than treating
    raw posterior odds as a class-conditional density ratio.
    """

    def __init__(self, threshold=0.5, random_state=42):
        self.model = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )
        self.threshold = threshold
        self.p_factual_prior = None
        self.p_hallucinated_prior = None

    def featurize(self, sentence, evidence, tokenizer, nli_model):
        segments = split_text(evidence)

        scores = [
            get_entailment_score(seg, sentence, tokenizer, nli_model)
            for seg in segments
        ]

        eps = 1e-8

        if scores:
            sorted_scores = sorted(scores, reverse=True)
            max_score = sorted_scores[0]
            avg_score = float(np.mean(scores))
            min_score = sorted_scores[-1]
            std_score = float(np.std(scores))
            median_score = float(np.median(scores))
            top_k_avg = float(np.mean(sorted_scores[:3]))
            top1_top2_gap = (
                float(sorted_scores[0] - sorted_scores[1])
                if len(sorted_scores) >= 2
                else float(sorted_scores[0])
            )
            prop_above_20 = float(np.mean(np.asarray(scores) >= 20.0))
            prop_above_30 = float(np.mean(np.asarray(scores) >= 30.0))

            # True log ratio of the strongest evidence score to the mean score.
            log_ratio_feature = float(
                np.log(max_score + eps) - np.log(avg_score + eps)
            )
        else:
            max_score = 0.0
            avg_score = 0.0
            min_score = 0.0
            std_score = 0.0
            median_score = 0.0
            top_k_avg = 0.0
            top1_top2_gap = 0.0
            prop_above_20 = 0.0
            prop_above_30 = 0.0
            log_ratio_feature = 0.0

        sent_len = len(sentence.split())
        evidence_len = len(evidence.split())
        num_segments = len(segments)

        features = np.array(
            [
                max_score,
                avg_score,
                min_score,
                std_score,
                median_score,
                top_k_avg,
                top1_top2_gap,
                prop_above_20,
                prop_above_30,
                log_ratio_feature,
                num_segments,
                sent_len,
                evidence_len,
            ],
            dtype=float,
        )

        return features, len(segments)

    def fit(self, data, tokenizer, nli_model, max_samples=None):
        X = []
        y = []

        subset = data if max_samples is None else data[:max_samples]

        for idx, item in enumerate(subset, start=1):
            print(f"DDRE training sample {idx}/{len(subset)}")

            feat, _ = self.featurize(
                item["sentence"],
                item["wiki_bio_text"],
                tokenizer,
                nli_model,
            )
            X.append(feat)
            y.append(item["label"])

        X = np.vstack(X)
        y = np.asarray(y, dtype=int)

        factual_count = int(np.sum(y == 1))
        hallucinated_count = int(np.sum(y == 0))
        total = len(y)

        if factual_count == 0 or hallucinated_count == 0:
            raise ValueError("DDRE training data must contain both classes.")

        self.p_factual_prior = factual_count / total
        self.p_hallucinated_prior = hallucinated_count / total

        self.model.fit(X, y)
        return self

    def predict_one(self, sentence, evidence, tokenizer, nli_model, threshold=None):
        if self.p_factual_prior is None or self.p_hallucinated_prior is None:
            raise RuntimeError("DDREModel must be fit before prediction.")

        x, nli_calls = self.featurize(sentence, evidence, tokenizer, nli_model)
        probs = self.model.predict_proba(x.reshape(1, -1))[0]

        classes = list(self.model.classes_)
        p_hallucinated_post = float(probs[classes.index(0)])
        p_factual_post = float(probs[classes.index(1)])

        eps = 1e-12
        posterior_odds = p_factual_post / max(p_hallucinated_post, eps)
        prior_odds_correction = self.p_hallucinated_prior / self.p_factual_prior
        density_ratio = posterior_odds * prior_odds_correction

        # Convert density ratio back to posterior probability using empirical priors.
        numerator = density_ratio * self.p_factual_prior
        denominator = numerator + self.p_hallucinated_prior
        p_factual = numerator / max(denominator, eps)

        decision_threshold = self.threshold if threshold is None else threshold
        pred = 1 if p_factual >= decision_threshold else 0

        return {
            "prediction": pred,
            "p_factual": float(p_factual),
            "p_hallucinated": float(1.0 - p_factual),
            "density_ratio": float(density_ratio),
            "posterior_odds": float(posterior_odds),
            "nli_calls": int(nli_calls),
        }
