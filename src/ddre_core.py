import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import split_text, get_entailment_score


class DDREModel:
    def __init__(self, threshold=0.6):
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.threshold = threshold

    def featurize(self, sentence, evidence, tokenizer, nli_model):
        segments = split_text(evidence)

        scores = []
        for seg in segments:
            score = get_entailment_score(seg, sentence, tokenizer, nli_model)
            scores.append(score)

        if scores:
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            std_score = float(np.std(scores))
            top_k_avg = float(np.mean(sorted(scores, reverse=True)[:3]))
        else:
            max_score = 0.0
            avg_score = 0.0
            min_score = 0.0
            std_score = 0.0
            top_k_avg = 0.0

        sent_len = len(sentence.split())
        evidence_len = len(evidence.split())
        num_segments = len(segments)

        return np.array(
            [
                max_score,
                avg_score,
                min_score,
                std_score,
                top_k_avg,
                num_segments,
                sent_len,
                evidence_len,
            ],
            dtype=float,
        )

    def fit(self, data, tokenizer, nli_model, max_samples=200):
        X = []
        y = []

        subset = data[:max_samples]

        for idx, item in enumerate(subset, start=1):
            print(f"DDRE training sample {idx}/{len(subset)}")

            sentence = item["sentence"]
            evidence = item["wiki_bio_text"]
            label = item["label"]

            feat = self.featurize(sentence, evidence, tokenizer, nli_model)
            X.append(feat)
            y.append(label)

        X = np.vstack(X)
        y = np.array(y)

        self.model.fit(X, y)

    def predict_one(self, sentence, evidence, tokenizer, nli_model):
        x = self.featurize(sentence, evidence, tokenizer, nli_model).reshape(1, -1)

        probs = self.model.predict_proba(x)[0]
        p_hallucinated = probs[0]
        p_factual = probs[1]

        ratio = p_factual / max(p_hallucinated, 1e-8)
        pred = 1 if p_factual >= self.threshold else 0

        return {
            "prediction": pred,
            "p_factual": float(p_factual),
            "p_hallucinated": float(p_hallucinated),
            "ratio": float(ratio),
        }