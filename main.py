import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.baseline_core import build_nbc_features, predict_one_sentence


def load_data():
    with open("data/processed/processed_sentences.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading model...")
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    data = load_data()

    print("Building NBC feature distributions...")
    pos_features, neg_features = build_nbc_features(data, tokenizer, model, max_samples=50)

    print("Positive feature buckets:", pos_features)
    print("Negative feature buckets:", neg_features)
    print()

    os.makedirs("results", exist_ok=True)
    with open("results/nbc_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pos_features": pos_features,
                "neg_features": neg_features,
            },
            f,
            indent=2,
        )

    print("Running test on 50 samples...\n")

    correct = 0
    total = min(50, len(data))

    for i in range(total):
        sample = data[i]

        sentence = sample["sentence"]
        evidence = sample["wiki_bio_text"]
        gold = sample["label"]

        P, score, bucket = predict_one_sentence(
            sentence,
            evidence,
            tokenizer,
            model,
            pos_features,
            neg_features,
        )

        pred = 1 if P >= 0.5 else 0

        if pred == gold:
            correct += 1

        print(
            f"[{i+1}] Gold: {gold} | Pred: {pred} | "
            f"P: {P:.4f} | Score: {score:.2f} | Bucket: {bucket}"
        )

    print(f"\nAccuracy on first {total} samples: {correct / total:.4f}")


if __name__ == "__main__":
    main()