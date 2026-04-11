import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.baseline_core import build_nbc_features, predict_one_sentence_iterative


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

    print("Running iterative baseline test on 50 samples...\n")

    correct = 0
    total = min(50, len(data))
    total_steps = 0

    start_time = time.time()

    for i in range(total):
        sample = data[i]

        sentence = sample["sentence"]
        evidence = sample["wiki_bio_text"]
        gold = sample["label"]

        results = predict_one_sentence_iterative(
            sentence,
            evidence,
            tokenizer,
            model,
            pos_features,
            neg_features,
            P0=0.5,
            C_M=28,
            C_FA=96,
            C_retrieve=1,

        )

        pred = results["prediction"]
        P = results["posterior"]
        steps_used = results["steps_used"]

        total_steps += steps_used

        if pred == gold:
            correct += 1

        print(
            f"[{i+1}] Gold: {gold} | Pred: {pred} | "
            f"P: {P:.4f} | Steps: {steps_used}"
        )

    elapsed = time.time() - start_time
    print(f"\nAccuracy on first {total} samples: {correct / total:.4f}")
    print(f"Average steps used: {total_steps / total:.2f}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()