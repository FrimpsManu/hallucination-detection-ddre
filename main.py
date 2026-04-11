import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.baseline_core import build_nbc_features, predict_one_sentence_iterative
from src.ddre_core import DDREModel


def load_data():
    with open("data/processed/processed_sentences.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(eval_data, tokenizer, model, pos_features, neg_features):
    print("\nRunning iterative baseline...\n")

    correct = 0
    total = len(eval_data)
    total_steps = 0

    start_time = time.time()

    for i, sample in enumerate(eval_data):
        sentence = sample["sentence"]
        evidence = sample["wiki_bio_text"]
        gold = sample["label"]

        result = predict_one_sentence_iterative(
            sentence=sentence,
            evidence=evidence,
            tokenizer=tokenizer,
            model=model,
            pos_features=pos_features,
            neg_features=neg_features,
            P0=0.5,
            C_M=28,
            C_FA=96,
            C_retrieve=1,
        )

        pred = result["prediction"]
        P = result["posterior"]
        steps_used = result["steps_used"]

        total_steps += steps_used

        if pred == gold:
            correct += 1

        print(
            f"[Baseline {i+1}] Gold: {gold} | Pred: {pred} | "
            f"P: {P:.4f} | Steps: {steps_used}"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    avg_steps = total_steps / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "avg_steps": avg_steps,
        "elapsed_time": elapsed,
    }


def run_ddre(train_data, eval_data, tokenizer, model):
    print("\nTraining DDRE model...\n")

    ddre = DDREModel()
    ddre.fit(train_data, tokenizer, model, max_samples=len(train_data))

    print("\nRunning DDRE evaluation...\n")

    correct = 0
    total = len(eval_data)

    start_time = time.time()

    for i, sample in enumerate(eval_data):
        sentence = sample["sentence"]
        evidence = sample["wiki_bio_text"]
        gold = sample["label"]

        result = ddre.predict_one(sentence, evidence, tokenizer, model)

        pred = result["prediction"]
        ratio = result["ratio"]
        p_factual = result["p_factual"]

        if pred == gold:
            correct += 1

        print(
            f"[DDRE {i+1}] Gold: {gold} | Pred: {pred} | "
            f"Ratio: {ratio:.4f} | P_factual: {p_factual:.4f}"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "elapsed_time": elapsed,
    }


def main():
    print("Loading model...")
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    data = load_data()

    train_samples = 50
    eval_samples = 20

    train_data = data[:train_samples]
    eval_data = data[train_samples:train_samples + eval_samples]

    print("Building NBC feature distributions...")
    pos_features, neg_features = build_nbc_features(
        data=train_data,
        tokenizer=tokenizer,
        model=model,
        max_samples=len(train_data),
    )

    print("Positive feature buckets:", pos_features)
    print("Negative feature buckets:", neg_features)

    os.makedirs("results", exist_ok=True)

    with open("results/nbc_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_samples": train_samples,
                "pos_features": pos_features,
                "neg_features": neg_features,
            },
            f,
            indent=2,
        )

    baseline_results = run_baseline(
        eval_data=eval_data,
        tokenizer=tokenizer,
        model=model,
        pos_features=pos_features,
        neg_features=neg_features,
    )

    ddre_results = run_ddre(
        train_data=train_data,
        eval_data=eval_data,
        tokenizer=tokenizer,
        model=model,
    )

    comparison = {
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "baseline": baseline_results,
        "ddre": ddre_results,
    }

    with open("results/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Baseline Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"Baseline Avg Steps: {baseline_results['avg_steps']:.2f}")
    print(f"Baseline Time: {baseline_results['elapsed_time']:.2f} seconds")
    print("-" * 70)
    print(f"DDRE Accuracy: {ddre_results['accuracy']:.4f}")
    print(f"DDRE Time: {ddre_results['elapsed_time']:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()