import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.baseline_core import build_nbc_features, predict_one_sentence_iterative
from src.ddre_core import DDREModel
from src.evaluation import classification_metrics, latency_metrics


def load_data():
    with open("data/processed/processed_sentences.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(eval_data, tokenizer, model, pos_features, neg_features):
    print("\nRunning iterative baseline...\n")

    y_true = []
    y_pred = []

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

        y_true.append(gold)
        y_pred.append(pred)
        total_steps += steps_used

        print(
            f"[Baseline {i+1}] Gold: {gold} | Pred: {pred} | "
            f"P: {P:.4f} | Steps: {steps_used}"
        )

    end_time = time.time()

    avg_steps = total_steps / total if total > 0 else 0.0

    class_results = classification_metrics(y_true, y_pred, positive_label=0)
    lat_results = latency_metrics(start_time, end_time, total, avg_steps=avg_steps)

    return {
        **class_results,
        **lat_results,
    }
   

def run_ddre(train_data, eval_data, tokenizer, model, threshold=0.6):
    print(f"\nTraining DDRE model (threshold={threshold})...\n")

    ddre = DDREModel(threshold=threshold)
    ddre.fit(train_data, tokenizer, model, max_samples=len(train_data))

    print("\nRunning DDRE evaluation...\n")

    y_true = []
    y_pred = []

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

        y_true.append(gold)
        y_pred.append(pred)

        print(
            f"[DDRE {i+1}] Gold: {gold} | Pred: {pred} | "
            f"Ratio: {ratio:.4f} | P_factual: {p_factual:.4f}"
        )

    end_time = time.time()

    class_results = classification_metrics(y_true, y_pred, positive_label=0)
    lat_results = latency_metrics(start_time, end_time, total, avg_steps=1.0)

    return {
        **class_results,
        **lat_results,
        "threshold": threshold,
    }


def main():
    print("Loading model...")
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    data = load_data()

    train_samples = 200
    eval_samples = 100

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

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    ddre_results_all = []

    for th in thresholds:
        result = run_ddre(
            train_data=train_data,
            eval_data=eval_data,
            tokenizer=tokenizer,
            model=model,
            threshold=th,
        )
        ddre_results_all.append(result)

    best_ddre = max(ddre_results_all, key=lambda x: x["f1_score"])

    comparison = {
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "baseline": baseline_results,
        "ddre_all": ddre_results_all,
        "best_ddre": best_ddre,
    }

    with open("results/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print("BASELINE")
    print(f"Precision: {baseline_results['precision']:.4f}")
    print(f"Recall:    {baseline_results['recall']:.4f}")
    print(f"F1-score:  {baseline_results['f1_score']:.4f}")
    print(f"Avg Steps: {baseline_results['avg_steps']:.2f}")
    print(f"Total Time: {baseline_results['total_inference_time']:.2f} seconds")
    print(f"Avg Time/Sample: {baseline_results['avg_inference_time_per_sample']:.4f} seconds")
    print("-" * 70)
    print("BEST DDRE")
    print(f"Threshold: {best_ddre['threshold']:.2f}")
    print(f"Precision: {best_ddre['precision']:.4f}")
    print(f"Recall:    {best_ddre['recall']:.4f}")
    print(f"F1-score:  {best_ddre['f1_score']:.4f}")
    print(f"Avg Steps: {best_ddre['avg_steps']:.2f}")
    print(f"Total Time: {best_ddre['total_inference_time']:.2f} seconds")
    print(f"Avg Time/Sample: {best_ddre['avg_inference_time_per_sample']:.4f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()