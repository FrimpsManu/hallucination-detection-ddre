import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline_core import build_nbc_features, predict_one_sentence_iterative
from src.ddre_core import DDREModel
from src.evaluation import (
    bootstrap_metric_ci,
    classification_metrics,
    latency_metrics,
)


RANDOM_STATE = 42
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def load_data():
    with open("data/processed/processed_sentences.json", "r", encoding="utf-8") as f:
        return json.load(f)


def group_train_val_test_split(
    data,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=RANDOM_STATE,
):
    """Split by biography ID so evidence from one biography cannot leak across sets."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    groups = np.asarray([item["wiki_bio_test_idx"] for item in data])
    indices = np.arange(len(data))

    outer = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(outer.split(indices, groups=groups))

    temp_groups = groups[temp_idx]
    relative_val_size = val_size / (val_size + test_size)

    inner = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=random_state + 1,
    )
    val_rel_idx, test_rel_idx = next(
        inner.split(temp_idx, groups=temp_groups)
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    train_groups = {item["wiki_bio_test_idx"] for item in train_data}
    val_groups = {item["wiki_bio_test_idx"] for item in val_data}
    test_groups = {item["wiki_bio_test_idx"] for item in test_data}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    metadata = {
        "random_state": random_state,
        "train_sentences": len(train_data),
        "validation_sentences": len(val_data),
        "test_sentences": len(test_data),
        "train_biographies": len(train_groups),
        "validation_biographies": len(val_groups),
        "test_biographies": len(test_groups),
        "train_bio_ids": sorted(train_groups),
        "validation_bio_ids": sorted(val_groups),
        "test_bio_ids": sorted(test_groups),
    }

    return train_data, val_data, test_data, metadata


def factual_prior(data):
    if not data:
        return 0.5
    return sum(item["label"] == 1 for item in data) / len(data)


def run_baseline(eval_data, tokenizer, model, pos_features, neg_features, prior):
    print("\nRunning sequential Bayesian baseline...\n")

    y_true = []
    y_pred = []
    hallucination_scores = []
    sample_times = []
    nli_calls = []
    steps_used = []

    for i, sample in enumerate(eval_data):
        start = time.perf_counter()
        result = predict_one_sentence_iterative(
            sentence=sample["sentence"],
            evidence=sample["wiki_bio_text"],
            tokenizer=tokenizer,
            model=model,
            pos_features=pos_features,
            neg_features=neg_features,
            P0=prior,
            C_M=28,
            C_FA=96,
            C_retrieve=1,
        )
        elapsed = time.perf_counter() - start

        y_true.append(sample["label"])
        y_pred.append(result["prediction"])
        hallucination_scores.append(1.0 - result["posterior"])
        sample_times.append(elapsed)
        nli_calls.append(result["nli_calls"])
        steps_used.append(result["steps_used"])

        print(
            f"[Baseline {i + 1}/{len(eval_data)}] "
            f"Gold={sample['label']} Pred={result['prediction']} "
            f"P(factual)={result['posterior']:.4f} Steps={result['steps_used']}"
        )

    class_results = classification_metrics(
        y_true,
        y_pred,
        y_score=hallucination_scores,
        positive_label=0,
    )
    class_results["f1_95_ci"] = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric="f1",
        positive_label=0,
        random_state=RANDOM_STATE,
    )
    lat_results = latency_metrics(
        sample_times,
        nli_calls=nli_calls,
        avg_steps=float(np.mean(steps_used)) if steps_used else 0.0,
    )

    return {
        **class_results,
        **lat_results,
        "positive_class": "hallucinated",
        "initial_factual_prior": prior,
    }


def collect_ddre_predictions(ddre, data, tokenizer, model, threshold):
    y_true = []
    y_pred = []
    p_factual = []
    density_ratios = []
    sample_times = []
    nli_calls = []

    for i, sample in enumerate(data):
        start = time.perf_counter()
        result = ddre.predict_one(
            sample["sentence"],
            sample["wiki_bio_text"],
            tokenizer,
            model,
            threshold=threshold,
        )
        elapsed = time.perf_counter() - start

        y_true.append(sample["label"])
        y_pred.append(result["prediction"])
        p_factual.append(result["p_factual"])
        density_ratios.append(result["density_ratio"])
        sample_times.append(elapsed)
        nli_calls.append(result["nli_calls"])

        print(
            f"[DDRE {i + 1}/{len(data)}] "
            f"Gold={sample['label']} Pred={result['prediction']} "
            f"P(factual)={result['p_factual']:.4f} "
            f"DensityRatio={result['density_ratio']:.4f}"
        )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "p_factual": p_factual,
        "density_ratios": density_ratios,
        "sample_times": sample_times,
        "nli_calls": nli_calls,
    }


def select_ddre_threshold(ddre, val_data, tokenizer, model):
    """Tune threshold on validation data only; the test set remains untouched."""
    print("\nComputing DDRE validation probabilities...\n")

    y_true = []
    p_factual = []

    for i, sample in enumerate(val_data):
        result = ddre.predict_one(
            sample["sentence"],
            sample["wiki_bio_text"],
            tokenizer,
            model,
            threshold=0.5,
        )
        y_true.append(sample["label"])
        p_factual.append(result["p_factual"])
        print(f"[Validation {i + 1}/{len(val_data)}] P(factual)={result['p_factual']:.4f}")

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    candidates = []

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in p_factual]
        metrics = classification_metrics(
            y_true,
            y_pred,
            y_score=[1.0 - p for p in p_factual],
            positive_label=0,
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "hallucination_f1": metrics["f1_score"],
                "hallucination_precision": metrics["precision"],
                "hallucination_recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    # Primary criterion: hallucination F1. Tie-break with macro-F1, then precision.
    best = max(
        candidates,
        key=lambda x: (
            x["hallucination_f1"],
            x["macro_f1"],
            x["hallucination_precision"],
        ),
    )

    return best["threshold"], candidates


def run_ddre(train_data, val_data, test_data, tokenizer, model):
    print("\nTraining classifier-based density-ratio estimator...\n")
    ddre = DDREModel(random_state=RANDOM_STATE)

    train_start = time.perf_counter()
    ddre.fit(train_data, tokenizer, model)
    training_time = time.perf_counter() - train_start

    threshold, validation_candidates = select_ddre_threshold(
        ddre,
        val_data,
        tokenizer,
        model,
    )

    print(f"\nSelected DDRE threshold from validation set: {threshold:.2f}\n")

    test = collect_ddre_predictions(
        ddre,
        test_data,
        tokenizer,
        model,
        threshold,
    )

    hallucination_scores = [1.0 - p for p in test["p_factual"]]
    class_results = classification_metrics(
        test["y_true"],
        test["y_pred"],
        y_score=hallucination_scores,
        positive_label=0,
    )
    class_results["f1_95_ci"] = bootstrap_metric_ci(
        test["y_true"],
        test["y_pred"],
        metric="f1",
        positive_label=0,
        random_state=RANDOM_STATE,
    )

    lat_results = latency_metrics(
        test["sample_times"],
        nli_calls=test["nli_calls"],
        avg_steps=1.0,
    )

    return {
        **class_results,
        **lat_results,
        "positive_class": "hallucinated",
        "selected_threshold": threshold,
        "validation_threshold_search": validation_candidates,
        "training_time_seconds": training_time,
        "factual_training_prior": ddre.p_factual_prior,
        "hallucinated_training_prior": ddre.p_hallucinated_prior,
        "density_ratio_definition": "p(x|factual) / p(x|hallucinated)",
        "estimator": "classifier-based density-ratio estimation via logistic regression with empirical-prior correction",
    }


def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    print("Loading NLI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    data = load_data()
    train_data, val_data, test_data, split_metadata = group_train_val_test_split(data)

    print("\nSplit summary:")
    print(json.dumps({k: v for k, v in split_metadata.items() if not k.endswith("_ids")}, indent=2))

    print("\nBuilding Bayesian score distributions from training data only...")
    pos_features, neg_features = build_nbc_features(
        train_data,
        tokenizer,
        model,
        max_samples=None,
    )

    prior = factual_prior(train_data)

    baseline_results = run_baseline(
        test_data,
        tokenizer,
        model,
        pos_features,
        neg_features,
        prior,
    )

    ddre_results = run_ddre(
        train_data,
        val_data,
        test_data,
        tokenizer,
        model,
    )

    comparison = {
        "experiment_version": "paper-v1-corrected",
        "random_state": RANDOM_STATE,
        "model_name": MODEL_NAME,
        "split": split_metadata,
        "label_definition": {
            "1": "factual (SelfCheckGPT label == accurate)",
            "0": "hallucinated (minor or major inaccuracy)",
        },
        "baseline": baseline_results,
        "ddre": ddre_results,
    }

    os.makedirs("results", exist_ok=True)

    with open("results/nbc_features.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_sentences": len(train_data),
                "factual_prior": prior,
                "pos_features": pos_features,
                "neg_features": neg_features,
            },
            f,
            indent=2,
        )

    with open("results/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 78)
    print("FINAL TEST-SET COMPARISON")
    print("=" * 78)
    print("Sequential Bayesian baseline")
    print(f"Hallucination Precision: {baseline_results['precision']:.4f}")
    print(f"Hallucination Recall:    {baseline_results['recall']:.4f}")
    print(f"Hallucination F1:        {baseline_results['f1_score']:.4f}")
    print(f"Macro F1:                {baseline_results['macro_f1']:.4f}")
    print(f"Avg NLI calls/sample:    {baseline_results['avg_nli_calls_per_sample']:.2f}")
    print(f"P95 inference latency:   {baseline_results['p95_inference_time']:.4f}s")
    print("-" * 78)
    print("Classifier-based density-ratio estimator")
    print(f"Threshold (validation):  {ddre_results['selected_threshold']:.2f}")
    print(f"Hallucination Precision: {ddre_results['precision']:.4f}")
    print(f"Hallucination Recall:    {ddre_results['recall']:.4f}")
    print(f"Hallucination F1:        {ddre_results['f1_score']:.4f}")
    print(f"Macro F1:                {ddre_results['macro_f1']:.4f}")
    print(f"Avg NLI calls/sample:    {ddre_results['avg_nli_calls_per_sample']:.2f}")
    print(f"P95 inference latency:   {ddre_results['p95_inference_time']:.4f}s")
    print("=" * 78)
    print("Results written to results/comparison_results.json")


if __name__ == "__main__":
    main()
