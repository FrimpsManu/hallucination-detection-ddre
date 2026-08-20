import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baseline_core import build_nbc_features, predict_one_sentence_iterative
from src.ddre_core import DDREModel
from src.evaluation import (
    bootstrap_metric_ci,
    classification_metrics,
    latency_metrics,
)
from src.utils import EntailmentScorer


RANDOM_STATE = 42
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hallucination detection experiment: Bayesian baseline vs density-ratio estimator."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small debugging experiment (80 train / 40 validation / 40 test). "
            "Smoke-test results are written to separate files and must not be used in the paper."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="NLI batch size for batched feature precomputation. Defaults to 16 on CUDA and 8 on CPU.",
    )
    parser.add_argument(
        "--cache-path",
        default="results/nli_cache.sqlite",
        help="Persistent SQLite cache for deterministic NLI entailment scores.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete the persistent NLI cache before running.",
    )
    return parser.parse_args()


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
    val_rel_idx, test_rel_idx = next(inner.split(temp_idx, groups=temp_groups))

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


def stratified_subset(data, n_samples, random_state):
    """Deterministic class-stratified subset used only for smoke testing."""
    if len(data) <= n_samples:
        return list(data)

    labels = [item["label"] for item in data]
    if len(set(labels)) < 2:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(data), size=n_samples, replace=False)
        return [data[i] for i in sorted(chosen)]

    _, subset = train_test_split(
        data,
        test_size=n_samples,
        random_state=random_state,
        stratify=labels,
    )
    return list(subset)


def factual_prior(data):
    if not data:
        return 0.5
    return sum(item["label"] == 1 for item in data) / len(data)


def synchronize_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_baseline(eval_data, scorer, pos_features, neg_features, prior):
    print("\nRunning sequential Bayesian baseline on the untouched test set...")

    y_true = []
    y_pred = []
    hallucination_scores = []
    sample_times = []
    nli_calls = []
    steps_used = []

    progress = tqdm(eval_data, desc="Bayesian test inference", unit="sample")
    for sample in progress:
        synchronize_if_cuda()
        start = time.perf_counter()
        result = predict_one_sentence_iterative(
            sentence=sample["sentence"],
            evidence=sample["wiki_bio_text"],
            scorer=scorer,
            pos_features=pos_features,
            neg_features=neg_features,
            P0=prior,
            C_M=28,
            C_FA=96,
            C_retrieve=1,
            use_cache=False,
        )
        synchronize_if_cuda()
        elapsed = time.perf_counter() - start

        y_true.append(sample["label"])
        y_pred.append(result["prediction"])
        hallucination_scores.append(1.0 - result["posterior"])
        sample_times.append(elapsed)
        nli_calls.append(result["nli_calls"])
        steps_used.append(result["steps_used"])
        progress.set_postfix(steps=result["steps_used"])

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
        "latency_cache_used": False,
    }


def collect_ddre_predictions(ddre, data, scorer, threshold, *, use_cache):
    y_true = []
    y_pred = []
    p_factual = []
    density_ratios = []
    sample_times = []
    nli_calls = []

    description = "DDRE cached validation" if use_cache else "DDRE test inference"
    progress = tqdm(data, desc=description, unit="sample")

    for sample in progress:
        synchronize_if_cuda()
        start = time.perf_counter()
        result = ddre.predict_one(
            sample["sentence"],
            sample["wiki_bio_text"],
            scorer,
            threshold=threshold,
            use_cache=use_cache,
        )
        synchronize_if_cuda()
        elapsed = time.perf_counter() - start

        y_true.append(sample["label"])
        y_pred.append(result["prediction"])
        p_factual.append(result["p_factual"])
        density_ratios.append(result["density_ratio"])
        sample_times.append(elapsed)
        nli_calls.append(result["nli_calls"])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "p_factual": p_factual,
        "density_ratios": density_ratios,
        "sample_times": sample_times,
        "nli_calls": nli_calls,
    }


def select_ddre_threshold(ddre, val_data, scorer):
    """Tune threshold on cached validation features only; test remains untouched."""
    print("\nSelecting DDRE threshold on the validation set...")

    validation = collect_ddre_predictions(
        ddre,
        val_data,
        scorer,
        threshold=0.5,
        use_cache=True,
    )

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    candidates = []

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in validation["p_factual"]]
        metrics = classification_metrics(
            validation["y_true"],
            y_pred,
            y_score=[1.0 - p for p in validation["p_factual"]],
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

    best = max(
        candidates,
        key=lambda x: (
            x["hallucination_f1"],
            x["macro_f1"],
            x["hallucination_precision"],
        ),
    )

    return best["threshold"], candidates


def run_ddre(train_data, val_data, test_data, scorer):
    print("\nTraining classifier-based density-ratio estimator from cached features...")
    ddre = DDREModel(random_state=RANDOM_STATE)

    train_start = time.perf_counter()
    ddre.fit(train_data, scorer)
    fit_time = time.perf_counter() - train_start

    threshold, validation_candidates = select_ddre_threshold(
        ddre,
        val_data,
        scorer,
    )

    print(f"Selected DDRE threshold from validation set: {threshold:.2f}")

    # The final test evaluation intentionally bypasses the cache. This keeps
    # latency metrics honest while training/validation remain fast and resumable.
    test = collect_ddre_predictions(
        ddre,
        test_data,
        scorer,
        threshold,
        use_cache=False,
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
        "classifier_fit_time_seconds_excluding_nli_precompute": fit_time,
        "factual_training_prior": ddre.p_factual_prior,
        "hallucinated_training_prior": ddre.p_hallucinated_prior,
        "density_ratio_definition": "p(x|factual) / p(x|hallucinated)",
        "estimator": "classifier-based density-ratio estimation via logistic regression with empirical-prior correction",
        "latency_cache_used": False,
    }


def clear_cache_files(cache_path):
    for suffix in ("", "-wal", "-shm"):
        path = cache_path + suffix
        if os.path.exists(path):
            os.remove(path)


def main():
    args = parse_args()

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    os.makedirs("results", exist_ok=True)

    if args.rebuild_cache:
        clear_cache_files(args.cache_path)

    print("Loading NLI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or (16 if torch.cuda.is_available() else 8)
    print(f"Device: {device}")
    print(f"Batched NLI precompute size: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    scorer = EntailmentScorer(
        tokenizer=tokenizer,
        model=model,
        model_name=MODEL_NAME,
        cache_path=args.cache_path,
        batch_size=batch_size,
    )

    try:
        data = load_data()
        train_data, val_data, test_data, split_metadata = group_train_val_test_split(data)

        if args.smoke_test:
            train_data = stratified_subset(train_data, 80, RANDOM_STATE)
            val_data = stratified_subset(val_data, 40, RANDOM_STATE + 1)
            test_data = stratified_subset(test_data, 40, RANDOM_STATE + 2)
            print("\n*** SMOKE TEST MODE: results are for debugging only, not the paper. ***")

        execution_sizes = {
            "train_sentences_used": len(train_data),
            "validation_sentences_used": len(val_data),
            "test_sentences_used": len(test_data),
        }

        print("\nSplit summary:")
        summary = {k: v for k, v in split_metadata.items() if not k.endswith("_ids")}
        summary.update(execution_sizes)
        print(json.dumps(summary, indent=2))

        print(
            "\nPrecomputing reusable NLI features for TRAIN + VALIDATION only. "
            "This stage is batched and resumable."
        )
        synchronize_if_cuda()
        cache_start = time.perf_counter()
        cache_stats = scorer.warm_cache(train_data + val_data)
        synchronize_if_cuda()
        cache_precompute_time = time.perf_counter() - cache_start
        cache_stats["elapsed_seconds_this_run"] = cache_precompute_time
        print("Cache summary:")
        print(json.dumps(cache_stats, indent=2))

        print("\nBuilding Bayesian score distributions from cached training features...")
        pos_features, neg_features = build_nbc_features(
            train_data,
            scorer,
            max_samples=None,
        )

        prior = factual_prior(train_data)

        baseline_results = run_baseline(
            test_data,
            scorer,
            pos_features,
            neg_features,
            prior,
        )

        ddre_results = run_ddre(
            train_data,
            val_data,
            test_data,
            scorer,
        )

        comparison = {
            "experiment_version": "paper-v2-batched-cache",
            "run_mode": "smoke-test" if args.smoke_test else "full-publication-experiment",
            "random_state": RANDOM_STATE,
            "model_name": MODEL_NAME,
            "device": str(device),
            "batch_size": batch_size,
            "nli_cache": {
                "path": args.cache_path,
                "training_validation_only": True,
                "final_test_latency_uses_cache": False,
                **cache_stats,
            },
            "split": split_metadata,
            "execution_sizes": execution_sizes,
            "label_definition": {
                "1": "factual (SelfCheckGPT label == accurate)",
                "0": "hallucinated (minor or major inaccuracy)",
            },
            "baseline": baseline_results,
            "ddre": ddre_results,
        }

        if args.smoke_test:
            comparison_path = "results/smoke_comparison_results.json"
            features_path = "results/smoke_nbc_features.json"
        else:
            comparison_path = "results/comparison_results.json"
            features_path = "results/nbc_features.json"

        with open(features_path, "w", encoding="utf-8") as f:
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

        with open(comparison_path, "w", encoding="utf-8") as f:
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
        print(f"Results written to {comparison_path}")

    finally:
        scorer.close()


if __name__ == "__main__":
    main()
