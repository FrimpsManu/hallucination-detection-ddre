import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score=None, positive_label=0):
    """Compute classification metrics with hallucination as the default positive class."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    if y_score is not None and len(set(y_true)) == 2:
        # y_score must represent probability/score for the requested positive class.
        y_binary = np.asarray([1 if y == positive_label else 0 for y in y_true])
        metrics["roc_auc"] = roc_auc_score(y_binary, y_score)
        metrics["pr_auc"] = average_precision_score(y_binary, y_score)

    return metrics


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric="f1",
    positive_label=0,
    n_bootstrap=2000,
    confidence=0.95,
    random_state=42,
):
    """Non-parametric bootstrap confidence interval for a classification metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {"lower": 0.0, "upper": 0.0, "confidence": confidence}

    rng = np.random.default_rng(random_state)
    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        yp = y_pred[idx]

        if metric == "f1":
            value = f1_score(yt, yp, pos_label=positive_label, zero_division=0)
        elif metric == "precision":
            value = precision_score(yt, yp, pos_label=positive_label, zero_division=0)
        elif metric == "recall":
            value = recall_score(yt, yp, pos_label=positive_label, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(yt, yp)
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")

        values.append(value)

    alpha = 1.0 - confidence
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))

    return {
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def latency_metrics(sample_times, nli_calls=None, avg_steps=None):
    sample_times = np.asarray(sample_times, dtype=float)

    if sample_times.size == 0:
        results = {
            "total_inference_time": 0.0,
            "avg_inference_time_per_sample": 0.0,
            "p50_inference_time": 0.0,
            "p95_inference_time": 0.0,
        }
    else:
        results = {
            "total_inference_time": float(np.sum(sample_times)),
            "avg_inference_time_per_sample": float(np.mean(sample_times)),
            "p50_inference_time": float(np.percentile(sample_times, 50)),
            "p95_inference_time": float(np.percentile(sample_times, 95)),
        }

    if avg_steps is not None:
        results["avg_steps"] = float(avg_steps)

    if nli_calls is not None:
        calls = np.asarray(nli_calls, dtype=float)
        results["total_nli_calls"] = int(np.sum(calls))
        results["avg_nli_calls_per_sample"] = float(np.mean(calls)) if calls.size else 0.0

    return results
