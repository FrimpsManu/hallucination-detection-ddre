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
