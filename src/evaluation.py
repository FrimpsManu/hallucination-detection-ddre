import time
from sklearn.metrics import precision_score, recall_score, f1_score


def classification_metrics(y_true, y_pred, positive_label=1):
    return {
        "precision": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
    }


def latency_metrics(start_time, end_time, num_samples, avg_steps=None):
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_samples if num_samples > 0 else 0.0

    results = {
        "total_inference_time": total_time,
        "avg_inference_time_per_sample": avg_time_per_sample,
    }

    if avg_steps is not None:
        results["avg_steps"] = avg_steps

    return results