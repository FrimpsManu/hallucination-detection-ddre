import time
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def _safe_corr(fn, x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(fn(x, y)[0])


def wang_pr_auc(y_binary, score):
    """Match Wang released code: precision_recall_curve followed by auc(recall, precision).

    Public because the confirmatory bootstrap must use exactly this definition.
    A second, subtly different PR-AUC would make the interval describe a
    different estimand from the point estimate it is meant to bracket, so both
    paths call this one function.
    """
    precision, recall, _ = precision_recall_curve(y_binary, score)
    return float(auc(recall, precision))


def summarize_method(records, results, elapsed_seconds=None):
    """Paper-aligned quality and computational-overhead metrics."""
    if len(records) != len(results):
        raise ValueError("records/results length mismatch")

    y_true = np.asarray([r.label for r in records], dtype=int)
    y_pred = np.asarray([r.prediction for r in results], dtype=int)
    p_factual = np.asarray([r.p_factual for r in results], dtype=float)
    p_nonfact = 1.0 - p_factual

    nonfact_true = 1 - y_true
    factual_true = y_true

    true_hallucinated = int(np.sum(y_true == 0))
    true_factual = int(np.sum(y_true == 1))
    predicted_hallucinated = int(np.sum(y_pred == 0))
    predicted_factual = int(np.sum(y_pred == 1))

    metrics = {
        "sentences": int(len(records)),
        "class_distribution": {
            "hallucinated": true_hallucinated,
            "factual": true_factual,
            "factual_prevalence": float(true_factual / len(y_true)) if len(y_true) else 0.0,
        },
        "prediction_distribution": {
            "hallucinated": predicted_hallucinated,
            "factual": predicted_factual,
            "predicted_factual_fraction": float(predicted_factual / len(y_pred)) if len(y_pred) else 0.0,
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix_labels_0_hallucinated_1_factual": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "nonfactual": {
            "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "auc_pr": wang_pr_auc(nonfact_true, p_nonfact),
            "average_precision": float(average_precision_score(nonfact_true, p_nonfact)),
        },
        "factual": {
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "auc_pr": wang_pr_auc(factual_true, p_factual),
            "average_precision": float(average_precision_score(factual_true, p_factual)),
        },
    }

    if len(np.unique(y_true)) == 2:
        metrics["nonfactual"]["roc_auc"] = float(
            roc_auc_score(nonfact_true, p_nonfact)
        )
        metrics["factual"]["roc_auc"] = float(
            roc_auc_score(factual_true, p_factual)
        )

    passage_gold = defaultdict(list)
    passage_pred = defaultdict(list)
    for record, result in zip(records, results):
        passage_gold[record.passage_index].append(record.label)
        passage_pred[record.passage_index].append(result.p_factual)

    passage_ids = sorted(passage_gold)
    gold_passage_scores = [float(np.mean(passage_gold[i])) for i in passage_ids]
    pred_passage_scores = [float(np.mean(passage_pred[i])) for i in passage_ids]
    metrics["passage_level"] = {
        "passages": len(passage_ids),
        "pearson": _safe_corr(pearsonr, gold_passage_scores, pred_passage_scores),
        "spearman": _safe_corr(spearmanr, gold_passage_scores, pred_passage_scores),
    }

    documents = np.asarray([r.documents_used for r in results], dtype=float)
    nli_calls = np.asarray([r.nli_calls for r in results], dtype=float)
    total_subclaims = sum(len(record.subclaims) for record in records)
    metrics["efficiency"] = {
        "total_retrieved_documents": int(np.sum(documents)),
        "avg_retrieved_documents_per_sentence": float(np.mean(documents)) if len(documents) else 0.0,
        "p50_retrieved_documents_per_sentence": float(np.percentile(documents, 50)) if len(documents) else 0.0,
        "p95_retrieved_documents_per_sentence": float(np.percentile(documents, 95)) if len(documents) else 0.0,
        "avg_retrieved_documents_per_subclaim": (
            float(np.sum(documents) / total_subclaims) if total_subclaims else 0.0
        ),
        "total_nli_span_calls": int(np.sum(nli_calls)),
        "avg_nli_span_calls_per_sentence": float(np.mean(nli_calls)) if len(nli_calls) else 0.0,
    }
    if elapsed_seconds is not None:
        metrics["efficiency"]["wall_clock_seconds"] = float(elapsed_seconds)
        metrics["efficiency"]["avg_wall_clock_seconds_per_sentence"] = (
            float(elapsed_seconds / len(records)) if records else 0.0
        )

    metrics["balanced_pr_auc"] = 0.5 * (
        metrics["nonfactual"]["auc_pr"] + metrics["factual"]["auc_pr"]
    )
    return metrics


def evaluate_detector(detector, records, scorer, *, description, use_cache=True):
    results = []
    start = time.perf_counter()
    for record in tqdm(records, desc=description, unit="sentence"):
        results.append(detector.detect_sentence(record, scorer, use_cache=use_cache))
    elapsed = time.perf_counter() - start
    metrics = summarize_method(records, results, elapsed_seconds=elapsed)
    return metrics, results


def prediction_rows(method_name, records, results):
    rows = []
    for record, result in zip(records, results):
        rows.append(
            {
                "method": method_name,
                "passage_index": record.passage_index,
                "sentence_index": record.sentence_index,
                "gold_label": record.label,
                "p_factual": result.p_factual,
                "prediction": result.prediction,
                "retrieved_documents": result.documents_used,
                "nli_span_calls": result.nli_calls,
            }
        )
    return rows
