import json
import time
from collections import defaultdict
from dataclasses import dataclass

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


@dataclass(frozen=True)
class EvaluatedSentence:
    """One detector result with the identity of the sentence that produced it.

    ``DetectionResult`` carries no identity, so two result lists of equal length
    look "paired" even when one has been permuted. Attaching the identity **at
    evaluation time** -- inside the loop that pairs a record with its result --
    is what makes a later paired analysis genuinely paired. Reconstructing it
    afterwards from a possibly-permuted list would assume exactly the property
    that needs checking.

    Frozen so an observation cannot be edited after the fact.

    ``n_subclaims`` and ``subclaim_traces`` are the D-10 additions. They are
    ``None`` -- not empty -- when the evaluation did not record them, so a
    trace-less observation can never be mistaken for a sentence that genuinely
    had no subclaims. The pairing identity deliberately does NOT include them:
    the confirmatory bootstrap pairs on ``(passage, sentence, label)`` and that
    must not change because an observation gained instrumentation.
    """

    passage_index: int
    sentence_index: int
    gold_label: int
    result: object
    n_subclaims: int = None
    subclaim_traces: tuple = None

    @property
    def identity(self):
        return (self.passage_index, self.sentence_index, self.gold_label)

    @property
    def has_traces(self):
        return self.subclaim_traces is not None


@dataclass(frozen=True)
class SubclaimTrace:
    """One subclaim's retrieval depth, as it actually happened.

    ``documents_available`` is ``min(len(subclaim.documents), max_docs)`` -- the
    budget of the *evaluated protocol*, not the raw corpus. Calling the raw
    length "available" would overstate what the detector could ever have used
    and would make a depth of ``max_docs`` look like early stopping.
    """

    subclaim_index: int
    documents_available: int
    documents_used: int
    nli_calls: int
    p_factual: float
    prediction: int

    @property
    def is_nonempty(self):
        return self.documents_available > 0

    @property
    def declined_first_retrieval(self):
        """Non-empty, yet no document was fetched. BSE can do this; DDRE cannot."""
        return self.is_nonempty and self.documents_used == 0


class SubclaimTraceMismatch(ValueError):
    """A recorded trace does not account for the sentence result it came with."""


def _traces_for(record, subclaim_results, max_docs):
    """Build the immutable trace, checking it against the record and the result."""
    if len(subclaim_results) != len(record.subclaims):
        raise SubclaimTraceMismatch(
            f"detector returned {len(subclaim_results)} subclaim results for a "
            f"sentence with {len(record.subclaims)} subclaims "
            f"(passage={record.passage_index}, sentence={record.sentence_index}). "
            "The trace cannot be aligned with the subclaims, so the per-subclaim "
            "retrieval depths would be attributed to the wrong claims."
        )
    return tuple(
        SubclaimTrace(
            subclaim_index=index,
            documents_available=min(len(subclaim.documents), int(max_docs)),
            documents_used=int(result.documents_used),
            nli_calls=int(result.nli_calls),
            p_factual=float(result.p_factual),
            prediction=int(result.prediction),
        )
        for index, (subclaim, result) in enumerate(
            zip(record.subclaims, subclaim_results)
        )
    )


def _check_trace_totals(record, sentence_result, traces):
    """The trace must ACCOUNT for the result, not merely accompany it.

    An integrity check, never a recomputation: the sentence result returned by
    the detector is what is kept. If these disagree, the trace describes a
    different computation and every subclaim-level number derived from it would
    be wrong in a way nothing downstream could see.
    """
    where = f"(passage={record.passage_index}, sentence={record.sentence_index})"
    documents = sum(trace.documents_used for trace in traces)
    if documents != int(sentence_result.documents_used):
        raise SubclaimTraceMismatch(
            f"subclaim retrieval depths sum to {documents} but the sentence "
            f"result records {sentence_result.documents_used} documents {where}"
        )
    nli_calls = sum(trace.nli_calls for trace in traces)
    if nli_calls != int(sentence_result.nli_calls):
        raise SubclaimTraceMismatch(
            f"subclaim NLI calls sum to {nli_calls} but the sentence result "
            f"records {sentence_result.nli_calls} calls {where}"
        )
    if traces:
        # Exact equality: the sentence posterior IS one of the subclaim
        # posteriors under the current min-aggregation, not an average of them.
        minimum = min(trace.p_factual for trace in traces)
        if float(minimum) != float(sentence_result.p_factual):
            raise SubclaimTraceMismatch(
                f"the minimum subclaim p_factual is {minimum!r} but the sentence "
                f"result records {sentence_result.p_factual!r} {where}"
            )


def _run_detector(detector, records, scorer, description, use_cache, with_traces=False):
    """The single evaluation loop. Builds results and identities together.

    The observation is constructed from the *same* ``record`` that produced the
    result, in the same iteration, so the identity cannot be attached to the
    wrong result. Zipping a record list against a returned result list
    afterwards would assume the ordering that the paired analysis exists to
    verify.

    The elapsed clock now spans the dataclass construction as well. That cost is
    microseconds against an NLI forward pass, and every method pays it equally,
    so no wall-clock comparison between methods is biased by it.
    """
    results = []
    observations = []
    start = time.perf_counter()
    for record in tqdm(records, desc=description, unit="sentence"):
        # ONE pass per sentence either way. The traced form returns the trace
        # from the same call that produces the result; it never evaluates the
        # sentence a second time to obtain one.
        if with_traces:
            result, subclaim_results = detector.detect_sentence_with_trace(
                record, scorer, use_cache=use_cache
            )
            traces = _traces_for(record, subclaim_results, detector.max_docs)
            _check_trace_totals(record, result, traces)
            n_subclaims = len(record.subclaims)
        else:
            result = detector.detect_sentence(record, scorer, use_cache=use_cache)
            traces = None
            n_subclaims = None
        results.append(result)
        observations.append(
            EvaluatedSentence(
                passage_index=record.passage_index,
                sentence_index=record.sentence_index,
                gold_label=int(record.label),
                result=result,
                n_subclaims=n_subclaims,
                subclaim_traces=traces,
            )
        )
    elapsed = time.perf_counter() - start
    return results, observations, elapsed


def evaluate_detector_with_identity(
    detector, records, scorer, *, description, use_cache=True
):
    """``evaluate_detector`` plus identity-bearing observations.

    A narrow addition rather than a change to ``evaluate_detector``'s return
    signature, which several diagnostic scripts and the Gate 1 reproduction
    already depend on. Returns ``(metrics, results, observations)``; the first
    two are exactly what ``evaluate_detector`` returns.
    """
    results, observations, elapsed = _run_detector(
        detector, records, scorer, description, use_cache
    )
    metrics = summarize_method(records, results, elapsed_seconds=elapsed)
    return metrics, results, observations


def evaluate_detector_with_traces(
    detector, records, scorer, *, description, use_cache=True
):
    """``evaluate_detector_with_identity`` plus per-subclaim retrieval traces.

    Returns ``(metrics, results, observations)`` where each observation carries
    ``n_subclaims`` and its ``subclaim_traces``. The saved efficiency trace is
    therefore from the same execution as the metrics -- there is no second pass
    to disagree with.
    """
    results, observations, elapsed = _run_detector(
        detector, records, scorer, description, use_cache, with_traces=True
    )
    metrics = summarize_method(records, results, elapsed_seconds=elapsed)
    return metrics, results, observations


def evaluate_detector(detector, records, scorer, *, description, use_cache=True):
    results, _, elapsed = _run_detector(
        detector, records, scorer, description, use_cache
    )
    metrics = summarize_method(records, results, elapsed_seconds=elapsed)
    return metrics, results


def _require_traces(observations, caller):
    """Trace-bearing observations, or a loud failure. Never a silent default."""
    if not observations:
        raise SubclaimTraceMismatch(f"{caller} received no observations")
    missing = [o.identity for o in observations if not o.has_traces]
    if missing:
        raise SubclaimTraceMismatch(
            f"{caller} requires observations carrying subclaim traces; "
            f"{len(missing)} of {len(observations)} have none (first: "
            f"{missing[0]}). They came from an evaluation that did not record "
            "them -- use evaluate_detector_with_traces. Per-subclaim numbers "
            "are NOT inferred by dividing sentence totals."
        )
    return observations


def summarize_subclaim_efficiency(observations):
    """Per-subclaim retrieval accounting from the ACTUAL traces (D-10).

    Every quantity here comes from individual ``SubclaimTrace`` values. None of
    it is inferred by dividing a sentence total by a subclaim count: that would
    reconstruct a mean and invent the distribution around it, which is exactly
    the information D-10 says was being lost.
    """
    _require_traces(observations, "summarize_subclaim_efficiency")
    traces = [t for o in observations for t in o.subclaim_traces]
    if not traces:
        raise SubclaimTraceMismatch(
            "summarize_subclaim_efficiency received observations with no "
            "subclaims at all; there is nothing to summarise"
        )

    depths = [t.documents_used for t in traces]
    nonempty = [t for t in traces if t.is_nonempty]
    histogram = {}
    for depth in depths:
        histogram[str(depth)] = histogram.get(str(depth), 0) + 1

    nonempty_zero = sum(1 for t in nonempty if t.documents_used == 0)
    return {
        "total_subclaims": len(traces),
        "nonempty_subclaims": len(nonempty),
        "empty_subclaims": len(traces) - len(nonempty),
        "total_documents_used": int(sum(depths)),
        "zero_retrieval_subclaims": int(sum(1 for d in depths if d == 0)),
        "zero_retrieval_nonempty_subclaims": int(nonempty_zero),
        "zero_retrieval_nonempty_fraction": (
            float(nonempty_zero / len(nonempty)) if nonempty else None
        ),
        "one_retrieval_nonempty_subclaims": int(
            sum(1 for t in nonempty if t.documents_used == 1)
        ),
        "avg_documents_per_subclaim": float(np.mean(depths)),
        "p50_documents_per_subclaim": float(np.percentile(depths, 50)),
        "p95_documents_per_subclaim": float(np.percentile(depths, 95)),
        "max_documents_per_subclaim_observed": int(max(depths)),
        "retrieval_depth_histogram": {
            key: histogram[key] for key in sorted(histogram, key=int)
        },
        "total_nli_calls": int(sum(t.nli_calls for t in traces)),
        "avg_nli_calls_per_subclaim": float(
            np.mean([t.nli_calls for t in traces])
        ),
        "source": "per-subclaim traces from the evaluating pass",
    }


ASYMMETRY_NOTE = (
    "BSE can decline the first retrieval through its decision-theoretic "
    "should_continue rule. The current DDRE stopping band is evaluated only "
    "after an evidence update, so every non-empty evaluated subclaim has a "
    "one-document structural floor. This asymmetry disadvantages DDRE in raw "
    "retrieval counts. No floor adjustment is applied to the confirmatory "
    "efficiency endpoint."
)


def retrieval_protocol_asymmetry(bse_observations, ddre_observations, *, max_docs):
    """Report the D-02 stopping-protocol asymmetry. Reporting only.

    This is an accounting statement, not a correction. The two methods have
    different stopping protocols -- BSE may stop before document 1, DDRE tests
    its band only after an evidence update -- and the honest response at this
    stage is to measure the difference and say so, not to give DDRE BSE's
    decision rule, which would be a new algorithm.

    The DDRE floor is defined PROSPECTIVELY from the protocol: the number of
    evaluated subclaims with ``documents_available > 0``, because the current
    protocol consumes at least one document for each of them. Deriving it from
    observed depths instead would make it a description of the data rather than
    a property of the protocol, and it could then never be violated.
    """
    bse = summarize_subclaim_efficiency(bse_observations)
    ddre = summarize_subclaim_efficiency(ddre_observations)

    ddre_traces = [t for o in ddre_observations for t in o.subclaim_traces]
    # The floor is a claim about the CURRENT DDRE protocol. If a non-empty
    # subclaim used no documents, the implementation no longer matches the
    # protocol whose floor is being reported, and the report would be false.
    # BSE is deliberately NOT subject to this: for BSE, exactly this case is
    # the pre-first-fetch stop being counted.
    declined = [t for t in ddre_traces if t.declined_first_retrieval]
    if declined:
        raise SubclaimTraceMismatch(
            f"{len(declined)} DDRE subclaims had documents available but "
            "retrieved none (first at subclaim index "
            f"{declined[0].subclaim_index}). The current DDRE protocol tests "
            "its stopping band only after an evidence update, so every "
            "non-empty subclaim must consume at least one document. The "
            "implementation no longer matches the protocol whose structural "
            "floor this report states."
        )

    floor = int(ddre["nonempty_subclaims"])
    above_floor = int(ddre["total_documents_used"]) - floor
    if above_floor < 0:
        raise SubclaimTraceMismatch(
            f"DDRE used {ddre['total_documents_used']} documents across "
            f"{floor} non-empty subclaims, which is below the protocol's "
            "one-document-per-non-empty-subclaim floor"
        )

    return {
        "max_documents_per_subclaim": int(max_docs),
        "bse_official": {
            "nonempty_subclaims": bse["nonempty_subclaims"],
            "zero_retrieval_nonempty_subclaims": bse[
                "zero_retrieval_nonempty_subclaims"
            ],
            "zero_retrieval_nonempty_fraction": bse[
                "zero_retrieval_nonempty_fraction"
            ],
            "observed_total_documents": bse["total_documents_used"],
            "may_decline_first_retrieval": True,
        },
        "ddre_ulsif": {
            "nonempty_subclaims": ddre["nonempty_subclaims"],
            "zero_retrieval_nonempty_subclaims": ddre[
                "zero_retrieval_nonempty_subclaims"
            ],
            "zero_retrieval_nonempty_fraction": ddre[
                "zero_retrieval_nonempty_fraction"
            ],
            "observed_total_documents": ddre["total_documents_used"],
            "may_decline_first_retrieval": False,
        },
        "ddre_first_retrieval_floor_documents": floor,
        "ddre_first_retrieval_floor_definition": (
            "number of evaluated subclaims with documents_available > 0; the "
            "current DDRE protocol consumes at least one document for each"
        ),
        "ddre_documents_above_first_retrieval_floor": above_floor,
        "finding": "D-02",
        "status": "reported protocol asymmetry; no algorithmic change made",
        "note": ASYMMETRY_NOTE,
        "confirmatory_endpoint_unchanged": (
            "The frozen primary efficiency endpoint remains BSE minus DDRE "
            "retrieved documents per SENTENCE, with no floor adjustment. "
            "documents_above_first_retrieval_floor is DESCRIPTIVE only and "
            "never enters a claim."
        ),
    }


def prediction_rows(method_name, records, results, observations=None):
    """Sentence rows for the predictions CSV.

    ``observations`` is optional so diagnostic scripts that never built traces
    keep working. When supplied it must be trace-bearing and aligned with
    ``records``, and the D-10 per-subclaim columns are added: the formal
    experiment passes it, so the saved CSV can reconstruct subclaim efficiency.
    """
    if observations is not None:
        _require_traces(observations, "prediction_rows")
        if len(observations) != len(records):
            raise SubclaimTraceMismatch(
                f"prediction_rows got {len(observations)} observations for "
                f"{len(records)} records"
            )

    rows = []
    for index, (record, result) in enumerate(zip(records, results)):
        row = {
            "method": method_name,
            "passage_index": record.passage_index,
            "sentence_index": record.sentence_index,
            "gold_label": record.label,
            "p_factual": result.p_factual,
            "prediction": result.prediction,
            "retrieved_documents": result.documents_used,
            "nli_span_calls": result.nli_calls,
        }
        if observations is not None:
            observation = observations[index]
            expected = (record.passage_index, record.sentence_index)
            if (observation.passage_index, observation.sentence_index) != expected:
                raise SubclaimTraceMismatch(
                    f"observation {index} is for sentence "
                    f"{(observation.passage_index, observation.sentence_index)} "
                    f"but the record at that position is {expected}"
                )
            traces = observation.subclaim_traces
            row.update(
                {
                    "n_subclaims": observation.n_subclaims,
                    # Compact JSON, not repr: a reader should be able to
                    # json.loads these columns without knowing Python.
                    "subclaim_documents_used": json.dumps(
                        [t.documents_used for t in traces], separators=(",", ":")
                    ),
                    "subclaim_documents_available": json.dumps(
                        [t.documents_available for t in traces],
                        separators=(",", ":"),
                    ),
                    "subclaim_nli_calls": json.dumps(
                        [t.nli_calls for t in traces], separators=(",", ":")
                    ),
                    "zero_retrieval_nonempty_subclaims": sum(
                        1 for t in traces if t.declined_first_retrieval
                    ),
                }
            )
        rows.append(row)
    return rows
