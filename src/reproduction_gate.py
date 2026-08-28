"""Wang et al. (EMNLP 2023) baseline reproduction gate.

This module turns the baseline reproduction from a report into a gate. It
holds the published Table 1 reference values, the predeclared pass/warn/fail
tolerances, and the verdict logic. It deliberately imports nothing beyond the
standard library so it can be unit tested without torch, scipy, or sklearn.

Table 1 evidence-count definition
---------------------------------
The paper states that it reports "the average number of retrieved documents at
the sentence-level", and Section 4.2 describes 3.05 and 6.22 as the average
number of retrieved documents per sentence. The reproduction comparison
therefore uses average retrieved documents per *sentence*.

Wang's released ``main.py`` prints both quantities, under variable names that
are swapped relative to their printed labels::

    sentence_search_time_list   -> appended once per subclaim
    hypothesis_search_time_list -> appended once per sentence
    print("avg_sentence_search_time:",   mean(hypothesis_search_time_list))
    print("hypothesis_avg_search_time:", mean(sentence_search_time_list))

Average documents per subclaim remains available as a diagnostic but is never
compared against Table 1.

Tolerances
----------
These thresholds are frozen before any reproduction run and are recorded in the
gate output so that a run can be shown to have been judged against them.

They are *not* derived from sampling standard error. This is a reproduction of
one fixed experiment on one fixed dataset, not an estimate of population
uncertainty: the sentences, subclaims, retrieved documents, and labels are
identical to Wang's. The only sources of divergence are environmental --
dependency and model-library versions, batching and padding, device numerics,
and any protocol deviation. A faithful reproduction should therefore land very
close to the published values, and the bands below express how much
environmental drift we are willing to call "reproduced" rather than how much
statistical noise we expect.

The correlation bands are wider than the sentence-level metrics because passage
correlations are computed over 238 passages rather than 1908 sentences, so a
single passage moves them further. The evidence-count band is relative because
the two published configurations differ roughly twofold (3.05 vs 6.22), and
sequential stopping amplifies small discretization differences.

A FAIL is a stop condition: do not proceed to DDRE work on a baseline that
fails. A WARN exits zero but must have a written explanation before Gate 2.

Protocol preconditions
----------------------
Tolerances judge how close the reproduction landed. Preconditions judge whether
the run was a reproduction at all. A run that scores a different model, against
different source data, or under a different truncation regime is measuring
something else, and no metric agreement can make it a valid Gate 1 result.

Three preconditions must hold for a formal Gate 1 run:

``official_model``
    The NLI model is exactly the one Wang et al. released against.
``wang_source_commit``
    ``data/wang/SOURCE.json`` records the pinned released data commit.
``truncation_equivalence``
    The tokenizer limit matches the configured ``max_length``, so this
    repository's explicit truncation is equivalent to Wang's ``truncation=True``.

These detect a protocol mismatch; they never repair one. In particular, a
truncation mismatch must be investigated at its source, not silenced by
changing ``src/utils.py`` to match the check.
"""

from collections import OrderedDict


STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

_SEVERITY = {STATUS_PASS: 0, STATUS_WARN: 1, STATUS_FAIL: 2}

# The only NLI model a formal Gate 1 run may use: the model named in Wang's
# released main.py argparse default.
OFFICIAL_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# The released-artifact commit the experiment is pinned to. Kept in sync with
# SOURCE_COMMIT in scripts/prepare_wang_data.py; a test asserts they agree.
PINNED_WANG_SOURCE_COMMIT = "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea"

TABLE1_EVIDENCE_NUM_DEFINITION = "average_retrieved_documents_per_sentence"
TABLE1_EVIDENCE_NUM_SOURCE = (
    "Wang et al. (EMNLP 2023) report the average number of retrieved documents "
    "at the sentence level; Section 4.2 describes 3.05 and 6.22 as documents "
    "per sentence."
)

# Published Table 1 reference values, keyed by cost configuration.
PUBLISHED_TABLE1 = OrderedDict(
    [
        (
            "CM_14_CFA_24",
            {
                "c_miss": 14,
                "c_false_alarm": 24,
                "nonfactual_auc_pr": 0.8242,
                "factual_auc_pr": 0.5701,
                "accuracy": 0.8024,
                "pearson": 0.7137,
                "spearman": 0.6455,
                "evidence_num_per_sentence": 3.05,
            },
        ),
        (
            "CM_28_CFA_96",
            {
                "c_miss": 28,
                "c_false_alarm": 96,
                "nonfactual_auc_pr": 0.8645,
                "factual_auc_pr": 0.6196,
                "accuracy": 0.8239,
                "pearson": 0.8118,
                "spearman": 0.7420,
                "evidence_num_per_sentence": 6.22,
            },
        ),
    ]
)

# Metrics compared against Table 1, in report order.
COMPARED_METRICS = (
    "nonfactual_auc_pr",
    "factual_auc_pr",
    "accuracy",
    "pearson",
    "spearman",
    "evidence_num_per_sentence",
)

# Predeclared tolerances. "absolute" compares |reproduced - published|;
# "relative" compares |reproduced - published| / |published|.
TOLERANCES = {
    "factual_auc_pr": {"kind": "absolute", "pass": 0.01, "warn": 0.03},
    "nonfactual_auc_pr": {"kind": "absolute", "pass": 0.01, "warn": 0.03},
    "accuracy": {"kind": "absolute", "pass": 0.01, "warn": 0.03},
    "pearson": {"kind": "absolute", "pass": 0.02, "warn": 0.05},
    "spearman": {"kind": "absolute", "pass": 0.02, "warn": 0.05},
    "evidence_num_per_sentence": {"kind": "relative", "pass": 0.05, "warn": 0.10},
}


def evaluate_protocol_preconditions(provenance):
    """Check the protocol preconditions for a formal Gate 1 run.

    Takes the provenance block produced by :func:`src.provenance.collect_provenance`
    and returns flat PASS/FAIL statuses plus per-check detail. Reports the
    mismatch; never repairs it.
    """
    nli = (provenance or {}).get("nli_model") or {}
    wang = (provenance or {}).get("wang_data") or {}
    checks = []

    observed_model = nli.get("model_name")
    model_ok = observed_model == OFFICIAL_NLI_MODEL
    checks.append(
        {
            "name": "official_model",
            "status": STATUS_PASS if model_ok else STATUS_FAIL,
            "expected": OFFICIAL_NLI_MODEL,
            "observed": observed_model,
            "message": (
                "NLI model matches the released Wang reproduction model."
                if model_ok
                else (
                    f"NLI model is {observed_model!r}, not the official "
                    f"{OFFICIAL_NLI_MODEL!r}. A different model may be useful for "
                    "diagnostics but cannot produce a valid Gate 1 PASS."
                )
            ),
        }
    )

    observed_commit = wang.get("source_commit")
    commit_ok = bool(wang.get("available")) and observed_commit == PINNED_WANG_SOURCE_COMMIT
    if not wang.get("available"):
        commit_message = (
            "data/wang/SOURCE.json is missing or unreadable, so the source data "
            "cannot be verified. Run scripts/prepare_wang_data.py."
        )
    elif commit_ok:
        commit_message = "Wang source data matches the pinned released commit."
    else:
        commit_message = (
            f"Wang source data records commit {observed_commit!r}, not the pinned "
            f"{PINNED_WANG_SOURCE_COMMIT!r}. The reproduction would run against "
            "different artifacts."
        )
    checks.append(
        {
            "name": "wang_source_commit",
            "status": STATUS_PASS if commit_ok else STATUS_FAIL,
            "expected": PINNED_WANG_SOURCE_COMMIT,
            "observed": observed_commit,
            "message": commit_message,
        }
    )

    # None means the tokenizer limit could not be read, which is as
    # disqualifying as a known mismatch: equivalence is unproven either way.
    truncation_ok = nli.get("truncation_matches_wang") is True
    checks.append(
        {
            "name": "truncation_equivalence",
            "status": STATUS_PASS if truncation_ok else STATUS_FAIL,
            "expected": nli.get("nli_max_length_configured"),
            "observed": nli.get("tokenizer_model_max_length"),
            "message": nli.get("truncation_note")
            or (
                "Truncation equivalence to Wang's truncation=True is unproven. "
                "Investigate the mismatch; do not change src/utils.py to silence "
                "this check."
            ),
        }
    )

    statuses = {check["name"]: check["status"] for check in checks}
    overall = worst_status(statuses.values())
    return {
        "official_model": statuses["official_model"],
        "wang_source_commit": statuses["wang_source_commit"],
        "truncation_equivalence": statuses["truncation_equivalence"],
        "overall": overall,
        "checks": checks,
        "failures": [c["name"] for c in checks if c["status"] != STATUS_PASS],
        "failure_messages": [
            c["message"] for c in checks if c["status"] != STATUS_PASS
        ],
    }


def format_preconditions(preconditions):
    """Render the protocol preconditions as a fixed-width block."""
    lines = ["-" * 100, "PROTOCOL PRECONDITIONS", "-" * 100]
    for check in preconditions["checks"]:
        lines.append(f"  {check['name']:<26}{check['status']}")
        if check["status"] != STATUS_PASS:
            lines.append(f"      expected: {check['expected']!r}")
            lines.append(f"      observed: {check['observed']!r}")
            lines.append(f"      {check['message']}")
    lines.append(f"  {'overall':<26}{preconditions['overall']}")
    if preconditions["overall"] != STATUS_PASS:
        lines.append(
            "  This run cannot be a formal Gate 1 result. Fix the protocol "
            "mismatch at its source."
        )
    lines.append("-" * 100)
    return "\n".join(lines)


def worst_status(statuses):
    """Return the most severe status in ``statuses`` (PASS if empty)."""
    worst = STATUS_PASS
    for status in statuses:
        if _SEVERITY.get(status, _SEVERITY[STATUS_FAIL]) > _SEVERITY[worst]:
            worst = status
    return worst


def classify(deviation, tolerance):
    """Classify a non-negative deviation against a tolerance specification."""
    if deviation is None:
        return STATUS_FAIL
    if deviation <= tolerance["pass"]:
        return STATUS_PASS
    if deviation <= tolerance["warn"]:
        return STATUS_WARN
    return STATUS_FAIL


def compare_metric(metric, published, reproduced, tolerance):
    """Compare one reproduced metric against its published reference value.

    ``signed_delta`` records the direction of the deviation for the reader;
    ``absolute_delta`` and ``relative_delta`` are magnitudes, and which one
    determines the status is fixed by ``tolerance["kind"]``.
    """
    row = {
        "metric": metric,
        "published": float(published),
        "reproduced": None if reproduced is None else float(reproduced),
        "signed_delta": None,
        "absolute_delta": None,
        "relative_delta": None,
        "tolerance_kind": tolerance["kind"],
        "pass_threshold": tolerance["pass"],
        "warn_threshold": tolerance["warn"],
        "status": STATUS_FAIL,
        "note": None,
    }

    if reproduced is None:
        row["note"] = "metric was not produced by the reproduction run"
        return row

    signed = float(reproduced) - float(published)
    absolute = abs(signed)
    row["signed_delta"] = signed
    row["absolute_delta"] = absolute
    if published != 0:
        row["relative_delta"] = absolute / abs(float(published))

    deviation = absolute if tolerance["kind"] == "absolute" else row["relative_delta"]
    row["status"] = classify(deviation, tolerance)
    return row


def evaluate_configuration(config_name, reproduced, published=None, tolerances=None):
    """Evaluate one cost configuration against Table 1.

    ``reproduced`` must contain the keys in :data:`COMPARED_METRICS` plus
    ``total_retrieved_documents``. Zero retrieval is an unconditional FAIL: a
    baseline that consumes no evidence is degenerate and cannot serve as a
    comparator, regardless of how its other metrics score.
    """
    published = PUBLISHED_TABLE1 if published is None else published
    tolerances = TOLERANCES if tolerances is None else tolerances
    if config_name not in published:
        raise KeyError(f"No published reference values for configuration {config_name!r}")
    reference = published[config_name]

    rows = [
        compare_metric(metric, reference[metric], reproduced.get(metric), tolerances[metric])
        for metric in COMPARED_METRICS
    ]

    total_documents = reproduced.get("total_retrieved_documents")
    zero_retrieval = total_documents is not None and int(total_documents) == 0

    verdict = worst_status([row["status"] for row in rows])
    if zero_retrieval:
        verdict = STATUS_FAIL

    return {
        "configuration": config_name,
        "c_miss": reference.get("c_miss"),
        "c_false_alarm": reference.get("c_false_alarm"),
        "metrics": rows,
        "zero_retrieval": bool(zero_retrieval),
        "zero_retrieval_note": (
            "bse_official consumed no documents; the baseline is degenerate and "
            "cannot be used as a comparator"
            if zero_retrieval
            else None
        ),
        "diagnostics": {
            "avg_retrieved_documents_per_subclaim": reproduced.get(
                "avg_retrieved_documents_per_subclaim"
            ),
            "total_retrieved_documents": total_documents,
            "total_nli_span_calls": reproduced.get("total_nli_span_calls"),
        },
        "verdict": verdict,
    }


def evaluate_gate(reproduced_by_configuration, preconditions, published=None, tolerances=None):
    """Evaluate every configuration and return the overall gate report.

    ``preconditions`` is required. A run whose protocol preconditions do not all
    pass is not a formal Gate 1 run, and its overall verdict is forced to FAIL
    however well the metrics agree: agreement on the wrong model, the wrong
    source data, or under a different truncation regime is not reproduction.
    """
    published = PUBLISHED_TABLE1 if published is None else published
    tolerances = TOLERANCES if tolerances is None else tolerances

    configurations = OrderedDict()
    for config_name in published:
        if config_name not in reproduced_by_configuration:
            raise KeyError(f"Reproduction did not produce configuration {config_name!r}")
        configurations[config_name] = evaluate_configuration(
            config_name,
            reproduced_by_configuration[config_name],
            published=published,
            tolerances=tolerances,
        )

    failed = []
    warned = []
    for config_name, result in configurations.items():
        for row in result["metrics"]:
            if row["status"] == STATUS_FAIL:
                failed.append(f"{config_name}.{row['metric']}")
            elif row["status"] == STATUS_WARN:
                warned.append(f"{config_name}.{row['metric']}")
        if result["zero_retrieval"]:
            failed.append(f"{config_name}.zero_retrieval")

    overall = worst_status([result["verdict"] for result in configurations.values()])

    formal = preconditions["overall"] == STATUS_PASS
    if not formal:
        overall = STATUS_FAIL
        failed.append("protocol_preconditions")

    return {
        "protocol_preconditions": preconditions,
        "formal_gate1_run": formal,
        "table1_evidence_num_definition": TABLE1_EVIDENCE_NUM_DEFINITION,
        "table1_evidence_num_source": TABLE1_EVIDENCE_NUM_SOURCE,
        "tolerances": tolerances,
        "tolerances_predeclared": True,
        "configurations": configurations,
        "failed_metrics": failed,
        "warned_metrics": warned,
        "overall_verdict": overall,
        "verdict_interpretation": {
            STATUS_PASS: "Baseline reproduced within predeclared tolerance; bse_official may be used as the primary baseline.",
            STATUS_WARN: "Baseline deviates beyond the pass band. Gate 2 must not begin until the discrepancy has a written explanation.",
            STATUS_FAIL: "Stop condition. Investigate before using bse_official as a baseline or interpreting any DDRE comparison.",
        }[overall],
    }


def format_report(gate_report):
    """Render the gate report as a fixed-width text table."""
    lines = []
    lines.append("=" * 100)
    lines.append("WANG ET AL. BASELINE REPRODUCTION GATE")
    lines.append("=" * 100)
    lines.append(f"Table 1 evidence-count definition: {gate_report['table1_evidence_num_definition']}")
    lines.append("")
    lines.append(format_preconditions(gate_report["protocol_preconditions"]))
    lines.append("")

    for config_name, result in gate_report["configurations"].items():
        lines.append(
            f"[{config_name}]  C_M={result['c_miss']}  C_FA={result['c_false_alarm']}"
        )
        lines.append(
            f"  {'metric':<28}{'published':>12}{'reproduced':>13}"
            f"{'delta':>12}{'|rel|':>12}  status"
        )
        for row in result["metrics"]:
            reproduced = "n/a" if row["reproduced"] is None else f"{row['reproduced']:.4f}"
            signed = "n/a" if row["signed_delta"] is None else f"{row['signed_delta']:+.4f}"
            relative = (
                "n/a"
                if row["relative_delta"] is None
                else f"{100 * row['relative_delta']:.2f}%"
            )
            marker = "" if row["tolerance_kind"] == "absolute" else " (rel)"
            lines.append(
                f"  {row['metric']:<28}{row['published']:>12.4f}{reproduced:>13}"
                f"{signed:>12}{relative:>12}  {row['status']}{marker}"
            )
        subclaim = result["diagnostics"]["avg_retrieved_documents_per_subclaim"]
        if subclaim is not None:
            lines.append(
                f"  diagnostic only (not compared to Table 1): "
                f"avg documents per subclaim = {subclaim:.4f}"
            )
        if result["zero_retrieval"]:
            lines.append(f"  *** {result['zero_retrieval_note']} ***")
        lines.append(f"  configuration verdict: {result['verdict']}")
        lines.append("")

    lines.append("-" * 100)
    if not gate_report["formal_gate1_run"]:
        lines.append(
            "NOT A FORMAL GATE 1 RUN: protocol preconditions failed "
            f"({', '.join(gate_report['protocol_preconditions']['failures'])})."
        )
    lines.append(f"OVERALL VERDICT: {gate_report['overall_verdict']}")
    lines.append(gate_report["verdict_interpretation"])
    if gate_report["failed_metrics"]:
        lines.append(f"Failed: {', '.join(gate_report['failed_metrics'])}")
    if gate_report["warned_metrics"]:
        lines.append(f"Warned: {', '.join(gate_report['warned_metrics'])}")
    lines.append("=" * 100)
    return "\n".join(lines)
