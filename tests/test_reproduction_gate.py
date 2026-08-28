"""Tests for the Wang baseline reproduction gate.

These run without torch, scipy, or sklearn so they execute in the minimal CI
environment.
"""

import pathlib
import unittest

from src.reproduction_gate import (
    COMPARED_METRICS,
    OFFICIAL_NLI_MODEL,
    PINNED_WANG_SOURCE_COMMIT,
    PUBLISHED_TABLE1,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    TABLE1_EVIDENCE_NUM_DEFINITION,
    TOLERANCES,
    classify,
    compare_metric,
    evaluate_configuration,
    evaluate_gate,
    evaluate_protocol_preconditions,
    format_preconditions,
    format_report,
    worst_status,
)


def perfect_reproduction(config_name):
    """Reproduced values identical to the published Table 1 references."""
    reference = PUBLISHED_TABLE1[config_name]
    values = {metric: reference[metric] for metric in COMPARED_METRICS}
    values["total_retrieved_documents"] = 11_869
    values["total_nli_span_calls"] = 71_214
    values["avg_retrieved_documents_per_subclaim"] = 3.97
    return values


def perfect_gate_input():
    return {name: perfect_reproduction(name) for name in PUBLISHED_TABLE1}


def provenance(
    model_name=OFFICIAL_NLI_MODEL,
    source_commit=PINNED_WANG_SOURCE_COMMIT,
    source_available=True,
    truncation_matches=True,
    tokenizer_limit=512,
):
    """A provenance block of the shape src.provenance.collect_provenance returns."""
    return {
        "wang_data": {"available": source_available, "source_commit": source_commit},
        "nli_model": {
            "model_name": model_name,
            "nli_max_length_configured": 512,
            "tokenizer_model_max_length": tokenizer_limit,
            "truncation_matches_wang": truncation_matches,
            "truncation_note": "note",
        },
    }


def passing_preconditions():
    return evaluate_protocol_preconditions(provenance())


class TestFrozenTolerances(unittest.TestCase):
    """The tolerances are predeclared; changing them silently is the failure mode."""

    def test_tolerance_values_are_the_predeclared_ones(self):
        self.assertEqual(
            TOLERANCES,
            {
                "factual_auc_pr": {"kind": "absolute", "pass": 0.01, "warn": 0.03},
                "nonfactual_auc_pr": {"kind": "absolute", "pass": 0.01, "warn": 0.03},
                "accuracy": {"kind": "absolute", "pass": 0.01, "warn": 0.03},
                "pearson": {"kind": "absolute", "pass": 0.02, "warn": 0.05},
                "spearman": {"kind": "absolute", "pass": 0.02, "warn": 0.05},
                "evidence_num_per_sentence": {
                    "kind": "relative",
                    "pass": 0.05,
                    "warn": 0.10,
                },
            },
        )

    def test_every_compared_metric_has_a_tolerance(self):
        for metric in COMPARED_METRICS:
            self.assertIn(metric, TOLERANCES)

    def test_table1_evidence_definition_is_per_sentence(self):
        self.assertEqual(
            TABLE1_EVIDENCE_NUM_DEFINITION, "average_retrieved_documents_per_sentence"
        )

    def test_published_reference_values(self):
        self.assertEqual(PUBLISHED_TABLE1["CM_14_CFA_24"]["evidence_num_per_sentence"], 3.05)
        self.assertEqual(PUBLISHED_TABLE1["CM_28_CFA_96"]["evidence_num_per_sentence"], 6.22)


class TestClassification(unittest.TestCase):
    def test_absolute_bands(self):
        tolerance = {"kind": "absolute", "pass": 0.01, "warn": 0.03}
        self.assertEqual(classify(0.0, tolerance), STATUS_PASS)
        self.assertEqual(classify(0.01, tolerance), STATUS_PASS)
        self.assertEqual(classify(0.02, tolerance), STATUS_WARN)
        self.assertEqual(classify(0.03, tolerance), STATUS_WARN)
        self.assertEqual(classify(0.031, tolerance), STATUS_FAIL)

    def test_missing_value_fails(self):
        self.assertEqual(classify(None, TOLERANCES["accuracy"]), STATUS_FAIL)

    def test_worst_status_picks_most_severe(self):
        self.assertEqual(worst_status([]), STATUS_PASS)
        self.assertEqual(worst_status([STATUS_PASS, STATUS_PASS]), STATUS_PASS)
        self.assertEqual(worst_status([STATUS_PASS, STATUS_WARN]), STATUS_WARN)
        self.assertEqual(worst_status([STATUS_WARN, STATUS_FAIL, STATUS_PASS]), STATUS_FAIL)


class TestCompareMetric(unittest.TestCase):
    def test_absolute_and_relative_deltas_both_reported(self):
        row = compare_metric("accuracy", 0.8239, 0.8189, TOLERANCES["accuracy"])
        self.assertAlmostEqual(row["signed_delta"], -0.005, places=9)
        self.assertAlmostEqual(row["absolute_delta"], 0.005, places=9)
        self.assertAlmostEqual(row["relative_delta"], 0.005 / 0.8239, places=9)
        self.assertEqual(row["status"], STATUS_PASS)

    def test_evidence_count_is_judged_relatively(self):
        tolerance = TOLERANCES["evidence_num_per_sentence"]
        # +0.30 on 6.22 is 4.8% -> PASS relatively, though 0.30 absolute would
        # exceed every absolute band in the table.
        row = compare_metric("evidence_num_per_sentence", 6.22, 6.52, tolerance)
        self.assertEqual(row["tolerance_kind"], "relative")
        self.assertEqual(row["status"], STATUS_PASS)

        self.assertEqual(
            compare_metric("evidence_num_per_sentence", 6.22, 6.72, tolerance)["status"],
            STATUS_WARN,
        )
        self.assertEqual(
            compare_metric("evidence_num_per_sentence", 6.22, 7.20, tolerance)["status"],
            STATUS_FAIL,
        )

    def test_direction_of_deviation_does_not_matter(self):
        tolerance = TOLERANCES["factual_auc_pr"]
        above = compare_metric("factual_auc_pr", 0.6196, 0.6196 + 0.02, tolerance)
        below = compare_metric("factual_auc_pr", 0.6196, 0.6196 - 0.02, tolerance)
        self.assertEqual(above["status"], below["status"])
        self.assertEqual(above["status"], STATUS_WARN)
        self.assertGreater(above["signed_delta"], 0)
        self.assertLess(below["signed_delta"], 0)

    def test_missing_reproduced_value_fails_with_note(self):
        row = compare_metric("pearson", 0.8118, None, TOLERANCES["pearson"])
        self.assertEqual(row["status"], STATUS_FAIL)
        self.assertIsNotNone(row["note"])


class TestConfigurationVerdict(unittest.TestCase):
    def test_exact_reproduction_passes(self):
        result = evaluate_configuration("CM_28_CFA_96", perfect_reproduction("CM_28_CFA_96"))
        self.assertEqual(result["verdict"], STATUS_PASS)
        self.assertFalse(result["zero_retrieval"])

    def test_zero_retrieval_fails_even_with_perfect_metrics(self):
        values = perfect_reproduction("CM_28_CFA_96")
        values["total_retrieved_documents"] = 0
        result = evaluate_configuration("CM_28_CFA_96", values)
        self.assertTrue(result["zero_retrieval"])
        self.assertEqual(result["verdict"], STATUS_FAIL)
        self.assertIsNotNone(result["zero_retrieval_note"])

    def test_subclaim_count_is_diagnostic_only(self):
        values = perfect_reproduction("CM_28_CFA_96")
        values["avg_retrieved_documents_per_subclaim"] = 999.0
        result = evaluate_configuration("CM_28_CFA_96", values)
        self.assertEqual(result["verdict"], STATUS_PASS)
        self.assertEqual(result["diagnostics"]["avg_retrieved_documents_per_subclaim"], 999.0)
        compared = {row["metric"] for row in result["metrics"]}
        self.assertNotIn("avg_retrieved_documents_per_subclaim", compared)

    def test_unknown_configuration_raises(self):
        with self.assertRaises(KeyError):
            evaluate_configuration("CM_99_CFA_99", perfect_reproduction("CM_28_CFA_96"))


class TestGateVerdict(unittest.TestCase):
    def test_exact_reproduction_of_both_configurations_passes(self):
        report = evaluate_gate(perfect_gate_input(), passing_preconditions())
        self.assertEqual(report["overall_verdict"], STATUS_PASS)
        self.assertEqual(report["failed_metrics"], [])
        self.assertEqual(report["warned_metrics"], [])
        self.assertTrue(report["tolerances_predeclared"])

    def test_overall_verdict_is_worst_across_configurations(self):
        payload = perfect_gate_input()
        payload["CM_14_CFA_24"]["accuracy"] = 0.8024 - 0.02  # WARN band
        report = evaluate_gate(payload, passing_preconditions())
        self.assertEqual(report["overall_verdict"], STATUS_WARN)
        self.assertIn("CM_14_CFA_24.accuracy", report["warned_metrics"])

        payload["CM_28_CFA_96"]["factual_auc_pr"] = 0.6196 - 0.10  # FAIL band
        report = evaluate_gate(payload, passing_preconditions())
        self.assertEqual(report["overall_verdict"], STATUS_FAIL)
        self.assertIn("CM_28_CFA_96.factual_auc_pr", report["failed_metrics"])

    def test_zero_retrieval_in_one_configuration_fails_the_gate(self):
        payload = perfect_gate_input()
        payload["CM_14_CFA_24"]["total_retrieved_documents"] = 0
        report = evaluate_gate(payload, passing_preconditions())
        self.assertEqual(report["overall_verdict"], STATUS_FAIL)
        self.assertIn("CM_14_CFA_24.zero_retrieval", report["failed_metrics"])

    def test_missing_configuration_raises(self):
        payload = perfect_gate_input()
        del payload["CM_14_CFA_24"]
        with self.assertRaises(KeyError):
            evaluate_gate(payload, passing_preconditions())

    def test_report_renders(self):
        text = format_report(evaluate_gate(perfect_gate_input(), passing_preconditions()))
        self.assertIn("OVERALL VERDICT: PASS", text)
        self.assertIn("evidence_num_per_sentence", text)
        self.assertIn("average_retrieved_documents_per_sentence", text)


if __name__ == "__main__":
    unittest.main()


class TestProtocolPreconditions(unittest.TestCase):
    def test_official_model_constant(self):
        self.assertEqual(
            OFFICIAL_NLI_MODEL,
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        )

    def test_pinned_commit_matches_the_download_script(self):
        """The gate and scripts/prepare_wang_data.py must pin the same commit."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "prepare_wang_data",
            str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "prepare_wang_data.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.SOURCE_COMMIT, PINNED_WANG_SOURCE_COMMIT)

    def test_all_preconditions_satisfied(self):
        result = evaluate_protocol_preconditions(provenance())
        self.assertEqual(result["official_model"], STATUS_PASS)
        self.assertEqual(result["wang_source_commit"], STATUS_PASS)
        self.assertEqual(result["truncation_equivalence"], STATUS_PASS)
        self.assertEqual(result["overall"], STATUS_PASS)
        self.assertEqual(result["failures"], [])

    def test_non_official_model_fails(self):
        result = evaluate_protocol_preconditions(
            provenance(model_name="cross-encoder/nli-deberta-v3-small")
        )
        self.assertEqual(result["official_model"], STATUS_FAIL)
        self.assertEqual(result["overall"], STATUS_FAIL)
        self.assertIn("official_model", result["failures"])

    def test_missing_source_provenance_fails(self):
        result = evaluate_protocol_preconditions(
            provenance(source_available=False, source_commit=None)
        )
        self.assertEqual(result["wang_source_commit"], STATUS_FAIL)
        self.assertEqual(result["overall"], STATUS_FAIL)

    def test_different_source_commit_fails(self):
        result = evaluate_protocol_preconditions(provenance(source_commit="deadbeef" * 5))
        self.assertEqual(result["wang_source_commit"], STATUS_FAIL)
        self.assertEqual(result["overall"], STATUS_FAIL)

    def test_truncation_mismatch_fails(self):
        result = evaluate_protocol_preconditions(
            provenance(truncation_matches=False, tokenizer_limit=10**30)
        )
        self.assertEqual(result["truncation_equivalence"], STATUS_FAIL)
        self.assertEqual(result["overall"], STATUS_FAIL)

    def test_unknown_truncation_fails_like_a_mismatch(self):
        # None means the tokenizer limit could not be read. Equivalence is
        # unproven, which disqualifies the run just as a known mismatch does.
        result = evaluate_protocol_preconditions(
            provenance(truncation_matches=None, tokenizer_limit=None)
        )
        self.assertEqual(result["truncation_equivalence"], STATUS_FAIL)
        self.assertEqual(result["overall"], STATUS_FAIL)

    def test_empty_provenance_fails_every_check(self):
        result = evaluate_protocol_preconditions({})
        self.assertEqual(result["overall"], STATUS_FAIL)
        self.assertEqual(len(result["failures"]), 3)

    def test_several_failures_are_all_reported(self):
        result = evaluate_protocol_preconditions(
            provenance(model_name="other", source_available=False, truncation_matches=None)
        )
        self.assertEqual(
            sorted(result["failures"]),
            ["official_model", "truncation_equivalence", "wang_source_commit"],
        )
        self.assertEqual(len(result["failure_messages"]), 3)

    def test_preconditions_render(self):
        text = format_preconditions(evaluate_protocol_preconditions(provenance(model_name="x")))
        self.assertIn("official_model", text)
        self.assertIn("cannot be a formal Gate 1 result", text)


class TestPreconditionsGateTheVerdict(unittest.TestCase):
    def test_perfect_metrics_still_fail_when_preconditions_fail(self):
        for bad in (
            provenance(model_name="cross-encoder/nli-deberta-v3-small"),
            provenance(source_available=False, source_commit=None),
            provenance(truncation_matches=None, tokenizer_limit=None),
        ):
            report = evaluate_gate(
                perfect_gate_input(), evaluate_protocol_preconditions(bad)
            )
            self.assertEqual(report["overall_verdict"], STATUS_FAIL)
            self.assertFalse(report["formal_gate1_run"])
            self.assertIn("protocol_preconditions", report["failed_metrics"])

    def test_passing_preconditions_allow_a_formal_pass(self):
        report = evaluate_gate(perfect_gate_input(), passing_preconditions())
        self.assertEqual(report["overall_verdict"], STATUS_PASS)
        self.assertTrue(report["formal_gate1_run"])

    def test_preconditions_are_recorded_in_the_report(self):
        report = evaluate_gate(perfect_gate_input(), passing_preconditions())
        self.assertEqual(report["protocol_preconditions"]["overall"], STATUS_PASS)
        self.assertIn("checks", report["protocol_preconditions"])

    def test_failed_preconditions_are_visible_in_the_rendered_report(self):
        report = evaluate_gate(
            perfect_gate_input(),
            evaluate_protocol_preconditions(provenance(model_name="x")),
        )
        self.assertIn("NOT A FORMAL GATE 1 RUN", format_report(report))
