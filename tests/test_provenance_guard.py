"""Tests for the pre-write provenance guard.

Synthetic provenance dictionaries only — no torch, no model, no network — so CI
runs the whole file with numpy alone.

The guard exists to stop a cache being extended with scores produced under
different semantics from the ones already in it. Its defining property is that
it fails closed: an unverifiable field is a failure, not a pass.
"""

import unittest

from src.provenance_guard import (
    OFFICIAL_NLI_MODEL,
    REQUIRED_BATCH_SIZE,
    REQUIRED_SCORE_VERSION,
    REQUIRED_WANG_SOURCE_COMMIT,
    SCORE_AFFECTING_LIBRARIES,
    STATUS_FAIL,
    STATUS_PASS,
    check_runtime_preconditions,
    check_static_preconditions,
    extract_reference,
    format_guard,
    guard_report,
)

REVISION = "0f3f1e6a8b2c4d5e6f708192a3b4c5d6e7f80912"
LIBRARIES = {
    "torch": "2.11.0+cu128",
    "transformers": "5.16.1",
    "tokenizers": "0.23.1",
    "numpy": "2.3.1",
    "scipy": "1.15.0",
    "sklearn": "1.6.0",
    "sentencepiece": "0.2.0",
}


def reference_payload(**overrides):
    """A collect_live_environment-shaped snapshot from the formal run."""
    payload = {
        "score_version": REQUIRED_SCORE_VERSION,
        "libraries": dict(LIBRARIES),
        "device": {"selected_device": "cuda", "gpu_name": "Tesla T4"},
        "model": {
            "dtype": "torch.float16",
            "training_mode": False,
            "config_commit_hash": REVISION,
        },
        "tokenizer": {"model_max_length": 512, "commit_hash": REVISION},
        "checkpoint_identity": {
            "model_name": OFFICIAL_NLI_MODEL,
            "resolved_revision": REVISION,
            "model_config_commit_hash": REVISION,
            "tokenizer_commit_hash": REVISION,
        },
    }
    payload.update(overrides)
    return payload


def observed_snapshot(**overrides):
    snapshot = reference_payload()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(snapshot.get(key), dict):
            snapshot[key] = {**snapshot[key], **value}
        else:
            snapshot[key] = value
    return snapshot


def static(reference, **overrides):
    kwargs = {
        "score_version": REQUIRED_SCORE_VERSION,
        "batch_size": REQUIRED_BATCH_SIZE,
        "wang_source_commit": REQUIRED_WANG_SOURCE_COMMIT,
        "model_name": OFFICIAL_NLI_MODEL,
    }
    kwargs.update(overrides)
    return check_static_preconditions(reference, **kwargs)


def full_guard(reference=None, observed=None, static_overrides=None, **runtime):
    reference = reference if reference is not None else extract_reference(reference_payload())
    observed = observed if observed is not None else observed_snapshot()
    checks = static(reference, **(static_overrides or {}))
    checks += check_runtime_preconditions(reference, observed, **runtime)
    return guard_report(checks)


class TestReferenceExtraction(unittest.TestCase):
    def test_live_environment_shape(self):
        reference = extract_reference(reference_payload())
        self.assertEqual(reference["model_name"], OFFICIAL_NLI_MODEL)
        self.assertEqual(reference["resolved_revision"], REVISION)
        self.assertEqual(reference["score_version"], REQUIRED_SCORE_VERSION)
        self.assertEqual(reference["model_dtype"], "torch.float16")
        self.assertEqual(reference["libraries"]["torch"], "2.11.0+cu128")

    def test_step2_report_shape(self):
        payload = {
            "environment_summary": {
                "model_name": OFFICIAL_NLI_MODEL,
                "resolved_hf_revision": REVISION,
                "model_dtype": "torch.float16",
                "torch_version": "2.11.0+cu128",
                "transformers_version": "5.16.1",
                "tokenizers_version": "0.23.1",
                "device": "cuda",
                "gpu_name": "Tesla T4",
            },
            "environment": reference_payload(),
        }
        reference = extract_reference(payload)
        self.assertEqual(reference["resolved_revision"], REVISION)
        self.assertEqual(reference["gpu_name"], "Tesla T4")
        self.assertEqual(reference["libraries"]["torch"], "2.11.0+cu128")

    def test_gate1_reproduction_shape(self):
        payload = {
            "provenance": {
                "libraries": dict(LIBRARIES),
                "runtime": {
                    "device": "cuda",
                    "batch_size": 1,
                    "score_version": REQUIRED_SCORE_VERSION,
                },
                "nli_model": {
                    "model_name": OFFICIAL_NLI_MODEL,
                    "tokenizer_model_max_length": 512,
                },
                "wang_data": {"source_commit": REQUIRED_WANG_SOURCE_COMMIT},
            }
        }
        reference = extract_reference(payload)
        self.assertEqual(reference["model_name"], OFFICIAL_NLI_MODEL)
        self.assertEqual(reference["batch_size"], 1)
        self.assertEqual(reference["wang_source_commit"], REQUIRED_WANG_SOURCE_COMMIT)

    def test_missing_fields_come_back_none(self):
        reference = extract_reference({})
        self.assertIsNone(reference["model_name"])
        self.assertIsNone(reference["resolved_revision"])
        self.assertEqual(reference["libraries"], {})


class TestStaticPreconditions(unittest.TestCase):
    def names(self, checks, status):
        return [c["name"] for c in checks if c["status"] == status]

    def test_a_matching_environment_passes(self):
        checks = static(extract_reference(reference_payload()))
        self.assertEqual(self.names(checks, STATUS_FAIL), [])

    def test_score_version_mismatch_fails(self):
        checks = static(
            extract_reference(reference_payload()),
            score_version="wang-emnlp23-temp5-seg400-overlap100-v1",
        )
        self.assertIn("score_version", self.names(checks, STATUS_FAIL))

    def test_batch_size_must_be_one(self):
        checks = static(extract_reference(reference_payload()), batch_size=8)
        self.assertIn("batch_size", self.names(checks, STATUS_FAIL))

    def test_wang_source_commit_mismatch_fails(self):
        checks = static(extract_reference(reference_payload()), wang_source_commit="deadbeef")
        self.assertIn("wang_source_commit", self.names(checks, STATUS_FAIL))

    def test_missing_wang_source_commit_fails(self):
        checks = static(extract_reference(reference_payload()), wang_source_commit=None)
        self.assertIn("wang_source_commit", self.names(checks, STATUS_FAIL))

    def test_non_official_model_fails(self):
        checks = static(extract_reference(reference_payload()), model_name="some/other-model")
        self.assertIn("official_model_name", self.names(checks, STATUS_FAIL))

    def test_reference_that_records_no_model_name_fails(self):
        reference = extract_reference({})
        checks = static(reference)
        self.assertIn("reference_model_name", self.names(checks, STATUS_FAIL))


class TestRuntimePreconditions(unittest.TestCase):
    def failed(self, report):
        return report["failed_checks"]

    def test_a_matching_environment_passes(self):
        report = full_guard()
        self.assertTrue(report["passed"], report["failure_messages"])

    def test_model_revision_mismatch_fails(self):
        observed = observed_snapshot(
            checkpoint_identity={"resolved_revision": "1111111111111111111111111111111111111111"}
        )
        report = full_guard(observed=observed)
        self.assertFalse(report["passed"])
        self.assertIn("checkpoint_revision", self.failed(report))

    def test_tokenizer_commit_mismatch_fails(self):
        observed = observed_snapshot(tokenizer={"commit_hash": "2222222"})
        report = full_guard(observed=observed)
        self.assertIn("tokenizer_commit_hash", self.failed(report))

    def test_truncation_mismatch_fails(self):
        observed = observed_snapshot(tokenizer={"model_max_length": 1000000000000000019884624838656})
        report = full_guard(observed=observed)
        self.assertFalse(report["passed"])
        self.assertIn("truncation_equivalence", self.failed(report))

    def test_dtype_mismatch_fails(self):
        observed = observed_snapshot(model={"dtype": "torch.float32"})
        report = full_guard(observed=observed)
        self.assertIn("model_dtype", self.failed(report))

    def test_device_mismatch_fails(self):
        observed = observed_snapshot(device={"selected_device": "cpu"})
        report = full_guard(observed=observed)
        self.assertIn("device", self.failed(report))

    def test_gpu_is_not_required_when_neither_run_used_one(self):
        # cpu-vs-cuda is gated by the device check; a CPU run has no GPU
        # identity to compare, so requiring one would be unsatisfiable.
        reference = extract_reference(
            reference_payload(device={"selected_device": "cpu", "gpu_name": None})
        )
        observed = observed_snapshot(device={"selected_device": "cpu", "gpu_name": None})
        report = full_guard(reference=reference, observed=observed)
        self.assertTrue(report["passed"], report["failure_messages"])

    def test_gpu_is_required_when_a_gpu_was_used(self):
        observed = observed_snapshot(device={"gpu_name": None})
        report = full_guard(observed=observed)
        self.assertFalse(report["passed"])
        self.assertIn("gpu_name", report["failed_checks"])

    def test_gpu_mismatch_fails(self):
        observed = observed_snapshot(device={"gpu_name": "A100-SXM4-40GB"})
        report = full_guard(observed=observed)
        self.assertIn("gpu_name", self.failed(report))

    def test_score_affecting_library_drift_fails(self):
        for name in SCORE_AFFECTING_LIBRARIES:
            with self.subTest(library=name):
                observed = observed_snapshot(
                    libraries={**LIBRARIES, name: "0.0.0-different"}
                )
                report = full_guard(observed=observed)
                self.assertIn(f"library_{name}", self.failed(report))

    def test_advisory_library_drift_does_not_block(self):
        # numpy cannot change an NLI forward pass; it is reported, not gated.
        observed = observed_snapshot(libraries={**LIBRARIES, "numpy": "1.26.0"})
        report = full_guard(observed=observed)
        self.assertTrue(report["passed"], report["failure_messages"])

    def test_train_mode_fails(self):
        observed = observed_snapshot(model={"training_mode": True})
        report = full_guard(observed=observed)
        self.assertIn("model_eval_mode", self.failed(report))

    def test_unverifiable_is_treated_as_failed(self):
        # The reference records no revision, so nothing can establish a match.
        reference = extract_reference(
            reference_payload(checkpoint_identity={"model_name": OFFICIAL_NLI_MODEL})
        )
        report = full_guard(reference=reference)
        self.assertFalse(report["passed"])
        self.assertIn("checkpoint_revision", self.failed(report))
        message = " ".join(report["failure_messages"])
        self.assertIn("Unverifiable is treated as failed", message)

    def test_local_checkpoint_override_downgrades_only_identity_checks(self):
        reference = extract_reference(
            reference_payload(checkpoint_identity={"model_name": OFFICIAL_NLI_MODEL})
        )
        report = full_guard(reference=reference, allow_local_checkpoint=True)
        self.assertTrue(report["passed"], report["failure_messages"])

    def test_the_override_does_not_excuse_a_dtype_mismatch(self):
        observed = observed_snapshot(model={"dtype": "torch.float32"})
        report = full_guard(observed=observed, allow_local_checkpoint=True)
        self.assertFalse(report["passed"])
        self.assertIn("model_dtype", self.failed(report))


class TestGuardReport(unittest.TestCase):
    def test_passed_is_false_if_any_check_fails(self):
        report = guard_report(
            [
                {"name": "a", "status": STATUS_PASS, "expected": 1, "observed": 1, "message": ""},
                {"name": "b", "status": STATUS_FAIL, "expected": 1, "observed": 2, "message": "no"},
            ]
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_checks"], ["b"])

    def test_the_policy_is_stated_in_the_report(self):
        report = guard_report([])
        self.assertIn("BEFORE the derived cache is opened for writes", report["policy"])
        self.assertIn("zero inference", report["policy"])
        self.assertIn("zero cache rows", report["policy"])

    def test_formatting_names_the_failed_check(self):
        report = full_guard(observed=observed_snapshot(model={"dtype": "torch.float32"}))
        text = format_guard(report)
        self.assertIn("model_dtype", text)
        self.assertIn("FAIL", text)
        self.assertIn("zero inference performed", text)


if __name__ == "__main__":
    unittest.main()
