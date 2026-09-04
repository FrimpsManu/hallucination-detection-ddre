"""Tests for the pre-write provenance guard.

Synthetic provenance dictionaries only — no torch, no model, no network — so CI
runs the whole file with numpy alone.

The guard exists to stop a cache being extended with scores produced under
different semantics from the ones already in it. Its defining property is that
it fails closed: an unverifiable field is a failure, not a pass.
"""

import unittest

from src.provenance_guard import (
    CHECKPOINT_SOURCE,
    FORMAL_SOURCE,
    OFFICIAL_NLI_MODEL,
    REQUIRED_BATCH_SIZE,
    REQUIRED_SCORE_VERSION,
    REQUIRED_WANG_SOURCE_COMMIT,
    SCORE_AFFECTING_LIBRARIES,
    STATUS_FAIL,
    STATUS_PASS,
    check_reference_bundle,
    check_runtime_preconditions,
    check_static_preconditions,
    extract_reference,
    format_bundle,
    format_guard,
    guard_report,
    merge_reference,
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
    overrides = dict(static_overrides or {})
    # The script passes the local-checkpoint override to both halves of the
    # guard; mirroring that here keeps the helper faithful to real usage.
    if "allow_local_checkpoint" in runtime:
        overrides.setdefault("allow_local_checkpoint", runtime["allow_local_checkpoint"])
    checks = static(reference, **overrides)
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


OLD_SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-v1"


def formal_payload(**overrides):
    """The formal v2 batch-1 Gate report shape.

    It records the corrected score version, the Wang commit, batch size, model
    name, the truncation precondition, device and its own library list. It does
    NOT record a resolved revision, the commit hashes, the dtype, the GPU, or
    the tokenizers/sentencepiece versions -- which is exactly why a supplement
    is needed.
    """
    payload = {
        "provenance": {
            "libraries": {
                "torch": LIBRARIES["torch"],
                "transformers": LIBRARIES["transformers"],
                "numpy": LIBRARIES["numpy"],
                "scipy": LIBRARIES["scipy"],
                "sklearn": LIBRARIES["sklearn"],
            },
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
    payload.update(overrides)
    return payload


def checkpoint_payload(**environment_overrides):
    """The Step 2 scoring-path diagnostic shape, pre-correction by construction."""
    environment = reference_payload(score_version=OLD_SCORE_VERSION)
    for key, value in environment_overrides.items():
        if isinstance(value, dict) and isinstance(environment.get(key), dict):
            environment[key] = {**environment[key], **value}
        else:
            environment[key] = value
    return {
        "environment_summary": {
            "model_name": (environment["checkpoint_identity"] or {}).get("model_name"),
            "resolved_hf_revision": (environment["checkpoint_identity"] or {}).get(
                "resolved_revision"
            ),
            "model_dtype": (environment["model"] or {}).get("dtype"),
            "device": (environment["device"] or {}).get("selected_device"),
            "gpu_name": (environment["device"] or {}).get("gpu_name"),
        },
        "environment": environment,
    }


def bundle_of(formal=None, checkpoint=None):
    return merge_reference(
        extract_reference(formal if formal is not None else formal_payload()),
        extract_reference(checkpoint) if checkpoint is not None
        else extract_reference(checkpoint_payload()),
        formal_path="formal.json",
        checkpoint_path="checkpoint.json",
    )


class TestReferenceBundle(unittest.TestCase):
    """The two-artifact reference: authority, supplement, and honest limits."""

    def failed(self, checks):
        return [c["name"] for c in checks if c["status"] == STATUS_FAIL]

    def source_of(self, bundle, field):
        for entry in bundle["field_sources"]:
            if entry["field"] == field:
                return entry["source"]
        raise AssertionError(f"{field} is not a guarded field")

    def test_a_matching_reference_bundle_passes(self):
        bundle = bundle_of()
        self.assertEqual(self.failed(check_reference_bundle(bundle)), [])
        report = full_guard(reference=bundle["reference"])
        self.assertTrue(report["passed"], report["failure_messages"])

    def test_the_formal_report_is_authoritative_where_it_records_a_field(self):
        # Both artifacts record the model name and the torch version; the
        # value used must come from the formal report.
        bundle = bundle_of()
        for field in ("model_name", "batch_size", "wang_source_commit",
                      "tokenizer_model_max_length", "library:torch"):
            with self.subTest(field=field):
                self.assertEqual(self.source_of(bundle, field), FORMAL_SOURCE)
        self.assertEqual(bundle["reference"]["score_version"], REQUIRED_SCORE_VERSION)

    def test_the_checkpoint_artifact_only_supplements_what_the_gate_report_lacks(self):
        bundle = bundle_of()
        for field in ("resolved_revision", "model_config_commit_hash",
                      "tokenizer_commit_hash", "model_dtype", "gpu_name",
                      "library:tokenizers", "library:sentencepiece"):
            with self.subTest(field=field):
                self.assertEqual(self.source_of(bundle, field), CHECKPOINT_SOURCE)
        self.assertEqual(bundle["reference"]["resolved_revision"], REVISION)

    def test_conflicting_formal_and_checkpoint_provenance_aborts(self):
        checkpoint = checkpoint_payload(libraries={**LIBRARIES, "torch": "1.0.0"})
        bundle = bundle_of(checkpoint=checkpoint)
        self.assertEqual(
            [entry["field"] for entry in bundle["conflicts"]], ["library:torch"]
        )
        checks = check_reference_bundle(bundle)
        self.assertIn("bundle_agreement_library_torch", self.failed(checks))
        self.assertIn("reference_bundle_consistent", self.failed(checks))

    def test_a_conflicting_model_name_aborts(self):
        checkpoint = checkpoint_payload(
            checkpoint_identity={"model_name": "some/other-model"}
        )
        bundle = bundle_of(checkpoint=checkpoint)
        self.assertIn(
            "bundle_agreement_model_name", self.failed(check_reference_bundle(bundle))
        )

    def test_the_score_version_divergence_is_recorded_but_does_not_gate(self):
        # Step 2 predates the PR #4 correction, so it records the old score
        # version by construction. That is not evidence of two machines.
        bundle = bundle_of()
        divergent = [entry["field"] for entry in bundle["recorded_divergences"]]
        self.assertEqual(divergent, ["score_version"])
        self.assertEqual(bundle["conflicts"], [])
        self.assertEqual(self.failed(check_reference_bundle(bundle)), [])
        self.assertTrue(
            any("predates the PR #4" in text for text in bundle["limitations"])
        )

    def test_an_advisory_library_divergence_does_not_gate(self):
        checkpoint = checkpoint_payload(libraries={**LIBRARIES, "numpy": "1.26.0"})
        bundle = bundle_of(checkpoint=checkpoint)
        self.assertEqual(bundle["conflicts"], [])
        self.assertEqual(self.failed(check_reference_bundle(bundle)), [])

    def test_exact_v2_checkpoint_identity_is_not_claimed(self):
        # The revision comes from the supplement, so nothing establishes that
        # the v2 cache rows were produced at it. Say so, do not imply it.
        bundle = bundle_of()
        self.assertFalse(bundle["checkpoint_identity_established"])
        text = " ".join(bundle["limitations"])
        self.assertIn("NO ARTIFACT ESTABLISHES", text)
        self.assertIn("cannot be established retrospectively", text)
        self.assertIn("NO ARTIFACT ESTABLISHES", format_bundle(bundle))

    def test_identity_is_established_when_the_formal_report_records_the_revision(self):
        formal = formal_payload()
        formal["provenance"]["nli_model"]["model_name"] = OFFICIAL_NLI_MODEL
        formal["checkpoint_identity"] = {"resolved_revision": REVISION}
        bundle = bundle_of(formal=formal)
        self.assertTrue(bundle["checkpoint_identity_established"])
        self.assertEqual(self.source_of(bundle, "resolved_revision"), FORMAL_SOURCE)

    def test_a_missing_checkpoint_revision_fails_a_static_check(self):
        # STATIC, so main() aborts on it before importing transformers and
        # before from_pretrained. There is no fallback to Hugging Face main.
        bundle = merge_reference(extract_reference(formal_payload()), None)
        self.assertIsNone(bundle["reference"]["resolved_revision"])
        checks = static(bundle["reference"])
        self.assertIn("resolved_revision_present", self.failed(checks))
        message = " ".join(
            c["message"] for c in checks if c["name"] == "resolved_revision_present"
        )
        self.assertIn("Aborting before from_pretrained", message)

    def test_the_bundle_records_where_every_guarded_field_came_from(self):
        bundle = bundle_of()
        recorded = {entry["field"] for entry in bundle["field_sources"]}
        for field in ("model_name", "score_version", "wang_source_commit",
                      "batch_size", "device", "tokenizer_model_max_length",
                      "resolved_revision", "model_config_commit_hash",
                      "tokenizer_commit_hash", "model_dtype", "gpu_name"):
            self.assertIn(field, recorded)
        for entry in bundle["field_sources"]:
            self.assertIn(
                entry["agreement"],
                {"both_agree", "conflict", "formal_only", "checkpoint_only",
                 "unrecorded"},
            )

    def test_a_bundle_with_no_supplement_still_uses_the_formal_report(self):
        bundle = merge_reference(extract_reference(formal_payload()), None)
        self.assertEqual(bundle["reference"]["model_name"], OFFICIAL_NLI_MODEL)
        self.assertEqual(bundle["reference"]["score_version"], REQUIRED_SCORE_VERSION)
        self.assertEqual(bundle["conflicts"], [])


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
