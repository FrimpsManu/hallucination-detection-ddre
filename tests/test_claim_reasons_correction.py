"""The derived correction of the frozen held-out run's NOT_CONFIRMATORY reasons.

The guarantee: the correction changes the REASONS and nothing else. It refuses
to produce a record if any non-reason field of the claim assessment would
differ, if the status would change, if the split does not verify, or if it
would write over the canonical artifact.

Synthetic results only. No model, no torch, no Wang data.
"""

import ast
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.paired_bootstrap import (
    CLAIM_NOT_CONFIRMATORY,
    CONFIRMATORY_PR_AUC_MARGIN,
    FROZEN_RUN_CONFIGURATION,
    VALIDATION_SELECTION_POST_D03_SWEEP,
    assess_claim,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "correct_frozen_heldout_claim_reasons.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load(SCRIPT, "correct_frozen_heldout_claim_reasons")
FIXTURES = load(PROJECT_ROOT / "tests" / "test_confirmatory_statistics.py", "_fixtures")
VALIDATION_IDS = list(FIXTURES.FROZEN_VALIDATION_PASSAGE_IDS)
TEST_IDS = list(FIXTURES.FROZEN_TEST_PASSAGE_IDS)


def frozen_result(bootstrap=None):
    """A result artifact shaped like the canonical one, with the original bugs."""
    bootstrap = bootstrap or FIXTURES.bootstrap_with(
        nonfactual_auc_pr_delta=-0.0155, factual_auc_pr_delta=-0.0367,
        balanced_pr_auc_delta=-0.0205,
    )
    split = {
        "validation_fraction": 0.2, "split_seed": 42,
        "validation_passage_ids": VALIDATION_IDS,
        "held_out_passage_ids": TEST_IDS,
        "observed_held_out_passage_ids": TEST_IDS,
    }
    # Exactly what the fae3eee runner passed: a bare False and no expected IDs.
    original = assess_claim(
        bootstrap,
        validation_selection_confirmatory=False,
        quality_tolerance=CONFIRMATORY_PR_AUC_MARGIN,
        run_configuration=dict(FROZEN_RUN_CONFIGURATION),
        split_metadata={"validation_passage_ids": VALIDATION_IDS, "test_passage_ids": TEST_IDS},
    )
    # The fae3eee assess_claim predates the validation_selection field.
    del original["validation_selection"]
    return {"bootstrap": bootstrap, "split": split, "claim_assessment": original}


def correct(result, **overrides):
    kwargs = dict(result_path="/x/result.json", result_sha256="abc",
                  expected_validation_ids=VALIDATION_IDS, expected_test_ids=TEST_IDS,
                  repository_commit="deadbeef")
    kwargs.update(overrides)
    return MODULE.build_correction(result, **kwargs)


class TestCorrection(unittest.TestCase):
    def test_the_original_carries_both_defects(self):
        reasons = frozen_result()["claim_assessment"]["confirmatory_disqualifiers"]
        self.assertTrue(any("fallback objective" in r for r in reasons))
        self.assertTrue(any("split was not verified" in r for r in reasons))

    def test_reasons_are_corrected_and_status_is_preserved(self):
        record = correct(frozen_result())
        corrected = record["corrected"]["confirmatory_disqualifiers"]
        self.assertEqual(len(corrected), 1)
        self.assertIn("after the D-03 diagnostic", corrected[0])
        self.assertNotIn("fallback", corrected[0])
        self.assertEqual(record["claim_status"],
                         {"original": CLAIM_NOT_CONFIRMATORY,
                          "corrected": CLAIM_NOT_CONFIRMATORY, "unchanged": True})
        self.assertTrue(record["corrected"]["split_matches_frozen"])
        self.assertEqual(record["corrected"]["validation_selection"],
                         VALIDATION_SELECTION_POST_D03_SWEEP)

    def test_original_reasons_are_kept_verbatim(self):
        result = frozen_result()
        record = correct(result)
        self.assertEqual(record["original"]["confirmatory_disqualifiers"],
                         result["claim_assessment"]["confirmatory_disqualifiers"])
        self.assertEqual(record["original"]["interpretation"],
                         result["claim_assessment"]["interpretation"])

    def test_every_non_reason_field_is_verified_identical(self):
        result = frozen_result()
        record = correct(result)
        expected = sorted(set(result["claim_assessment"]) - set(MODULE.REGENERATED_FIELDS))
        self.assertEqual(record["fields_verified_identical"], expected)
        for field in ("performance_noninferiority", "retrieval_efficiency_ci_lower",
                      "nli_efficiency_ci_lower", "claim_status", "primary_claim_supported"):
            self.assertIn(field, record["fields_verified_identical"])

    def test_the_record_references_the_canonical_artifact(self):
        record = correct(frozen_result(), result_sha256="b5bc")
        self.assertEqual(record["corrects"]["sha256"], "b5bc")
        self.assertFalse(record["canonical_artifact_modified"])
        self.assertIn("no scores, predictions, bootstrap samples or metrics", record["statement"])

    def test_a_tampered_bootstrap_is_refused(self):
        # The saved bootstrap no longer matches the saved assessment, so a
        # non-reason field would change. The correction must refuse.
        result = frozen_result()
        tampered = copy.deepcopy(result)
        tampered["bootstrap"] = FIXTURES.bootstrap_with(nonfactual_auc_pr_delta=0.5)
        with self.assertRaises(MODULE.CorrectionRefused):
            correct(tampered)

    def test_a_split_that_does_not_verify_is_refused(self):
        with self.assertRaises(MODULE.CorrectionRefused):
            correct(frozen_result(), expected_test_ids=TEST_IDS[:-1] + [999])

    def test_a_result_without_the_defects_is_refused(self):
        result = frozen_result()
        result["claim_assessment"]["confirmatory_disqualifiers"] = ["something else"]
        with self.assertRaises(MODULE.CorrectionRefused):
            correct(result)


class TestCanonicalArtifactIsUntouchable(unittest.TestCase):
    def test_pinned_digest(self):
        self.assertEqual(MODULE.CANONICAL_RESULT_SHA256,
                         "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67")

    def test_refuses_to_write_over_the_canonical_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / MODULE.CANONICAL_RESULT_NAME
            canonical.write_text("{}")
            args = mock.Mock(artifacts_dir=tmp, data_root="unused", output=str(canonical))
            with mock.patch.object(MODULE, "parse_args", return_value=args):
                with self.assertRaises(MODULE.CorrectionRefused):
                    MODULE.main()
            self.assertEqual(canonical.read_text(), "{}")

    def test_refuses_a_wrong_canonical_digest_before_reading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / MODULE.CANONICAL_RESULT_NAME).write_text("{}")
            args = mock.Mock(artifacts_dir=tmp, data_root="unused", output=None)
            with mock.patch.object(MODULE, "parse_args", return_value=args), \
                    mock.patch.object(MODULE.frozen, "frozen_split") as split:
                with self.assertRaises(MODULE.CorrectionRefused):
                    MODULE.main()
            split.assert_not_called()
            self.assertFalse((Path(tmp) / MODULE.CORRECTION_NAME).exists())

    def test_the_canonical_path_is_only_ever_read(self):
        tree = ast.parse(SCRIPT.read_text())
        writes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write_text", "write_bytes", "unlink", "rename", "replace"}
        ]
        self.assertEqual(len(writes), 1)
        self.assertEqual(ast.unparse(writes[0].func.value), "output")

    def test_nothing_is_recomputed(self):
        source = SCRIPT.read_text()
        for name in ("paired_passage_bootstrap", "evaluate_detector", "BSEDetector",
                     "DDREDetector", "CachedDocumentScorer", "EntailmentScorer",
                     "from_pretrained"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
