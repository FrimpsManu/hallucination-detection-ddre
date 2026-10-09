"""``scripts/analyze_frozen_heldout.py`` can only describe the canonical artifacts.

It has no model-loading path, no selection machinery, it pins the canonical
digests, and it refuses mismatched inputs before reconstructing anything.
No model, no torch, no Wang data.
"""

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import heldout_analysis as A

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "analyze_frozen_heldout.py"


def load_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze_frozen_heldout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScriptCannotInfer(unittest.TestCase):
    def setUp(self):
        self.source = SCRIPT.read_text()
        self.tree = ast.parse(self.source)

    def _names(self):
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name for a in node.names)
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module)
        return names

    def test_no_model_or_inference_path(self):
        banned = {"transformers", "torch", "from_pretrained", "AutoModelForSequenceClassification",
                  "AutoTokenizer", "EntailmentScorer", "select_device", "warm_pairs"}
        self.assertEqual(self._names() & banned, set())

    def test_no_selection_machinery(self):
        banned = {"select_threshold_configuration", "cost_consistent_thresholds",
                  "tune_ddre_thresholds", "paired_passage_bootstrap", "assess_claim"}
        self.assertEqual(self._names() & banned, set())

    def test_scores_only_through_the_read_only_cache(self):
        self.assertIn("CachedDocumentScorer", self._names())

    def test_pins_the_canonical_artifacts(self):
        module = load_script()
        pins = module.pinned_digests()
        self.assertEqual(pins["result"],
                         "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67")
        self.assertEqual(pins["predictions"],
                         "a78b09b7aa09803a21a761b3bede7600180014b97142a381b509d7a1267211c1")
        self.assertEqual(pins["freeze"],
                         "744abf4afc05f586ed138dbbb5a503ac8014af40ab14c2a6d13b5f512ecdf9fa")

    def test_refuses_mismatched_inputs_before_any_work(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            for path in module.input_paths(Path(tmp)).values():
                path.write_text("not the canonical artifact")
            args = mock.Mock(artifacts_dir=tmp, output_dir=tmp, data_root="unused",
                             write_traces=False)
            with mock.patch.object(module, "parse_args", return_value=args), \
                    mock.patch.object(module.frozen, "frozen_split") as split, \
                    mock.patch.object(module, "build_detectors") as build:
                with self.assertRaises(A.ArtifactDigestMismatch):
                    module.main()
            split.assert_not_called()
            build.assert_not_called()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             sorted(p.name for p in module.input_paths(Path(tmp)).values()))

    def test_result_cross_reference_check(self):
        module = load_script()
        good = {
            "freeze_artifact_sha256": module.frozen.FREEZE_SHA256,
            "frozen_sigma": module.frozen.FROZEN_SIGMA,
            "frozen_lambda": module.frozen.FROZEN_LAMBDA,
            "frozen_lower": module.frozen.FROZEN_LOWER,
            "frozen_upper": module.frozen.FROZEN_UPPER,
            "d03_cache": {"d03_cache_sha256": module.frozen.D03_CACHE_SHA256},
            "checkpoint": {"d03_report_sha256": module.D03_REPORT_SHA256},
        }
        module.check_result_cross_references(good)
        with self.assertRaises(A.ArtifactDigestMismatch):
            module.check_result_cross_references(dict(good, frozen_lower=0.25))


if __name__ == "__main__":
    unittest.main()
