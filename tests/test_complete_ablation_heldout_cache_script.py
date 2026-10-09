"""``scripts/complete_ablation_heldout_cache.py``: pins, ordering, and no forbidden path.

The script pins the canonical source cache, the canonical result and the
checkpoint revision; loads the model only at that revision; stops before any
model load on --dry-run; stops before completion on an environment mismatch;
and reaches no detector, metric or selection code.

No model, no torch required (two ordering tests skip without torch).
"""

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import ablation_cache as AC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "complete_ablation_heldout_cache.py"


class TestScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("complete_ablation_heldout_cache", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.source = SCRIPT.read_text()
        cls.names = set()
        for node in ast.walk(ast.parse(cls.source + "\n" + Path(AC.__file__).read_text())):
            if isinstance(node, ast.Name):
                cls.names.add(node.id)
            elif isinstance(node, ast.Attribute):
                cls.names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                cls.names.update(a.name for a in node.names)

    def test_no_detector_metric_or_selection_path(self):
        banned = {
            "BSEDetector", "DDREDetector", "ULSIFDensityRatio", "candidate_estimator",
            "build_nbc_histograms", "evaluate_detector", "evaluate_detector_with_traces",
            "summarize_method", "wang_pr_auc", "paired_passage_bootstrap", "assess_claim",
            "select_threshold_configuration", "candidate_record", "cost_consistent_thresholds",
            "tune_ddre_thresholds", "class_metrics", "heldout_analysis",
        }
        self.assertEqual(self.names & banned, set())

    def test_pins(self):
        m = self.module
        self.assertEqual(m.SOURCE_CACHE_SHA256,
                         "dce6c49f813cb0ebecaf15d813e35ef940d259f97ad246d42ab8208c7a85be25")
        self.assertEqual(m.RESULT_SHA256,
                         "b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67")
        self.assertEqual(m.REVISION, "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7")
        self.assertEqual(AC.BATCH_SIZE, 1)
        self.assertEqual((AC.SEGMENT_LENGTH, AC.OVERLAP_LENGTH, AC.MAX_DOCS), (400, 100, 10))

    def test_model_load_pins_the_revision(self):
        calls = [n for n in ast.walk(ast.parse(self.source))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "from_pretrained"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            revision = [k for k in call.keywords if k.arg == "revision"]
            self.assertEqual(len(revision), 1)
            self.assertEqual(ast.unparse(revision[0].value), "REVISION")

    def _run_main(self, tmp, *, dry_run, env_problems=(), load_return=None):
        args = mock.Mock(artifacts_dir=tmp, data_root="unused", dry_run=dry_run)
        m = self.module
        fake_cov = {"keys": [], "pairs": [], "counts": {}, "coverage_sha256": "c"}
        with mock.patch.object(m, "parse_args", return_value=args), \
                mock.patch.object(m, "verify_inputs", return_value={"environment": {}}), \
                mock.patch.object(m, "held_out_records", return_value=([], {})), \
                mock.patch.object(m.AC, "enumerate_coverage", return_value=fake_cov), \
                mock.patch.object(m.AC, "present_keys", return_value=set()), \
                mock.patch.object(m.AC, "check_environment", return_value=list(env_problems)), \
                mock.patch.object(m, "load_model", return_value=load_return) as load, \
                mock.patch("src.diagnostic_probe.collect_live_environment", return_value={}), \
                mock.patch.object(m.AC, "complete_cache") as complete:
            try:
                code = m.main()
            except AC.CompletionAborted:
                code = "aborted"
        return code, load, complete

    def test_dry_run_loads_no_model_and_writes_nothing(self):
        try:
            import src.utils  # noqa: F401  (needs torch; the dry run imports it)
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            code, load, complete = self._run_main(tmp, dry_run=True)
            self.assertEqual(code, 0)
            load.assert_not_called()
            complete.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_environment_mismatch_aborts_before_completion(self):
        try:
            import src.utils  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            code, load, complete = self._run_main(
                tmp, dry_run=False, env_problems=["x"],
                load_return=(mock.Mock(), mock.Mock(), "mps"),
            )
            load.assert_called_once()
            self.assertEqual(code, "aborted")
            complete.assert_not_called()

    def test_verify_inputs_refuses_wrong_digests_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.module.paths(Path(tmp))
            p["source"].write_text("not canonical")
            p["result"].write_text("{}")
            with self.assertRaises(AC.CompletionAborted):
                self.module.verify_inputs(p)


if __name__ == "__main__":
    unittest.main()
