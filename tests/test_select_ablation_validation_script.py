"""``scripts/select_ablation_validation.py`` is walled off from held-out data.

It must read only the D-03 validation cache, evaluate only validation records,
never name a held-out artifact, never reach held-out evaluation or statistics,
and refuse to run twice or on unpinned inputs.

No model, no torch, no Wang data: the Wang loader and the evaluation are
replaced by fakes.
"""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import ablation_selection as S
from src.hyperparameter_outcome_sensitivity import HeldOutLeak

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_ablation_validation.py"
MODULE_SOURCE = (PROJECT_ROOT / "src" / "ablation_selection.py").read_text()


def load():
    spec = importlib.util.spec_from_file_location("select_ablation_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load()
SOURCE = SCRIPT.read_text()


def rec(passage):
    return SimpleNamespace(passage_index=passage, subclaims=[object()])


class TestFirewall(unittest.TestCase):
    def test_no_held_out_artifact_is_named(self):
        for text in (SOURCE, MODULE_SOURCE):
            for name in ("ablation_heldout_full_cache", "frozen_heldout_nli_cache",
                         "frozen_heldout_predictions", "frozen_heldout_result",
                         "run_frozen_heldout", "2497487664e2df55", "dce6c49f813cb0eb"):
                self.assertNotIn(name, text)

    def test_no_held_out_evaluation_or_statistics_path(self):
        names = set()
        for node in ast.walk(ast.parse(SOURCE + "\n" + MODULE_SOURCE)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name for a in node.names)
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module)
        banned = {"paired_passage_bootstrap", "assess_claim", "evaluate_detector_with_traces",
                  "retrieval_protocol_asymmetry", "src.heldout_analysis", "src.ablation_cache",
                  "transformers", "torch", "from_pretrained", "EntailmentScorer",
                  "prepare_derived_cache"}
        self.assertEqual(names & banned, set())

    def test_the_only_scorer_is_the_validation_cache(self):
        calls = [n for n in ast.walk(ast.parse(SOURCE)) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "CachedDocumentScorer"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "p['validation_cache']")
        self.assertEqual(M.VALIDATION_CACHE_SHA256,
                         "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776")

    def test_evaluate_refuses_held_out_records_before_any_detection(self):
        detector = mock.Mock()
        with self.assertRaises(HeldOutLeak):
            M.evaluate(detector, [rec(1), rec(7)], [7, 9], scorer=None, where="test")
        detector.detect_sentence.assert_not_called()

    def test_validation_split_returns_validation_records_only(self):
        validation, held_out = [rec(1), rec(2)], [rec(3), rec(4)]
        meta = {"test_passage_ids": [3, 4]}
        sensitivity = {"split": {"sha256": M.passage_ids_sha256([1, 2])}}
        with mock.patch("src.wang_data.load_sentence_records", return_value=[]), \
                mock.patch("src.wang_data.group_split_records",
                           return_value=(validation, held_out, meta)), \
                mock.patch.multiple(M, EXPECTED_VALIDATION_PASSAGES=2,
                                    EXPECTED_VALIDATION_SENTENCES=2,
                                    EXPECTED_VALIDATION_SUBCLAIMS=2):
            records, held_ids, identity = M.validation_split("unused", sensitivity)
        self.assertEqual([r.passage_index for r in records], [1, 2])
        self.assertEqual(held_ids, [3, 4])
        self.assertEqual(identity["held_out_passages_excluded"], 2)

    def test_validation_split_refuses_a_leaked_passage(self):
        validation, meta = [rec(1), rec(3)], {"test_passage_ids": [3]}
        sensitivity = {"split": {"sha256": M.passage_ids_sha256([1, 3])}}
        with mock.patch("src.wang_data.load_sentence_records", return_value=[]), \
                mock.patch("src.wang_data.group_split_records",
                           return_value=(validation, [], meta)), \
                mock.patch.multiple(M, EXPECTED_VALIDATION_PASSAGES=2,
                                    EXPECTED_VALIDATION_SENTENCES=2,
                                    EXPECTED_VALIDATION_SUBCLAIMS=2):
            with self.assertRaises(HeldOutLeak):
                M.validation_split("unused", sensitivity)


class TestCells(unittest.TestCase):
    def test_run_cells_evaluates_exactly_the_preregistered_candidates(self):
        seen = []

        def fake_evaluate(detector, records, held_out_ids, scorer, where):
            seen.append((where, type(detector.ratio_estimator).__name__,
                         getattr(detector.ratio_estimator, "kappa", None),
                         detector.lower_threshold, detector.upper_threshold))
            return {"nonfactual": {"auc_pr": 0.9, "precision": 1, "recall": 1},
                    "factual": {"auc_pr": 0.7, "precision": 1, "recall": 1},
                    "balanced_pr_auc": 0.8, "accuracy": 1, "macro_f1": 1,
                    "efficiency": {"avg_retrieved_documents_per_sentence": 5,
                                   "avg_retrieved_documents_per_subclaim": 3,
                                   "avg_nli_span_calls_per_sentence": 15}}

        pos = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
        neg = [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]
        d0 = SimpleNamespace(ratio=lambda s: 1.0)
        ref = {"nonfactual_auc_pr": 0.9, "factual_auc_pr": 0.7, "balanced_pr_auc": 0.8}
        with mock.patch.object(M, "evaluate", side_effect=fake_evaluate):
            rows = M.run_cells([rec(1)], [9], None, pos, neg, d0, ref)
        self.assertEqual({c: len(r) for c, r in rows.items()}, {"B": 32, "C": 224, "D0": 32})
        self.assertEqual({w for w, *_ in seen}, {"cell B", "cell C", "cell D0"})
        self.assertEqual(sorted({k for w, _, k, *_ in seen if w == "cell C"}),
                         list(S.KAPPA_GRID))
        self.assertTrue(all(k == 1.0 for w, _, k, *_ in seen if w == "cell B"))
        self.assertTrue(all(t == "SimpleNamespace" for w, t, *_ in seen if w == "cell D0"))


class TestInputs(unittest.TestCase):
    def test_wrong_digests_and_an_existing_freeze_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = M.paths(Path(tmp))
            for name in ("validation_cache", "d03_report", "sensitivity"):
                p[name].write_text("x")
            p["freeze"].write_text("{}")
            with self.assertRaises(S.SelectionRefused) as ctx:
                M.verify_inputs(p)
            text = str(ctx.exception)
            for name in ("validation_cache", "d03_report", "sensitivity", "freeze"):
                self.assertIn(name, text)
            self.assertEqual(p["freeze"].read_text(), "{}")

    def test_dry_run_evaluates_nothing_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sensitivity = {
                "bse_official_validation": {"nonfactual": {"auc_pr": 0.9},
                                            "factual": {"auc_pr": 0.7},
                                            "balanced_pr_auc": 0.8},
                "hyperparameter_rows": [{"is_production_pair": True, "sigma": S.D0_SIGMA,
                                         "lambda": S.D0_LAMBDA, "threshold_candidates": []}],
            }
            (Path(tmp) / M.SENSITIVITY_NAME).write_text(json.dumps(sensitivity))
            args = SimpleNamespace(artifacts_dir=tmp, data_root="unused", dry_run=True)
            with mock.patch.object(M, "parse_args", return_value=args), \
                    mock.patch.object(M, "verify_inputs", return_value={}), \
                    mock.patch.object(M, "validation_split", return_value=([], [], {})), \
                    mock.patch.object(M, "evaluate") as evaluate, \
                    mock.patch.object(M, "run_cells") as run_cells:
                self.assertEqual(M.main(), 0)
            evaluate.assert_not_called()
            run_cells.assert_not_called()
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [M.SENSITIVITY_NAME])


if __name__ == "__main__":
    unittest.main()
