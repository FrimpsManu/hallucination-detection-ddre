"""``scripts/run_ablation_heldout_replay.py`` cannot run anything but the frozen cells.

No CLI option touches a configuration; no selection or tuning code is
reachable; D0 is never built; every pinned input is checked; a consistency-gate
failure stops the run before B or C and writes nothing; a dry run replays
nothing. No model, no torch, no Wang data.
"""

import argparse
import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import ablation_replay as R
from src import ablation_selection as S

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_ablation_heldout_replay.py"
SOURCE = SCRIPT.read_text()
TREE = ast.parse(SOURCE)


def load():
    spec = importlib.util.spec_from_file_location("run_ablation_heldout_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load()


def names(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
    return out


class TestNoOverrides(unittest.TestCase):
    def test_cli_has_no_configuration_options(self):
        with mock.patch("sys.argv", ["x"]):
            args = M.parse_args()
        self.assertEqual(set(vars(args)), {"artifacts_dir", "data_root", "dry_run"})
        parser_calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                        and getattr(n.func, "attr", None) == "add_argument"]
        flags = [a.value for c in parser_calls for a in c.args if isinstance(a, ast.Constant)]
        for flag in flags:
            for word in ("kappa", "sigma", "lambda", "lower", "upper", "threshold", "grid",
                         "tolerance", "margin", "config", "cell", "resample", "seed"):
                self.assertNotIn(word, flag)

    def test_no_selection_tuning_or_inference_path(self):
        banned = {
            "select", "order_key", "candidate_row", "cell_parameters", "verify_band_grid",
            "BAND_GRID", "KAPPA_GRID", "D0_SIGMA", "D0_LAMBDA", "PRODUCTION_SIGMA",
            "select_threshold_configuration", "cost_consistent_thresholds",
            "tune_ddre_thresholds", "candidate_record", "assess_claim",
            "paired_passage_bootstrap", "transformers", "torch", "from_pretrained",
            "EntailmentScorer", "prepare_derived_cache", "write_cache",
        }
        self.assertEqual(names(TREE) & banned, set())
        module_tree = ast.parse((PROJECT_ROOT / "src" / "ablation_replay.py").read_text())
        self.assertEqual(names(module_tree) & banned, set())

    def test_exactly_the_four_cells_are_built_and_d0_is_not(self):
        build = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
                     and n.name == "build")
        dicts = [n for n in ast.walk(build) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "A" for k in n.keys)]
        self.assertEqual(len(dicts), 1)
        self.assertEqual([k.value for k in dicts[0].keys], ["A", "B", "C", "D"])
        estimators = [n for n in ast.walk(build) if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) == "candidate_estimator"]
        self.assertEqual(len(estimators), 1)
        self.assertEqual(ast.unparse(estimators[0].args[3]), "d['sigma']")
        self.assertNotIn('"D0"]', SOURCE.replace("cells['D0']", "").replace('cells["D0"]', ""))

    def test_band_cells_read_their_parameters_from_the_frozen_cells(self):
        build = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
                     and n.name == "build")
        text = ast.unparse(build)
        self.assertIn("kappa=cells['B']['kappa']", text)
        self.assertIn("kappa=cells['C']['kappa']", text)
        self.assertNotIn("2.5", text)
        self.assertNotIn("0.85", text)


class TestPins(unittest.TestCase):
    def test_pinned_digests(self):
        self.assertEqual(M.PINNED["validation_freeze"][1],
                         "a1824f663653b66df9deec590146de67992bd6a63320ccfc5f18e9a7bd7967ac")
        self.assertEqual(M.PINNED["full_cache"][1],
                         "2497487664e2df552ba9bf28947c4ccae7b5f66d2feccbada05ffa5aaea88f28")
        self.assertEqual(M.PINNED["full_cache_manifest"][1],
                         "5163c8a2ed8b8730e245964e65ce669d7bb30c6603e40de7be6d25d42d2c12d7")
        self.assertEqual(M.PINNED["canonical_predictions"][1],
                         "a78b09b7aa09803a21a761b3bede7600180014b97142a381b509d7a1267211c1")

    def test_wrong_digests_and_existing_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = M.paths(Path(tmp))
            for name in M.PINNED:
                p[name].write_text("not pinned")
            p["result"].write_text("{}")
            p["preregistration"] = Path(tmp) / "prereg.md"
            p["preregistration"].write_text("edited")
            with self.assertRaises(R.ReplayRefused) as ctx:
                M.verify_inputs(p)
            text = str(ctx.exception)
            for name in list(M.PINNED) + ["preregistration", "result"]:
                self.assertIn(name, text)

    def test_an_incomplete_cache_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = M.paths(Path(tmp))
            p["full_cache_manifest"].write_text(json.dumps(
                {"status": "incomplete", "derived_sha256": M.PINNED["full_cache"][1],
                 "coverage": {"missing_after": 3}}))
            with mock.patch.object(M, "sha256_file",
                                   side_effect=lambda path: _pinned_for(M, p, path)), \
                    mock.patch.object(Path, "exists", lambda self: self.name not in
                                      (M.RESULT_NAME, M.PREDICTIONS_NAME)):
                with self.assertRaises(R.ReplayRefused) as ctx:
                    M.verify_inputs(p)
            self.assertIn("manifest", str(ctx.exception))


def _pinned_for(module, p, path):
    for name, (filename, sha) in module.PINNED.items():
        if Path(path) == p[name]:
            return sha
    return S.PREREGISTRATION_SHA256


class TestRunOrder(unittest.TestCase):
    def _main(self, tmp, *, dry_run, replay_rows):
        canonical = Path(tmp) / M.PINNED["canonical_predictions"][0]
        row = {"method": "bse_official", "passage_index": 0, "sentence_index": 0,
               "gold_label": 0, "p_factual": 0.1, "prediction": 0, "retrieved_documents": 4,
               "nli_span_calls": 9, "subclaim_documents_used": "[4]",
               "subclaim_nli_calls": "[9]"}
        with canonical.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
            w.writerow(dict(row, method="frozen_ddre"))
        (Path(tmp) / M.PINNED["validation_freeze"][0]).write_text("{}")
        (Path(tmp) / M.PINNED["canonical_result"][0]).write_text("{}")
        args = SimpleNamespace(artifacts_dir=tmp, data_root="unused", dry_run=dry_run)
        cells = {"A": {}, "B": {}, "C": {}, "D": {}, "D0": {"reason": "x" * 80}}
        scorer = mock.Mock(misses=0)
        calls = []

        def fake_replay(cell, detector, records, scorer):
            calls.append(cell)
            return {}, [], [dict(r) for r in replay_rows]

        with mock.patch.object(M, "parse_args", return_value=args), \
                mock.patch.object(M, "verify_inputs", return_value={}), \
                mock.patch.object(M.R, "frozen_cells", return_value=cells), \
                mock.patch.object(M, "held_out", return_value=([], {})), \
                mock.patch.object(M, "build", return_value=({c: mock.Mock() for c in "ABCD"}, scorer)) as build, \
                mock.patch.object(M, "replay", side_effect=fake_replay):
            try:
                code = M.main()
            except R.ReplayRefused as exc:
                code = type(exc).__name__
        return code, calls, build, row

    def test_gate_failure_stops_before_b_and_c_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = [{"passage_index": 0, "sentence_index": 0, "p_factual": 0.2,
                    "prediction": 0, "retrieved_documents": 4, "nli_span_calls": 9,
                    "subclaim_documents_used": "[4]", "subclaim_nli_calls": "[9]"}]
            code, calls, _, _ = self._main(tmp, dry_run=False, replay_rows=bad)
            self.assertEqual(code, "ConsistencyGateFailed")
            self.assertEqual(calls, ["A"])
            self.assertFalse((Path(tmp) / M.RESULT_NAME).exists())
            self.assertFalse((Path(tmp) / M.PREDICTIONS_NAME).exists())

    def test_dry_run_replays_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, build, _ = self._main(tmp, dry_run=True, replay_rows=[])
            self.assertEqual(code, 0)
            self.assertEqual(calls, [])
            build.assert_not_called()
            self.assertFalse((Path(tmp) / M.RESULT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
