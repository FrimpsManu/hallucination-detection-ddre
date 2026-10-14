"""FActScore transfer analysis and replay script (transfer preregistration §5-§8).

Synthetic observations only: no model, no cache, no Wang or FActScore data.
"""

import ast
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import factscore_transfer as T
from src.baseline_core import DetectionResult
from src.evaluation import EvaluatedSentence

ROOT = Path(__file__).resolve().parents[1]
PREREG = (ROOT / "docs" / "factscore_transfer_preregistration.md").read_text()


def obs(entity, gen, s, label, p, docs):
    return EvaluatedSentence(passage_index=entity, sentence_index=gen * 1_000_000 + s,
                             gold_label=label,
                             result=DetectionResult(p_factual=p, prediction=int(p > 0.2258),
                                                    documents_used=docs, nli_calls=docs))


def cell(docs_fn, shift=0.0):
    out = []
    for entity in range(10):
        for gen in (0, 1):
            for s in range(2):
                label = (entity + gen + s) % 2
                p = min(max(0.2 + 0.5 * label + 0.03 * s + shift, 0.0), 1.0)
                out.append(obs(entity, gen, s, label, p, docs_fn(entity)))
    return out


CELLS = {"A": cell(lambda e: 6), "B": cell(lambda e: 7), "C": cell(lambda e: 5),
         "D": cell(lambda e: 4, shift=0.01)}


class TestBootstrap(unittest.TestCase):
    def test_entity_is_the_cluster_across_generators(self):
        out = T.bootstrap(CELLS, n_resamples=50)
        self.assertEqual(out["clusters"], 10)
        self.assertEqual(out["unit"], "entity")
        self.assertEqual((T.N_RESAMPLES, T.SEED, T.CI_LEVEL), (10_000, 42, 0.95))

    def test_d_minus_a_endpoints_are_added(self):
        e = T.bootstrap(CELLS, n_resamples=50)["endpoints"]
        for key in ("balanced_D_minus_A", "nonfactual_auc_pr_D_minus_A",
                    "factual_auc_pr_D_minus_A", "docs_S_D", "docs_C_minus_D"):
            self.assertIn(key, e)
        self.assertEqual(e["docs_S_D"]["observed"], 2.0)
        self.assertEqual((e["docs_S_D"]["ci_lower"], e["docs_S_D"]["ci_upper"]), (2.0, 2.0))

    def test_subset_by_generator_keeps_alignment(self):
        sub = T.subset(CELLS, lambda o: o.sentence_index // 1_000_000 == 1)
        self.assertEqual({c: len(v) for c, v in sub.items()}, {c: 20 for c in "ABCD"})
        self.assertEqual(T.bootstrap(sub, n_resamples=20)["clusters"], 10)


def ep(sd_lo, bal_lo):
    return {"docs_S_D": {"observed": 1.0, "ci_lower": sd_lo, "ci_upper": 2.0},
            "balanced_D_minus_A": {"observed": 0.0, "ci_lower": bal_lo, "ci_upper": 0.05}}


class TestRQ1(unittest.TestCase):
    def test_margin_matches_the_preregistration(self):
        self.assertEqual(T.QUALITY_MARGIN, 0.03)
        self.assertIn("is ≥ −0.03", PREREG)

    def test_rules(self):
        self.assertTrue(T.rq1(ep(0.1, -0.03))["rq1_supported_on_factscore"])
        self.assertFalse(T.rq1(ep(0.1, math.nextafter(-0.03, -1)))["quality_comparable"])
        self.assertFalse(T.rq1(ep(0.0, 0.01))["efficiency_transfers"])
        self.assertTrue(T.rq1(ep(0.1, 0.001))["quality_improvement_shown"])
        self.assertFalse(T.rq1(ep(0.1, 0.0))["quality_improvement_shown"])


class TestSupport(unittest.TestCase):
    def test_thresholds_match_the_preregistration(self):
        self.assertEqual(T.SUPPORT_THRESHOLDS, (76.26953125, 83.35, 95.0))
        for text in ("above 76.27", "above 83.35", "at or above 95", "more than 10%"):
            self.assertIn(text, PREREG.replace("\n", " "))

    def test_shares_and_label(self):
        inside = T.support_diagnostic({c: [10, 50, 80, 84] + [30] * 36 for c in "ABCD"},
                                      [10, 99])
        self.assertEqual(inside["consumed"]["D"]["share_above_83.35"], 1 / 40)
        self.assertFalse(inside["outside_frozen_estimator_support"])
        self.assertEqual(inside["all_retrieved"]["share_at_or_above_95"], 0.5)
        outside = T.support_diagnostic({**{c: [30] for c in "ABC"}, "D": [84, 30]}, [84])
        self.assertTrue(outside["outside_frozen_estimator_support"])


def load_script():
    spec = importlib.util.spec_from_file_location(
        "run_factscore_transfer", ROOT / "scripts" / "run_factscore_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load_script()
SOURCE = (ROOT / "scripts" / "run_factscore_transfer.py").read_text()


class TestScript(unittest.TestCase):
    def test_cli_has_no_configuration_options(self):
        with mock.patch("sys.argv", ["x"]):
            args = S.parse_args()
        self.assertEqual(set(vars(args)),
                         {"release_dir", "artifacts_dir", "wang_data_root", "dry_run"})

    def test_no_selection_tuning_or_inference_path(self):
        names = set()
        for node in ast.walk(ast.parse(SOURCE)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name for a in node.names)
        banned = {"select", "cell_parameters", "candidate_row", "KAPPA_GRID", "BAND_GRID",
                  "select_threshold_configuration", "cost_consistent_thresholds",
                  "tune_ddre_thresholds", "transformers", "torch", "from_pretrained",
                  "EntailmentScorer", "complete_cache"}
        self.assertEqual(names & banned, set())

    def test_pins(self):
        pins = S.expected_digests()
        self.assertEqual(pins["validation_freeze"],
                         "a1824f663653b66df9deec590146de67992bd6a63320ccfc5f18e9a7bd7967ac")
        self.assertEqual(pins["evidence"],
                         "44f479128d41818fe5a056f125407dbde75a8735669c398a3b945cfeaa310d09")
        self.assertEqual(pins["preregistration"], S.PREREGISTRATION_SHA256)

    def test_unpinned_or_existing_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = S.paths(Path(tmp), Path(tmp))
            p["result"].write_text("{}")
            with self.assertRaises(S.TransferRefused) as ctx:
                S.verify_inputs(p)
            text = str(ctx.exception)
            self.assertIn("already exists", text)
            self.assertIn("evidence", text)

    def test_dry_run_replays_nothing(self):
        args = SimpleNamespace(release_dir="r", artifacts_dir="a", wang_data_root="w",
                               dry_run=True)
        with mock.patch.object(S, "parse_args", return_value=args), \
                mock.patch.object(S, "verify_inputs", return_value={}), \
                mock.patch.object(S.AR, "frozen_cells", return_value={}), \
                mock.patch.object(S, "load_analysis_set", return_value=[]), \
                mock.patch.object(S, "read_evidence", return_value=[]), \
                mock.patch.object(S, "build_records", return_value=([], {})), \
                mock.patch.object(Path, "read_text", return_value="{}"), \
                mock.patch.object(S.RM, "build") as build, \
                mock.patch.object(S.RM, "replay") as replay:
            self.assertEqual(S.main(), 0)
        build.assert_not_called()
        replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
