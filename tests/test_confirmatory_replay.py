"""Confirmatory endpoints and decision rule (preregistration §4-§5), synthetic only."""

import unittest

import numpy as np

from src import confirmatory_replay as CR
from src.mddre_replay import NOT_REACHED


def pr_auc(y, s):
    from src.evaluation import wang_pr_auc
    return wang_pr_auc(y, s)


def make(labels, quality_noise, docs, op_band, rng):
    """A cell whose curve has len(docs) bands, cleaner with lower noise."""
    nb, n = len(docs), len(labels)
    p = np.clip(labels * 0.6 + 0.2 + rng.normal(0, quality_noise, (nb, n)), 0, 1)
    d = np.tile(np.asarray(docs, float)[:, None], (1, n))
    return {"p_factual": p, "documents": d, "nli": 2 * d,
            "op_p_factual": p[op_band], "op_documents": d[op_band], "op_nli": 2 * d[op_band]}


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.labels = np.array([i % 2 for i in range(80)])
        docs = [1, 2, 3, 4, 5, 6]
        self.curves = {"A": make(self.labels, 0.45, docs, 3, rng),
                       "M": make(self.labels, 0.05, docs, 3, rng)}
        self.ref = {k: self.curves["A"][f"op_{k}"] for k in ("p_factual", "documents", "nli")}

    def test_clean_cell_saves_documents_and_gains_quality(self):
        out, _ = CR.endpoints(self.curves, self.ref, ["A", "M"], self.labels, pr_auc,
                              np.arange(80), chain=[("A", "M")])
        self.assertGreater(out["M.E1"], 0)
        self.assertGreater(out["M.E2"], 0)
        self.assertLessEqual(out["A.E1"], 0)  # D_A is capped at BSE's own cost
        self.assertAlmostEqual(out["M-A.E2"], out["M.E2"] - out["A.E2"])
        self.assertAlmostEqual(out["M.E1_nli"], 2 * out["M.E1"])
        self.assertEqual(out["A.same_band_docs_saving"], 0.0)

    def test_not_reached_counts_against(self):
        bad = dict(self.curves["M"])
        bad["p_factual"] = np.full_like(bad["p_factual"], 0.5)  # constant: worst quality
        out, _ = CR.endpoints({"A": self.curves["A"], "M": bad}, self.ref, ["A", "M"],
                              self.labels, pr_auc, np.arange(80))
        self.assertEqual(out["M.E1"], NOT_REACHED)

    def test_bootstrap_runs_and_is_paired(self):
        clusters = [i // 4 for i in range(80)]
        b = CR.bootstrap(self.curves, self.ref, ["A", "M"], self.labels, clusters, pr_auc,
                         n_resamples=30)
        self.assertEqual(b["clusters"], 20)
        self.assertIn("M.E1", b["endpoints"])
        self.assertLessEqual(b["endpoints"]["M.E2"]["ci_lower"],
                             b["endpoints"]["M.E2"]["ci_upper"])


def summary(e1, e2):
    s = lambda o, lo, hi, p: {"observed": o, "ci_lower": lo, "ci_upper": hi, "p_two_sided": p}
    return {"M.E1": s(*e1), "M.E2": s(*e2)}


class TestDecision(unittest.TestCase):
    def test_confirmed_needs_both_datasets(self):
        good = summary((1, .5, 2, 0.0001), (.05, .02, .08, 0.0001))
        d = CR.decide({"expertqa": good, "bingcheck": good})
        self.assertTrue(d["lower_cost_and_better_detection_confirmed"])
        self.assertTrue(d["statements"]["E2"].startswith("confirmed"))

    def test_one_dataset_is_partial_and_negative_ci_is_reported(self):
        good = summary((1, .5, 2, 0.0001), (.05, .02, .08, 0.0001))
        worse = summary((-1, -2, -.5, 0.0001), (-.05, -.08, -.02, 0.0001))
        d = CR.decide({"expertqa": good, "bingcheck": worse})
        self.assertTrue(d["statements"]["E2"].startswith("partial support (expertqa)"))
        self.assertTrue(d["tests"]["bingcheck.E2"]["m_worse"])
        self.assertFalse(d["lower_cost_and_better_detection_confirmed"])

    def test_holm_is_over_all_four_tests(self):
        # p = 0.02 passes alone but not at Holm's first threshold 0.05 / 4.
        weak = summary((1, .5, 2, 0.02), (.05, .02, .08, 0.02))
        d = CR.decide({"expertqa": weak, "bingcheck": weak})
        self.assertEqual(len(d["tests"]), 4)
        self.assertFalse(any(t["supported"] for t in d["tests"].values()))
        self.assertTrue(d["statements"]["E1"].startswith("not confirmed"))


class TestReplayScript(unittest.TestCase):
    def test_structure(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_confirmatory_replay.py"
        spec = importlib.util.spec_from_file_location("run_confirmatory_replay", path)
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        self.assertEqual(S.DATASETS, ("expertqa", "bingcheck"))
        self.assertEqual(S.CELLS, ("A", "A_prime", "U", "M", "L"))
        self.assertEqual({m for _, m in S.SENSITIVITY},
                         {"unsure_hallucinated", "refuted_only", "not_supported_hallucinated"})
        self.assertTrue(S.output_paths(Path("/x"))["result"].name.startswith(
            "confirmatory_replay_result_"))


if __name__ == "__main__":
    unittest.main()
