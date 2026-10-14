"""Multivariate DDRE estimators on synthetic data (design §3)."""

import math
import unittest

import numpy as np

from src import mddre_estimators as E


def docs(n, e_mean, k_mean, seed):
    rng = np.random.default_rng(seed)
    return [{"e": float(np.clip(rng.normal(e_mean, 10), 1, 99)),
             "k": float(np.clip(rng.normal(k_mean, 10), 1, 99)),
             "rank": int(rng.integers(1, 11))} for _ in range(n)]


FACT = docs(400, 60, 20, 1)
HALL = docs(400, 30, 40, 2)


class TestFixedGrids(unittest.TestCase):
    def test_grids_are_fixed_before_fitting(self):
        self.assertEqual(E.LAMBDA_GRID, (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0))
        self.assertEqual(E.SIGMA_MULTIPLIERS, (0.5, 1.0, 2.0, 4.0))
        self.assertEqual((E.N_CENTERS, E.CV_FOLDS, E.SEED), (100, 5, 42))
        self.assertEqual(E.FEATURE_SETS, {"U": ("e",), "M": ("e", "k", "log_rank")})


class TestHistogram(unittest.TestCase):
    def test_laplace_counts_and_document_binning(self):
        h = E.HistogramEvidence.fit([55.0, 65.0], [15.0])
        self.assertEqual(sum(h.pos_hist), 12)
        self.assertEqual(sum(h.neg_hist), 11)
        self.assertGreater(h.log_ratio({"e": 55.0}), 0)
        self.assertLess(h.log_ratio({"e": 15.0}), 0)


class TestULSIF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = E.MultiULSIF(E.FEATURE_SETS["M"]).fit(FACT, HALL)

    def test_self_normalised_on_hallucinated_training_samples(self):
        self.assertAlmostEqual(self.m.diagnostics["mean_r_hallucinated"], 1.0, places=6)
        self.assertGreater(self.m.diagnostics["mean_r_factual"], 1.0)

    def test_evidence_direction(self):
        self.assertGreater(self.m.log_ratio({"e": 65, "k": 15, "rank": 1}), 0)
        self.assertLess(self.m.log_ratio({"e": 25, "k": 45, "rank": 1}), 0)

    def test_selected_from_the_fixed_grid_by_cv_only(self):
        self.assertEqual(len(self.m.cv_table), len(E.SIGMA_MULTIPLIERS) * len(E.LAMBDA_GRID))
        best = min(self.m.cv_table, key=lambda r: r["cv_objective"])
        self.assertEqual((self.m.sigma, self.m.lam), (best["sigma"], best["lambda"]))

    def test_deterministic(self):
        again = E.MultiULSIF(E.FEATURE_SETS["M"]).fit(FACT, HALL)
        self.assertEqual(again.to_json()["alpha"], self.m.to_json()["alpha"])

    def test_one_dimensional_cell_u(self):
        u = E.MultiULSIF(E.FEATURE_SETS["U"]).fit(FACT, HALL)
        self.assertAlmostEqual(u.diagnostics["mean_r_hallucinated"], 1.0, places=6)
        self.assertGreater(u.log_ratio({"e": 65, "k": 0, "rank": 1}),
                           u.log_ratio({"e": 25, "k": 0, "rank": 1}))


class TestLogistic(unittest.TestCase):
    def test_direction_and_prior_correction(self):
        lr = E.LogisticRatio(E.FEATURE_SETS["M"]).fit(FACT, HALL[:200])
        self.assertAlmostEqual(lr.prior_correction, math.log(200 / 400))
        self.assertGreater(lr.log_ratio({"e": 65, "k": 15, "rank": 1}), 0)
        self.assertLess(lr.log_ratio({"e": 25, "k": 45, "rank": 1}), 0)


class TestFeatures(unittest.TestCase):
    def test_log_rank_and_refusal(self):
        self.assertEqual(E.features({"e": 1, "k": 2, "rank": 1}, ("log_rank",)).tolist(), [0.0])
        with self.assertRaises(E.EstimatorError):
            E.features({"e": float("nan"), "k": 0, "rank": 1}, ("e",))


if __name__ == "__main__":
    unittest.main()


class TestFitScriptReadsTrainingOnly(unittest.TestCase):
    def test_no_test_set_path(self):
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "scripts"
                  / "fit_mddre_estimators.py").read_text()
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        banned = {"load_factcheck", "held_out_records", "enumerate_datasets",
                  "load_analysis_set", "test", "DDREDetector", "BSEDetector"}
        self.assertEqual(names & banned, set())
        self.assertIn("training, _, split = M.factscore_split", source)
        self.assertIn("out.chmod(0o444)", source)
