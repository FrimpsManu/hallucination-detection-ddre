"""Plan-v2 estimator additions (src/mddre_estimators.py, src/sequential.py)
and the run-once fit script (scripts/fit_v2_estimators.py)."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import mddre_estimators as E
from src import sequential as S

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def docs(n, e_mean, k_mean, seed, groups=20):
    rng = np.random.default_rng(seed)
    out = [{"e": float(np.clip(rng.normal(e_mean, 10), 1, 99)),
            "k": float(np.clip(rng.normal(k_mean, 10), 1, 99)),
            "rank": int(rng.integers(1, 11))} for _ in range(n)]
    return out, [f"g{seed}-{i % groups}" for i in range(n)]


FACT, FG = docs(400, 60, 20, 1)
HALL, HG = docs(400, 30, 40, 2)


class GroupFolds(unittest.TestCase):
    def test_a_group_never_straddles_folds(self):
        groups = [f"e{i % 37}" for i in range(500)]
        fold = E.group_folds(groups)
        seen = {}
        for g, k in zip(groups, fold):
            self.assertEqual(seen.setdefault(g, k), k)
        self.assertEqual(set(fold), set(range(E.CV_FOLDS)))

    def test_deterministic_and_needs_enough_groups(self):
        groups = [i % 9 for i in range(90)]
        self.assertTrue(np.array_equal(E.group_folds(groups), E.group_folds(groups)))
        with self.assertRaises(E.EstimatorError):
            E.group_folds(["a", "b", "c"])


class GroupedULSIF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = E.MultiULSIF(E.FEATURE_SETS["M"]).fit(
            FACT, HALL, factual_groups=FG, hallucinated_groups=HG,
            sigma_multipliers=E.SIGMA_MULTIPLIERS_V2)

    def test_v2_grid_and_grouped_cv_recorded(self):
        self.assertEqual(len(self.m.sigma_grid), len(E.SIGMA_MULTIPLIERS_V2))
        self.assertTrue(self.m.diagnostics["grouped_cv"])
        self.assertEqual(self.m.diagnostics["sigma_multipliers"], list(E.SIGMA_MULTIPLIERS_V2))
        self.assertEqual(len(self.m.cv_table), len(E.SIGMA_MULTIPLIERS_V2) * len(E.LAMBDA_GRID))

    def test_self_normalised_and_directional(self):
        self.assertAlmostEqual(self.m.diagnostics["mean_r_hallucinated"], 1.0, places=6)
        self.assertGreater(self.m.log_ratio({"e": 70, "k": 15, "rank": 1}),
                           self.m.log_ratio({"e": 25, "k": 45, "rank": 1}))

    def test_default_fit_is_the_historical_one(self):
        a = E.MultiULSIF(E.FEATURE_SETS["U"]).fit(FACT, HALL)
        b = E.MultiULSIF(E.FEATURE_SETS["U"]).fit(FACT, HALL, factual_groups=None,
                                                    hallucinated_groups=None)
        self.assertEqual(a.cv_table, b.cv_table)
        self.assertFalse(a.diagnostics["grouped_cv"])
        self.assertEqual(a.diagnostics["sigma_multipliers"], list(E.SIGMA_MULTIPLIERS))

    def test_groups_for_one_class_only_refused(self):
        with self.assertRaises(E.EstimatorError):
            E.MultiULSIF(("e",)).fit(FACT, HALL, factual_groups=FG)


class NaiveBayes(unittest.TestCase):
    def test_log_ratio_is_the_sum_of_per_feature_histograms(self):
        nb = E.NaiveBayesEvidence().fit(FACT, HALL)
        doc = {"e": 55.0, "k": 25.0, "rank": 3}
        want = 0.0
        for name in E.NB_FEATURES:
            if name == "rank":
                b = 2
                pos = [1] * 10
                neg = [1] * 10
                for d in FACT:
                    pos[d["rank"] - 1] += 1
                for d in HALL:
                    neg[d["rank"] - 1] += 1
            else:
                h = E.HistogramEvidence.fit([d[name] for d in FACT], [d[name] for d in HALL])
                pos, neg = h.pos_hist, h.neg_hist
                from src.baseline_core import discretize_document_score
                b = discretize_document_score(doc[name])
            want += math.log((pos[b] / sum(pos)) / (neg[b] / sum(neg)))
        self.assertAlmostEqual(nb.log_ratio(doc), want, places=12)

    def test_normalised_mean_is_one_on_hallucinated_training(self):
        nb = E.NaiveBayesEvidence().fit(FACT, HALL, normalise=True)
        self.assertAlmostEqual(nb.diagnostics["mean_r_hallucinated"], 1.0, places=10)
        self.assertGreater(nb.diagnostics["mean_r_factual"], 1.0)

    def test_round_trip_and_rank_range(self):
        nb = E.NaiveBayesEvidence().fit(FACT, HALL, normalise=True)
        back = E.from_json(json.loads(json.dumps(nb.to_json())))
        for d in FACT[:30] + HALL[:30]:
            self.assertEqual(back.log_ratio(d), nb.log_ratio(d))
        with self.assertRaises(E.EstimatorError):
            nb.log_ratio({"e": 50.0, "k": 10.0, "rank": 11})


class Logistic(unittest.TestCase):
    def test_grouped_and_normalised(self):
        lr = E.LogisticRatio(E.FEATURE_SETS["M"]).fit(
            FACT, HALL[:200], factual_groups=FG, hallucinated_groups=HG[:200], normalise=True)
        self.assertTrue(lr.diagnostics["grouped_cv"])
        self.assertAlmostEqual(lr.diagnostics["mean_r_hallucinated"], 1.0, places=6)
        back = E.from_json(json.loads(json.dumps(lr.to_json())))
        self.assertEqual(back.log_ratio(FACT[0]), lr.log_ratio(FACT[0]))

    def test_historical_record_without_normaliser_loads(self):
        lr = E.LogisticRatio(E.FEATURE_SETS["M"]).fit(FACT, HALL)
        record = lr.to_json()
        del record["log_normaliser"]
        self.assertEqual(E.from_json(record).log_ratio(HALL[0]), lr.log_ratio(HALL[0]))


class Supports(unittest.TestCase):
    def test_rank_reading_estimators_get_ranked_support(self):
        m = E.MultiULSIF(E.FEATURE_SETS["M"]).fit(FACT, HALL)
        u = E.MultiULSIF(E.FEATURE_SETS["U"]).fit(FACT, HALL)
        sm, su = S.build_support(m, HALL), S.build_support(u, HALL)
        self.assertIsInstance(sm, S.RankedSupport)
        self.assertEqual(set(sm.by_rank), {d["rank"] for d in HALL})
        at3 = [m.log_ratio(d) for d in HALL if d["rank"] == 3]
        self.assertTrue(np.allclose(sm.at(3).log_ratios, at3))
        self.assertIsInstance(su, S.Support)
        self.assertEqual(su.log_ratios.size, len(HALL))
        self.assertAlmostEqual(su.mean_ratio, 1.0, places=6)

    def test_serialisation_round_trip(self):
        m = E.NaiveBayesEvidence().fit(FACT, HALL, normalise=True)
        support = S.build_support(m, HALL)
        back = S.support_from_json(json.loads(json.dumps(S.support_to_json(support))))
        for r in support.by_rank:
            self.assertTrue(np.array_equal(back.at(r).log_ratios, support.at(r).log_ratios))
            self.assertTrue(np.allclose(back.at(r).weights, support.at(r).weights))
        with self.assertRaises(S.SequentialError):
            S.build_support(m, [])


class FitScript(unittest.TestCase):
    def setUp(self):
        import fit_v2_estimators as F
        self.F = F

    def test_gate_x1_compares_with_gate1(self):
        F = self.F
        recorded = {"nbc_histograms_laplace_smoothed": {
            "positive": [1, 78, 19, 13, 6, 5, 28, 57, 1, 1],
            "negative": [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            path.write_text(json.dumps(recorded))
            with mock.patch.object(F, "GATE1", (path, "x")):
                ok = E.HistogramEvidence(recorded["nbc_histograms_laplace_smoothed"]["positive"],
                                         recorded["nbc_histograms_laplace_smoothed"]["negative"])
                self.assertTrue(F.gate_bse_histogram(ok)["passed"])
                bad = E.HistogramEvidence([2] * 10, [2] * 10)
                with self.assertRaises(F.FitRefused):
                    F.gate_bse_histogram(bad)

    def test_refuses_when_the_output_exists_or_the_plan_changed(self):
        F = self.F
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / F.OUT_NAME
            out.write_text("{}")
            with mock.patch.object(F, "PLAN", (F.PLAN[0], "0" * 64)):
                with self.assertRaises(F.FitRefused) as ctx:
                    F.verify_inputs(out)
            msg = str(ctx.exception)
            self.assertIn("already exists", msg)
            self.assertIn("approved plan", msg)

    def test_missing_artifacts_are_reported_not_raised(self):
        F = self.F
        with tempfile.TemporaryDirectory() as tmp:
            gone = (Path(tmp) / "absent.sqlite", F.FEATURES[1])
            with mock.patch.object(F, "FEATURES", gone), \
                    mock.patch.object(F, "GATE1", (Path(tmp) / "absent.json", "x")):
                with self.assertRaises(F.FitRefused) as ctx:
                    F.verify_inputs(Path(tmp) / F.OUT_NAME)
            self.assertIn("absent.sqlite is missing", str(ctx.exception))
            self.assertIn("absent.json is missing", str(ctx.exception))

    def test_plan_pin_matches_the_merged_plan(self):
        from src.factscore_data import sha256_path
        self.assertEqual(sha256_path(ROOT / self.F.PLAN[0]), self.F.PLAN[1])

    def test_fit_tier_supports(self):
        F = self.F
        tier = {"factual": (FACT, FG), "hallucinated": (HALL, HG)}
        fitted, supports = F.fit_tier(tier, multivariate=True)
        self.assertEqual(set(fitted), {"Hist-1D", "DDRE-1D'", "NB-M", "DDRE-M", "Logit-M"})
        self.assertIsInstance(supports["Hist-1D"], S.Support)
        self.assertEqual(supports["Hist-1D"].log_ratios.size, 10)
        for name in ("NB-M", "DDRE-M", "Logit-M"):
            self.assertIsInstance(supports[name], S.RankedSupport)
        self.assertNotIsInstance(supports["DDRE-1D'"], S.RankedSupport)
        for name in ("NB-M", "DDRE-M", "Logit-M", "DDRE-1D'"):
            self.assertAlmostEqual(fitted[name].diagnostics["mean_r_hallucinated"], 1.0, places=5)
        # Each support yields a valid Eq. 8 policy.
        for s in supports.values():
            S.Eq8Policy(s, S.Costs(28, 96, 1))


if __name__ == "__main__":
    unittest.main()
