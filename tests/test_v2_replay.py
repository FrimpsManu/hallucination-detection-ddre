"""Plan-v2 replay library (src/v2_replay.py) on synthetic sentences."""

import unittest

import numpy as np

from src import sequential as S
from src import v2_replay as R
from src.baseline_core import BSEDetector
from src.mddre_estimators import HistogramEvidence
from src.mddre_replay import bse_subclaim

POS = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
NEG = [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]


def synthetic(n=120, seed=0, max_docs=5):
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        label = int(rng.random() < 0.6)
        subs = []
        for _ in range(int(rng.integers(1, 4))):
            docs = []
            for rank in range(1, int(rng.integers(0, max_docs + 2)) + 1):
                e = float(np.clip(rng.normal(65 if label else 30, 18), 0.5, 99))
                docs.append({"e": e, "k": float(rng.uniform(1, 60)), "rank": min(rank, 10),
                             "spans": int(rng.integers(1, 4))})
            subs.append(docs)
        out.append({"id": str(j), "cluster": j // 4, "label": label, "subclaims": subs})
    return out


class Running(unittest.TestCase):
    def setUp(self):
        self.prepared = synthetic()
        self.hist = HistogramEvidence(POS, NEG)
        self.lrs = R.log_ratios(self.prepared, self.hist)

    def test_official_rule_is_bse_as_it_runs(self):
        costs = S.Costs(28, 96, 1)
        got = R.run_bse_official(self.prepared, self.hist, costs, 5)
        bse = BSEDetector(POS, NEG, mode="official")
        for j, s in enumerate(self.prepared):
            subs = [bse_subclaim([self.hist.likelihoods(d) for d in sub],
                                 [d["spans"] for d in sub], bse, 5) for sub in s["subclaims"]]
            self.assertEqual(got["p_factual"][j], min(r[0] for r in subs))
            self.assertEqual(got["documents"][j], sum(r[1] for r in subs))

    def test_eq8_with_the_histogram_is_bse_eq8(self):
        for cm, cfa, cr in ((14, 24, 0.25), (28, 96, 1)):
            costs = S.Costs(cm, cfa, cr)
            policy = S.Eq8Policy(S.Support.from_histogram(POS, NEG), costs)
            got = R.run_eq8(self.prepared, self.lrs, policy, costs, 5)
            bse = BSEDetector(POS, NEG, mode="eq8", c_miss=cm, c_false_alarm=cfa, c_retrieve=cr)
            for j, s in enumerate(self.prepared):
                docs = sum(bse_subclaim([self.hist.likelihoods(d) for d in sub],
                                        [d["spans"] for d in sub], bse, 5)[1]
                           for sub in s["subclaims"])
                self.assertEqual(got["documents"][j], docs)

    def test_backward_induction_reads_and_respects_max_docs(self):
        costs = S.Costs(28, 96, 0.25)
        policies = S.PolicyByHorizon(S.Support.from_histogram(POS, NEG), costs, 10)
        got = R.run_backward_induction(self.prepared, self.lrs, policies, costs, 3)
        self.assertGreater(got["documents"].sum(), 0)
        for j, s in enumerate(self.prepared):
            self.assertLessEqual(got["documents"][j], 3 * len(s["subclaims"]))
            self.assertEqual(len(got["subclaim_documents"][j]), len(s["subclaims"]))
            self.assertEqual(got["prediction"][j], S.decision(got["p_factual"][j], 28, 96))

    def test_fixed_k(self):
        got = R.run_fixed_k(self.prepared, self.lrs, 2, "max", 5)
        for j, s in enumerate(self.prepared):
            want = min((max(d["e"] for d in sub[:2]) / 100 if sub else 0.5)
                       for sub in s["subclaims"])
            self.assertAlmostEqual(got["score"][j], want, places=12)
            self.assertEqual(got["documents"][j], sum(min(2, len(sub)) for sub in s["subclaims"]))

    def test_a_sentence_without_subclaims_is_refused(self):
        bad = [{"id": "x", "cluster": 0, "label": 1, "subclaims": []}]
        with self.assertRaises(R.ReplayError):
            R.run_bse_official(bad, self.hist, S.Costs(), 5)


def fake_results(methods, n, seed, better=None):
    """Sweep results for each method; ``better`` gets sharper posteriors."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.6).astype(int)
    base = rng.normal(0, 1, n)
    out = {}
    for m in methods:
        out[m] = {}
        for cm, cfa, cr in R.settings():
            reads = 6.0 / (1 + cr)
            signal = 1.0 + (0.8 if m == better else 0.0)
            logit = signal * (2 * labels - 1) * min(reads, 3) / 3 + base
            p = 1 / (1 + np.exp(-logit))
            out[m][R.setting_key(cm, cfa, cr)] = {
                "p_factual": p, "nli_calls": np.full(n, 3 * reads),
                "documents": np.full(n, reads), "prediction": (p > 0.3).astype(int)}
    return R.with_rounded(out), labels


class Endpoints(unittest.TestCase):
    METHODS = ("BSE", "DDRE-1D", "Hist-1D", "DDRE-1D'", "NB-M", "DDRE-M", "Logit-M",
               "BSE-official")

    def test_identical_methods_give_zero_everywhere(self):
        results, labels = fake_results(self.METHODS, 300, 1)
        e = R.contrast_endpoints(results, labels, np.arange(300))
        for k, v in e.items():
            self.assertIsNotNone(v, k)
            self.assertAlmostEqual(v, 0.0, places=12, msg=k)
        self.assertIn("DDRE-M|NB-M|28/96|brier_reduction", e)
        self.assertNotIn("DDRE-M|Logit-M|28/96|brier_reduction", e)

    def test_a_sharper_method_gains_quality_at_equal_cost(self):
        results, labels = fake_results(self.METHODS, 400, 2, better="DDRE-1D")
        e = R.contrast_endpoints(results, labels, np.arange(400))
        self.assertGreater(e["DDRE-1D|BSE|28/96|e_quality"], 0)
        self.assertGreater(e["DDRE-1D|BSE|28/96|op_quality_gain"], 0)
        self.assertEqual(e["DDRE-1D|BSE|28/96|op_nli_saved"], 0)
        self.assertGreater(e["DDRE-1D|BSE|28/96|auc_gap"], 0)
        self.assertGreaterEqual(e["DDRE-1D|BSE|28/96|e_cost_nli"], 0)

    def test_grid(self):
        self.assertEqual(len(R.settings()), 2 * len(R.C_RETRIEVE_GRID))
        self.assertIn(1.0, R.C_RETRIEVE_GRID)
        self.assertAlmostEqual(R.C_RETRIEVE_GRID[0], 0.125)
        self.assertAlmostEqual(R.C_RETRIEVE_GRID[-1], 16.0)
        self.assertTrue(all(b > a for a, b in zip(R.C_RETRIEVE_GRID, R.C_RETRIEVE_GRID[1:])))


if __name__ == "__main__":
    unittest.main()
