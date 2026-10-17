"""The HELM preregistration's decision rule (src/helm_decision.py)."""

import unittest

import numpy as np

from src import helm_decision as D


def draws(center, spread=0.005, n=2000, seed=0):
    return list(np.random.default_rng(seed).normal(center, spread, n))


def build(**centers):
    obs, dr = {}, {}
    for t in D.TESTS:
        c = centers.get(t, 0.0)
        obs[D.endpoint_key(t)] = c
        dr[D.endpoint_key(t)] = draws(c, seed=hash(t) % 1000)
    return obs, dr


class Decision(unittest.TestCase):
    def test_all_positive_confirms_both(self):
        d = D.decide(*build(H1a=0.02, H1b=0.02, H2a=0.02, H2b=0.02))
        self.assertTrue(all(s["supported"] for s in d["tests"].values()))
        self.assertTrue(d["statements"]["H1"].startswith("confirmed"))
        self.assertTrue(d["statements"]["H2"].startswith("confirmed"))

    def test_one_cost_setting_is_partial_support(self):
        d = D.decide(*build(H1a=0.02, H1b=0.0, H2a=0.0, H2b=0.0))
        self.assertTrue(d["tests"]["H1a"]["supported"])
        self.assertFalse(d["tests"]["H1b"]["supported"])
        self.assertIn("partial support at C_M/C_FA 28/96", d["statements"]["H1"])
        self.assertTrue(d["statements"]["H2"].startswith("not confirmed"))

    def test_below_is_reported(self):
        d = D.decide(*build(H1a=-0.02, H1b=-0.02))
        self.assertTrue(d["tests"]["H1a"]["below"])
        self.assertIn("lies below BSE at 28/96, 14/24", d["statements"]["H1"])

    def test_undefined_replicates_count_against(self):
        obs, dr = build(H1a=0.02, H1b=0.02, H2a=0.02, H2b=0.02)
        key = D.endpoint_key("H1a")
        dr[key] = dr[key][:1900] + [None] * 100      # 5% undefined
        s = D.test_summary(obs[key], dr[key])
        self.assertEqual(s["undefined"], 100)
        self.assertAlmostEqual(s["p_one_sided"], 0.05, places=3)
        self.assertEqual(s["q05"], float("-inf"))
        self.assertFalse(D.decide(obs, dr)["tests"]["H1a"]["supported"])

    def test_holm_step_down(self):
        h = D.holm({"a": 0.001, "b": 0.013, "c": 0.02, "d": 0.5})
        self.assertEqual([h[k]["reject"] for k in "abcd"], [True, True, True, False])
        self.assertAlmostEqual(h["a"]["threshold"], 0.0125)
        h = D.holm({"a": 0.001, "b": 0.06, "c": 0.02, "d": 0.01})
        self.assertEqual([h[k]["reject"] for k in "abcd"], [True, False, True, True])

    def test_keys_match_the_budget_endpoints(self):
        self.assertEqual(D.endpoint_key("H2b"), "DDRE-1D|fixed-max|14/24|auc_gap")
        self.assertEqual(D.endpoint_key("H1a"), "DDRE-1D|BSE|28/96|auc_gap")

    def test_no_replicates_refused(self):
        with self.assertRaises(D.DecisionError):
            D.test_summary(0.0, [])


if __name__ == "__main__":
    unittest.main()
