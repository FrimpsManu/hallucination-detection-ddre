"""Amendment 2: budget-knob curves (src/v2_budget.py, scripts/run_v2_budget.py)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_v2_budget as RB  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_budget as B  # noqa: E402
from src import v2_replay as R  # noqa: E402
from test_run_v2_replay import histogram_everything  # noqa: E402
from test_v2_replay import synthetic  # noqa: E402


class Grid(unittest.TestCase):
    def test_grid_and_operating_point(self):
        self.assertEqual(len(B.grid(5)), 5 * len(B.BUDGET_C_RETRIEVE))
        self.assertEqual(B.operating_key(5), "K5/cr1")
        self.assertIn(1.0, B.BUDGET_C_RETRIEVE)
        self.assertTrue(set(B.BUDGET_C_RETRIEVE) <= set(R.C_RETRIEVE_GRID))


class Curves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = synthetic(50, seed=6, max_docs=4)
        cls.est, cls.sup = histogram_everything()
        cls.policies = RB.build_policies(cls.sup)
        cls.curves = RB.budget_curves(cls.prepared, cls.est, cls.sup, cls.policies, 4)
        cls.labels = np.array([s["label"] for s in cls.prepared])

    def test_operating_point_is_phase5s(self):
        lrs = R.log_ratios(self.prepared, self.est["BSE"])
        costs = S.Costs(28, 96, 1)
        want = R.run_backward_induction(self.prepared, lrs,
                                        S.PolicyByHorizon(self.sup["BSE"], costs, 10), costs, 4)
        got = self.curves[("BSE", (28.0, 96.0))]["K4/cr1"]
        self.assertTrue(np.array_equal(got["p_factual"], want["p_factual"]))
        self.assertTrue(np.array_equal(got["documents"], want["documents"]))

    def test_k_caps_reading(self):
        for k in range(1, 5):
            r = self.curves[("DDRE-M", (28.0, 96.0))][f"K{k}/cr0.125"]
            for j, s in enumerate(self.prepared):
                self.assertLessEqual(r["documents"][j], k * len(s["subclaims"]))

    def test_fixed_k_is_wangs_maximum(self):
        for k in range(1, 5):
            want = R.run_fixed_k(self.prepared, None, k, "max", 4)
            self.assertTrue(np.array_equal(self.curves[(B.FIXED, None)][f"K{k}"]["p_factual"],
                                           want["score"]))

    def test_identical_methods_give_zero(self):
        e = B.contrast_endpoints(self.curves, self.labels, np.arange(len(self.labels)), 4,
                                 R.COST_PAIRS, contrasts=B.CONTRASTS[:2])
        for k, v in e.items():
            self.assertIsNotNone(v, k)
            self.assertAlmostEqual(v, 0.0, places=12, msg=k)

    def test_fixed_contrasts_are_anchored_at_the_method(self):
        e = B.contrast_endpoints(self.curves, self.labels, np.arange(len(self.labels)), 4,
                                 R.COST_PAIRS)
        self.assertIn("DDRE-1D|fixed-max|28/96|reference_never_reaches_nli", e)
        self.assertIn("DDRE-M|fixed-max|14/24|e_quality", e)

    def test_gate_x4(self):
        rows = {}
        for m in B.METHODS:
            for cm, cfa in R.COST_PAIRS:
                r = self.curves[(m, (cm, cfa))]["K4/cr1"]
                for j, s in enumerate(self.prepared):
                    rows[("d", m, R.setting_key(cm, cfa, 1.0), s["id"])] = {
                        "p_factual": str(float(r["p_factual"][j])),
                        "documents": str(int(r["documents"][j])),
                        "nli_calls": str(int(r["nli_calls"][j]))}
        self.assertEqual(RB.gate_x4("d", self.prepared, self.curves, 4, rows), 0)
        key = next(iter(rows))
        rows[key] = dict(rows[key], nli_calls="999")
        self.assertEqual(RB.gate_x4("d", self.prepared, self.curves, 4, rows), 1)


class Inputs(unittest.TestCase):
    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = RB.output_path(Path(tmp))
            out.write_text("{}")
            pins = {k: (Path(tmp) / f"{k}.missing", v[1]) for k, v in RB.PINS.items()}
            with mock.patch.object(RB, "PINS", pins):
                with self.assertRaises(RB.BudgetRefused) as ctx:
                    RB.verify_inputs(out, 3)
            msg = str(ctx.exception)
            self.assertIn("phase5_result:", msg)
            self.assertIn("already exists", msg)
            self.assertIn("n_resamples is fixed", msg)

    def test_plan_at_amendment_2(self):
        from src import plan_v2
        self.assertEqual(plan_v2.plan_problems(ROOT, require_amendment_2=True), [])


if __name__ == "__main__":
    unittest.main()
