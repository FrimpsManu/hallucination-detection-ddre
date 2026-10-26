"""The running-time measurement (src/runtime.py, scripts/run_runtime.py), without a model."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_runtime as RT  # noqa: E402
from src import mddre_estimators as E  # noqa: E402
from src import runtime as T  # noqa: E402
from src import sequential as S  # noqa: E402

POS = [3, 2, 2, 3, 4, 5, 8, 12, 30, 140]
NEG = [60, 30, 25, 20, 15, 12, 10, 8, 6, 15]


def estimators():
    h = E.HistogramEvidence(POS, NEG)
    s = S.Support.from_histogram(POS, NEG)
    return {"BSE": h, "DDRE-1D": h}, {"BSE": s, "DDRE-1D": s}


def score(claim, text):
    e = float((sum(map(ord, text)) * 7) % 100)
    return {"e": e, "k": 1.0, "spans": 1}, [(text, e)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


class Protocol(unittest.TestCase):
    def test_subset(self):
        idx = T.subset()
        self.assertEqual(len(idx), 200)
        self.assertEqual(idx, sorted(set(idx)))
        self.assertTrue(0 <= idx[0] and idx[-1] < 3736)
        self.assertEqual(idx, T.subset())

    def test_cells_match_protocol(self):
        self.assertEqual([c[0] for c in T.CELLS], ["D1", "D2", "D3", "D4", "E1", "E2", "E3", "E4"])
        self.assertEqual(T.setting_key("E1"), ("28/96", "K5/cr0.125"))
        self.assertEqual(T.setting_key("E2"), ("28/96", "K2/cr0.25"))
        self.assertEqual(T.setting_key("E3"), ("14/24", "K3/cr1"))
        self.assertEqual(T.setting_key("E4"), ("14/24", "K1/cr0.125"))
        self.assertEqual(T.setting_key("D2"), ("28/96", "K10/cr1"))

    def test_pass_order_interleaves_every_cell(self):
        order, cells = T.pass_order(200, 1)
        self.assertEqual(sorted(order), list(range(200)))
        self.assertTrue(all(sorted(c) == sorted(x[0] for x in T.CELLS) for c in cells))
        self.assertNotEqual(T.pass_order(200, 1)[0], T.pass_order(200, 2)[0])


class Live(unittest.TestCase):
    def test_live_equals_cached_loop(self):
        est, sup = estimators()
        pol = T.policies(sup)
        rng = np.random.default_rng(0)
        for trial in range(40):
            docs = [f"doc {trial} {i} {rng.integers(1e6)}" for i in range(int(rng.integers(1, 11)))]
            for cell, cm, cfa, method, k, cr in T.CELLS:
                live = T.run_live("claim", docs, cell, est[method], pol[cell], score, Clock())
                feats = [dict(score("claim", d)[0], rank=r) for r, d in enumerate(docs, 1)]
                lrs = [est[method].log_ratio(f) for f in feats]
                p, n, calls = S.run_subclaim_bi(lrs, [1] * len(lrs), pol[cell], k)
                self.assertEqual((live["p_factual"], live["documents"], live["nli_calls"]),
                                 (p, n, calls))
                self.assertEqual(live["prediction"], S.decision(p, cm, cfa))
                self.assertLessEqual(live["documents"], k)
                self.assertEqual(len(live["spans"]), live["documents"])

    def test_timers_fit_inside_total(self):
        est, sup = estimators()
        pol = T.policies(sup)
        live = T.run_live("c", [f"d{i}" for i in range(10)], "D1", est["BSE"], pol["D1"],
                          score, Clock())
        self.assertLessEqual(sum(live[n] for n in T.TIMERS), live["total"])
        self.assertGreater(live["nli"], 0)

    def test_policies_built_before_timing(self):
        _, sup = estimators()
        pol = T.policies(sup)
        self.assertEqual(sorted(pol["E2"]._cache), [1, 2])
        self.assertEqual(list(T.policies(sup, cells=("E4",))), ["E4"])


class Summaries(unittest.TestCase):
    def rows(self):
        out = []
        for p in (1, 2, 3):
            for s in ("a", "b", "c", "d"):
                for cell, *_ in T.CELLS:
                    slow = 2.0 if cell in ("E1", "E3", "D2", "D4") else 1.0
                    out.append({"pass": p, "sentence": s, "cell": cell, "documents": 2,
                                "nli_calls": 2, "total": slow + 0.01 * p,
                                "nli": slow, "estimator": 0.001, "decide_update": 0.001})
        return out

    def test_medians_and_cells(self):
        med = T.medians(self.rows())
        self.assertAlmostEqual(med[("a", "E1")]["total"], 2.02)
        cs = T.cell_summary(med, ["a", "b", "c", "d"])
        self.assertAlmostEqual(cs["E1"]["seconds_per_sentence"], 2.02)
        self.assertAlmostEqual(cs["E1"]["ms_per_nli_call"], 1000.0)

    def test_medians_refuse_missing_or_unequal_passes(self):
        rows = self.rows()
        with self.assertRaises(T.RuntimeMeasurementError):
            T.medians(rows[1:])
        rows[0] = dict(rows[0], documents=3)
        with self.assertRaises(T.RuntimeMeasurementError):
            T.medians(rows)

    def test_contrasts_sign_and_wording(self):
        med = T.medians(self.rows())
        c = T.contrasts(med, ["a", "b", "c", "d"], ["x", "x", "y", "z"])
        eq = c["T1 28/96 equal quality"]
        self.assertAlmostEqual(eq["observed"], 1.0)
        self.assertAlmostEqual(eq["percent_saved"], 100.0 / 2.02)
        self.assertEqual(eq["wording"], "DDRE-1D is faster than BSE")
        self.assertEqual(c["T2 28/96 default"]["wording"], "DDRE-1D is slower than BSE")

    def test_bootstrap(self):
        e = T.cluster_bootstrap([1.0] * 6, list("aabbcc"), n_resamples=200)
        self.assertEqual((e["ci_lower"], e["ci_upper"], e["clusters"]), (1.0, 1.0, 3))
        e = T.cluster_bootstrap([0.0, 2.0, -1.0, 1.0], list("abcd"), n_resamples=2000)
        self.assertEqual(T.wording(e), "no measurable difference")
        self.assertAlmostEqual(e["observed"], 0.5)


class Script(unittest.TestCase):
    def test_check_live_catches_differences(self):
        class Reader:
            def pair(self, premise, hypothesis):
                return {"entailment": 50.0}

        replay = {"D1": {"documents": np.array([1.0]), "nli_calls": np.array([1.0]),
                         "p_factual": np.array([0.7])}}
        ok = {"cell": "D1", "documents": 1, "nli_calls": 1, "p_factual": 0.7,
              "spans": [[("seg", 50.0)]]}
        RT.check_live(ok, 0, replay, Reader(), "c", "u")
        with self.assertRaises(RT.CheckFailed) as e:
            RT.check_live(dict(ok, spans=[[("seg", 50.000001)]]), 0, replay, Reader(), "c", "u")
        self.assertEqual(e.exception.check, "V2")
        with self.assertRaises(RT.CheckFailed) as e:
            RT.check_live(dict(ok, documents=2), 0, replay, Reader(), "c", "u")
        self.assertEqual(e.exception.check, "V3")

    def test_power_source(self):
        out = "Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged\n"
        with mock.patch.object(RT.subprocess, "check_output", return_value=out):
            self.assertEqual(RT.power_source(), "AC Power")
        with mock.patch.object(RT.subprocess, "check_output", side_effect=OSError):
            self.assertIsNone(RT.power_source())

    def test_pins_are_full_digests(self):
        self.assertEqual(set(RT.PINS), {"estimators", "helm_evidence", "helm_features",
                                        "helm_result", "helm_predictions"})
        self.assertTrue(all(len(sha) == 64 for _, sha in RT.PINS.values()))


if __name__ == "__main__":
    unittest.main()
