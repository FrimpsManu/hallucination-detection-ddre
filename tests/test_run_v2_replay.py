"""The plan-v2 replay script (scripts/run_v2_replay.py), without artifacts."""

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

import run_v2_replay as RV  # noqa: E402
from src import sequential as S  # noqa: E402
from src import v2_replay as R  # noqa: E402
from src.mddre_estimators import HistogramEvidence  # noqa: E402
from test_v2_replay import NEG, POS, synthetic  # noqa: E402


def histogram_everything():
    h = HistogramEvidence(POS, NEG)
    names = R.TIER1 + R.TIER2
    return ({n: h for n in names},
            {n: S.Support.from_histogram(POS, NEG) for n in names})


class Inputs(unittest.TestCase):
    def test_missing_pins_and_existing_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = RV.output_paths(Path(tmp))
            out["result"].write_text("{}")
            pins = {k: (Path(tmp) / f"{k}.missing", v[1]) for k, v in RV.PINS.items()}
            with mock.patch.object(RV, "PINS", pins):
                with self.assertRaises(RV.V2ReplayRefused) as ctx:
                    RV.verify_inputs(out, 5, dry_run=False)
            msg = str(ctx.exception)
            for k in RV.PINS:
                self.assertIn(f"{k}:", msg)
            self.assertIn("already exists", msg)
            self.assertIn("n_resamples is fixed", msg)

    def test_dry_run_ignores_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = RV.output_paths(Path(tmp))
            out["result"].write_text("{}")
            pins = {k: (Path(tmp) / f"{k}.missing", v[1]) for k, v in RV.PINS.items()}
            with mock.patch.object(RV, "PINS", pins):
                with self.assertRaises(RV.V2ReplayRefused) as ctx:
                    RV.verify_inputs(out, RV.N_RESAMPLES, dry_run=True)
            self.assertNotIn("already exists", str(ctx.exception))

    def test_reader_dtype_comes_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps({"dtype": "torch.float16", "cache_sha256": "x"}))
            pins = dict(RV.PINS, mddre_manifest=(manifest, "y"))
            with mock.patch.object(RV, "PINS", pins):
                self.assertEqual(RV.cache_dtype("mddre_features"), "torch.float16")

    def test_a_manifest_must_certify_its_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps({"dtype": "torch.float16", "cache_sha256": "other"}))
            pins = {k: (Path(tmp) / f"{k}.missing", v[1]) for k, v in RV.PINS.items()}
            pins["mddre_manifest"] = (manifest, "y")
            with mock.patch.object(RV, "PINS", pins):
                with self.assertRaises(RV.V2ReplayRefused) as ctx:
                    RV.verify_inputs(RV.output_paths(Path(tmp)), RV.N_RESAMPLES, True)
            self.assertIn("does not certify the pinned mddre cache", str(ctx.exception))

    def test_the_plan_must_be_at_amendment_1(self):
        from src import plan_v2
        self.assertEqual(plan_v2.plan_problems(ROOT, require_amendment_1=True), [])


class Gates(unittest.TestCase):
    def setUp(self):
        self.prepared = synthetic(60, seed=4)
        self.hist = HistogramEvidence(POS, NEG)
        self.official = R.run_bse_official(self.prepared, self.hist, S.Costs(28, 96, 1), 5)

    def frozen_rows(self, name):
        return {(name, s["id"]): {
            "p_factual": str(float(self.official["p_factual"][j])),
            "documents": str(int(self.official["documents"][j])),
            "nli_calls": str(int(self.official["nli_calls"][j])),
            "subclaim_documents": json.dumps(self.official["subclaim_documents"][j])}
            for j, s in enumerate(self.prepared)}

    def test_x2_passes_on_identical_rows_and_counts_differences(self):
        rows = self.frozen_rows("d")
        self.assertEqual(RV.gate_x2("d", self.prepared, self.official, rows), 0)
        first = ("d", self.prepared[0]["id"])
        rows[first] = dict(rows[first], documents="99")
        self.assertEqual(RV.gate_x2("d", self.prepared, self.official, rows), 1)
        del rows[first]
        self.assertGreater(RV.gate_x2("d", self.prepared, self.official, rows), 0)

    def test_x3_engine_equals_bsedetector(self):
        self.assertEqual(RV.gate_x3(self.prepared, self.hist, 5), 0)


class EndToEnd(unittest.TestCase):
    def test_run_dataset_and_prediction_rows(self):
        prepared = synthetic(40, seed=5, max_docs=3)
        est, sup = histogram_everything()
        policies, eq8 = RV.build_policies(sup)
        self.assertEqual(len(policies), len(R.settings()) * len(R.TIER1 + R.TIER2))
        self.assertEqual(len(eq8), len(R.COST_PAIRS) * len(R.TIER1 + R.TIER2))
        res, eq8_res, fixed = RV.run_dataset(prepared, est, sup, policies, eq8, 3)
        self.assertEqual(set(res), {"BSE-official"} | set(R.TIER1 + R.TIER2))
        self.assertEqual(len(res["BSE"]), len(R.settings()))
        # identical estimators: identical results for every backward-induction method
        for m in R.TIER2:
            for key in res["BSE"]:
                self.assertTrue(np.array_equal(res[m][key]["p_factual"],
                                               res["BSE"][key]["p_factual"]))
        self.assertEqual(set(fixed), {f"{h}@{k}" for h in RV.FIXED_K_AGGREGATES
                                      for k in (1, 2, 3)})
        rows = RV.prediction_rows("d", prepared, {**res, **eq8_res})
        methods = len(res) + len(eq8_res)
        self.assertEqual(len(rows), methods * len(R.COST_PAIRS) * len(prepared))
        labels = np.array([s["label"] for s in prepared])
        e = R.contrast_endpoints(res, labels, np.arange(len(prepared)))
        self.assertAlmostEqual(e["DDRE-M|NB-M|28/96|e_quality"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
