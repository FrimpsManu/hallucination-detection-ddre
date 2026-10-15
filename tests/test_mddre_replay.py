"""Replay engine, curves and endpoints (design §5-§7), synthetic only."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from src import mddre_replay as R
from src.baseline_core import BSEDetector, discretize_document_score

POS = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
NEG = [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]


class StubScorer:
    def __init__(self, scores):
        self.scores = scores

    def score_document(self, claim, text, **_):
        return self.scores[int(text)], 2


class TestBand(unittest.TestCase):
    def test_grid_and_threshold(self):
        self.assertEqual(R.BAND_SCALES[0], 0.25)
        self.assertEqual(R.BAND_SCALES[-1], 4.0)
        self.assertEqual(len(R.BAND_SCALES), 16)
        self.assertIn(1.0, R.BAND_SCALES)
        self.assertAlmostEqual(R.decision_threshold(), 28 / 124)

    def test_scale_one_is_the_operating_point(self):
        band = R.bse_band(BSEDetector(POS, NEG, mode="official"))
        l, u = R.scaled_band(band, 1.0)
        self.assertAlmostEqual(l, band[0], places=12)
        self.assertAlmostEqual(u, band[1], places=12)
        l, u = R.scaled_band(band, 2.0)
        self.assertAlmostEqual(R._logit(l), 2 * R._logit(band[0]))
        self.assertAlmostEqual(R._logit(u), 2 * R._logit(band[1]))

    def test_every_curve_band_reads_at_least_one_document(self):
        # Amendment 1 regression: the preregistered symmetric sweep had bands
        # below P0 that stopped before any document, a constant P = 0.5.
        band = R.bse_band(BSEDetector(POS, NEG, mode="official"))
        for lo, hi in R.curve_bands(band):
            self.assertLess(lo, R.P0)
            self.assertGreater(hi, R.P0)
            for lr in ([-9.0] * 5, [9.0] * 5, [0.0] * 5):
                self.assertGreaterEqual(R.band_subclaim(lr, [1] * 5, lo, hi, 5)[1], 1)

    def test_curve_refuses_a_band_without_p0(self):
        with self.assertRaises(ValueError):
            R.curve_bands((0.6, 0.8))

    def test_bse_official_continue_region_is_the_published_band(self):
        l, u = R.bse_band(BSEDetector(POS, NEG, mode="official"))
        self.assertAlmostEqual(l, 0.2041, places=3)
        self.assertAlmostEqual(u, 0.7689, places=3)

    def test_band_semantics_check_before_each_document(self):
        p, d, n = R.band_subclaim([-3.0, -3.0], [2, 2], 0.2, 0.8, 10)
        self.assertEqual((d, n), (1, 2))
        p, d, n = R.band_subclaim([5.0], [1], 0.6, 0.8, 10)  # P0 outside the band
        self.assertEqual((p, d), (0.5, 0))
        p, d, n = R.band_subclaim([0.1] * 12, [1] * 12, 0.01, 0.99, 10)
        self.assertEqual(d, 10)

    def test_bse_replay_equals_bse_detector(self):
        bse = BSEDetector(POS, NEG, mode="official")
        scores = [35.0, 22.0, 61.0, 18.0, 74.0, 33.0, 12.0, 45.0, 29.0, 66.0]
        sub = SimpleNamespace(text="c", documents=[SimpleNamespace(page_content=str(i))
                                                   for i in range(10)])
        want = bse.detect_subclaim(sub, StubScorer(scores))
        tp, tn = sum(POS), sum(NEG)
        lik = [(POS[discretize_document_score(s)] / tp, NEG[discretize_document_score(s)] / tn)
               for s in scores]
        p, d, n = R.bse_subclaim(lik, [2] * 10, bse, 10)
        self.assertEqual((p, d, n), (want.p_factual, want.documents_used, want.nli_calls))

    def test_sentence_is_the_minimum(self):
        s = R.sentence_result([(0.9, 2, 4), (0.1, 3, 3)])
        self.assertEqual((s["p_factual"], s["documents"], s["nli_calls"], s["prediction"]),
                         (0.1, 5, 7, 0))


class TestMatching(unittest.TestCase):
    def test_matched_quality_interpolates(self):
        self.assertAlmostEqual(R.matched_quality_docs([1, 2, 3], [0.5, 0.6, 0.7], 0.65), 2.5)
        self.assertEqual(R.matched_quality_docs([1, 2], [0.7, 0.8], 0.6), 1)
        self.assertIsNone(R.matched_quality_docs([1, 2], [0.5, 0.6], 0.7))

    def test_quality_at_docs(self):
        self.assertAlmostEqual(R.quality_at_docs([1, 3], [0.5, 0.7], 2), 0.6)
        self.assertIsNone(R.quality_at_docs([1, 3], [0.5, 0.7], 4))

    def test_holm(self):
        h = R.holm({"RQ1": 0.01, "RQ2": 0.04})
        self.assertTrue(h["RQ1"]["reject"] and h["RQ2"]["reject"])
        h = R.holm({"RQ1": 0.03, "RQ2": 0.04})
        self.assertFalse(h["RQ1"]["reject"] or h["RQ2"]["reject"])


def synthetic(n_clusters=12):
    rng = np.random.default_rng(0)
    labels, clusters = [], []
    for c in range(n_clusters):
        for _ in range(4):
            labels.append(int(rng.random() < 0.5))
            clusters.append(c)
    labels = np.array(labels)
    nb = len(R.BAND_SCALES)

    def cell(noise, docs_scale):
        p = np.clip(labels * 0.6 + 0.2 + rng.normal(0, noise, (nb, len(labels))), 0, 1)
        d = np.tile(np.arange(1, nb + 1)[:, None] * docs_scale, (1, len(labels))).astype(float)
        return {"p_factual": p, "documents": d, "op_p_factual": p[8], "op_documents": d[8]}

    curves = {"A": cell(0.3, 1.0), "M": cell(0.1, 0.5)}
    reference = {"p_factual": curves["A"]["p_factual"][8], "documents": curves["A"]["documents"][8]}
    return curves, reference, labels, clusters


class TestBootstrap(unittest.TestCase):
    def test_endpoints_and_decision(self):
        from src.evaluation import wang_pr_auc

        curves, ref, labels, clusters = synthetic()
        out = R.bootstrap(curves, ref, ["A", "M"], [("A", "M")], labels, clusters,
                          wang_pr_auc, n_resamples=50)
        e = out["endpoints"]
        self.assertEqual(out["clusters"], 12)
        self.assertGreater(e["M.matched_quality_saving"]["observed"], 0)
        self.assertEqual(e["A.same_band_docs_saving"]["observed"], 0.0)
        self.assertIn("M-A.matched_cost_gain", e)
        d = R.decide(e)
        self.assertIn("holm", d["RQ1"])


if __name__ == "__main__":
    unittest.main()


class TestReplayScript(unittest.TestCase):
    def test_structure(self):
        import ast
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_mddre_replay.py"
        source = path.read_text()
        spec = importlib.util.spec_from_file_location("run_mddre_replay", path)
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        self.assertEqual(S.CELLS, ("A", "A_prime", "U", "M", "L"))
        self.assertEqual(S.CHAIN, (("A", "A_prime"), ("A_prime", "U"), ("U", "M"), ("L", "M")))
        self.assertEqual(S.DATASETS["factcheck_bench"]["max_docs"], 5)
        self.assertTrue(all(v["status"].startswith("exploratory under amendment 1")
                            for v in S.DATASETS.values()))
        self.assertTrue(S.output_paths(Path("/x"), "float16")["result"].name
                        .endswith("_amendment1.json"))
        names = {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)}
        self.assertEqual(names & {"fit", "cv_table_select", "LogisticRegressionCV"}, set())
        self.assertLess(source.index("g1 = gate_engine"), source.index("sets, split = test_sets"))
        self.assertIn('p[k].chmod(0o444)', source)


class TestGateG3(unittest.TestCase):
    def test_exact_rows_pass_and_any_difference_fails(self):
        import csv
        import importlib.util
        import tempfile
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_mddre_replay.py"
        spec = importlib.util.spec_from_file_location("run_mddre_replay", path)
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        rows = [{"dataset": "d", "cell": "M", "id": "1", "cluster": 0, "gold_label": 1,
                 "p_factual": 0.19336384439359272, "prediction": 0, "documents": 3,
                 "nli_calls": 3, "subclaim_documents": [3]}]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "frozen.csv"
            with f.open("w", newline="") as handle:
                w = csv.DictWriter(handle, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            frozen = S.frozen_rows(f)
        self.assertEqual(S.gate_frozen_operating_point(frozen, "d", rows), 0)
        changed = [{**rows[0], "p_factual": 0.19336384439359267}]
        self.assertEqual(S.gate_frozen_operating_point(frozen, "d", changed), 1)
        self.assertEqual(S.gate_frozen_operating_point(frozen, "d", []), 1)
