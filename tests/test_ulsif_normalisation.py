"""uLSIF normalisation diagnostic: synthetic estimators only."""

import ast
import math
import unittest
from pathlib import Path

from src import ulsif_normalisation as N

ROOT = Path(__file__).resolve().parents[1]


class Scaled:
    """r(s) = c * exp(a * (s - 50) / 50): a known normaliser on fixed samples."""

    def __init__(self, c, a=1.0):
        self.c, self.a = c, a

    def ratio(self, s):
        return self.c * math.exp(self.a * (s - 50.0) / 50.0)


HALLUC = [10.0, 30.0, 50.0]
FACTUAL = [50.0, 70.0, 90.0]


class TestNormalisation(unittest.TestCase):
    def test_normaliser_is_the_mean_ratio_on_hallucinated_samples(self):
        est = Scaled(0.5)
        want = sum(est.ratio(s) for s in HALLUC) / 3
        self.assertAlmostEqual(N.normaliser(est, HALLUC), want)

    def test_normalised_curve_does_not_depend_on_the_scale(self):
        a = N.normalisation_row(Scaled(0.5), FACTUAL, HALLUC)
        b = N.normalisation_row(Scaled(2.0), FACTUAL, HALLUC)
        for s in a["log_r"]:
            self.assertAlmostEqual(a["log_r_normalised"][s], b["log_r_normalised"][s])
        self.assertAlmostEqual(a["log_r"]["50"], math.log(0.5))

    def test_a_shrunk_ratio_can_flip_sign_at_the_mode(self):
        row = N.normalisation_row(Scaled(0.3, a=0.2), FACTUAL, HALLUC)
        self.assertEqual(row["sign_at_mode"], -1)
        self.assertEqual(row["sign_at_mode_normalised"], 1)
        self.assertLess(row["mean_r_hallucinated"], 1.0)

    def test_sign_agreement(self):
        rows = {"a": {"sign_at_mode": 1}, "b": {"sign_at_mode": -1}, "c": {"sign_at_mode": 1}}
        self.assertEqual(N.sign_agreement(rows, "a"),
                         {"reference": "a", "reference_sign": 1, "same": 2, "total": 3})

    def test_grid_and_mode(self):
        self.assertEqual(N.GRID[0], 0)
        self.assertEqual(N.GRID[-1], 100)
        self.assertEqual(N.MODE_SCORE, 35.0)


class TestScriptTouchesTrainingDataOnly(unittest.TestCase):
    def test_no_held_out_or_detector_path(self):
        source = (ROOT / "scripts" / "diagnose_ulsif_normalisation.py").read_text()
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name for a in node.names)
        banned = {"load_sentence_records", "split_records", "DDREDetector", "BSEDetector",
                  "evaluate_detector", "replay", "build", "EntailmentScorer",
                  "transformers", "torch", "from_pretrained"}
        self.assertEqual(names & banned, set())
        self.assertIn("output.chmod(0o444)", source)
        self.assertIn("already exists", source)


if __name__ == "__main__":
    unittest.main()
