"""Feature-scoring script for multivariate DDRE: structure only (no model, no data)."""

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "score_mddre_features.py").read_text()


def load():
    spec = importlib.util.spec_from_file_location(
        "score_mddre_features", ROOT / "scripts" / "score_mddre_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load()


class TestStructure(unittest.TestCase):
    def test_precision_is_required_and_limited(self):
        self.assertEqual(S.DTYPES, ("float16", "float32"))
        self.assertIn("required=True", SOURCE)

    def test_references_are_precision_matched(self):
        self.assertEqual(set(S.references("float16")),
                         {"d03_cache", "heldout_full_cache", "factscore_cache"})
        self.assertEqual(set(S.references("float32")), {"fp32_gate1_cache"})
        self.assertTrue(all(sha for _, sha in S.references("float16").values()))

    def test_outputs_are_named_by_precision(self):
        out = S.output_paths(Path("/x"), "float32")
        self.assertEqual(out["cache"].name, "mddre_features_float32.sqlite")

    def test_no_detector_or_selection_path(self):
        names = set()
        for node in ast.walk(ast.parse(SOURCE)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name for a in node.names)
        banned = {"BSEDetector", "DDREDetector", "evaluate_detector", "replay",
                  "ULSIFDensityRatio", "summarize_method", "EntailmentScorer"}
        self.assertEqual(names & banned, set())

    def test_equality_runs_before_bulk_scoring_and_outputs_are_frozen(self):
        self.assertLess(SOURCE.index("NBC equality failed"), SOURCE.index("feature scoring"))
        self.assertIn('out["cache"].chmod(0o444)', SOURCE)
        self.assertIn("scoring runs once", SOURCE)

    def test_factcheck_records_keep_order(self):
        from src.mddre_data import Claim, FactcheckSentence

        s = FactcheckSentence(0, "sentence1", "t", (Claim("c", 1, ("d0", "d1", "d2", "d3", "d4")),))
        recs = S.factcheck_records([s])
        self.assertEqual([d.page_content for d in recs[0].subclaims[0].documents],
                         ["d0", "d1", "d2", "d3", "d4"])


if __name__ == "__main__":
    unittest.main()
