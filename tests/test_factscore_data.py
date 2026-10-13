"""FActScore analysis-set loader (transfer preregistration §1-§2). Synthetic lines only."""

import json
import tempfile
import unittest
from pathlib import Path

from src import factscore_data as F


def row(topic, annotations):
    return json.dumps({"input": "", "output": "", "topic": topic, "cat": [],
                       "annotations": annotations})


def ann(text, facts, relevant=True):
    return {"text": text, "is-relevant": relevant,
            "human-atomic-facts": [{"text": t, "label": l} for t, l in facts]}


LINES = [
    row("Ann Lee", [ann("s0", [("Ann is a poet.", "S"), ("She was born in 1900.", "NS")]),
                    ann("s1", [("Ann won a prize.", "S")]),
                    ann("s2", [("irrelevant fact", "IR")]),
                    ann("s3", [("not judged", "S")], relevant=False)]),
    row("Bridget Moynahan", [ann("x", [("excluded entity", "S")])]),
    json.dumps({"topic": "Declined", "annotations": None}),
    row("Bo Ray", [ann("t0", [("  ", "S"), ("Bo is a chef.", "S")])]),
]


class TestParse(unittest.TestCase):
    def test_exclusions_and_units(self):
        out = F.parse_generator(LINES, "ChatGPT")
        self.assertEqual([(s.entity, s.sentence_index) for s in out],
                         [("Ann Lee", 0), ("Ann Lee", 1), ("Bo Ray", 0)])
        self.assertEqual([s.label for s in out], [0, 1, 1])
        self.assertEqual(out[0].facts, (F.Fact("Ann is a poet.", "S"),
                                        F.Fact("She was born in 1900.", "NS")))
        self.assertEqual(out[2].facts, (F.Fact("Bo is a chef.", "S"),))
        self.assertEqual(out[0].key, ("ChatGPT", 0, 0))
        self.assertEqual(out[2].response_index, 3)

    def test_counts(self):
        c = F.counts(F.parse_generator(LINES, "ChatGPT"))
        self.assertEqual(c, {"entities": 2, "sentences": 3, "factual": 2, "hallucinated": 1,
                             "facts": 4, "S": 3, "NS": 1})

    def test_preregistered_constants(self):
        self.assertEqual(F.GENERATORS, ("ChatGPT", "PerplexityAI"))
        self.assertEqual(F.EXCLUDED_ENTITIES, {"Bridget Moynahan"})
        self.assertEqual(sum(F.EXPECTED[g]["sentences"] for g in F.GENERATORS), 2573)
        self.assertEqual(sum(F.EXPECTED[g]["facts"] for g in F.GENERATORS), 10386)
        self.assertEqual(F.EXPECTED_POOLED_ENTITIES, 172)

    def test_preregistration_is_the_pinned_one(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(F.sha256_path(root / F.PREREGISTRATION_PATH), F.PREREGISTRATION_SHA256)


class TestLoaderRefuses(unittest.TestCase):
    def test_unpinned_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "labeled").mkdir()
            for g in F.GENERATORS:
                (Path(tmp) / "labeled" / f"{g}.jsonl").write_text("\n".join(LINES))
            with self.assertRaises(F.FactScoreDataRefused) as ctx:
                F.load_analysis_set(tmp)
            self.assertIn("pinned", str(ctx.exception))

    def test_missing_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(F.FactScoreDataRefused):
                F.load_analysis_set(tmp)


if __name__ == "__main__":
    unittest.main()
