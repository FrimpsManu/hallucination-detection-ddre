"""Multivariate DDRE data: entity split and Factcheck-Bench rules (synthetic)."""

import json
import unittest
from pathlib import Path

from src import mddre_data as M

ROOT = Path(__file__).resolve().parents[1]


def claim_block(claims):
    return {
        "text": "s",
        "claims": [c[0] for c in claims],
        "claim_checkworthiness": [c[1] for c in claims],
        "claims_factuality_label": [c[2] for c in claims],
        "auto_evidence": [[f"doc{j}" for j in range(5)] for _ in claims],
    }


class TestEntitySplit(unittest.TestCase):
    def test_split_is_deterministic_halves_by_hash(self):
        names = [f"Person {i}" for i in range(11)]
        a, b = M.split_entities(names)
        self.assertEqual((len(a), len(b)), (5, 6))
        self.assertEqual(M.split_entities(reversed(names)), (a, b))
        self.assertTrue(all(M.entity_hash(x) < M.entity_hash(y) for x in a for y in b))

    def test_pins(self):
        self.assertEqual(M.TRAINING_GENERATORS, ("InstructGPT", "ChatGPT", "PerplexityAI"))
        self.assertEqual(M.EXPECTED_ENTITIES, 182)
        self.assertIn("Bridget Moynahan", M.EXCLUDED_ENTITIES)


class TestFactcheckRules(unittest.TestCase):
    def lines(self, sentences):
        return [json.dumps({"sentences": sentences})]

    def test_inclusion_and_sentence_label(self):
        out = M.parse_factcheck(self.lines({
            "sentence1": claim_block([("a", "factual", True), ("b", "factual", "unknown"),
                                      ("c", "opinion", True)]),
            "sentence2": claim_block([("d", "factual", True), ("e", "factual", False)]),
            "sentence3": claim_block([("f", "not_a_claim", False)]),
        }))
        self.assertEqual([s.sentence_key for s in out], ["sentence1", "sentence2"])
        self.assertEqual([c.text for c in out[0].claims], ["a"])
        self.assertEqual((out[0].label, out[1].label), (1, 0))
        self.assertEqual(out[1].claims[1].documents, tuple(f"doc{j}" for j in range(5)))

    def test_wrong_evidence_count_is_refused(self):
        block = claim_block([("a", "factual", True)])
        block["auto_evidence"] = [["only one"]]
        with self.assertRaises(M.MDDREDataRefused):
            M.parse_factcheck(self.lines({"sentence1": block}))

    def test_pins(self):
        self.assertEqual(M.FACTCHECK_COMMIT[:7], "a9e6a04")
        self.assertEqual(M.FACTCHECK_DOCS, 5)
        self.assertEqual(M.DESIGN_MERGE_COMMIT[:7], "e5c8bda")


class TestRetrievalScript(unittest.TestCase):
    def test_reuses_the_frozen_retrieval_and_records_no_labels(self):
        source = (ROOT / "scripts" / "retrieve_mddre_training_evidence.py").read_text()
        self.assertIn("RF._retrieve_one", source)
        self.assertIn('"labels_recorded": False', source)
        self.assertIn('s.generator == "InstructGPT"', source)
        self.assertNotIn(".label", source)


if __name__ == "__main__":
    unittest.main()


class TestDesignPins(unittest.TestCase):
    def test_amendments_only_append_to_the_preregistration(self):
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(M.design_problems(root, require_amendment_1=True), [])
        text = (root / M.DESIGN_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / M.DESIGN_PATH
            f.parent.mkdir(parents=True)
            f.write_bytes(text[:M.DESIGN_PREREGISTERED_BYTES])
            self.assertEqual(M.design_problems(tmp), [])
            self.assertTrue(M.design_problems(tmp, require_amendment_1=True))
            f.write_bytes(b"X" + text[1:])
            self.assertIn("the design preregistration differs from its pinned digest",
                          M.design_problems(tmp))
            f.write_bytes(text + b"\nmore\n")
            self.assertIn("the design file has an unpinned amendment", M.design_problems(tmp))
