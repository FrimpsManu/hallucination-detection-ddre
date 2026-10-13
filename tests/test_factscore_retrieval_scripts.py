"""The index and retrieval scripts: pinned inputs, run-once, and no labels in retrieval."""

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src import factscore_data as F

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load("build_wikipedia_index")
RETRIEVE = load("retrieve_factscore_evidence")


class TestPins(unittest.TestCase):
    def test_snapshot_pins_agree(self):
        self.assertEqual(BUILD.SNAPSHOT_SHA256,
                         "31cf7b6b4465459844bb00f3a6ac75560fc7d1525112205a21859323dc5d33d7")
        self.assertEqual(RETRIEVE.SNAPSHOT_SHA256, BUILD.SNAPSHOT_SHA256)
        self.assertEqual(BUILD.SNAPSHOT_PAGES, 6_187_531)
        self.assertEqual(RETRIEVE.EXPECTED_SUBCLAIMS, 10_386)
        self.assertEqual(RETRIEVE.INDEX_NAME, BUILD.INDEX_NAME)


class TestRunOnce(unittest.TestCase):
    def test_build_refuses_an_existing_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / BUILD.INDEX_NAME).write_text("x")
            BUILD.parse_args = lambda: SimpleNamespace(release_dir=tmp)
            with self.assertRaises(BUILD.IndexRefused):
                BUILD.main()

    def test_retrieval_refuses_missing_index_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (RETRIEVE.SNAPSHOT_NAME, RETRIEVE.INDEX_NAME, RETRIEVE.EVIDENCE_NAME):
                (Path(tmp) / name).write_text("x")
            with self.assertRaises(RETRIEVE.EvidenceRefused) as ctx:
                RETRIEVE.verify_inputs(Path(tmp))
            text = str(ctx.exception)
            self.assertIn("manifest is missing", text)
            self.assertIn("snapshot is not the pinned one", text)
            self.assertIn("already exists", text)


class TestNoLabelsInRetrieval(unittest.TestCase):
    def test_subclaim_queries_carry_no_label(self):
        s = F.FactSentence("ChatGPT", "Ann Lee", 3, 1, "x",
                           (F.Fact("Ann is a poet.", "S"), F.Fact("Ann won.", "NS")))
        out = RETRIEVE.subclaim_queries([s])
        self.assertEqual(out, [(("ChatGPT", 3, 1, 0), "Ann Lee", "Ann is a poet."),
                               (("ChatGPT", 3, 1, 1), "Ann Lee", "Ann won.")])

    def test_the_script_never_reads_a_label(self):
        tree = ast.parse((ROOT / "scripts" / "retrieve_factscore_evidence.py").read_text())
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
        self.assertNotIn("label", attrs)
        self.assertNotIn("label", consts)

    def test_the_retriever_module_never_sees_a_label(self):
        text = (ROOT / "src" / "wikipedia_bm25.py").read_text()
        self.assertNotIn("label", text)
        self.assertNotIn("factscore_data", text)


if __name__ == "__main__":
    unittest.main()
