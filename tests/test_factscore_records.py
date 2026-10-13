"""FActScore evidence -> detector records, and the scoring script's guarantees."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import factscore_data as F
from src import factscore_records as R

ROOT = Path(__file__).resolve().parents[1]


def sentence(gen, entity, resp, sent, facts):
    return F.FactSentence(gen, entity, resp, sent, f"{entity} sentence",
                          tuple(F.Fact(t, l) for t, l in facts))


def evidence(s, fact_index, n=3, **over):
    row = {"generator": s.generator, "response_index": s.response_index,
           "sentence_index": s.sentence_index, "fact_index": fact_index,
           "entity": s.entity, "fact": s.facts[fact_index].text, "query_terms": [],
           "documents": [{"rank": r + 1, "title": f"P{r}", "pidx": r, "bm25": -1.0 - r,
                          "text": f"P{r}\npassage {r}"} for r in range(n)]}
    row.update(over)
    return row


SENTS = [
    sentence("ChatGPT", "Bo Ray", 4, 0, [("Bo is a chef.", "S"), ("Bo was born in 1900.", "NS")]),
    sentence("PerplexityAI", "Bo Ray", 4, 0, [("Bo cooks.", "S")]),
    sentence("ChatGPT", "Ann Lee", 1, 2, [("Ann is a poet.", "S")]),
]


def all_evidence():
    return [evidence(s, i) for s in SENTS for i in range(len(s.facts))]


class TestBuildRecords(unittest.TestCase):
    def test_records_mirror_selfcheckgpt_inputs(self):
        records, clusters = R.build_records(SENTS, all_evidence())
        self.assertEqual(clusters, {"Ann Lee": 0, "Bo Ray": 1})
        self.assertEqual([r.passage_index for r in records], [1, 1, 0])
        self.assertEqual([r.label for r in records], [0, 1, 1])
        sub = records[0].subclaims[1]
        self.assertEqual(sub.text, "Bo was born in 1900.")
        self.assertEqual([d.url for d in sub.documents], ["P0#0", "P1#1", "P2#2"])
        self.assertEqual(sub.documents[0].page_content, "P0\npassage 0")

    def test_identities_are_unique_across_generators(self):
        records, _ = R.build_records(SENTS, all_evidence())
        self.assertEqual(records[0].sentence_index, 4_000)
        self.assertEqual(records[1].sentence_index, 1_004_000)
        self.assertEqual(len({(r.passage_index, r.sentence_index) for r in records}), 3)

    def test_missing_duplicate_extra_or_mismatched_evidence_is_refused(self):
        rows = all_evidence()
        for bad in (rows[:-1],
                    rows + [rows[0]],
                    rows + [evidence(SENTS[2], 0, sentence_index=9)],
                    rows[:-1] + [evidence(SENTS[2], 0, fact="Ann is a painter.")],
                    rows[:-1] + [dict(rows[-1], documents=list(reversed(rows[-1]["documents"])))]):
            with self.assertRaises(R.EvidenceMismatch):
                R.build_records(SENTS, bad)

    def test_read_evidence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for row in all_evidence():
                f.write(json.dumps(row) + "\n")
        self.assertEqual(len(R.read_evidence(f.name)), 4)
        Path(f.name).unlink()


def load_script():
    spec = importlib.util.spec_from_file_location(
        "score_factscore_evidence", ROOT / "scripts" / "score_factscore_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load_script()
SOURCE = (ROOT / "scripts" / "score_factscore_evidence.py").read_text()


class TestScoringScript(unittest.TestCase):
    def test_pins(self):
        self.assertEqual(S.SOURCE_CACHE_SHA256,
                         "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776")
        self.assertEqual(S.EXPECTED_SUBCLAIMS, 10_386)

    def test_no_detector_metric_or_selection_path(self):
        names = set()
        for node in ast.walk(ast.parse(SOURCE)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.name for a in node.names)
        banned = {"BSEDetector", "DDREDetector", "HistogramRatio", "evaluate_detector",
                  "evaluate_detector_with_traces", "summarize_method", "wang_pr_auc",
                  "paired_passage_bootstrap", "multi_cell_bootstrap", "label"}
        self.assertEqual(names & banned, set())

    def test_inputs_are_refused_when_unpinned_or_already_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            release, artifacts = Path(tmp) / "r", Path(tmp) / "a"
            release.mkdir()
            artifacts.mkdir()
            (release / S.EVIDENCE_NAME).write_text("{}")
            (release / S.EVIDENCE_MANIFEST_NAME).write_text(json.dumps(
                {"preregistration_sha256": "x", "labels_recorded": True, "subclaims": 5,
                 "evidence_sha256": "y"}))
            (release / S.CACHE_NAME).write_text("x")
            for name in (S.SOURCE_CACHE_NAME, S.CA.RESULT_NAME):
                (artifacts / name).write_text("x")
            with self.assertRaises(S.AC.CompletionAborted) as ctx:
                S.verify_inputs(release, artifacts)
            text = str(ctx.exception)
            for fragment in ("not retrieved under this preregistration", "label-free",
                             "10,386", "differs from its manifest digest", "not the pinned file",
                             "already exists"):
                self.assertIn(fragment, text)

    def test_dry_run_loads_no_model(self):
        try:
            import src.utils  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        args = SimpleNamespace(release_dir="r", artifacts_dir="a", wang_data_root="w",
                               dry_run=True)
        cov = {"keys": [], "pairs": [], "counts": {}, "coverage_sha256": "c"}
        with mock.patch.object(S, "parse_args", return_value=args), \
                mock.patch.object(S, "verify_inputs", return_value=({}, {})), \
                mock.patch.object(S, "load_analysis_set", return_value=[]), \
                mock.patch.object(S, "read_evidence", return_value=[]), \
                mock.patch.object(S, "build_records", return_value=([], {})), \
                mock.patch.object(S.AC, "enumerate_coverage", return_value=cov), \
                mock.patch.object(S.AC, "present_keys", return_value=set()), \
                mock.patch.object(S.CA, "load_model") as load, \
                mock.patch.object(S.AC, "complete_cache") as complete:
            self.assertEqual(S.main(), 0)
        load.assert_not_called()
        complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
