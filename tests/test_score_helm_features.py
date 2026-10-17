"""HELM feature scoring script (scripts/score_helm_features.py), without artifacts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_helm_features as SH  # noqa: E402
from src import helm_data as H  # noqa: E402


def units():
    return [H.Unit("g1:7:0", "7", "g1", "Alpha", "Alpha is red .", 1),
            H.Unit("g1:7:1", "7", "g1", "Alpha", "Alpha is big .", 0)]


def write_evidence(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class Evidence(unittest.TestCase):
    def rows(self):
        return [{"id": u.id, "title": u.title, "sentence": u.sentence,
                 "documents": [{"text": f"Alpha passage {i} " * 3} for i in range(12)]}
                for u in units()]

    def test_records_follow_the_units_and_cap_nothing_here(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.jsonl"
            write_evidence(p, self.rows())
            recs = SH.evidence_records(units(), p)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[1].subclaims[0].text, "Alpha is big .")
        self.assertEqual(len(recs[0].subclaims[0].documents), 12)

    def test_order_and_content_mismatches_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.jsonl"
            write_evidence(p, self.rows()[::-1])
            with self.assertRaises(SH.HelmScoringRefused):
                SH.evidence_records(units(), p)
            rows = self.rows()
            rows[0]["sentence"] = "changed"
            write_evidence(p, rows)
            with self.assertRaises(SH.HelmScoringRefused):
                SH.evidence_records(units(), p)

    def test_enumeration_reads_at_most_ten_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.jsonl"
            write_evidence(p, self.rows())
            with mock.patch.object(SH, "EVIDENCE", (p, "x")):
                cov = SH.enumerate_pairs(units())
        self.assertEqual(cov["counts"]["document_occurrences"], 20)
        self.assertEqual(len(cov["pairs"]), 20)


class Inputs(unittest.TestCase):
    def test_refusals_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = SH.output_paths(Path(tmp))
            out["cache"].write_text("")
            gone = Path(tmp) / "absent"
            with mock.patch.object(SH, "EVIDENCE", (gone, "x")), \
                    mock.patch.object(SH, "EVIDENCE_MANIFEST", gone), \
                    mock.patch.object(SH.SC, "ESTIMATORS", (gone, "x")), \
                    mock.patch.object(SH.SC, "MDDRE_FEATURES", (gone, "x")), \
                    mock.patch.object(SH.SF, "FLOAT16_REFERENCES", {"r": (gone, "x")}):
                with self.assertRaises(SH.HelmScoringRefused) as ctx:
                    SH.verify_inputs(out, dry_run=False)
            msg = str(ctx.exception)
            self.assertIn("is missing", msg)
            self.assertIn("evidence manifest", msg)
            self.assertIn("reference r", msg)
            self.assertIn("already exists", msg)


if __name__ == "__main__":
    unittest.main()
