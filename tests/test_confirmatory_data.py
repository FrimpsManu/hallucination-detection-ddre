"""Confirmatory data loaders (preregistration §2), synthetic plus pinned-release checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import confirmatory_data as C

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def passages(*texts):
    return "\n\n".join(f"Passage ID {i}: {t}" for i, t in enumerate(texts, start=1))


class TestPassages(unittest.TestCase):
    def test_split_in_id_order(self):
        self.assertEqual(C.split_passages(passages("a b", "c")), ("a b", "c"))
        self.assertEqual(C.split_passages(""), ())

    def test_refuses_out_of_order_or_leading_text(self):
        with self.assertRaises(C.ConfirmatoryDataRefused):
            C.split_passages("Passage ID 2: x\n\nPassage ID 1: y")
        with self.assertRaises(C.ConfirmatoryDataRefused):
            C.split_passages("stray Passage ID 1: x")


class TestMappings(unittest.TestCase):
    def test_primary_mappings_are_the_preregistered_ones(self):
        e = C.EXPERTQA_MAPPINGS["primary"]
        self.assertEqual({k for k, v in e.items() if v == 0},
                         {"Likely incorrect", "Definitely incorrect"})
        self.assertNotIn("Unsure", e)
        b = C.BINGCHECK_MAPPINGS["primary"]
        self.assertEqual({k for k, v in b.items() if v == 0}, {"refuted", "partially supported"})
        self.assertNotIn("not supported", b)

    def test_sensitivity_mappings_are_supersets(self):
        for maps in (C.EXPERTQA_MAPPINGS, C.BINGCHECK_MAPPINGS):
            for name, m in maps.items():
                if name != "refuted_only":
                    self.assertLessEqual(set(maps["primary"]), set(m))


class TestExpertQALoader(unittest.TestCase):
    def test_units_labels_exclusions_and_document_cap(self):
        claims = [
            {"claim_string": "s1", "correctness": "Definitely correct",
             "atomic_claims": ["a1", "a2"],
             "atomic_evidences": [passages(*"pqrstuv"), ""],
             "revised_claim": "IGNORED", "revised_evidence": "IGNORED"},
            {"claim_string": "s2", "correctness": "Likely incorrect",
             "atomic_claims": ["b1"], "atomic_evidences": [passages("x")]},
            {"claim_string": "s3", "correctness": "Unsure",
             "atomic_claims": ["c1"], "atomic_evidences": [passages("y")]},
            {"claim_string": "s4", "correctness": "Probably correct",
             "atomic_claims": [], "atomic_evidences": []},
        ]
        row = {"question": "q", "annotator_id": "u7",
               "answers": {"post_hoc_gs_gpt4": {"claims": claims}}}
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / C.EXPERTQA_FILE
            f.write_text(json.dumps(row) + "\n")
            with mock.patch.object(C, "EXPERTQA_SHA256", sha(f)):
                units = C.load_expertqa(tmp)
                sens = C.load_expertqa(tmp, mapping="unsure_hallucinated")
        self.assertEqual([(u.id, u.cluster, u.label) for u in units],
                         [("0:0", "u7", 1), ("0:1", "u7", 0)])
        self.assertEqual(units[0].subclaims[0].documents, tuple("pqrst"))
        self.assertEqual(units[0].subclaims[1].documents, ())
        self.assertEqual([u.label for u in sens], [1, 0, 0])
        self.assertNotIn("IGNORED", repr(units))

    def test_refuses_an_unpinned_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / C.EXPERTQA_FILE).write_text("{}\n")
            with self.assertRaises(C.ConfirmatoryDataRefused):
                C.load_expertqa(tmp)


class TestBingCheckLoader(unittest.TestCase):
    def test_units_labels_pages_and_human_evidence_unused(self):
        item = {"r1": {"query": "q", "response": "r", "fact-checking": {
            "subclaims": ["c1", "c2", "c3"],
            "search_results": {"c1": [{"title": "T", "url": "u", "content": ["a", "b"]}] * 4,
                               "c2": [{"title": "T", "url": "u", "content": "z"}]},
            "evidence": {"c1": ["HUMAN"], "c2": ["HUMAN"]},
            "subconclusions": {"c1": "supported", "c2": "partially supported",
                               "c3": "not supported"}}}}
        with tempfile.TemporaryDirectory() as tmp:
            rel = Path(tmp) / "release"
            rel.mkdir()
            (rel / "x_r1.json").write_text(json.dumps(item))
            (rel / C.BINGCHECK_PLACEHOLDER).write_text("version https://git-lfs")
            listing = Path(tmp) / "file_sha256.txt"
            listing.write_text("".join(f"{sha(p)}  {p.name}\n" for p in rel.glob("*.json")))
            with mock.patch.object(C, "BINGCHECK_FILES", 2), \
                    mock.patch.object(C, "BINGCHECK_LISTING_SHA256", sha(listing)):
                units = C.load_bingcheck(rel, listing=listing)
                sens = C.load_bingcheck(rel, mapping="not_supported_hallucinated",
                                        listing=listing)
                (rel / "x_r1.json").write_text(json.dumps(item) + " ")
                with self.assertRaises(C.ConfirmatoryDataRefused):
                    C.load_bingcheck(rel, listing=listing)
        self.assertEqual([(u.id, u.cluster, u.label) for u in units],
                         [("r1:0", "r1", 1), ("r1:1", "r1", 0)])
        self.assertEqual(units[0].subclaims[0].documents, ("a b",) * 3)
        self.assertEqual(units[1].subclaims[0].documents, ("z",))
        self.assertEqual(sens[2].subclaims[0].documents, ())
        self.assertNotIn("HUMAN", repr(units))


@unittest.skipUnless(C.EXPERTQA_DIR.exists() and C.BINGCHECK_DIR.exists(),
                     "pinned releases not present")
class TestPinnedReleases(unittest.TestCase):
    def test_gate_g0_and_preregistration_pins(self):
        self.assertEqual(C.prereg_problems(ROOT), [])
        for name, loader in C.LOADERS.items():
            got = C.check_expected(name, loader())
            self.assertEqual({k: got[k] for k in C.EXPECTED[name]}, C.EXPECTED[name])


class TestPreregistrationPins(unittest.TestCase):
    def test_amendments_only_append(self):
        self.assertEqual(C.prereg_problems(ROOT), [])
        text = (ROOT / C.PREREG_PATH).read_bytes()
        self.assertEqual(hashlib.sha256(text[:C.PREREG_BYTES]).hexdigest(), C.PREREG_SHA256)


if __name__ == "__main__":
    unittest.main()
