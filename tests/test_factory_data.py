"""FACTORY loader and retrieval script (src/factory_data.py, scripts/retrieve_factory_evidence.py)."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src import factory_data as F  # noqa: E402


def record(question, sentence, *tags):
    r = {"question": question, "response": "r", "target_sentence": sentence}
    for i in range(1, 6):
        t = tags[i - 1] if i <= len(tags) else ""
        r[f"Claim {i}"], r[f"Claim {i} Tag"] = (f"c{i}" if t else ""), t
        r[f"Source Claim {i}"], r[f"Claim {i} Snippet"] = "", ""
    return r


RECORDS = [
    record("Q1", "s0", "Factual", "factual"),               # factual
    record("Q1", "s1", "Factual", "NonFactual"),            # hallucinated (A and B)
    record("Q2", "s2", "Factual", " Inconclusive "),        # hallucinated in A; excluded in B
    record("Q2", "s3"),                                     # excluded: no labelled claim
    record("Q3", "s4", "No Verifiable Facts"),              # excluded: only no claim
    record("Q3", "s5", "Factual", "No Verifiable Facts"),   # factual
    record("Q1", "s0", "Nonfactual"),                       # duplicate text, own unit
]


def fake_release(tmp, records=RECORDS):
    path = Path(tmp) / "human_annotations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class Labels(unittest.TestCase):
    def test_rules(self):
        cases = [(["factual"], 1, 1), (["factual", "nonfactual"], 0, 0),
                 (["factual", "inconclusive"], 0, None), (["inconclusive"], 0, None),
                 ([], None, None), (["no verifiable facts"], None, None),
                 (["factual", "no verifiable facts"], 1, 1)]
        for tags, a, b in cases:
            self.assertEqual((F.rule_a(tags), F.rule_b(tags)), (a, b), tags)

    def test_tags_are_normalised_in_claim_order(self):
        r = record("q", "s", "Factual", " NonFactual", "inconclusive")
        self.assertEqual(F.claim_tags(r), ["factual", "nonfactual", "inconclusive"])

    def test_unknown_tag_stops(self):
        with self.assertRaises(F.FactoryDataRefused):
            F.claim_tags(record("q", "s", "Mostly true"))


class Loader(unittest.TestCase):
    def load(self, rule=F.rule_a, records=RECORDS):
        with tempfile.TemporaryDirectory() as tmp:
            path, sha = fake_release(tmp, records)
            with mock.patch.object(F, "RELEASE_SHA256", sha):
                return F.load_factory(path, rule)

    def test_units_clusters_and_exclusions(self):
        units, counts = self.load()
        self.assertEqual([u.id for u in units], ["0", "1", "2", "5", "6"])
        self.assertEqual([u.label for u in units], [1, 0, 0, 1, 0])
        self.assertEqual([u.cluster for u in units], ["q0", "q0", "q1", "q2", "q0"])
        self.assertEqual(units[0].title, "Q1")
        self.assertEqual(counts, {"records": 7, "excluded_no_labelled_claim": 1,
                                  "excluded_only_no_verifiable_facts": 1})
        self.assertEqual(F.totals(units, counts)["hallucinated"], 3)

    def test_rule_b(self):
        units, _ = self.load(F.rule_b)
        self.assertEqual([u.id for u in units], ["0", "1", "5", "6"])

    def test_queries_carry_no_labels(self):
        units, _ = self.load()
        self.assertEqual(F.queries(units), [("Q1", "s0"), ("Q1", "s1"), ("Q2", "s2"),
                                            ("Q3", "s5")])

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fake_release(tmp)
            with self.assertRaises(F.FactoryDataRefused):
                F.load_factory(path)  # digest does not match the pin
        with self.assertRaises(F.FactoryDataRefused):
            self.load(records=[record("Q", " ", "Factual")])
        units, counts = self.load()
        with self.assertRaises(F.FactoryDataRefused):
            F.check_expected(units, counts)

    def test_expected_totals_are_consistent(self):
        e = F.EXPECTED
        self.assertEqual(e["units"] + e["excluded_no_labelled_claim"]
                         + e["excluded_only_no_verifiable_facts"], e["records"])
        self.assertEqual(e["units"] - e["hallucinated"], 1938)


class Prereg(unittest.TestCase):
    def test_preregistration_prefix_pinned(self):
        self.assertEqual(F.prereg_problems(ROOT), [])


class Script(unittest.TestCase):
    def test_outputs_and_refusal(self):
        import retrieve_factory_evidence as RE

        out = RE.output_paths(Path("/x"))
        self.assertEqual(out["evidence"].name, "factory_evidence_top10.jsonl")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(F, "RELEASE_FILE", Path(tmp) / "missing.jsonl"):
                with self.assertRaises(RE.FactoryRetrievalRefused):
                    RE.verify_inputs(Path(tmp), RE.output_paths(Path(tmp)), dry_run=True)


if __name__ == "__main__":
    unittest.main()
