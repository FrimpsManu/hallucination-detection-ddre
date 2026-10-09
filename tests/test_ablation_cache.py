"""Full-coverage held-out cache completion (preregistration §5).

Covered: the source is never modified; existing rows are copied and preserved
exactly; the coverage set is deterministic and detector-independent; the
398-pair probe gates every write; incomplete coverage fails closed and is not
frozen; the manifest's hashes and counts are the real ones; and the script
has no detector, metric or selection path.

Synthetic SQLite caches and a fake scorer only. No model, no torch, no Wang data.
"""

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import ablation_cache as AC
from src.cache_completion import sha256_file
from src.score_compatibility import production_cache_key
from src.wang_data import EvidenceDocument, SentenceRecord, Subclaim

MODEL = "test-model"
VERSION = "test-version"


def split_text(text, segment_length, overlap_length):
    assert (segment_length, overlap_length) == (400, 100)
    return [part for part in text.split("|") if part]


def record(passage, sentence, subclaims):
    return SentenceRecord(
        passage_index=passage, sentence_index=sentence, sentence="s", label=0,
        raw_label="major_inaccurate",
        subclaims=[Subclaim(text=t, documents=[EvidenceDocument(url="", page_content=d)
                                               for d in docs])
                   for t, docs in subclaims],
    )


RECORDS = [
    record(0, 0, [("claim a", ["x|y", "z"]), ("claim b", ["x"])]),
    record(1, 0, [("claim c", [f"d{i}" for i in range(12)])]),  # beyond max_docs
]


def coverage():
    return AC.enumerate_coverage(RECORDS, split_text, model_name=MODEL, score_version=VERSION)


def make_cache(path, pairs, score=50.0):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE nli_scores (cache_key TEXT PRIMARY KEY, model_name TEXT "
                 "NOT NULL, score_version TEXT NOT NULL, score REAL NOT NULL)")
    conn.executemany(
        "INSERT INTO nli_scores VALUES (?, ?, ?, ?)",
        [(production_cache_key(MODEL, VERSION, p, h), MODEL, VERSION, score + i)
         for i, (p, h) in enumerate(pairs)],
    )
    conn.commit()
    conn.close()
    os.chmod(path, 0o444)  # the canonical cache is read-only, like the real one


class FakeScorer:
    """Writes rows the way EntailmentScorer does. Records what it was asked."""

    def __init__(self, path, skip=0, tamper_existing=False):
        self.conn = sqlite3.connect(path)
        self.skip = skip
        self.tamper_existing = tamper_existing
        self.requested = []

    def score_pairs(self, pairs, **kwargs):
        self.requested.extend(pairs)
        rows = [(production_cache_key(MODEL, VERSION, p, h), MODEL, VERSION, 33.3)
                for p, h in pairs[self.skip:]]
        self.conn.executemany("INSERT OR REPLACE INTO nli_scores VALUES (?, ?, ?, ?)", rows)
        if self.tamper_existing:
            self.conn.execute("UPDATE nli_scores SET score = -1 WHERE rowid = 1")
        self.conn.commit()
        return [33.3] * len(pairs)

    def close(self):
        self.conn.close()


PASSING_PROBE = {"score_compatibility_established": True, "pairs_probed": 398,
                 "exact_raw_matches": 398, "raw_mismatches": 0}


class Fixture:
    def __init__(self, tmp, present_pairs=2):
        self.tmp = Path(tmp)
        self.cov = coverage()
        self.source = self.tmp / "source.sqlite"
        make_cache(self.source, self.cov["pairs"][:present_pairs])
        self.sha = sha256_file(self.source)
        self.mode = stat.S_IMODE(self.source.stat().st_mode)
        self.dest = self.tmp / "derived.sqlite"
        self.scorers = []

    def probe(self, ok=True):
        return lambda: dict(PASSING_PROBE, score_compatibility_established=ok,
                            source_sha256_after_probe=sha256_file(self.source))

    def opener(self, **kw):
        def open_scorer(path):
            scorer = FakeScorer(path, **kw)
            self.scorers.append(scorer)
            return scorer
        return open_scorer

    def run(self, probe_ok=True, **scorer_kw):
        return AC.complete_cache(
            source=self.source, destination=self.dest, source_sha256=self.sha,
            coverage=self.cov, run_probe=self.probe(probe_ok),
            open_scorer=self.opener(**scorer_kw), environment={"device": "mps"},
        )


class TestCoverage(unittest.TestCase):
    def test_enumeration_is_deterministic_and_ordered(self):
        a, b = coverage(), coverage()
        self.assertEqual(a["keys"], b["keys"])
        self.assertEqual(a["coverage_sha256"], b["coverage_sha256"])
        self.assertEqual(a["pairs"][:3], [("x", "claim a"), ("y", "claim a"), ("z", "claim a")])

    def test_positions_one_to_ten_only_and_every_span(self):
        counts = coverage()["counts"]
        self.assertEqual(counts["subclaims"], 3)
        self.assertEqual(counts["document_occurrences"], 2 + 1 + 10)
        self.assertEqual(counts["span_occurrences"], 3 + 1 + 10)
        self.assertNotIn(("d10", "claim c"), coverage()["pairs"])

    def test_pairs_are_deduplicated_by_key(self):
        records = [record(0, 0, [("c", ["x", "x|x"])])]
        cov = AC.enumerate_coverage(records, split_text, model_name=MODEL, score_version=VERSION)
        self.assertEqual(cov["counts"]["span_occurrences"], 3)
        self.assertEqual(cov["counts"]["unique_pairs"], 1)

    def test_keys_use_the_production_convention(self):
        cov = coverage()
        self.assertEqual(cov["keys"][0], production_cache_key(MODEL, VERSION, "x", "claim a"))

    def test_enumeration_takes_no_detector(self):
        params = AC.enumerate_coverage.__code__.co_varnames[
            :AC.enumerate_coverage.__code__.co_argcount + AC.enumerate_coverage.__code__.co_kwonlyargcount]
        self.assertEqual(set(params), {"records", "split_text", "model_name",
                                       "score_version", "max_docs"})


class TestCompletion(unittest.TestCase):
    def test_complete_run_preserves_source_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            m = f.run()
            self.assertEqual(m["status"], AC.STATUS_COMPLETE)
            # source immutability: digest and mode
            self.assertEqual(sha256_file(f.source), f.sha)
            self.assertEqual(stat.S_IMODE(f.source.stat().st_mode), f.mode)
            self.assertTrue(m["source_unchanged"])
            # faithful copy of existing rows
            self.assertTrue(m["preserved_rows"]["preserved"])
            self.assertEqual(m["source_row_count"], 2)
            self.assertEqual(m["added_rows"], len(f.cov["keys"]) - 2)
            self.assertEqual(m["resulting_row_count"], len(f.cov["keys"]))
            self.assertEqual(m["coverage"]["missing_after"], 0)

    def test_only_missing_pairs_are_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, present_pairs=3)
            f.run()
            self.assertEqual(f.scorers[0].requested, f.cov["pairs"][3:])

    def test_manifest_hashes_are_the_real_ones_and_cache_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            m = f.run()
            self.assertEqual(m["derived_sha256"], sha256_file(f.dest))
            self.assertEqual(m["source_sha256"], f.sha)
            self.assertEqual(m["coverage"]["coverage_sha256"], f.cov["coverage_sha256"])
            self.assertEqual(m["derived_frozen_mode"], "0o444")
            self.assertFalse(os.access(f.dest, os.W_OK) and os.geteuid() != 0)
            for field in ("source_row_count", "resulting_row_count", "added_rows",
                          "environment", "compatibility", "status"):
                self.assertIn(field, m)
            self.assertEqual(m["coverage"]["expected_pairs"], len(f.cov["keys"]))
            self.assertFalse(m["new_score_values_inspected"])
            self.assertFalse(m["detectors_run"])
            json.dumps(m)

    def test_manifest_carries_no_new_score_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = Fixture(tmp).run()
            self.assertNotIn("33.3", json.dumps(m))

    def test_probe_failure_aborts_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            with self.assertRaises(AC.CompletionAborted):
                f.run(probe_ok=False)
            self.assertFalse(f.dest.exists())
            self.assertEqual(f.scorers, [])
            self.assertEqual(sha256_file(f.source), f.sha)

    def test_wrong_source_digest_aborts_before_the_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            probe = mock.Mock()
            with self.assertRaises(AC.CompletionAborted):
                AC.complete_cache(source=f.source, destination=f.dest, source_sha256="0" * 64,
                                  coverage=f.cov, run_probe=probe, open_scorer=f.opener(),
                                  environment={})
            probe.assert_not_called()
            self.assertFalse(f.dest.exists())

    def test_unbound_probe_digest_aborts_and_discards_the_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            with self.assertRaises(AC.CompletionAborted):
                AC.complete_cache(
                    source=f.source, destination=f.dest, source_sha256=f.sha, coverage=f.cov,
                    run_probe=lambda: dict(PASSING_PROBE, source_sha256_after_probe="other"),
                    open_scorer=f.opener(), environment={})
            self.assertFalse(f.dest.exists())
            self.assertEqual(f.scorers, [])

    def test_existing_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            f.dest.write_text("x")
            with self.assertRaises(AC.CompletionAborted):
                f.run()
            self.assertEqual(f.dest.read_text(), "x")

    def test_incomplete_coverage_fails_closed_and_is_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            m = f.run(skip=1)
            self.assertEqual(m["status"], AC.STATUS_INCOMPLETE)
            self.assertEqual(m["coverage"]["missing_after"], 1)
            self.assertIsNone(m["derived_frozen_mode"])

    def test_a_changed_existing_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            m = f.run(tamper_existing=True)
            self.assertEqual(m["status"], AC.STATUS_INCOMPLETE)
            self.assertEqual(m["preserved_rows"]["source_rows_changed_in_derived"], 1)


class TestPreservedRows(unittest.TestCase):
    def test_missing_and_changed_rows_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "s.sqlite", Path(tmp) / "d.sqlite"
            make_cache(src, [("a", "h"), ("b", "h"), ("c", "h")])
            make_cache(dst, [("a", "h"), ("b", "h")])
            os.chmod(dst, 0o644)
            conn = sqlite3.connect(dst)
            conn.execute("UPDATE nli_scores SET score = 0 WHERE score = 51.0")
            conn.commit()
            conn.close()
            out = AC.verify_preserved_rows(src, dst)
            self.assertEqual(out["source_rows_missing_from_derived"], 1)
            self.assertEqual(out["source_rows_changed_in_derived"], 1)
            self.assertFalse(out["preserved"])


class TestEnvironmentGate(unittest.TestCase):
    REFERENCE = {"tokenizer": {"model_max_length": 512, "class": "T"}, "model": {"class": "M"},
                 "libraries": {"torch": "2", "transformers": "5", "tokenizers": "0.2"}}

    def observed(self, **overrides):
        snap = {
            "checkpoint_identity": {"resolved_revision": "r", "model_config_commit_hash": "r",
                                    "tokenizer_commit_hash": "r"},
            "model": {"dtype": "torch.float16", "training_mode": False, "class": "M"},
            "device_placement": {"selected_device": "mps", "matches": True},
            "tokenizer": {"model_max_length": 512, "class": "T"},
            "tokenizer_emission_probe": {"input_ids_identical": True},
            "score_version": "v",
            "libraries": {"torch": "2", "transformers": "5", "tokenizers": "0.2"},
        }
        for dotted, value in overrides.items():
            node = snap
            *head, last = dotted.split("__")
            for part in head:
                node = node[part]
            node[last] = value
        return snap

    def expectations(self, reference=None):
        return AC.environment_expectations(reference or self.REFERENCE, revision="r",
                                           score_version="v")

    def test_matching_environment_passes(self):
        self.assertEqual(AC.check_environment(self.observed(), self.expectations()), [])

    def test_each_pinned_field_gates(self):
        for override in ({"model__dtype": "torch.float32"},
                         {"device_placement__selected_device": "cpu"},
                         {"checkpoint_identity__resolved_revision": "main"},
                         {"libraries__torch": "3"},
                         {"model__training_mode": True}):
            self.assertTrue(AC.check_environment(self.observed(**override), self.expectations()),
                            override)

    def test_unobservable_or_unrecorded_fields_fail(self):
        missing = self.observed()
        del missing["checkpoint_identity"]
        self.assertTrue(AC.check_environment(missing, self.expectations()))
        reference = json.loads(json.dumps(self.REFERENCE))
        del reference["libraries"]["tokenizers"]
        self.assertTrue(AC.check_environment(self.observed(), self.expectations(reference)))


if __name__ == "__main__":
    unittest.main()
