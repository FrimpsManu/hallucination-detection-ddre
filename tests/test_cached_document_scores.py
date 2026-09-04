"""Tests for the read-only cached document scorer.

Builds real SQLite caches in a temporary directory, so no torch, no model and
no released data are needed and CI runs the file with numpy alone.

Two properties matter most and are pinned here: the cache is opened read-only,
so a formal Gate 1 cache cannot be modified even by a bug; and a score that is
absent raises rather than defaulting, so a sensitivity analysis can never
silently invent the documents it has not seen.
"""

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.cached_document_scores import CachedDocumentScorer, MissingDocumentScore

MODEL = "test-model"
VERSION = "test-version-v2"


def cache_key(premise, hypothesis, model=MODEL, version=VERSION):
    payload = "\0".join([model, version, str(premise), str(hypothesis)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_cache(path, entries, model=MODEL, version=VERSION):
    connection = sqlite3.connect(str(path))
    connection.execute(
        "CREATE TABLE nli_scores ("
        "cache_key TEXT PRIMARY KEY, model_name TEXT NOT NULL, "
        "score_version TEXT NOT NULL, score REAL NOT NULL)"
    )
    connection.executemany(
        "INSERT OR REPLACE INTO nli_scores VALUES (?, ?, ?, ?)",
        [
            (cache_key(premise, hypothesis, model, version), model, version, score)
            for premise, hypothesis, score in entries
        ],
    )
    connection.commit()
    connection.close()


def fake_split_text(text, segment_length=400, overlap_length=100):
    """Deterministic stand-in: one span per whitespace-separated word."""
    return str(text).split()


class CachedScorerTestCase(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix="cached-scores-")
        self.path = Path(self._tempdir.name) / "cache.sqlite"

    def tearDown(self):
        self._tempdir.cleanup()

    def scorer(self, model=MODEL, version=VERSION):
        return CachedDocumentScorer(self.path, model, version, fake_split_text)


class TestDocumentScoring(CachedScorerTestCase):
    def test_document_score_is_the_maximum_across_spans(self):
        build_cache(
            self.path,
            [("alpha", "claim", 10.0), ("beta", "claim", 42.5), ("gamma", "claim", 7.0)],
        )
        scorer = self.scorer()
        try:
            score, spans = scorer.score_document("claim", "alpha beta gamma")
        finally:
            scorer.close()
        self.assertEqual(score, 42.5)
        self.assertEqual(spans, 3)

    def test_empty_document_scores_zero_with_no_spans(self):
        build_cache(self.path, [])
        scorer = self.scorer()
        try:
            self.assertEqual(scorer.score_document("claim", ""), (0.0, 0))
        finally:
            scorer.close()

    def test_repeated_lookups_are_memoized(self):
        build_cache(self.path, [("alpha", "claim", 1.0)])
        scorer = self.scorer()
        try:
            scorer.score_document("claim", "alpha")
            first = scorer.lookups
            scorer.score_document("claim", "alpha")
            self.assertEqual(scorer.lookups, first)
        finally:
            scorer.close()

    def test_signature_matches_the_production_scorer(self):
        # BSEDetector passes use_cache/write_cache; they must be accepted.
        build_cache(self.path, [("alpha", "claim", 3.0)])
        scorer = self.scorer()
        try:
            score, _ = scorer.score_document(
                "claim", "alpha", use_cache=True, write_cache=True
            )
        finally:
            scorer.close()
        self.assertEqual(score, 3.0)


class TestMissingScores(CachedScorerTestCase):
    def test_a_missing_span_raises_rather_than_defaulting(self):
        build_cache(self.path, [("alpha", "claim", 5.0)])
        scorer = self.scorer()
        try:
            with self.assertRaises(MissingDocumentScore):
                scorer.score_document("claim", "alpha missing")
        finally:
            scorer.close()

    def test_the_error_names_the_span_and_claim(self):
        build_cache(self.path, [("alpha", "claim", 5.0)])
        scorer = self.scorer()
        try:
            with self.assertRaises(MissingDocumentScore) as caught:
                scorer.score_document("claim", "alpha missing")
        finally:
            scorer.close()
        self.assertEqual(caught.exception.span_index, 1)
        self.assertEqual(caught.exception.total_spans, 2)
        self.assertEqual(caught.exception.claim, "claim")

    def test_misses_are_counted(self):
        build_cache(self.path, [("alpha", "claim", 5.0)])
        scorer = self.scorer()
        try:
            with self.assertRaises(MissingDocumentScore):
                scorer.score_document("claim", "nope")
            self.assertEqual(scorer.misses, 1)
        finally:
            scorer.close()

    def test_a_wrong_score_version_misses_everything(self):
        # This is what stops a v1 cache being replayed as if it were v2.
        build_cache(self.path, [("alpha", "claim", 5.0)], version="other-version-v1")
        scorer = self.scorer()
        try:
            with self.assertRaises(MissingDocumentScore):
                scorer.score_document("claim", "alpha")
        finally:
            scorer.close()

    def test_a_wrong_model_name_misses_everything(self):
        build_cache(self.path, [("alpha", "claim", 5.0)], model="other-model")
        scorer = self.scorer()
        try:
            with self.assertRaises(MissingDocumentScore):
                scorer.score_document("claim", "alpha")
        finally:
            scorer.close()


class TestReadOnlyGuarantee(CachedScorerTestCase):
    def test_the_connection_refuses_writes(self):
        build_cache(self.path, [("alpha", "claim", 1.0)])
        scorer = self.scorer()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                scorer._connection.execute("DELETE FROM nli_scores")
        finally:
            scorer.close()

    def test_scoring_leaves_the_cache_byte_identical(self):
        build_cache(self.path, [("alpha", "claim", 1.0), ("beta", "claim", 2.0)])
        before = self.path.read_bytes()
        scorer = self.scorer()
        try:
            scorer.score_document("claim", "alpha beta")
        finally:
            scorer.close()
        self.assertEqual(self.path.read_bytes(), before)

    def test_row_count_is_reported(self):
        build_cache(self.path, [("alpha", "claim", 1.0), ("beta", "claim", 2.0)])
        scorer = self.scorer()
        try:
            self.assertEqual(scorer.cache_rows(), 2)
        finally:
            scorer.close()

    def test_stats_record_the_replay_identity(self):
        build_cache(self.path, [("alpha", "claim", 1.0)])
        scorer = self.scorer()
        try:
            scorer.score_document("claim", "alpha")
            stats = scorer.stats()
        finally:
            scorer.close()
        self.assertEqual(stats["model_name"], MODEL)
        self.assertEqual(stats["score_version"], VERSION)
        self.assertEqual(stats["span_misses"], 0)
        self.assertEqual(stats["documents_memoized"], 1)

    def test_close_is_idempotent(self):
        build_cache(self.path, [])
        scorer = self.scorer()
        scorer.close()
        scorer.close()


if __name__ == "__main__":
    unittest.main()
