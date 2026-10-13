"""BM25 evidence retrieval (transfer preregistration §3) on a tiny synthetic snapshot."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import wikipedia_bm25 as W

SEP = W.SNAPSHOT_SEPARATOR


def make_snapshot(path, pages):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE documents (title PRIMARY KEY, text)")
    con.executemany("INSERT INTO documents VALUES (?, ?)", pages)
    con.commit()
    con.close()


class TestQuery(unittest.TestCase):
    def test_tokenize_folds_case_and_diacritics(self):
        self.assertEqual(W.tokenize("José Martí, born 1853!"), ["jose", "marti", "born", "1853"])

    def test_query_terms_remove_stopwords_and_duplicates(self):
        self.assertEqual(W.query_terms("Lanny Flaherty", "He was born in Pensacola, Florida."),
                         ["lanny", "flaherty", "born", "pensacola", "florida"])
        self.assertEqual(W.query_terms("Ann Lee", "Ann Lee is Ann."), ["ann", "lee"])

    def test_empty_query_is_refused(self):
        with self.assertRaises(W.RetrievalRefused):
            W.query_terms("", "it is")

    def test_match_expression_quotes_each_term(self):
        self.assertEqual(W.match_expression(["a1", "b"]), '"a1" OR "b"')

    def test_frozen_parameters(self):
        self.assertEqual((W.TOP_K, W.COLUMN_WEIGHTS), (10, (1.0, 1.0)))
        self.assertEqual(W.TOKENIZER, "unicode61 remove_diacritics 2")
        self.assertEqual(len(W.STOPWORDS), 179)

    def test_document_text(self):
        self.assertEqual(W.document_text("Ann Lee", "A poet."), "Ann Lee\nA poet.")


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.snapshot, self.index = root / "snap.db", root / "idx.sqlite"
        make_snapshot(self.snapshot, [
            ("Ann Lee", SEP.join(["Ann Lee is a poet from Lisbon.", "  ", "She won the Lisbon prize."])),
            ("Bo Ray", "Bo Ray is a chef."),
            ("Zed", "unrelated text about rivers"),
            ("Copy B", "identical twin passage"),
            ("Copy A", "identical twin passage"),
            ("Copy C", "identical twin passage"),
        ])
        self.counts = W.build_index(self.snapshot, self.index, batch=2)
        self.con = sqlite3.connect(f"file:{self.index}?mode=ro", uri=True)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_every_nonempty_passage_is_indexed_with_snapshot_indices(self):
        self.assertEqual(self.counts, {"pages": 6, "passages": 7})
        maps = self.con.execute("SELECT title, pidx FROM passage_map WHERE title = 'Ann Lee'").fetchall()
        self.assertEqual(maps, [("Ann Lee", 0), ("Ann Lee", 2)])

    def test_ranking_and_snapshot_text(self):
        hits = W.retrieve(self.con, W.query_terms("Ann Lee", "won the Lisbon prize"))
        self.assertEqual([(t, p) for _, t, p in hits[:2]], [("Ann Lee", 2), ("Ann Lee", 0)])
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(hits, hits[1:])))
        snap = sqlite3.connect(f"file:{self.snapshot}?mode=ro", uri=True)
        self.assertEqual(W.passage_text(snap, "Ann Lee", 2), "She won the Lisbon prize.")
        with self.assertRaises(W.RetrievalRefused):
            W.passage_text(snap, "Ann Lee", 1)

    def test_ties_break_by_title_then_passage_and_never_split_at_the_cutoff(self):
        terms = W.query_terms("", "identical twin")
        full = W.retrieve(self.con, terms, k=10)
        self.assertEqual([t for _, t, _ in full], ["Copy A", "Copy B", "Copy C"])
        # Pool smaller than the tie: it must grow rather than cut the tie arbitrarily.
        cut = W.retrieve(self.con, terms, k=2, pool=2)
        self.assertEqual([t for _, t, _ in cut], ["Copy A", "Copy B"])

    def test_fewer_matches_than_k(self):
        self.assertEqual(len(W.retrieve(self.con, ["rivers"])), 1)


if __name__ == "__main__":
    unittest.main()
