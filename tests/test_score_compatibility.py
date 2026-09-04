"""Tests for the pre-write 398-pair score-compatibility probe.

Synthetic pairs and a fake scorer only -- no torch, no model, no released data
-- so CI runs the whole file with numpy alone.

The property under test is that the probe is *evidence*, not decoration: it
must compare fresh scores against stored ones by exact equality, refuse to be
satisfied by bucket agreement, refuse when a sentinel is missing, and write
nothing anywhere.
"""

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.score_compatibility import (
    EXPECTED_NEGATIVE_PAIRS,
    EXPECTED_PAIR_COUNT,
    EXPECTED_POSITIVE_PAIRS,
    PROBE_BATCH_SIZE,
    compare_pair_scores,
    compatibility_report,
    source_integrity_check,
    format_compatibility,
    production_cache_key,
    read_stored_scores,
    recompute_scores,
    run_compatibility_probe,
)

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2"


def synthetic_pairs(n_positive=EXPECTED_POSITIVE_PAIRS,
                    n_negative=EXPECTED_NEGATIVE_PAIRS):
    pairs = [(f"premise {i}", f"positive hypothesis {i}") for i in range(n_positive)]
    pairs += [(f"premise {i}", f"negative hypothesis {i}") for i in range(n_negative)]
    polarities = ["positive"] * n_positive + ["negative"] * n_negative
    return pairs, polarities, n_positive, n_negative


def truth_score(index):
    """A score with enough mantissa to make bucket-vs-raw distinguishable."""
    return 19.9462890625 + index * 0.0009765625


class FakeScorer:
    """Records how it was called and returns whatever it was told to return.

    ``during`` is invoked inside the scoring call, which is where the real
    398 forward passes happen -- the window a post-probe integrity check has to
    cover and a pre-probe one does not.
    """

    def __init__(self, scores, during=None):
        self.scores = list(scores)
        self.calls = []
        self.during = during
        self.finalized_after_calls = None

    def score_pairs(self, pairs, *, use_cache, write_cache, show_progress=False,
                    description=""):
        self.calls.append(
            {"pairs": len(pairs), "use_cache": use_cache, "write_cache": write_cache}
        )
        if self.during is not None:
            self.during()
        return self.scores[: len(pairs)]


def build_source_cache(path, keys_and_scores):
    connection = sqlite3.connect(str(path))
    connection.execute(
        "CREATE TABLE nli_scores (cache_key TEXT PRIMARY KEY, model_name TEXT "
        "NOT NULL, score_version TEXT NOT NULL, score REAL NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO nli_scores VALUES (?, ?, ?, ?)",
        [(key, MODEL, SCORE_VERSION, score) for key, score in keys_and_scores],
    )
    connection.commit()
    connection.close()


class ProbeTestCase(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix="compat-probe-")
        self.root = Path(self._tempdir.name)
        self.source = self.root / "formal_v2.sqlite"
        self.pairs, self.polarities, self.n_pos, self.n_neg = synthetic_pairs()
        self.keys = [
            production_cache_key(MODEL, SCORE_VERSION, premise, hypothesis)
            for premise, hypothesis in self.pairs
        ]
        self.truth = [truth_score(i) for i in range(len(self.pairs))]

    def tearDown(self):
        self._tempdir.cleanup()

    def probe(self, stored=None, fresh=None, scratch_rows=0, mutate_during=None):
        """Drive the probe end to end against a real on-disk source cache.

        ``mutate_during`` runs from inside ``FakeScorer.score_pairs``, i.e.
        between the pre-probe digest and the post-probe digest, so it simulates
        the source cache changing while the forward passes are running.
        """
        stored = self.truth if stored is None else stored
        fresh = self.truth if fresh is None else fresh
        build_source_cache(
            self.source,
            [
                (key, score)
                for key, score in zip(self.keys, stored)
                if score is not None
            ],
        )
        scorer = FakeScorer(fresh, during=mutate_during)
        self.digest_calls = []

        def source_digest():
            digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
            self.digest_calls.append(digest)
            return digest

        report = run_compatibility_probe(
            pairs=self.pairs,
            polarities=self.polarities,
            positive_pairs=self.n_pos,
            negative_pairs=self.n_neg,
            source_cache=str(self.source),
            model_name=MODEL,
            score_version=SCORE_VERSION,
            scorer=scorer,
            finalize=lambda: self._finalize(scorer, scratch_rows),
            source_digest=source_digest,
        )
        self.scorer = scorer
        return report

    def _finalize(self, scorer, scratch_rows):
        scorer.finalized_after_calls = len(scorer.calls)
        return scratch_rows


class TestCacheKey(unittest.TestCase):
    def test_the_probe_key_equals_the_production_key(self):
        # The probe reimplements the key deliberately, so that a change to the
        # production key construction is caught here rather than silently
        # making every sentinel "missing". This test is the pin.
        import hashlib

        premise, hypothesis = "a premise", "a hypothesis"
        expected = hashlib.sha256(
            "\0".join([MODEL, SCORE_VERSION, premise, hypothesis]).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            production_cache_key(MODEL, SCORE_VERSION, premise, hypothesis), expected
        )

    def test_the_key_depends_on_every_component(self):
        base = production_cache_key(MODEL, SCORE_VERSION, "p", "h")
        for other in (
            production_cache_key("other/model", SCORE_VERSION, "p", "h"),
            production_cache_key(MODEL, "other-version", "p", "h"),
            production_cache_key(MODEL, SCORE_VERSION, "p2", "h"),
            production_cache_key(MODEL, SCORE_VERSION, "p", "h2"),
        ):
            self.assertNotEqual(base, other)


class TestCompatibilityVerdict(ProbeTestCase):
    def test_398_of_398_exact_matches_passes(self):
        report = self.probe()
        self.assertEqual(report["pairs_probed"], EXPECTED_PAIR_COUNT)
        self.assertEqual(report["cached_rows_found"], EXPECTED_PAIR_COUNT)
        self.assertEqual(report["exact_raw_matches"], EXPECTED_PAIR_COUNT)
        self.assertEqual(report["raw_mismatches"], 0)
        self.assertEqual(report["maximum_absolute_raw_delta"], 0.0)
        self.assertEqual(report["mismatch_examples"], [])
        self.assertTrue(report["score_compatibility_established"])

    def test_one_raw_mismatch_fails(self):
        fresh = list(self.truth)
        fresh[7] = fresh[7] + 0.0009765625
        report = self.probe(fresh=fresh)
        self.assertEqual(report["exact_raw_matches"], EXPECTED_PAIR_COUNT - 1)
        self.assertEqual(report["raw_mismatches"], 1)
        self.assertFalse(report["score_compatibility_established"])
        self.assertEqual(report["mismatch_examples"][0]["index"], 7)

    def test_a_missing_cached_score_fails(self):
        stored = list(self.truth)
        stored[42] = None
        report = self.probe(stored=stored)
        self.assertEqual(report["cached_rows_found"], EXPECTED_PAIR_COUNT - 1)
        self.assertEqual(report["missing_from_source_cache"], 1)
        # Present pairs all matched, and it still fails.
        self.assertEqual(report["exact_raw_matches"], EXPECTED_PAIR_COUNT - 1)
        self.assertFalse(report["score_compatibility_established"])
        self.assertIn("absent from", report["verdict_reason"])

    def test_the_same_bucket_with_a_different_raw_score_fails(self):
        # The whole point of exact equality. 19.9462890625 and 19.94 round to
        # the same one decimal and land in the same NBC bucket, and they are
        # still different numbers produced by different behaviour. A verdict
        # built on buckets would call this a pass.
        fresh = list(self.truth)
        fresh[0] = 19.94
        report = self.probe(fresh=fresh)
        example = report["mismatch_examples"][0]
        self.assertEqual(example["stored_bucket"], example["fresh_bucket"])
        self.assertEqual(report["nbc_bucket_matches"], EXPECTED_PAIR_COUNT)
        self.assertEqual(report["one_decimal_matches"], EXPECTED_PAIR_COUNT)
        self.assertEqual(report["raw_mismatches"], 1)
        self.assertFalse(report["score_compatibility_established"])

    def test_bucket_agreement_is_reported_as_diagnostic_only(self):
        report = self.probe(fresh=[19.94] + self.truth[1:])
        self.assertIn("never substitute", report["equality_rule"])
        self.assertIn("diagnostic only", format_compatibility(report))

    def test_an_incomplete_sentinel_set_fails(self):
        self.pairs, self.polarities, self.n_pos, self.n_neg = synthetic_pairs(10, 10)
        self.keys = [
            production_cache_key(MODEL, SCORE_VERSION, p, h) for p, h in self.pairs
        ]
        self.truth = [truth_score(i) for i in range(len(self.pairs))]
        report = self.probe()
        self.assertEqual(report["exact_raw_matches"], 20)
        self.assertFalse(report["score_compatibility_established"])
        self.assertIn("must be used whole", report["verdict_reason"])

    def test_a_source_cache_changed_during_the_probe_fails(self):
        # The mutation lands between the two digests, i.e. while the forward
        # passes are running. A pre-probe-only check could not see this.
        def append_row():
            connection = sqlite3.connect(str(self.source))
            connection.execute(
                "INSERT INTO nli_scores VALUES ('intruder', ?, ?, 1.0)",
                (MODEL, SCORE_VERSION),
            )
            connection.commit()
            connection.close()

        report = self.probe(mutate_during=append_row)
        self.assertEqual(report["exact_raw_matches"], EXPECTED_PAIR_COUNT)
        self.assertFalse(report["source_cache_unchanged"])
        self.assertFalse(report["score_compatibility_established"])
        self.assertIn("CHANGED DURING THE COMPATIBILITY PROBE",
                      report["verdict_reason"])

    def test_the_maximum_delta_is_reported(self):
        fresh = list(self.truth)
        fresh[3] = fresh[3] + 0.5
        fresh[9] = fresh[9] + 0.125
        report = self.probe(fresh=fresh)
        self.assertAlmostEqual(report["maximum_absolute_raw_delta"], 0.5)
        self.assertEqual(report["raw_mismatches"], 2)


class TestTheProbeWritesNothing(ProbeTestCase):
    def test_the_recompute_bypasses_the_cache_in_both_directions(self):
        # use_cache=False is what makes this evidence: a cached read would
        # compare the stored value against itself. write_cache=False is what
        # keeps it read-only.
        self.probe()
        self.assertEqual(len(self.scorer.calls), 1)
        self.assertFalse(self.scorer.calls[0]["use_cache"])
        self.assertFalse(self.scorer.calls[0]["write_cache"])

    def test_a_scratch_cache_that_gained_rows_fails(self):
        report = self.probe(scratch_rows=1)
        self.assertFalse(report["score_compatibility_established"])
        self.assertIn("must hold zero", report["verdict_reason"])

    def test_zero_writes_are_reported(self):
        report = self.probe()
        self.assertEqual(report["cache_writes_performed"], 0)
        self.assertEqual(report["scratch_cache_rows_after"], 0)

    def test_the_source_cache_is_opened_read_only(self):
        build_source_cache(self.source, list(zip(self.keys, self.truth)))
        connection = sqlite3.connect(f"file:{self.source}?mode=ro", uri=True)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM nli_scores")
        finally:
            connection.close()

    def test_the_source_is_byte_identical_after_a_probe(self):
        import hashlib

        build_source_cache(self.source, list(zip(self.keys, self.truth)))
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        rows_before = sqlite3.connect(str(self.source)).execute(
            "SELECT COUNT(*) FROM nli_scores"
        ).fetchone()[0]

        found = read_stored_scores(self.source, self.keys)
        recompute_scores(FakeScorer(self.truth), self.pairs)

        self.assertEqual(len(found), EXPECTED_PAIR_COUNT)
        self.assertEqual(
            hashlib.sha256(self.source.read_bytes()).hexdigest(), digest
        )
        self.assertEqual(
            sqlite3.connect(str(self.source)).execute(
                "SELECT COUNT(*) FROM nli_scores"
            ).fetchone()[0],
            rows_before,
        )


class TestSourceIntegrityOrdering(ProbeTestCase):
    """The digest that decides the verdict must be the one taken afterwards."""

    def test_the_digest_is_taken_before_and_after_the_forward_passes(self):
        report = self.probe()
        self.assertEqual(len(self.digest_calls), 2)
        self.assertEqual(
            report["source_sha256_before_probe"], self.digest_calls[0]
        )
        self.assertEqual(report["source_sha256_after_probe"], self.digest_calls[1])
        self.assertTrue(report["source_cache_unchanged"])
        self.assertTrue(report["score_compatibility_established"])

    def test_the_post_probe_digest_is_the_one_used_in_the_verdict(self):
        # The pre-probe digest is taken while the file is still pristine, so a
        # verdict built on it would pass here. The post-probe digest is not.
        pristine = {}

        def append_row():
            pristine["digest"] = self.digest_calls[0]
            connection = sqlite3.connect(str(self.source))
            connection.execute(
                "INSERT INTO nli_scores VALUES ('intruder', ?, ?, 1.0)",
                (MODEL, SCORE_VERSION),
            )
            connection.commit()
            connection.close()

        report = self.probe(mutate_during=append_row)
        self.assertEqual(report["source_sha256_before_probe"], pristine["digest"])
        self.assertNotEqual(
            report["source_sha256_after_probe"], report["source_sha256_before_probe"]
        )
        self.assertFalse(report["score_compatibility_established"])

    def test_the_scratch_scorer_is_finalized_before_the_post_probe_digest(self):
        # finalize() closes the scratch scorer; the digest that follows is
        # therefore taken with nothing still holding a database open.
        self.probe()
        self.assertEqual(self.scorer.finalized_after_calls, 1)

    def test_the_report_carries_both_digests_and_the_verdict(self):
        report = self.probe()
        for key in ("source_sha256_before_probe", "source_sha256_after_probe",
                    "source_cache_unchanged"):
            self.assertIn(key, report)
        self.assertEqual(
            report["source_cache_check"]["sha256_after_probe"],
            report["source_sha256_after_probe"],
        )


class TestMissingMeasurementsFailClosed(unittest.TestCase):
    """An unmeasured integrity property is a failed one, never a passing one."""

    def entries(self):
        pairs, polarities, n_pos, n_neg = synthetic_pairs()
        keys = [f"key-{i}" for i in range(len(pairs))]
        truth = [truth_score(i) for i in range(len(pairs))]
        return (
            compare_pair_scores(
                pairs, polarities, keys, dict(zip(keys, truth)), truth
            ),
            n_pos,
            n_neg,
        )

    def report(self, **overrides):
        entries, n_pos, n_neg = self.entries()
        kwargs = {
            "source_cache": "formal.sqlite",
            "model_name": MODEL,
            "score_version": SCORE_VERSION,
            "positive_pairs": n_pos,
            "negative_pairs": n_neg,
            "scratch_cache_rows_after": 0,
            "source_check": source_integrity_check("formal.sqlite", "abc", "abc"),
        }
        kwargs.update(overrides)
        return compatibility_report(entries, **kwargs)

    def test_the_baseline_passes(self):
        report = self.report()
        self.assertEqual(report["exact_raw_matches"], EXPECTED_PAIR_COUNT)
        self.assertTrue(report["score_compatibility_established"])

    def test_no_source_check_fails(self):
        report = self.report(source_check=None)
        self.assertFalse(report["source_cache_unchanged"])
        self.assertFalse(report["score_compatibility_established"])
        self.assertIn("No post-probe source-integrity check",
                      report["verdict_reason"])

    def test_an_unmeasured_scratch_row_count_fails(self):
        # None must never be read as "zero writes".
        report = self.report(scratch_cache_rows_after=None)
        self.assertFalse(report["score_compatibility_established"])
        self.assertIn("never measured", report["verdict_reason"])

    def test_a_half_measured_digest_fails(self):
        for before, after in (("abc", None), (None, "abc"), (None, None)):
            with self.subTest(before=before, after=after):
                check = source_integrity_check("formal.sqlite", before, after)
                self.assertFalse(check["unchanged"])
                self.assertIn("unverified", check["message"])
                self.assertFalse(
                    self.report(source_check=check)[
                        "score_compatibility_established"
                    ]
                )

    def test_differing_digests_fail(self):
        check = source_integrity_check("formal.sqlite", "abc", "def")
        self.assertFalse(check["unchanged"])
        self.assertFalse(
            self.report(source_check=check)["score_compatibility_established"]
        )


class TestComparisonEntries(unittest.TestCase):
    def test_a_missing_pair_is_never_counted_as_exact(self):
        entries = compare_pair_scores(
            [("p", "h")], ["positive"], ["key"], {}, [19.9462890625]
        )
        self.assertFalse(entries[0]["present_in_source_cache"])
        self.assertFalse(entries[0]["exact_match"])
        self.assertIsNone(entries[0]["absolute_delta"])
        self.assertFalse(entries[0]["nbc_bucket_match"])

    def test_exactness_is_not_approximate(self):
        # One ULP apart: the smallest difference two float64 scores can have,
        # and still a mismatch. No tolerance is applied anywhere.
        import math

        stored = 19.9462890625
        entries = compare_pair_scores(
            [("p", "h")], ["positive"], ["key"],
            {"key": stored}, [math.nextafter(stored, math.inf)],
        )
        self.assertFalse(entries[0]["exact_match"])
        self.assertTrue(entries[0]["one_decimal_match"])
        self.assertTrue(entries[0]["nbc_bucket_match"])
        self.assertGreater(entries[0]["absolute_delta"], 0.0)

    def test_batch_size_one_is_the_probe_default(self):
        self.assertEqual(PROBE_BATCH_SIZE, 1)

    def test_the_expected_sentinel_counts_are_the_released_ones(self):
        self.assertEqual(EXPECTED_POSITIVE_PAIRS, 199)
        self.assertEqual(EXPECTED_NEGATIVE_PAIRS, 199)
        self.assertEqual(EXPECTED_PAIR_COUNT, 398)


class TestReportShape(ProbeTestCase):
    def test_the_report_separates_identity_from_compatibility(self):
        report = self.probe()
        self.assertIn("does not, and cannot, establish which", report["scope_note"])
        self.assertNotIn("checkpoint_identity_established", report)

    def test_an_empty_probe_is_never_established(self):
        report = compatibility_report(
            [], source_cache="x", model_name=MODEL, score_version=SCORE_VERSION,
            positive_pairs=0, negative_pairs=0, scratch_cache_rows_after=0,
        )
        self.assertFalse(report["score_compatibility_established"])

    def test_the_formatted_report_names_the_verdict(self):
        text = format_compatibility(self.probe(fresh=[0.0] + self.truth[1:]))
        self.assertIn("score_compatibility_established: False", text)
        self.assertIn("ABORTING", text)


if __name__ == "__main__":
    unittest.main()
