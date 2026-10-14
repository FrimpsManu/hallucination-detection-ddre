"""Feature scorer for multivariate DDRE (preregistration §2, §8 step 2)."""

import tempfile
import unittest

from src import feature_scorer as F
from tests.test_scorer_extraction_fidelity import (
    PROBE_LOGITS,
    TORCH_AVAILABLE,
    WANG_SCORE,
    StubModel,
    StubTokenizer,
)


class TestPureHelpers(unittest.TestCase):
    def test_label_indices(self):
        self.assertEqual(F.label_indices({0: "entailment", 1: "neutral", 2: "contradiction"}),
                         {"entailment": 0, "neutral": 1, "contradiction": 2})
        with self.assertRaises(F.FeatureScoreError):
            F.label_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})

    def test_key_depends_on_dtype_and_version(self):
        a = F.feature_key("m", "torch.float16", "p", "h")
        self.assertNotEqual(a, F.feature_key("m", "torch.float32", "p", "h"))
        self.assertNotEqual(a, F.feature_key("m", "torch.float16", "p", "h2"))
        self.assertIn("temp5-seg400-overlap100", F.FEATURE_VERSION)

    def test_document_features_take_the_max_of_each_class(self):
        spans = [{"entailment": 30.0, "contradiction": 60.0},
                 {"entailment": 70.0, "contradiction": 10.0}]
        self.assertEqual(F.document_features(spans), {"e": 70.0, "k": 60.0, "spans": 2})
        with self.assertRaises(F.FeatureScoreError):
            F.document_features([])

    def test_entailment_mismatches_is_exact(self):
        rows = {"a": {"entailment": 19.9462890625}, "b": {"entailment": 50.0}}
        self.assertEqual(F.entailment_mismatches(rows, {"a": 19.9462890625, "b": 50.0}), [])
        self.assertEqual(F.entailment_mismatches(rows, {"a": 19.953125, "b": 50.0}), ["a"])
        with self.assertRaises(F.FeatureScoreError):
            F.entailment_mismatches(rows, {"c": 1.0})


@unittest.skipUnless(TORCH_AVAILABLE, "the feature scorer's forward pass needs torch")
class TestEqualsTheFrozenScorer(unittest.TestCase):
    ROWS = [PROBE_LOGITS, [0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 1.0, 4.0]]
    PAIRS = [(f"p{i}", f"h{i}") for i in range(4)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def scorers(self):
        from src.utils import EntailmentScorer

        frozen = EntailmentScorer(StubTokenizer(), StubModel(self.ROWS), "stub",
                                  cache_path=f"{self.tmp.name}/frozen.sqlite", batch_size=8)
        new = F.FeatureScorer(StubTokenizer(), StubModel(self.ROWS), "stub",
                              dtype="torch.float16",
                              cache_path=f"{self.tmp.name}/features.sqlite", batch_size=8)
        return frozen, new

    def test_entailment_is_bit_identical_to_entailment_scorer(self):
        frozen, new = self.scorers()
        try:
            want = frozen._infer_batch(self.PAIRS)
            got = [r["entailment"] for r in new._infer_batch(self.PAIRS)]
        finally:
            frozen.close()
            new.close()
        self.assertEqual(got, want)
        self.assertEqual(got[0], WANG_SCORE)

    def test_three_probabilities_sum_to_100_and_logits_are_kept(self):
        _, new = self.scorers()
        try:
            rows = new._infer_batch(self.PAIRS)
        finally:
            new.close()
        for row, logits in zip(rows, self.ROWS):
            self.assertAlmostEqual(row["entailment"] + row["neutral"] + row["contradiction"],
                                   100.0, places=1)
            self.assertEqual([row["logit_entailment"], row["logit_neutral"],
                              row["logit_contradiction"]], logits)

    def test_cache_round_trip(self):
        _, new = self.scorers()
        try:
            first = new.score_pairs([("p", "h")])
            self.assertEqual(new.cache_size(), 1)
            again = new.score_pairs([("p", "h"), ("p", "h")])
            self.assertEqual(new.cache_size(), 1)
        finally:
            new.close()
        self.assertEqual(again, [first[0], first[0]])


if __name__ == "__main__":
    unittest.main()
