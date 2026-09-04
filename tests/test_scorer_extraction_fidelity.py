"""Regression tests for the Wang-fidelity scaling correction in EntailmentScorer.

Gate 1 Step 2 isolated the cause of the A1/A3 scorer-path discrepancy on the
real T4 run: the forward path was bit-identical across all 398 NBC pairs, the
softmax 1-D/2-D shape was bit-identical too, and the *scaling order alone*
reproduced the recorded pair-169 discrepancy and the exact Step 1 positive NBC
histogram movement.

Released ``utils.py:59-62`` takes the probabilities out of the tensor and only
then multiplies by 100, in Python float64. The previous implementation
multiplied while still inside the tensor, so the product was rounded to the
tensor dtype. On a half-precision run that rounding is coarse: float16 spacing
in [16, 32) is 0.015625, and the probe probability 0.199462890625 gives
19.9462890625 host-side but 19.953125 tensor-side.

Two layers of coverage:

* an arithmetic-contract test using ``numpy.float16``, which needs no model and
  no torch, so CI (numpy only) always runs it;
* a torch-gated test that drives the real ``_infer_batch`` with fp16 logits
  chosen to produce exactly that probability, so the production code path
  itself is pinned.
"""

import importlib.util
import tempfile
import unittest

import numpy as np


def _torch_available():
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


TORCH_AVAILABLE = _torch_available()

# The entailment probability recorded on the T4 run for positive NBC pair 169,
# exactly representable in float16.
PROBE_PROBABILITY = 0.199462890625

# What Wang's released extraction produces, and what the pre-correction
# tensor-side scaling produced.
WANG_SCORE = 19.9462890625
TENSOR_SCALED_SCORE = 19.953125

# fp16 logits whose fp16 softmax(logits / 5) yields PROBE_PROBABILITY exactly.
PROBE_LOGITS = [-3.484375, 0.0, 0.0]


class TestScalingOrderArithmetic(unittest.TestCase):
    """The contract, demonstrated without torch so CI always covers it."""

    def test_host_side_scaling_gives_the_wang_score(self):
        p = np.float16(PROBE_PROBABILITY)
        self.assertEqual(float(p) * 100.0, WANG_SCORE)

    def test_tensor_side_scaling_gives_the_wrong_score(self):
        p = np.float16(PROBE_PROBABILITY)
        self.assertEqual(float(np.float16(p * np.float16(100.0))), TENSOR_SCALED_SCORE)

    def test_the_two_orders_disagree_by_the_recorded_gap(self):
        p = np.float16(PROBE_PROBABILITY)
        gap = float(np.float16(p * np.float16(100.0))) - float(p) * 100.0
        self.assertEqual(gap, 0.0068359375)

    def test_the_corrected_order_is_the_one_the_scorer_must_use(self):
        self.assertNotEqual(WANG_SCORE, TENSOR_SCALED_SCORE)


class StubInputs(dict):
    """Mapping that also answers ``.to(device)``, like a BatchEncoding."""

    def to(self, device):
        return self


class StubTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.calls.append({"premises": list(premises), "kwargs": dict(kwargs)})
        return StubInputs(batch=len(premises))


class StubConfig:
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}


class StubOutput:
    def __init__(self, logits):
        self.logits = logits


class StubModel:
    """Returns fixed fp16 logits, one row per pair in the batch."""

    def __init__(self, rows):
        import torch

        self.config = StubConfig()
        self._rows = rows
        self._torch = torch
        self._parameter = torch.zeros(1)

    def parameters(self):
        yield self._parameter

    def __call__(self, **kwargs):
        batch = kwargs.get("batch", len(self._rows))
        logits = self._torch.tensor(
            self._rows[:batch], dtype=self._torch.float16
        )
        return StubOutput(logits)


@unittest.skipUnless(TORCH_AVAILABLE, "src.utils imports torch at module scope")
class TestInferBatchScalingFidelity(unittest.TestCase):
    """The real ``_infer_batch``, pinned against the recorded T4 values."""

    def scorer(self, rows):
        from src.utils import EntailmentScorer

        self._tempdir = tempfile.TemporaryDirectory(prefix="scorer-fidelity-")
        return EntailmentScorer(
            StubTokenizer(),
            StubModel(rows),
            "stub-model",
            cache_path=f"{self._tempdir.name}/scratch.sqlite",
            batch_size=8,
        )

    def tearDown(self):
        tempdir = getattr(self, "_tempdir", None)
        if tempdir is not None:
            tempdir.cleanup()

    def test_probe_pair_returns_the_wang_score(self):
        scorer = self.scorer([PROBE_LOGITS])
        try:
            scores = scorer._infer_batch([("premise", "hypothesis")])
        finally:
            scorer.close()
        self.assertEqual(scores, [WANG_SCORE])

    def test_probe_pair_does_not_return_the_tensor_scaled_score(self):
        # The exact regression: this is what the pre-correction code returned.
        scorer = self.scorer([PROBE_LOGITS])
        try:
            scores = scorer._infer_batch([("premise", "hypothesis")])
        finally:
            scorer.close()
        self.assertNotEqual(scores[0], TENSOR_SCALED_SCORE)

    def test_the_underlying_probability_is_the_recorded_one(self):
        # Guards the fixture: if the fp16 softmax stopped producing exactly
        # 0.199462890625, the two assertions above would be testing nothing.
        import torch

        logits = torch.tensor([PROBE_LOGITS], dtype=torch.float16)
        probability = float(torch.softmax(logits / 5.0, dim=-1)[0, 0])
        self.assertEqual(probability, PROBE_PROBABILITY)

    def test_no_one_decimal_rounding_is_introduced(self):
        # Scores must stay continuous for DDRE. Rounding belongs to BSE, in
        # src/baseline_core.py, not to the scorer.
        scorer = self.scorer([PROBE_LOGITS])
        try:
            scores = scorer._infer_batch([("premise", "hypothesis")])
        finally:
            scorer.close()
        self.assertNotEqual(scores[0], round(scores[0], 1))
        self.assertGreater(len(str(scores[0]).split(".")[1]), 1)

    def test_batching_returns_one_score_per_pair(self):
        rows = [PROBE_LOGITS, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        pairs = [("p0", "h0"), ("p1", "h1"), ("p2", "h2")]
        scorer = self.scorer(rows)
        try:
            scores = scorer._infer_batch(pairs)
        finally:
            scorer.close()
        self.assertEqual(len(scores), len(pairs))
        self.assertEqual(scores[0], WANG_SCORE)
        # Row 1 has equal logits, so the entailment probability is exactly 1/3.
        self.assertAlmostEqual(scores[1], 100.0 / 3.0, places=1)
        self.assertNotEqual(scores[1], scores[2])

    def test_scores_are_plain_floats(self):
        scorer = self.scorer([PROBE_LOGITS])
        try:
            scores = scorer._infer_batch([("premise", "hypothesis")])
        finally:
            scorer.close()
        self.assertIsInstance(scores[0], float)

    def test_score_pairs_returns_one_score_per_pair_uncached(self):
        rows = [PROBE_LOGITS, [1.0, 0.0, 0.0]]
        pairs = [("p0", "h0"), ("p1", "h1")]
        scorer = self.scorer(rows)
        try:
            scores = scorer.score_pairs(pairs, use_cache=False, write_cache=False)
        finally:
            scorer.close()
        self.assertEqual(len(scores), len(pairs))
        self.assertEqual(scores[0], WANG_SCORE)


class TestScoreVersionBump(unittest.TestCase):
    """Old fp16-scaled cache rows must not be reusable under the new scorer."""

    @unittest.skipUnless(TORCH_AVAILABLE, "src.utils imports torch at module scope")
    def test_score_version_is_v2_and_names_the_change(self):
        from src.utils import SCORE_VERSION

        self.assertNotEqual(SCORE_VERSION, "wang-emnlp23-temp5-seg400-overlap100-v1")
        self.assertTrue(SCORE_VERSION.endswith("v2"))
        self.assertIn("hostscale", SCORE_VERSION)

    @unittest.skipUnless(TORCH_AVAILABLE, "src.utils imports torch at module scope")
    def test_cache_keys_differ_from_the_v1_convention(self):
        import hashlib

        from src.utils import SCORE_VERSION, EntailmentScorer

        with tempfile.TemporaryDirectory() as tempdir:
            scorer = EntailmentScorer(
                StubTokenizer(),
                StubModel([PROBE_LOGITS]),
                "stub-model",
                cache_path=f"{tempdir}/scratch.sqlite",
                batch_size=1,
            )
            try:
                key = scorer._cache_key("premise", "hypothesis")
            finally:
                scorer.close()

        legacy = hashlib.sha256(
            "\0".join(
                [
                    "stub-model",
                    "wang-emnlp23-temp5-seg400-overlap100-v1",
                    "premise",
                    "hypothesis",
                ]
            ).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(key, legacy)
        self.assertIn(SCORE_VERSION, "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2")


if __name__ == "__main__":
    unittest.main()
