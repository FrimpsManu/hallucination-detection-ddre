"""Fidelity tests against Wang et al.'s released implementation.

Each test transcribes the released code literally and asserts that this
repository reproduces it. References are to the released repository at commit
3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea:

    https://github.com/xhwang22/HallucinationDetection

These tests exist so that a future reviewer who suspects one of these choices
is a bug can see, in the diff, that it is a deliberate reproduction of the
published implementation. They must fail loudly if anyone "improves" the
baseline. See the baseline protection rule in the project research brief.

No torch, scipy, or sklearn is required except where explicitly skipped.
"""

import importlib.util
import unittest

from src.baseline_core import (
    BSEDetector,
    cost_based_prediction,
    discretize_document_score,
    discretize_nbc_score,
    stop_cost,
)
from src.wang_data import EvidenceDocument, Subclaim, label_to_int


def _torch_available():
    """find_spec can raise for partially-initialised or stubbed modules."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


TORCH_AVAILABLE = _torch_available()

# Representative Laplace-smoothed histogram shapes. The middle entry is the
# degenerate symmetric case; the last is the shape under which the released
# control flow retrieves nothing at P0 = 0.5.
HISTOGRAM_SHAPES = (
    ([3, 3, 3, 4, 6, 9, 16, 31, 61, 121], [101, 41, 21, 11, 9, 7, 6, 5, 5, 4]),
    ([21] * 10, [21] * 10),
    ([6, 6, 6, 6, 11, 16, 26, 41, 61, 81], [81, 61, 41, 26, 16, 11, 6, 6, 6, 6]),
)


# --------------------------------------------------------------------------
# Literal transcriptions of the released implementation.
# --------------------------------------------------------------------------

def wang_min_cost(p, C_M, C_FA):
    """Released main.py:121-125."""
    return min((1 - p) * C_M, p * C_FA)


def wang_cal_En_plus1(neg_features, pos_features, P_n):
    """Released main.py:127-140.

    Note the uniform average over the ten bins: the released look-ahead does
    not weight bins by their predictive probability. That is precisely where
    the released code and the paper's Equation 8 diverge, which is why this
    repository keeps bse_official and bse_equation8 as separate modes.
    """
    P_nplus1 = [0] * 10
    for i in range(10):
        P_nplus1_given_1 = pos_features[i] / sum(pos_features)
        P_nplus1_given_0 = neg_features[i] / sum(neg_features)
        P_nplus1[i] = (
            P_n * P_nplus1_given_1
            / ((1 - P_n) * P_nplus1_given_0 + P_n * P_nplus1_given_1)
        )
    return sum(P_nplus1) / len(P_nplus1)


def wang_split_text(text, segment_length, overlap_length):
    """Released utils.py:71-88 (no deduplication of the repeated tail span)."""
    segments = []
    start = 0
    end = segment_length
    text_list = text.split()[0:4000]
    while start < len(text_list):
        if end >= len(text_list):
            segment = text_list[-segment_length:]
        else:
            segment = text_list[start:end]
        segments.append(" ".join(segment))
        start += segment_length - overlap_length
        end = start + segment_length
    return segments


def wang_detect_subclaim(documents, pos, neg, P0, C_M, C_FA, C_retrieve, score_of):
    """Released main.py:226-258, the per-subclaim sequential loop.

    The stop/continue test is evaluated BEFORE the first retrieval. This is the
    released behaviour and must not be changed.
    """
    P = P0
    search_time = 0
    stop = wang_min_cost(P, C_M, C_FA)
    search = C_retrieve + wang_min_cost(wang_cal_En_plus1(neg, pos, P), C_M, C_FA)
    for web in documents:
        if stop > search:
            search_time += 1
            entailment_prob = score_of(web)
            bucket = int((entailment_prob - 0.1) / 10)
            p_given_1 = pos[bucket] / sum(pos)
            p_given_0 = neg[bucket] / sum(neg)
            P = P * p_given_1 / ((1 - P) * p_given_0 + P * p_given_1)
            stop = wang_min_cost(P, C_M, C_FA)
            search = C_retrieve + wang_min_cost(
                wang_cal_En_plus1(neg, pos, P), C_M, C_FA
            )
    return P, search_time


class StubScorer:
    """Deterministic stand-in for EntailmentScorer, so no NLI model is needed."""

    def __init__(self, scores_by_content):
        self.scores_by_content = scores_by_content

    def score_document(self, claim, page_content, *, use_cache=True, write_cache=True):
        return self.scores_by_content[page_content], 3


# --------------------------------------------------------------------------

class TestCostFunctions(unittest.TestCase):
    def test_stop_cost_matches_min_cost(self):
        for c_miss, c_false_alarm in ((14, 24), (28, 96)):
            for i in range(0, 101):
                p = i / 100.0
                self.assertAlmostEqual(
                    stop_cost(p, c_miss, c_false_alarm),
                    wang_min_cost(p, c_miss, c_false_alarm),
                    places=12,
                )

    def test_final_decision_matches_released_rule(self):
        for c_miss, c_false_alarm in ((14, 24), (28, 96)):
            for i in range(0, 1001):
                p = i / 1000.0
                expected = 1 if ((1 - p) * c_miss) < (p * c_false_alarm) else 0
                self.assertEqual(
                    cost_based_prediction(p, c_miss, c_false_alarm), expected
                )


class TestLookAhead(unittest.TestCase):
    def test_official_look_ahead_matches_cal_En_plus1(self):
        for pos, neg in HISTOGRAM_SHAPES:
            detector = BSEDetector(pos, neg, mode="official")
            for i in range(1, 100):
                p = i / 100.0
                # Private accessor used deliberately: this is the quantity being
                # checked for fidelity against the released look-ahead.
                self.assertAlmostEqual(
                    detector._official_expected_next_posterior(p),
                    wang_cal_En_plus1(neg, pos, p),
                    places=12,
                )

    def test_should_continue_matches_released_comparison(self):
        for pos, neg in HISTOGRAM_SHAPES:
            for c_miss, c_false_alarm in ((14, 24), (28, 96)):
                detector = BSEDetector(
                    pos,
                    neg,
                    mode="official",
                    c_miss=c_miss,
                    c_false_alarm=c_false_alarm,
                    c_retrieve=1,
                )
                for i in range(1, 100):
                    p = i / 100.0
                    expected = wang_min_cost(p, c_miss, c_false_alarm) > (
                        1
                        + wang_min_cost(
                            wang_cal_En_plus1(neg, pos, p), c_miss, c_false_alarm
                        )
                    )
                    self.assertEqual(detector.should_continue(p), expected)


class TestDiscretization(unittest.TestCase):
    """The released code uses two different discretizers. Both are preserved."""

    def test_nbc_discretizer_matches_NBC_feature_py(self):
        # NBC_feature.py:34 -> int(score / 10), on scores already rounded to one
        # decimal place by utils.py:62.
        for i in range(0, 1000):
            score = round(i / 10.0, 1)
            self.assertEqual(discretize_nbc_score(score), int(score / 10))

    def test_document_discretizer_matches_released_main_py(self):
        # main.py:249-250 -> int((score - 0.1) / 10)
        for i in range(0, 1000):
            score = round(i / 10.0, 1)
            self.assertEqual(
                discretize_document_score(score), int((score - 0.1) / 10)
            )

    def test_the_two_discretizers_really_do_differ(self):
        self.assertEqual(discretize_nbc_score(10.0), 1)
        self.assertEqual(discretize_document_score(10.0), 0)

    def test_clamp_is_unreachable_for_in_range_scores(self):
        # The clamp only engages at exactly 100.0, where the released
        # NBC_feature.py would index a ten-element list with 10 and raise
        # IndexError. Reaching 100.0 requires an entailment probability above
        # 0.99995 after temperature-5 softmax, which does not occur in practice.
        self.assertEqual(int(100.0 / 10), 10)
        self.assertEqual(discretize_nbc_score(100.0), 9)
        self.assertEqual(discretize_document_score(100.0), 9)


class TestSequentialLoop(unittest.TestCase):
    def test_subclaim_loop_matches_released_control_flow(self):
        contents = [f"document-{i}" for i in range(10)]
        scores = {
            content: round(3.7 + 9.4 * i, 1) for i, content in enumerate(contents)
        }
        subclaim = Subclaim(
            text="a subclaim",
            documents=[EvidenceDocument(url="", page_content=c) for c in contents],
        )
        scorer = StubScorer(scores)

        for pos, neg in HISTOGRAM_SHAPES:
            for c_miss, c_false_alarm in ((14, 24), (28, 96)):
                detector = BSEDetector(
                    pos,
                    neg,
                    mode="official",
                    p0=0.5,
                    c_miss=c_miss,
                    c_false_alarm=c_false_alarm,
                    c_retrieve=1,
                    max_docs=10,
                )
                result = detector.detect_subclaim(subclaim, scorer)
                expected_p, expected_docs = wang_detect_subclaim(
                    contents,
                    pos,
                    neg,
                    0.5,
                    c_miss,
                    c_false_alarm,
                    1,
                    lambda content: scores[content],
                )
                self.assertAlmostEqual(result.p_factual, expected_p, places=12)
                self.assertEqual(result.documents_used, expected_docs)

    def test_first_iteration_decision_is_dataset_wide_constant(self):
        # Every subclaim starts at P = P0 and the histograms are global, so
        # whether the first document is retrieved does not depend on the
        # subclaim. This is what makes zero retrieval an all-or-nothing
        # degeneracy worth probing before a full run.
        pos, neg = HISTOGRAM_SHAPES[2]
        detector = BSEDetector(
            pos, neg, mode="official", p0=0.5, c_miss=28, c_false_alarm=96, c_retrieve=1
        )
        self.assertFalse(detector.should_continue(0.5))

        scorer = StubScorer({"anything": 55.0})
        subclaim = Subclaim(
            text="s", documents=[EvidenceDocument(url="", page_content="anything")]
        )
        result = detector.detect_subclaim(subclaim, scorer)
        self.assertEqual(result.documents_used, 0)
        self.assertEqual(result.p_factual, 0.5)


class TestLabelMapping(unittest.TestCase):
    def test_label_mapping_matches_released_main_py(self):
        # main.py:205-209: "accurate" is factual; both minor_inaccurate and
        # major_inaccurate are nonfactual.
        for raw in ("accurate", "minor_inaccurate", "major_inaccurate"):
            expected = 1 if raw == "accurate" else 0
            self.assertEqual(label_to_int(raw), expected)


@unittest.skipUnless(TORCH_AVAILABLE, "src.utils imports torch at module scope")
class TestSegmentation(unittest.TestCase):
    """Deduplication is this repository's only deviation from released split_text."""

    LENGTHS = (0, 1, 50, 299, 300, 350, 399, 400, 401, 700, 1000, 1500, 4000, 5000)

    def _document(self, n_words):
        return " ".join(f"w{i}" for i in range(n_words))

    def test_dedup_is_the_only_difference(self):
        from src.utils import split_text

        for n in self.LENGTHS:
            text = self._document(n)
            ours = split_text(text, segment_length=400, overlap_length=100)
            theirs = wang_split_text(text, 400, 100)
            self.assertEqual(ours, list(dict.fromkeys(theirs)))
            self.assertEqual(set(ours), set(theirs))
            self.assertLessEqual(len(ours), len(theirs))

    def test_max_over_spans_is_unaffected_by_dedup(self):
        from src.utils import split_text

        def pseudo_score(segment):
            return (hash(segment) % 1000) / 10.0

        for n in self.LENGTHS:
            text = self._document(n)
            ours = split_text(text, segment_length=400, overlap_length=100)
            theirs = wang_split_text(text, 400, 100)
            if not theirs:
                self.assertEqual(ours, [])
                continue
            self.assertEqual(
                max(pseudo_score(s) for s in ours),
                max(pseudo_score(s) for s in theirs),
            )

    def test_four_thousand_word_cap_is_released_behaviour(self):
        from src.utils import split_text

        # utils.py:78 -> text.split()[0:4000]. This cap is Wang's, not ours.
        long_text = self._document(12000)
        covered = {w for s in split_text(long_text) for w in s.split()}
        self.assertEqual(len(covered), 4000)
        self.assertEqual(
            split_text(long_text), list(dict.fromkeys(wang_split_text(long_text, 400, 100)))
        )


if __name__ == "__main__":
    unittest.main()
