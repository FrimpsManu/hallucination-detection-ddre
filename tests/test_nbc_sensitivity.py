"""Tests for the NBC missing-examples sensitivity analysis.

Synthetic values throughout. No torch, no model, no released data, so CI runs
the whole file with numpy alone.

The analysis is a sensitivity study, and these tests pin the properties that
keep it one: the released histograms are never mutated, an unevaluated
combination is never folded into a PASS or FAIL tally, and an empty sample never
produces a conclusion.
"""

import unittest

from src.nbc_sensitivity import (
    MISSING_EXAMPLES_PER_CLASS,
    N_BINS,
    PAPER_EXAMPLES_PER_CLASS,
    RELEASED_EXAMPLES_PER_CLASS,
    RELEASED_NEGATIVE_HISTOGRAM,
    RELEASED_POSITIVE_HISTOGRAM,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    add_one_to_bin,
    closest_combinations,
    combination_histograms,
    enumerate_combinations,
    interpretation,
    metric_distance,
    raw_count,
    summarize,
)

PRIMARY = "CM_14_CFA_24"
SECONDARY = "CM_28_CFA_96"

METRIC_NAMES = (
    "nonfactual_auc_pr",
    "factual_auc_pr",
    "accuracy",
    "pearson",
    "spearman",
    "evidence_num_per_sentence",
)


def configuration_block(verdict, deltas=None):
    """A minimal stand-in for evaluate_configuration()'s return shape."""
    deltas = deltas or {name: 0.0 for name in METRIC_NAMES}
    return {
        "verdict": verdict,
        "metrics": [
            {
                "metric": name,
                "published": 1.0,
                "reproduced": 1.0 + deltas[name],
                "signed_delta": deltas[name],
                "absolute_delta": abs(deltas[name]),
                "status": verdict,
            }
            for name in METRIC_NAMES
        ],
    }


def row(positive_bin, negative_bin, primary, secondary, deltas=None, incomplete=False):
    entry = {
        "positive_bin": positive_bin,
        "negative_bin": negative_bin,
        "incomplete": incomplete,
        "incomplete_reason": "cache miss" if incomplete else None,
        "configurations": {},
    }
    if not incomplete:
        entry["configurations"] = {
            PRIMARY: configuration_block(primary, deltas),
            SECONDARY: configuration_block(secondary),
        }
    return entry


class TestReleasedHistograms(unittest.TestCase):
    def test_released_histograms_are_the_recorded_ones(self):
        self.assertEqual(
            list(RELEASED_POSITIVE_HISTOGRAM), [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
        )
        self.assertEqual(
            list(RELEASED_NEGATIVE_HISTOGRAM), [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]
        )

    def test_both_encode_199_observed_examples(self):
        # Laplace smoothing adds one per bin, so raw = sum - 10.
        self.assertEqual(raw_count(RELEASED_POSITIVE_HISTOGRAM), 199)
        self.assertEqual(raw_count(RELEASED_NEGATIVE_HISTOGRAM), 199)
        self.assertEqual(RELEASED_EXAMPLES_PER_CLASS, 199)

    def test_the_paper_protocol_is_one_example_larger(self):
        self.assertEqual(PAPER_EXAMPLES_PER_CLASS, 200)
        self.assertEqual(MISSING_EXAMPLES_PER_CLASS, 1)


class TestAddOneToBin(unittest.TestCase):
    def test_exactly_one_count_is_added_to_the_named_bin(self):
        updated = add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, 3)
        self.assertEqual(updated[3], RELEASED_POSITIVE_HISTOGRAM[3] + 1)
        for index in range(N_BINS):
            if index != 3:
                self.assertEqual(updated[index], RELEASED_POSITIVE_HISTOGRAM[index])

    def test_total_rises_by_exactly_one(self):
        for bin_index in range(N_BINS):
            updated = add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, bin_index)
            self.assertEqual(sum(updated), sum(RELEASED_POSITIVE_HISTOGRAM) + 1)

    def test_the_result_encodes_the_paper_count(self):
        for bin_index in range(N_BINS):
            updated = add_one_to_bin(RELEASED_NEGATIVE_HISTOGRAM, bin_index)
            self.assertEqual(raw_count(updated), PAPER_EXAMPLES_PER_CLASS)

    def test_the_released_histogram_is_never_mutated(self):
        before = list(RELEASED_POSITIVE_HISTOGRAM)
        add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, 0)
        add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, 9)
        self.assertEqual(list(RELEASED_POSITIVE_HISTOGRAM), before)

    def test_a_mutable_input_is_also_left_alone(self):
        # The released constants are tuples, so a copy-vs-mutate bug would hide
        # there. A list input catches it.
        original = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
        updated = add_one_to_bin(original, 4)
        self.assertEqual(original, [1, 78, 19, 13, 6, 5, 28, 57, 1, 1])
        self.assertEqual(updated[4], 7)
        self.assertIsNot(updated, original)

    def test_length_is_preserved(self):
        self.assertEqual(len(add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, 5)), N_BINS)

    def test_out_of_range_bins_are_rejected(self):
        for bad in (-1, N_BINS, 99):
            with self.assertRaises(IndexError):
                add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, bad)


class TestCombinationGrid(unittest.TestCase):
    def test_the_grid_is_ten_by_ten(self):
        combinations = enumerate_combinations()
        self.assertEqual(len(combinations), 100)
        self.assertEqual(len(set(combinations)), 100)

    def test_every_bin_pair_appears_once(self):
        combinations = set(enumerate_combinations())
        for positive in range(N_BINS):
            for negative in range(N_BINS):
                self.assertIn((positive, negative), combinations)

    def test_combination_histograms_move_one_bin_on_each_side(self):
        positive, negative = combination_histograms(2, 7)
        self.assertEqual(positive[2], RELEASED_POSITIVE_HISTOGRAM[2] + 1)
        self.assertEqual(negative[7], RELEASED_NEGATIVE_HISTOGRAM[7] + 1)
        self.assertEqual(raw_count(positive), 200)
        self.assertEqual(raw_count(negative), 200)

    def test_the_two_sides_are_independent(self):
        positive, negative = combination_histograms(0, 9)
        self.assertEqual(positive[0], RELEASED_POSITIVE_HISTOGRAM[0] + 1)
        self.assertEqual(negative[0], RELEASED_NEGATIVE_HISTOGRAM[0])
        self.assertEqual(negative[9], RELEASED_NEGATIVE_HISTOGRAM[9] + 1)


class TestSummary(unittest.TestCase):
    def test_counts_each_verdict_category(self):
        rows = [
            row(0, 0, STATUS_PASS, STATUS_PASS),
            row(0, 1, STATUS_PASS, STATUS_FAIL),
            row(0, 2, STATUS_WARN, STATUS_PASS),
            row(0, 3, STATUS_FAIL, STATUS_PASS),
        ]
        summary = summarize(rows, primary=PRIMARY, secondary=SECONDARY)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 2)
        self.assertEqual(summary[f"{PRIMARY}_warn_not_fail"], 1)
        self.assertEqual(summary[f"{PRIMARY}_fail"], 1)
        self.assertEqual(summary[f"{SECONDARY}_pass"], 3)
        self.assertEqual(summary["both_configurations_pass"], 1)

    def test_warn_count_excludes_fail(self):
        rows = [row(0, 0, STATUS_FAIL, STATUS_FAIL)]
        summary = summarize(rows)
        self.assertEqual(summary[f"{PRIMARY}_warn_not_fail"], 0)

    def test_incomplete_combinations_are_never_counted_as_pass_or_fail(self):
        rows = [
            row(1, 1, STATUS_PASS, STATUS_PASS),
            row(2, 2, None, None, incomplete=True),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["combinations_evaluated"], 2)
        self.assertEqual(summary["combinations_complete"], 1)
        self.assertEqual(summary["combinations_incomplete"], 1)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 1)
        self.assertEqual(summary[f"{PRIMARY}_fail"], 0)
        self.assertEqual(summary["incomplete_combinations"][0]["positive_bin"], 2)

    def test_passing_bins_are_reported(self):
        rows = [row(4, 6, STATUS_PASS, STATUS_PASS), row(1, 2, STATUS_FAIL, STATUS_FAIL)]
        summary = summarize(rows)
        self.assertEqual(summary["bins_that_make_primary_pass"], [(4, 6)])
        self.assertEqual(summary["bins_where_both_pass"], [(4, 6)])

    def test_both_pass_requires_both_configurations(self):
        rows = [row(3, 3, STATUS_PASS, STATUS_WARN)]
        summary = summarize(rows)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 1)
        self.assertEqual(summary["both_configurations_pass"], 0)

    def test_empty_input_is_handled(self):
        summary = summarize([])
        self.assertEqual(summary["combinations_evaluated"], 0)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 0)


class TestClosestCombinations(unittest.TestCase):
    def near(self, positive_bin, negative_bin, size):
        return row(
            positive_bin,
            negative_bin,
            STATUS_FAIL,
            STATUS_FAIL,
            deltas={name: size for name in METRIC_NAMES},
        )

    def test_ranked_by_summed_absolute_delta(self):
        rows = [self.near(0, 0, 0.5), self.near(1, 1, 0.1), self.near(2, 2, 0.3)]
        closest = closest_combinations(rows, configuration=PRIMARY, limit=3)
        self.assertEqual(
            [(c["positive_bin"], c["negative_bin"]) for c in closest],
            [(1, 1), (2, 2), (0, 0)],
        )

    def test_limit_is_respected(self):
        rows = [self.near(i, i, i / 10.0) for i in range(N_BINS)]
        self.assertEqual(len(closest_combinations(rows, limit=3)), 3)

    def test_incomplete_rows_are_excluded(self):
        rows = [self.near(0, 0, 0.5), row(1, 1, None, None, incomplete=True)]
        closest = closest_combinations(rows)
        self.assertEqual(len(closest), 1)
        self.assertEqual(closest[0]["positive_bin"], 0)

    def test_distance_sums_the_six_metrics(self):
        self.assertAlmostEqual(
            metric_distance(self.near(0, 0, 0.25), PRIMARY), 6 * 0.25, places=12
        )

    def test_distance_is_none_for_an_unknown_configuration(self):
        self.assertIsNone(metric_distance(self.near(0, 0, 0.1), "NOT_A_CONFIG"))


class TestInterpretation(unittest.TestCase):
    def test_zero_passes_says_the_missing_examples_cannot_explain_the_gap(self):
        summary = summarize([row(0, 0, STATUS_FAIL, STATUS_FAIL)])
        reading = interpretation(summary, primary=PRIMARY)
        self.assertEqual(reading["headline"], "MISSING_EXAMPLES_CANNOT_EXPLAIN_THE_GAP")
        self.assertIn("cannot explain", reading["message"])
        self.assertIn("another source", reading["message"])

    def test_some_passes_says_only_that_it_is_plausible(self):
        summary = summarize([row(0, 0, STATUS_PASS, STATUS_PASS)])
        reading = interpretation(summary, primary=PRIMARY)
        self.assertEqual(
            reading["headline"], "MISSING_EXAMPLES_ARE_A_PLAUSIBLE_EXPLANATION"
        )
        self.assertIn("PLAUSIBLE", reading["message"])
        self.assertIn("does NOT identify their true bins", reading["message"])

    def test_a_positive_result_never_licenses_adopting_a_histogram(self):
        summary = summarize([row(0, 0, STATUS_PASS, STATUS_PASS)])
        reading = interpretation(summary, primary=PRIMARY)
        self.assertIn("released 199+199 data remain", reading["message"])
        self.assertTrue(reading["released_data_remain_the_baseline"])
        self.assertTrue(reading["does_not_identify_true_bins"])

    def test_no_complete_combinations_is_inconclusive_not_a_conclusion(self):
        # An empty sample must not yield "cannot explain the gap".
        summary = summarize([row(0, 0, None, None, incomplete=True)])
        reading = interpretation(summary, primary=PRIMARY)
        self.assertEqual(reading["headline"], "INCONCLUSIVE_INSUFFICIENT_CACHE_COVERAGE")
        self.assertIn("Nothing has been established", reading["message"])
        self.assertNotIn("cannot explain", reading["message"])

    def test_partial_coverage_is_caveated(self):
        rows = [
            row(0, 0, STATUS_FAIL, STATUS_FAIL),
            row(0, 1, None, None, incomplete=True),
        ]
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertEqual(reading["headline"], "MISSING_EXAMPLES_CANNOT_EXPLAIN_THE_GAP")
        self.assertIn("could not be evaluated", reading["message"])

    def test_every_reading_is_flagged_as_sensitivity_only(self):
        for rows in (
            [row(0, 0, STATUS_PASS, STATUS_PASS)],
            [row(0, 0, STATUS_FAIL, STATUS_FAIL)],
            [row(0, 0, None, None, incomplete=True)],
        ):
            reading = interpretation(summarize(rows), primary=PRIMARY)
            self.assertTrue(reading["is_sensitivity_analysis_only"])


if __name__ == "__main__":
    unittest.main()
