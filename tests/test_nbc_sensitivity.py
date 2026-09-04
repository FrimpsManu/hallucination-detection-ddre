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
    HEADLINE_CANNOT_EXPLAIN,
    HEADLINE_NO_COVERAGE,
    HEADLINE_PARTIAL_COVERAGE,
    HEADLINE_PLAUSIBLE,
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
    is_complete,
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


def config_block(verdict, deltas=None, complete=True):
    """A minimal stand-in for evaluate_configuration() plus the complete flag."""
    if not complete:
        return {
            "complete": False,
            "incomplete_reason": "required document score absent from the cache",
            "verdict": None,
            "metrics": [],
        }
    deltas = deltas or {name: 0.0 for name in METRIC_NAMES}
    return {
        "complete": True,
        "incomplete_reason": None,
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


def row(
    positive_bin,
    negative_bin,
    primary=STATUS_FAIL,
    secondary=STATUS_FAIL,
    deltas=None,
    primary_complete=True,
    secondary_complete=True,
):
    """One combination, with each configuration's completeness set separately."""
    return {
        "positive_bin": positive_bin,
        "negative_bin": negative_bin,
        "configurations": {
            PRIMARY: config_block(primary, deltas, complete=primary_complete),
            SECONDARY: config_block(secondary, complete=secondary_complete),
        },
    }


def full_grid(primary=STATUS_FAIL, secondary=STATUS_FAIL):
    """All 100 placements, every configuration complete."""
    return [
        row(p, n, primary, secondary)
        for p in range(N_BINS)
        for n in range(N_BINS)
    ]


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
            row(2, 2, primary_complete=False, secondary_complete=False),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["combinations_evaluated"], 2)
        self.assertEqual(summary[f"{PRIMARY}_complete"], 1)
        self.assertEqual(summary[f"{PRIMARY}_incomplete"], 1)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 1)
        self.assertEqual(summary[f"{PRIMARY}_fail"], 0)
        self.assertEqual(
            summary["incomplete_combinations"][PRIMARY][0]["positive_bin"], 2
        )

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
        rows = [
            self.near(0, 0, 0.5),
            row(1, 1, primary_complete=False, secondary_complete=False),
        ]
        closest = closest_combinations(rows)
        self.assertEqual(len(closest), 1)
        self.assertEqual(closest[0]["positive_bin"], 0)

    def test_distance_sums_the_six_metrics(self):
        self.assertAlmostEqual(
            metric_distance(self.near(0, 0, 0.25), PRIMARY), 6 * 0.25, places=12
        )

    def test_distance_is_none_for_an_unknown_configuration(self):
        self.assertIsNone(metric_distance(self.near(0, 0, 0.1), "NOT_A_CONFIG"))


class TestPerConfigurationCompleteness(unittest.TestCase):
    """A completed primary result must survive a secondary cache miss."""

    def test_primary_pass_is_counted_when_the_secondary_is_incomplete(self):
        rows = [row(3, 4, STATUS_PASS, None, secondary_complete=False)]
        summary = summarize(rows, expected_combinations=1)
        self.assertEqual(summary[f"{PRIMARY}_complete"], 1)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 1)
        self.assertEqual(summary[f"{SECONDARY}_complete"], 0)
        self.assertEqual(summary[f"{SECONDARY}_incomplete"], 1)
        self.assertEqual(summary["bins_that_make_primary_pass"], [(3, 4)])

    def test_secondary_pass_is_counted_when_the_primary_is_incomplete(self):
        rows = [row(1, 1, None, STATUS_PASS, primary_complete=False)]
        summary = summarize(rows, expected_combinations=1)
        self.assertEqual(summary[f"{SECONDARY}_pass"], 1)
        self.assertEqual(summary[f"{PRIMARY}_complete"], 0)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 0)

    def test_both_pass_requires_both_to_complete(self):
        rows = [row(2, 2, STATUS_PASS, None, secondary_complete=False)]
        summary = summarize(rows, expected_combinations=1)
        self.assertEqual(summary[f"{PRIMARY}_pass"], 1)
        self.assertEqual(summary["both_configurations_pass"], 0)
        self.assertEqual(summary["both_configurations_complete"], 0)

    def test_both_pass_requires_both_to_pass(self):
        rows = [row(2, 2, STATUS_PASS, STATUS_WARN)]
        summary = summarize(rows, expected_combinations=1)
        self.assertEqual(summary["both_configurations_complete"], 1)
        self.assertEqual(summary["both_configurations_pass"], 0)

    def test_completion_counts_are_reported_per_configuration(self):
        rows = [
            row(0, 0, STATUS_PASS, STATUS_PASS),
            row(0, 1, STATUS_FAIL, None, secondary_complete=False),
            row(0, 2, None, STATUS_PASS, primary_complete=False),
        ]
        summary = summarize(rows, expected_combinations=3)
        self.assertEqual(summary[f"{PRIMARY}_complete"], 2)
        self.assertEqual(summary[f"{PRIMARY}_incomplete"], 1)
        self.assertEqual(summary[f"{SECONDARY}_complete"], 2)
        self.assertEqual(summary[f"{SECONDARY}_incomplete"], 1)

    def test_incomplete_lists_are_keyed_by_configuration(self):
        rows = [row(5, 6, STATUS_PASS, None, secondary_complete=False)]
        summary = summarize(rows, expected_combinations=1)
        self.assertEqual(summary["incomplete_combinations"][PRIMARY], [])
        self.assertEqual(
            summary["incomplete_combinations"][SECONDARY][0]["negative_bin"], 6
        )

    def test_is_complete_reads_the_per_configuration_flag(self):
        entry = row(0, 0, STATUS_PASS, None, secondary_complete=False)
        self.assertTrue(is_complete(entry, PRIMARY))
        self.assertFalse(is_complete(entry, SECONDARY))

    def test_an_incomplete_configuration_has_no_verdict(self):
        entry = row(0, 0, STATUS_PASS, None, secondary_complete=False)
        self.assertIsNone(entry["configurations"][SECONDARY]["verdict"])
        self.assertIsNotNone(
            entry["configurations"][SECONDARY]["incomplete_reason"]
        )


class TestInterpretation(unittest.TestCase):
    """The four predeclared branches for the primary CM=14/CFA=24 question."""

    def test_a_all_primary_complete_with_a_pass_is_plausible(self):
        rows = full_grid(STATUS_FAIL, STATUS_FAIL)
        rows[17] = row(1, 7, STATUS_PASS, STATUS_PASS)
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertEqual(reading["headline"], HEADLINE_PLAUSIBLE)
        self.assertIn("PLAUSIBLE", reading["message"])
        self.assertIn("does NOT identify their true bins", reading["message"])

    def test_a_one_pass_under_partial_coverage_is_still_plausible(self):
        # A single reproducing placement is sufficient; unevaluated placements
        # cannot take that away.
        rows = full_grid(STATUS_FAIL, STATUS_FAIL)
        rows[0] = row(0, 0, STATUS_PASS, STATUS_PASS)
        for index in range(1, 40):
            rows[index] = row(
                rows[index]["positive_bin"],
                rows[index]["negative_bin"],
                primary_complete=False,
            )
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertEqual(reading["headline"], HEADLINE_PLAUSIBLE)
        self.assertIn("Coverage was partial", reading["message"])
        self.assertIn("does not weaken this finding", reading["message"])

    def test_b_zero_passes_with_partial_coverage_is_inconclusive(self):
        # The review's key case: an unevaluated placement could still pass, so
        # "cannot explain" is not supported.
        rows = full_grid(STATUS_FAIL, STATUS_FAIL)
        rows[5] = row(0, 5, primary_complete=False)
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertNotEqual(reading["headline"], HEADLINE_CANNOT_EXPLAIN)
        self.assertEqual(reading["headline"], HEADLINE_PARTIAL_COVERAGE)
        # The phrase appears only inside an explicit negation.
        self.assertIn(
            "does NOT establish that the missing 200th examples cannot explain",
            reading["message"],
        )
        self.assertIn("could still pass", reading["message"])

    def test_c_full_grid_complete_with_zero_passes_cannot_explain(self):
        reading = interpretation(summarize(full_grid()), primary=PRIMARY)
        self.assertEqual(reading["headline"], HEADLINE_CANNOT_EXPLAIN)
        self.assertIn("All 100", reading["message"])
        self.assertIn("cannot explain", reading["message"])
        self.assertIn("another source", reading["message"])

    def test_d_zero_primary_complete_is_insufficient_coverage(self):
        rows = [
            row(p, n, primary_complete=False)
            for p in range(N_BINS)
            for n in range(N_BINS)
        ]
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertEqual(reading["headline"], HEADLINE_NO_COVERAGE)
        self.assertIn("Nothing has been established", reading["message"])
        self.assertNotIn("cannot explain", reading["message"])

    def test_a_truncated_grid_can_never_conclude_cannot_explain(self):
        # --limit-combinations must not produce a negative conclusion.
        rows = [row(0, n, STATUS_FAIL, STATUS_FAIL) for n in range(4)]
        summary = summarize(rows)
        self.assertFalse(summary["grid_fully_enumerated"])
        reading = interpretation(summary, primary=PRIMARY)
        self.assertEqual(reading["headline"], HEADLINE_PARTIAL_COVERAGE)

    def test_secondary_incompleteness_does_not_affect_the_primary_reading(self):
        rows = full_grid(STATUS_FAIL, STATUS_FAIL)
        rows = [
            row(r["positive_bin"], r["negative_bin"], STATUS_FAIL, secondary_complete=False)
            for r in rows
        ]
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertEqual(reading["headline"], HEADLINE_CANNOT_EXPLAIN)

    def test_a_positive_result_never_licenses_adopting_a_histogram(self):
        rows = full_grid()
        rows[0] = row(0, 0, STATUS_PASS, STATUS_PASS)
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertIn("released 199+199 data remain", reading["message"])
        self.assertTrue(reading["released_data_remain_the_baseline"])
        self.assertTrue(reading["does_not_identify_true_bins"])

    def test_every_reading_is_flagged_as_sensitivity_only(self):
        cases = [
            full_grid(),
            [row(p, n, primary_complete=False) for p in range(N_BINS) for n in range(N_BINS)],
            [row(0, 0, STATUS_PASS, STATUS_PASS)],
        ]
        for rows in cases:
            reading = interpretation(summarize(rows), primary=PRIMARY)
            self.assertTrue(reading["is_sensitivity_analysis_only"])
            self.assertTrue(reading["released_data_remain_the_baseline"])

    def test_reading_reports_the_primary_completion_counts(self):
        rows = full_grid()
        rows[3] = row(0, 3, primary_complete=False)
        reading = interpretation(summarize(rows), primary=PRIMARY)
        self.assertEqual(reading["primary_complete"], 99)
        self.assertEqual(reading["primary_incomplete"], 1)


if __name__ == "__main__":
    unittest.main()
