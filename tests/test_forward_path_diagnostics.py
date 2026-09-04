"""Tests for the Step 2 forward-path isolation logic.

Synthetic scores only. No torch, no transformers, no model download, so CI --
which installs numpy alone -- runs the whole file.

What these pin:

* the two findings stay separate, and a raw-score change alone never sets
  ``bse_decision_impact``;
* all five predeclared isolation rules, each with a fixture that satisfies only
  that rule;
* the determinism control gates every attribution;
* the reference observation from Step 1 is reproduced exactly by this module's
  own rounding and bucketing, so a future change to either cannot silently
  invalidate the comparison against the recorded numbers.
"""

import unittest

from src.forward_path_diagnostics import (
    ARM_SPECS,
    DETERMINISM_CONTROL,
    DIFFERENCE_IDENTICAL,
    DIFFERENCE_NEGLIGIBLE,
    DIFFERENCE_SUBSTANTIAL,
    ENTAILMENT_INDEX,
    NEGLIGIBLE_MAX_ABS_DELTA,
    NO,
    REFERENCE_OBSERVATION,
    SOFTMAX_TEMPERATURE,
    YES,
    arm_by_name,
    assess_reference_reproduction,
    build_verdict,
    classify_difference,
    compare_forward_arms,
    entailment_score_from_probabilities,
    flat_index,
    histograms_by_polarity,
    interpret_factor_isolation,
    isolated_factor,
    probe_report,
    score_report,
)
from src.scoring_diagnostics import delta_stats, nbc_bucket, round_one_decimal


# 199 positives then 199 negatives, matching the released NBC files.
N_POSITIVE = 199
N_NEGATIVE = 199
POLARITIES = ["positive"] * N_POSITIVE + ["negative"] * N_NEGATIVE

# The Step 1 divergence: 19.9462890625 rounds to 19.9 (bucket 1) while
# 19.953125 rounds to 20.0 (bucket 2), a 6.8e-3 raw difference.
WANG_169 = REFERENCE_OBSERVATION["literal_wang"]["raw"]
REPO_169 = REFERENCE_OBSERVATION["repository"]["raw"]

# A perturbation far below the negligible bound: visible in floating point,
# invisible to every discretizer.
TINY = 1e-9


def base_scores():
    """A flat score vector that puts every pair in a stable bucket interior."""
    return [45.0] * (N_POSITIVE + N_NEGATIVE)


def with_probe(value, scores=None):
    """Set positive pair 169 to a given raw score."""
    out = list(scores) if scores is not None else base_scores()
    out[flat_index("positive", 169, POLARITIES)] = value
    return out


def perturb(scores, amount):
    return [s + amount for s in scores]


def compare(left, right, left_name="C0", right_name="C1"):
    return compare_forward_arms(left_name, right_name, left, right, POLARITIES)


def matrix_from(c0, c1, c2, c3):
    """Build the comparison matrix the isolation rules consume."""
    return {
        "C0_vs_C1": compare(c0, c1, "C0", "C1"),
        "C0_vs_C2": compare(c0, c2, "C0", "C2"),
        "C0_vs_C3": compare(c0, c3, "C0", "C3"),
        "C1_vs_C3": compare(c1, c3, "C1", "C3"),
        "C2_vs_C3": compare(c2, c3, "C2", "C3"),
        "C1_vs_C2": compare(c1, c2, "C1", "C2"),
    }


class TestReferenceObservation(unittest.TestCase):
    """The recorded Step 1 numbers must reproduce under this module's rules."""

    def test_literal_wang_score_rounds_and_buckets_as_recorded(self):
        self.assertEqual(round_one_decimal(WANG_169), 19.9)
        self.assertEqual(nbc_bucket(WANG_169), 1)
        self.assertEqual(
            score_report(WANG_169)["nbc_bucket"],
            REFERENCE_OBSERVATION["literal_wang"]["nbc_bucket"],
        )

    def test_repository_score_rounds_and_buckets_as_recorded(self):
        self.assertEqual(round_one_decimal(REPO_169), 20.0)
        self.assertEqual(nbc_bucket(REPO_169), 2)
        self.assertEqual(
            score_report(REPO_169)["nbc_bucket"],
            REFERENCE_OBSERVATION["repository"]["nbc_bucket"],
        )

    def test_recorded_delta_is_far_above_the_negligible_bound(self):
        # The bound must sit well under the effect it is used to attribute.
        self.assertGreater(REFERENCE_OBSERVATION["raw_delta"], 0)
        self.assertGreater(
            REFERENCE_OBSERVATION["raw_delta"], 50 * NEGLIGIBLE_MAX_ABS_DELTA
        )

    def test_recorded_histograms_differ_by_one_pair_moving_bucket_1_to_2(self):
        wang = REFERENCE_OBSERVATION["positive_histogram_literal_wang"]
        repository = REFERENCE_OBSERVATION["positive_histogram_repository"]
        self.assertEqual(repository[1], wang[1] - 1)
        self.assertEqual(repository[2], wang[2] + 1)
        self.assertEqual(sum(wang), sum(repository))


class TestArmSpecs(unittest.TestCase):
    def test_the_four_factorial_arms_cover_both_factors(self):
        combinations = {
            (spec["attention_mask"], spec["inference_mode"])
            for spec in ARM_SPECS
            if spec["role"] in ("reference", "factor")
        }
        self.assertEqual(
            combinations, {(False, False), (False, True), (True, False), (True, True)}
        )

    def test_c0_is_the_literal_wang_reference(self):
        c0 = arm_by_name("C0")
        self.assertFalse(c0["attention_mask"])
        self.assertFalse(c0["token_type_ids"])
        self.assertFalse(c0["inference_mode"])
        self.assertEqual(c0["role"], "reference")

    def test_c4_is_labelled_confirmation_only(self):
        c4 = arm_by_name("C4")
        self.assertEqual(c4["role"], "confirmation")
        self.assertIn("CONFIRMATION ONLY", c4["note"])
        self.assertIn("type_vocab_size is 0", c4["note"])

    def test_control_repeats_c0_exactly(self):
        c0 = arm_by_name("C0")
        for key in ("attention_mask", "token_type_ids", "inference_mode"):
            self.assertEqual(DETERMINISM_CONTROL[key], c0[key])
        self.assertEqual(DETERMINISM_CONTROL["role"], "control")

    def test_unknown_arm_raises(self):
        with self.assertRaises(KeyError):
            arm_by_name("C9")

    def test_wang_scoring_constants(self):
        self.assertEqual(ENTAILMENT_INDEX, 0)
        self.assertEqual(SOFTMAX_TEMPERATURE, 5.0)

    def test_score_extraction_scales_the_entailment_class_to_0_100(self):
        self.assertAlmostEqual(
            entailment_score_from_probabilities([0.199462890625, 0.5, 0.3]),
            19.9462890625,
            places=10,
        )


class TestDifferenceClassification(unittest.TestCase):
    def test_bit_identical_is_identical(self):
        scores = base_scores()
        self.assertEqual(
            classify_difference(delta_stats(scores, scores)), DIFFERENCE_IDENTICAL
        )

    def test_tiny_perturbation_is_negligible(self):
        scores = base_scores()
        self.assertEqual(
            classify_difference(delta_stats(scores, perturb(scores, TINY))),
            DIFFERENCE_NEGLIGIBLE,
        )

    def test_perturbation_above_the_bound_is_substantial(self):
        scores = base_scores()
        moved = perturb(scores, 10 * NEGLIGIBLE_MAX_ABS_DELTA)
        self.assertEqual(
            classify_difference(delta_stats(scores, moved)), DIFFERENCE_SUBSTANTIAL
        )

    def test_the_bound_itself_is_still_negligible(self):
        # Fed directly rather than via a float addition: 45.0 + 1e-4 does not
        # differ from 45.0 by exactly 1e-4, and this test is about the
        # comparison operator at the boundary, not about float arithmetic.
        self.assertEqual(
            classify_difference({"max_absolute": NEGLIGIBLE_MAX_ABS_DELTA}),
            DIFFERENCE_NEGLIGIBLE,
        )

    def test_just_above_the_bound_is_substantial(self):
        self.assertEqual(
            classify_difference({"max_absolute": NEGLIGIBLE_MAX_ABS_DELTA * 1.001}),
            DIFFERENCE_SUBSTANTIAL,
        )

    def test_missing_max_absolute_is_treated_as_identical(self):
        self.assertEqual(classify_difference({"max_absolute": None}), DIFFERENCE_IDENTICAL)


class TestSeparateReporting(unittest.TestCase):
    """The two findings must never collapse into one word."""

    def test_identical_arms_report_no_on_both(self):
        scores = base_scores()
        block = compare(scores, scores)
        self.assertEqual(block["numerical_difference"], NO)
        self.assertEqual(block["bse_decision_impact"], NO)

    def test_raw_change_without_a_bucket_change_is_not_decision_impact(self):
        # The central discipline: floating point moved, BSE did not.
        scores = base_scores()
        block = compare(scores, perturb(scores, TINY))
        self.assertEqual(block["numerical_difference"], YES)
        self.assertEqual(block["bse_decision_impact"], NO)
        self.assertIn("changes no NBC bucket", block["causal_candidate"])

    def test_a_large_raw_change_inside_one_bucket_is_still_not_decision_impact(self):
        # 41.0 and 48.0 are both bucket 4. A 7-point raw move changes nothing.
        block = compare([41.0] * len(POLARITIES), [48.0] * len(POLARITIES))
        self.assertEqual(block["numerical_difference"], YES)
        self.assertEqual(block["bse_decision_impact"], NO)

    def test_bucket_change_sets_decision_impact(self):
        block = compare(with_probe(WANG_169), with_probe(REPO_169))
        self.assertEqual(block["numerical_difference"], YES)
        self.assertEqual(block["bse_decision_impact"], YES)
        self.assertEqual(block["nbc_bucket_disagreements"]["count"], 1)

    def test_no_verdict_field_uses_a_single_conflated_word(self):
        block = compare(with_probe(WANG_169), with_probe(REPO_169))
        self.assertNotIn("MATERIAL", block["numerical_difference"])
        self.assertNotIn("MATERIAL", block["bse_decision_impact"])
        self.assertIn("numerical_difference", block)
        self.assertIn("bse_decision_impact", block)
        self.assertIn("causal_candidate", block)

    def test_comparison_records_the_factor_actually_toggled(self):
        scores = base_scores()
        self.assertEqual(compare(scores, scores, "C0", "C1")["isolated_factor"],
                         "torch.inference_mode()")
        self.assertEqual(compare(scores, scores, "C0", "C2")["isolated_factor"],
                         "attention_mask")

    def test_histograms_are_reported_for_both_sides(self):
        block = compare(with_probe(WANG_169), with_probe(REPO_169))
        left = block["histograms"]["left"]["positive"]
        right = block["histograms"]["right"]["positive"]
        self.assertEqual(right[1], left[1] - 1)
        self.assertEqual(right[2], left[2] + 1)


class TestIsolatedFactorDerivation(unittest.TestCase):
    """The factor is derived from what differs, not from the right arm's label.

    C1_vs_C3 and C2_vs_C3 are the comparisons interpret_factor_isolation() uses
    to tell a dominant factor from an interaction. If either were labelled with
    C3's "both factors" description, the JSON would misdescribe the experiment.
    """

    def test_c0_baselines_toggle_one_factor_each(self):
        self.assertEqual(isolated_factor("C0", "C1"), "torch.inference_mode()")
        self.assertEqual(isolated_factor("C0", "C2"), "attention_mask")

    def test_c0_vs_c3_toggles_both(self):
        self.assertEqual(
            isolated_factor("C0", "C3"), "attention_mask + torch.inference_mode()"
        )

    def test_c1_vs_c3_toggles_attention_mask_alone(self):
        # Both arms already run under inference_mode.
        self.assertEqual(isolated_factor("C1", "C3"), "attention_mask")

    def test_c2_vs_c3_toggles_inference_mode_alone(self):
        # Both arms already pass an attention mask.
        self.assertEqual(isolated_factor("C2", "C3"), "torch.inference_mode()")

    def test_c3_vs_c4_toggles_token_type_ids(self):
        self.assertEqual(isolated_factor("C3", "C4"), "token_type_ids")

    def test_control_toggles_nothing(self):
        self.assertIsNone(isolated_factor("C0", "C0R"))

    def test_control_comparison_is_described_as_a_control(self):
        scores = base_scores()
        block = compare(scores, scores, "C0", "C0R")
        self.assertIsNone(block["isolated_factor"])
        self.assertIn("determinism control", block["causal_candidate"])

    def test_derivation_is_symmetric(self):
        for left, right in (("C0", "C1"), ("C1", "C3"), ("C2", "C3")):
            self.assertEqual(
                isolated_factor(left, right), isolated_factor(right, left)
            )


class TestHistograms(unittest.TestCase):
    def test_polarities_are_split_correctly(self):
        scores = [5.0] * N_POSITIVE + [95.0] * N_NEGATIVE
        histograms = histograms_by_polarity(scores, POLARITIES)
        self.assertEqual(histograms["positive"][0], N_POSITIVE + 1)
        self.assertEqual(histograms["negative"][9], N_NEGATIVE + 1)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            histograms_by_polarity([1.0], POLARITIES)

    def test_flat_index_finds_the_nth_pair_of_a_polarity(self):
        self.assertEqual(flat_index("positive", 169, POLARITIES), 169)
        self.assertEqual(flat_index("negative", 0, POLARITIES), N_POSITIVE)

    def test_flat_index_raises_when_out_of_range(self):
        with self.assertRaises(IndexError):
            flat_index("positive", 10_000, POLARITIES)


class TestIsolationRules(unittest.TestCase):
    """Each of the five predeclared rules, with a fixture matching only it."""

    def test_no_effect_anywhere_is_inconclusive(self):
        scores = base_scores()
        result = interpret_factor_isolation(
            matrix_from(scores, scores, scores, scores)
        )
        self.assertEqual(result["headline"], "NO_FORWARD_PATH_EFFECT")
        self.assertIn("inconclusive", result["causal_candidate"])
        self.assertEqual(result["inference_mode_contributes"], NO)
        self.assertEqual(result["attention_mask_contributes"], NO)

    def test_attention_mask_primary(self):
        # attention_mask moves the score in both grad contexts; inference_mode
        # moves it in neither.
        c0 = base_scores()
        c1 = list(c0)
        c2 = perturb(c0, 0.01)
        c3 = list(c2)
        result = interpret_factor_isolation(matrix_from(c0, c1, c2, c3))
        self.assertEqual(result["headline"], "ATTENTION_MASK_PRIMARY")
        self.assertEqual(result["attention_mask_contributes"], YES)
        self.assertEqual(result["inference_mode_contributes"], NO)

    def test_inference_mode_primary(self):
        c0 = base_scores()
        c1 = perturb(c0, 0.01)
        c2 = list(c0)
        c3 = list(c1)
        result = interpret_factor_isolation(matrix_from(c0, c1, c2, c3))
        self.assertEqual(result["headline"], "INFERENCE_MODE_PRIMARY")
        self.assertEqual(result["inference_mode_contributes"], YES)
        self.assertEqual(result["attention_mask_contributes"], NO)

    def test_interaction_when_only_both_together_move_the_score(self):
        c0 = base_scores()
        c1 = list(c0)
        c2 = list(c0)
        c3 = perturb(c0, 0.01)
        result = interpret_factor_isolation(matrix_from(c0, c1, c2, c3))
        self.assertEqual(result["headline"], "FACTOR_INTERACTION")
        self.assertIn("interaction", result["causal_candidate"])

    def test_both_contribute_when_each_moves_the_score_independently(self):
        c0 = base_scores()
        c1 = perturb(c0, 0.01)
        c2 = perturb(c0, 0.02)
        c3 = perturb(c0, 0.05)
        result = interpret_factor_isolation(matrix_from(c0, c1, c2, c3))
        self.assertEqual(result["headline"], "BOTH_FACTORS_CONTRIBUTE")
        self.assertEqual(result["inference_mode_contributes"], YES)
        self.assertEqual(result["attention_mask_contributes"], YES)

    def test_negligible_differences_do_not_implicate_a_factor(self):
        # Floating-point noise below the bound must not be read as an effect.
        c0 = base_scores()
        c1 = perturb(c0, TINY)
        c2 = perturb(c0, TINY)
        c3 = perturb(c0, TINY)
        result = interpret_factor_isolation(matrix_from(c0, c1, c2, c3))
        self.assertEqual(result["headline"], "NO_FORWARD_PATH_EFFECT")


class TestReferenceReproduction(unittest.TestCase):
    def build(self, arm_scores, control_scores=None):
        scores_by_arm = dict(arm_scores)
        scores_by_arm["C0R"] = control_scores or list(arm_scores["C0"])
        probe = probe_report("positive", 169, scores_by_arm, POLARITIES)
        control = compare(
            scores_by_arm["C0"], scores_by_arm["C0R"], "C0", "C0R"
        )
        return probe, control

    def test_reproduced_when_an_arm_moves_the_probe_to_the_repository_bucket(self):
        wang = with_probe(WANG_169)
        repo = with_probe(REPO_169)
        probe, control = self.build({"C0": wang, "C1": wang, "C2": repo, "C3": repo})
        result = assess_reference_reproduction(probe, control)
        self.assertEqual(result["explains_reference_observation"], YES)
        self.assertEqual(result["reproduced_by_arms"], ["C2", "C3"])

    def test_not_reproduced_when_every_arm_stays_in_the_wang_bucket(self):
        wang = with_probe(WANG_169)
        probe, control = self.build({"C0": wang, "C1": wang, "C2": wang, "C3": wang})
        result = assess_reference_reproduction(probe, control)
        self.assertEqual(result["explains_reference_observation"], NO)
        self.assertIn("another difference remains", result["message"])

    def test_nondeterministic_device_blocks_any_attribution(self):
        # The control gate: if C0 will not reproduce itself, nothing else counts.
        wang = with_probe(WANG_169)
        repo = with_probe(REPO_169)
        probe, control = self.build(
            {"C0": wang, "C1": wang, "C2": repo, "C3": repo},
            control_scores=perturb(wang, 0.02),
        )
        result = assess_reference_reproduction(probe, control)
        self.assertEqual(result["explains_reference_observation"], "UNDETERMINED")
        self.assertFalse(result["determinism_control_passed"])
        self.assertIn("nondeterministic", result["message"])

    def test_c0_not_matching_wang_blocks_attribution(self):
        repo = with_probe(REPO_169)
        probe, control = self.build({"C0": repo, "C1": repo, "C2": repo, "C3": repo})
        result = assess_reference_reproduction(probe, control)
        self.assertEqual(result["explains_reference_observation"], "UNDETERMINED")
        self.assertFalse(result["c0_matches_literal_wang"])
        self.assertIn("not comparable to Step 1", result["message"])

    def test_the_control_arm_is_never_credited_with_reproducing(self):
        wang = with_probe(WANG_169)
        probe, control = self.build(
            {"C0": wang, "C1": wang, "C2": wang, "C3": wang},
            control_scores=wang,
        )
        result = assess_reference_reproduction(probe, control)
        self.assertNotIn("C0R", result["reproduced_by_arms"])

    def test_probe_reports_every_arm(self):
        wang = with_probe(WANG_169)
        probe, _ = self.build({"C0": wang, "C1": wang, "C2": wang, "C3": wang})
        self.assertEqual(set(probe["arms"]), {"C0", "C1", "C2", "C3", "C0R"})
        self.assertEqual(probe["arms"]["C0"]["nbc_bucket"], 1)
        self.assertEqual(probe["flat_index"], 169)


class TestBuildVerdict(unittest.TestCase):
    def build(self, c0, c1, c2, c3, control=None):
        matrix = matrix_from(c0, c1, c2, c3)
        scores_by_arm = {"C0": c0, "C1": c1, "C2": c2, "C3": c3, "C0R": control or list(c0)}
        probe = probe_report("positive", 169, scores_by_arm, POLARITIES)
        control_block = compare(c0, scores_by_arm["C0R"], "C0", "C0R")
        return build_verdict(matrix, probe, control_block)

    def test_clean_run_reports_no_on_both_findings(self):
        scores = with_probe(WANG_169)
        verdict = self.build(scores, scores, scores, scores)
        self.assertEqual(verdict["numerical_difference"], NO)
        self.assertEqual(verdict["bse_decision_impact"], NO)
        self.assertEqual(verdict["headline"], "NO_FORWARD_PATH_EFFECT")

    def test_numerical_difference_without_decision_impact_is_representable(self):
        c0 = with_probe(WANG_169)
        moved = perturb(c0, TINY)
        verdict = self.build(c0, moved, moved, moved)
        self.assertEqual(verdict["numerical_difference"], YES)
        self.assertEqual(verdict["bse_decision_impact"], NO)

    def test_decision_impact_lists_the_responsible_comparisons(self):
        c0 = with_probe(WANG_169)
        repo = with_probe(REPO_169)
        verdict = self.build(c0, c0, repo, repo)
        self.assertEqual(verdict["bse_decision_impact"], YES)
        # attention_mask is the toggled factor in every listed comparison, and
        # the one comparison that toggles only inference_mode is absent.
        self.assertEqual(
            verdict["comparisons_with_bse_decision_impact"],
            ["C0_vs_C2", "C0_vs_C3", "C1_vs_C2", "C1_vs_C3"],
        )
        self.assertNotIn("C0_vs_C1", verdict["comparisons_with_bse_decision_impact"])
        self.assertNotIn("C2_vs_C3", verdict["comparisons_with_bse_decision_impact"])

    def test_verdict_carries_the_reporting_discipline_note(self):
        scores = with_probe(WANG_169)
        verdict = self.build(scores, scores, scores, scores)
        self.assertIn("NOT evidence", verdict["reporting_note"])
        self.assertIn("causal_candidate", verdict)
        self.assertIn("reference_reproduction", verdict)


if __name__ == "__main__":
    unittest.main()
