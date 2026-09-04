"""Tests for the Step 2 primary decomposition: forward path vs extraction path.

The central claim under test is arithmetic, not empirical: on a half-precision
tensor, scaling by 100 *inside* the tensor rather than after leaving it moves
the Step 1 probe score from 19.9462890625 to 19.953125 -- exactly the recorded
A1 and A3 values. numpy.float16 is enough to demonstrate that, and numpy is
already a CI dependency, so no model download is needed.

This is diagnostic evidence about precision and order of operations. It is not
proof about the real model; the Colab run supplies that.
"""

import unittest

import numpy as np

from src.extraction_path_diagnostics import (
    CELL_BY_NAME,
    CELL_COMPARISONS,
    CELL_SPECS,
    EXTRACTION_SPECS,
    FACTOR_EXTRACTION,
    FACTOR_FORWARD,
    FORWARD_SPECS,
    GUARD_ENDPOINTS_NOT_RECONSTRUCTED,
    GUARD_NONDETERMINISTIC,
    HEADLINE_BOTH_CONTRIBUTE,
    HEADLINE_EXTRACTION_EXPLAINS,
    HEADLINE_EXTRACTION_IMPLICATED,
    HEADLINE_FORWARD_EXPLAINS,
    HEADLINE_NEITHER,
    HEADLINE_UNDETERMINED_ENDPOINTS,
    HEADLINE_UNDETERMINED_NONDETERMINISTIC,
    build_primary_verdict,
    cell_isolated_factor,
    compare_cells,
    histogram_movement_matches_step1,
    interpret_shape_and_scaling,
    probe_bucket_flip_reproduced,
    probe_cell_report,
    repository_extraction,
    score_pair_both_extractions,
    secondary_expectation,
    shape_and_scaling_probe,
    summarize_shape_and_scaling,
    wang_extraction,
)
from src.forward_path_diagnostics import (
    NO,
    REFERENCE_OBSERVATION,
    YES,
    histograms_by_polarity,
)
from src.scoring_diagnostics import nbc_bucket

N_POSITIVE = 199
N_NEGATIVE = 199
POLARITIES = ["positive"] * N_POSITIVE + ["negative"] * N_NEGATIVE

WANG_169 = REFERENCE_OBSERVATION["literal_wang"]["raw"]
REPO_169 = REFERENCE_OBSERVATION["repository"]["raw"]

# The underlying entailment probability implied by the recorded A1 score.
PROBE_PROBABILITY = 0.199462890625


# --------------------------------------------------------------------------
# The arithmetic itself.
# --------------------------------------------------------------------------

class TestHalfPrecisionScalingArithmetic(unittest.TestCase):
    """Precision and order of operations, demonstrated directly."""

    def test_the_probability_is_exactly_representable_in_float16(self):
        # If it were not, the reconstruction below would be approximate and the
        # exact match with the recorded values would be a coincidence.
        self.assertEqual(float(np.float16(PROBE_PROBABILITY)), PROBE_PROBABILITY)

    def test_converting_before_scaling_gives_the_recorded_a1_value(self):
        p = np.float16(PROBE_PROBABILITY)
        self.assertEqual(float(p) * 100, 19.9462890625)
        self.assertEqual(float(p) * 100, WANG_169)

    def test_scaling_inside_the_tensor_gives_the_recorded_a3_value(self):
        p = np.float16(PROBE_PROBABILITY)
        self.assertEqual(float(np.float16(p * np.float16(100.0))), 19.953125)
        self.assertEqual(float(np.float16(p * np.float16(100.0))), REPO_169)

    def test_the_gap_is_exactly_the_recorded_divergence(self):
        p = np.float16(PROBE_PROBABILITY)
        gap = float(np.float16(p * np.float16(100.0))) - float(p) * 100
        self.assertEqual(gap, 0.0068359375)
        self.assertEqual(gap, REFERENCE_OBSERVATION["raw_delta"])

    def test_float32_does_not_reproduce_the_divergence(self):
        # The mechanism requires half precision. In float32 both orders agree,
        # so this is evidence about the dtype the recorded run used.
        p = np.float32(PROBE_PROBABILITY)
        self.assertEqual(float(np.float32(p * np.float32(100.0))), float(p) * 100)

    def test_the_rounding_is_explained_by_float16_spacing(self):
        # float16 spacing in [16, 32) is 2^4 * 2^-10 = 0.015625, and
        # 19.9462890625 sits 0.5625 of a step above 19.9375, so it rounds up.
        spacing = 2.0**4 * 2.0**-10
        self.assertEqual(spacing, 0.015625)
        self.assertEqual(float(np.float16(19.9375)), 19.9375)
        self.assertEqual(float(np.float16(19.953125)), 19.953125)
        self.assertAlmostEqual((WANG_169 - 19.9375) / spacing, 0.5625, places=12)


# --------------------------------------------------------------------------
# The same arithmetic, through the real extraction functions.
# --------------------------------------------------------------------------

class HalfTensor:
    """A minimal float16 tensor standing in for torch, backed by numpy."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float16)

    def __truediv__(self, divisor):
        return HalfTensor(self.values / np.float16(divisor))

    def __mul__(self, factor):
        return HalfTensor(self.values * np.float16(factor))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return HalfTensor(self.values[key])
        return HalfTensor(self.values[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        # Mirrors torch: a 0-dim tensor yields a scalar, not a list.
        return self.values.tolist()


class HalfTorch:
    """Injected torch stand-in whose softmax returns a float16 tensor."""

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def softmax(self, tensor, dim):
        # The exact softmax is irrelevant here; what matters is that the result
        # is float16 and keeps the input's shape, as it does for a
        # half-precision model. Wang's path softmaxes a 1-D row; the repository
        # path softmaxes the 2-D batch and then indexes a column.
        values = np.asarray(tensor.values)
        if values.ndim == 1:
            return HalfTensor(self.probabilities)
        return HalfTensor([self.probabilities] * values.shape[0])

    def inference_mode(self):
        class Context:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Context()


PROBE_PROBABILITIES = [PROBE_PROBABILITY, 0.5, 0.300537109375]


class TestExtractionFunctionsOnHalfPrecision(unittest.TestCase):
    """The production code paths, not just the arithmetic in isolation."""

    def setUp(self):
        self.torch = HalfTorch(PROBE_PROBABILITIES)
        self.logits = HalfTensor([PROBE_PROBABILITIES])

    def test_wang_extraction_reproduces_step1_a1(self):
        self.assertEqual(wang_extraction(self.logits, self.torch), WANG_169)

    def test_repository_extraction_reproduces_step1_a3(self):
        self.assertEqual(repository_extraction(self.logits, self.torch), REPO_169)

    def test_the_two_extractions_differ_by_the_recorded_gap(self):
        gap = repository_extraction(self.logits, self.torch) - wang_extraction(
            self.logits, self.torch
        )
        self.assertEqual(gap, REFERENCE_OBSERVATION["raw_delta"])

    def test_one_forward_serves_both_extractions(self):
        # D00 and D01 must share bit-identical logits, so the extraction
        # comparison is exact by construction.
        calls = []

        class Model:
            def __call__(self, input_ids, **kwargs):
                calls.append(kwargs)
                return {"logits": HalfTensor([PROBE_PROBABILITIES])}

        scores = score_pair_both_extractions(
            Model(), {"input_ids": "IDS"}, FORWARD_SPECS["F0"], self.torch
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(scores["X0"], WANG_169)
        self.assertEqual(scores["X1"], REPO_169)

    def test_forward_specs_pass_their_own_arguments(self):
        seen = []

        class Model:
            def __call__(self, input_ids, **kwargs):
                seen.append(set(kwargs))
                return {"logits": HalfTensor([PROBE_PROBABILITIES])}

        encoded = {"input_ids": "IDS", "attention_mask": "M", "token_type_ids": "T"}
        score_pair_both_extractions(Model(), encoded, FORWARD_SPECS["F0"], self.torch)
        score_pair_both_extractions(Model(), encoded, FORWARD_SPECS["F1"], self.torch)
        self.assertEqual(seen[0], set())
        self.assertEqual(seen[1], {"attention_mask", "token_type_ids"})


# --------------------------------------------------------------------------
# Factor structure
# --------------------------------------------------------------------------

class TestCellStructure(unittest.TestCase):
    def test_the_four_cells_span_both_factors(self):
        combinations = {(c["forward"], c["extraction"]) for c in CELL_SPECS}
        self.assertEqual(
            combinations, {("F0", "X0"), ("F0", "X1"), ("F1", "X0"), ("F1", "X1")}
        )

    def test_endpoints_are_labelled(self):
        self.assertEqual(CELL_BY_NAME["D00"]["reconstructs"], "Step 1 A1")
        self.assertEqual(CELL_BY_NAME["D11"]["reconstructs"], "Step 1 A3")

    def test_each_comparison_isolates_exactly_one_factor(self):
        for left, right in CELL_COMPARISONS:
            with self.subTest(pair=(left, right)):
                factor = cell_isolated_factor(left, right)
                self.assertIn(factor, (FACTOR_FORWARD, FACTOR_EXTRACTION))

    def test_extraction_comparisons_hold_the_forward_fixed(self):
        self.assertEqual(cell_isolated_factor("D00", "D01"), FACTOR_EXTRACTION)
        self.assertEqual(cell_isolated_factor("D10", "D11"), FACTOR_EXTRACTION)

    def test_forward_comparisons_hold_the_extraction_fixed(self):
        self.assertEqual(cell_isolated_factor("D00", "D10"), FACTOR_FORWARD)
        self.assertEqual(cell_isolated_factor("D01", "D11"), FACTOR_FORWARD)

    def test_the_diagonal_toggles_both(self):
        self.assertEqual(
            cell_isolated_factor("D00", "D11"),
            f"{FACTOR_FORWARD} + {FACTOR_EXTRACTION}",
        )

    def test_a_cell_against_itself_toggles_nothing(self):
        self.assertIsNone(cell_isolated_factor("D00", "D00"))

    def test_f1_carries_every_repository_forward_argument(self):
        f1 = FORWARD_SPECS["F1"]
        self.assertTrue(f1["attention_mask"])
        self.assertTrue(f1["token_type_ids"])
        self.assertTrue(f1["inference_mode"])

    def test_f0_carries_none_of_them(self):
        f0 = FORWARD_SPECS["F0"]
        self.assertFalse(f0["attention_mask"])
        self.assertFalse(f0["token_type_ids"])
        self.assertFalse(f0["inference_mode"])

    def test_extraction_specs_name_where_the_scaling_happens(self):
        self.assertIn("float64", EXTRACTION_SPECS["X0"]["scales_in"])
        self.assertIn("tensor dtype", EXTRACTION_SPECS["X1"]["scales_in"])


# --------------------------------------------------------------------------
# Fixtures for the verdict
# --------------------------------------------------------------------------

def flat(polarity, index):
    return index if polarity == "positive" else N_POSITIVE + index


def vector(probe_value, base=45.0):
    scores = [base] * (N_POSITIVE + N_NEGATIVE)
    scores[flat("positive", 169)] = probe_value
    return scores


def step1_positive_vectors():
    """Two 199-pair positive vectors modelling the exact Step 1 movement.

    The first has the recorded A1 histogram; the second is identical except
    that pair 169 moves from bucket 1 to bucket 2, which is precisely how the
    recorded A3 histogram differs from the A1 one.
    """
    target = REFERENCE_OBSERVATION["positive_histogram_literal_wang"]
    scores = []
    for bucket, smoothed in enumerate(target):
        scores.extend([bucket * 10.0 + 5.0] * (smoothed - 1))
    assert len(scores) == N_POSITIVE, len(scores)

    # Move a bucket-1 score to position 169 (a swap preserves the histogram),
    # then set it to the recorded A1 raw value, which is also bucket 1.
    source = next(i for i, v in enumerate(scores) if nbc_bucket(v) == 1)
    scores[source], scores[169] = scores[169], scores[source]
    assert nbc_bucket(scores[169]) == 1
    a1 = list(scores)
    a1[169] = WANG_169

    a3 = list(a1)
    a3[169] = REPO_169
    return a1, a3


def perturb(scores, amount):
    return [v + amount for v in scores]


def build_matrix(d00, d01, d10, d11):
    scores = {"D00": d00, "D01": d01, "D10": d10, "D11": d11}
    matrix = {}
    for left, right in CELL_COMPARISONS:
        block = compare_cells(left, right, scores[left], scores[right], POLARITIES)
        matrix[block["comparison"]] = block
    return matrix, scores


def build(d00, d01, d10, d11, controls=None):
    matrix, scores = build_matrix(d00, d01, d10, d11)
    probe = probe_cell_report("positive", 169, scores, POLARITIES)
    return build_primary_verdict(matrix, probe, scores, POLARITIES, controls=controls)


def clean_controls(d00, d10):
    return {
        "F0": compare_cells("D00", "D00", d00, d00, POLARITIES),
        "F1": compare_cells("D10", "D10", d10, d10, POLARITIES),
    }


class TestCompareCells(unittest.TestCase):
    def test_identical_cells_report_no_on_both_findings(self):
        v = vector(WANG_169)
        block = compare_cells("D00", "D01", v, v, POLARITIES)
        self.assertEqual(block["numerical_difference"], NO)
        self.assertEqual(block["bse_decision_impact"], NO)

    def test_the_recorded_divergence_is_a_bucket_change(self):
        block = compare_cells(
            "D00", "D01", vector(WANG_169), vector(REPO_169), POLARITIES
        )
        self.assertEqual(block["numerical_difference"], YES)
        self.assertEqual(block["bse_decision_impact"], YES)
        self.assertEqual(block["nbc_bucket_disagreements"]["count"], 1)
        self.assertEqual(block["isolated_factor"], FACTOR_EXTRACTION)

    def test_raw_change_without_a_bucket_change_stays_separate(self):
        block = compare_cells(
            "D00", "D10", vector(WANG_169), vector(WANG_169 + 1e-9), POLARITIES
        )
        self.assertEqual(block["numerical_difference"], YES)
        self.assertEqual(block["bse_decision_impact"], NO)


class TestHistogramMovement(unittest.TestCase):
    def test_recorded_movement_is_recognised(self):
        result = histogram_movement_matches_step1(
            REFERENCE_OBSERVATION["positive_histogram_literal_wang"],
            REFERENCE_OBSERVATION["positive_histogram_repository"],
        )
        self.assertTrue(result["movement_reproduced"])

    def test_wrong_movement_is_rejected(self):
        result = histogram_movement_matches_step1(
            REFERENCE_OBSERVATION["positive_histogram_literal_wang"],
            REFERENCE_OBSERVATION["positive_histogram_literal_wang"],
        )
        self.assertFalse(result["movement_reproduced"])
        self.assertTrue(result["left_matches_step1_a1"])
        self.assertFalse(result["right_matches_step1_a3"])

    def test_probe_flip_detection(self):
        scores = {"D00": vector(WANG_169), "D01": vector(REPO_169)}
        probe = probe_cell_report("positive", 169, scores, POLARITIES)
        self.assertTrue(probe_bucket_flip_reproduced(probe, "D00", "D01"))
        self.assertFalse(probe_bucket_flip_reproduced(probe, "D01", "D00"))

    def test_probe_flip_false_for_missing_cells(self):
        scores = {"D00": vector(WANG_169)}
        probe = probe_cell_report("positive", 169, scores, POLARITIES)
        self.assertFalse(probe_bucket_flip_reproduced(probe, "D00", "D01"))


class TestPrimaryVerdict(unittest.TestCase):
    def test_the_strong_extraction_result(self):
        # D00 == D10, D01 == D11, D00 ~ A1, D11 ~ A3: extraction explains it.
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        verdict = build(wang, repo, wang, repo, controls=clean_controls(wang, wang))
        self.assertEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)
        self.assertFalse(verdict["causal_attribution_withheld"])
        self.assertEqual(verdict["extraction_path"]["moves_raw_score"], YES)
        self.assertEqual(verdict["extraction_path"]["changes_nbc_bucket"], YES)
        self.assertEqual(verdict["forward_path"]["moves_raw_score"], NO)
        self.assertTrue(verdict["probe_bucket_flip_reproduced"])
        self.assertIn("No forward-path change is required", verdict["causal_candidate"])

    def test_the_strong_extraction_result_reproduces_the_histogram_movement(self):
        a1, a3 = step1_positive_vectors()
        d00 = a1 + [45.0] * N_NEGATIVE
        d01 = a3 + [45.0] * N_NEGATIVE
        verdict = build(d00, d01, d00, d01, controls=clean_controls(d00, d00))
        self.assertEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)
        self.assertTrue(verdict["histogram_movement"]["movement_reproduced"])
        self.assertTrue(verdict["required_links_passed"])

    def test_forward_only_result(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        verdict = build(wang, wang, repo, repo, controls=clean_controls(wang, repo))
        self.assertEqual(verdict["headline"], HEADLINE_FORWARD_EXPLAINS)
        self.assertEqual(verdict["forward_path"]["moves_raw_score"], YES)
        self.assertEqual(verdict["extraction_path"]["moves_raw_score"], NO)

    def test_both_contribute(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        middle = vector(WANG_169 + 0.003)
        verdict = build(wang, middle, middle, repo, controls=clean_controls(wang, middle))
        self.assertEqual(verdict["headline"], HEADLINE_BOTH_CONTRIBUTE)
        self.assertEqual(verdict["forward_path"]["moves_raw_score"], YES)
        self.assertEqual(verdict["extraction_path"]["moves_raw_score"], YES)

    def test_nondeterministic_control_withholds_attribution(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        controls = {
            "F0": compare_cells("D00", "D00", wang, vector(WANG_169 + 0.02), POLARITIES)
        }
        verdict = build(wang, repo, wang, repo, controls=controls)
        self.assertEqual(verdict["headline"], HEADLINE_UNDETERMINED_NONDETERMINISTIC)
        self.assertEqual(verdict["guard"], GUARD_NONDETERMINISTIC)
        self.assertTrue(verdict["causal_attribution_withheld"])

    def test_endpoints_not_reconstructed_withholds_attribution(self):
        # D00 lands nowhere near A1, so the 2x2 does not span the divergence.
        off = vector(30.0)
        repo = vector(REPO_169)
        verdict = build(off, repo, off, repo, controls=clean_controls(off, off))
        self.assertEqual(verdict["headline"], HEADLINE_UNDETERMINED_ENDPOINTS)
        self.assertEqual(verdict["guard"], GUARD_ENDPOINTS_NOT_RECONSTRUCTED)
        self.assertTrue(verdict["causal_attribution_withheld"])
        self.assertIn("D00 does not reconstruct", verdict["causal_candidate"])

    def test_d11_not_reconstructed_withholds_attribution(self):
        wang = vector(WANG_169)
        off = vector(30.0)
        verdict = build(wang, off, wang, off, controls=clean_controls(wang, wang))
        self.assertEqual(verdict["guard"], GUARD_ENDPOINTS_NOT_RECONSTRUCTED)
        self.assertIn("D11 does not reconstruct", verdict["causal_candidate"])

    def test_a_withheld_verdict_names_no_factor(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        controls = {
            "F0": compare_cells("D00", "D00", wang, vector(WANG_169 + 0.02), POLARITIES)
        }
        verdict = build(wang, repo, wang, repo, controls=controls)
        self.assertNotIn("extraction path is sufficient", verdict["causal_candidate"])

    def test_determinism_takes_precedence_over_endpoints(self):
        off = vector(30.0)
        controls = {
            "F0": compare_cells("D00", "D00", off, vector(30.02), POLARITIES)
        }
        verdict = build(off, off, off, off, controls=controls)
        self.assertEqual(verdict["guard"], GUARD_NONDETERMINISTIC)

    def test_chain_reports_both_endpoints_and_the_controls(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        verdict = build(wang, repo, wang, repo, controls=clean_controls(wang, wang))
        links = {link["link"]: link for link in verdict["causal_chain"]}
        self.assertTrue(links[1]["passed"])
        self.assertTrue(links[2]["passed"])
        self.assertTrue(links[3]["passed"])
        self.assertEqual(
            [k for k, v in links.items() if v["required"]], [1, 2, 3]
        )

    def test_findings_stay_separate(self):
        wang = vector(WANG_169)
        nudged = vector(WANG_169 + 1e-9)
        verdict = build(wang, nudged, wang, nudged, controls=clean_controls(wang, wang))
        self.assertEqual(verdict["numerical_difference"], YES)
        self.assertEqual(verdict["bse_decision_impact"], NO)
        self.assertIn("NOT evidence", verdict["reporting_note"])


class TestSecondaryExpectation(unittest.TestCase):
    def test_extraction_result_predicts_the_secondary_chain_will_not_reproduce(self):
        message = secondary_expectation({"headline": HEADLINE_EXTRACTION_EXPLAINS})
        self.assertIn("EXPECTED to report that it did not", message)
        self.assertIn("not a contradiction", message)

    def test_withheld_primary_blocks_secondary_interpretation(self):
        message = secondary_expectation(
            {"headline": HEADLINE_UNDETERMINED_NONDETERMINISTIC}
        )
        self.assertIn("cannot be interpreted", message)

    def test_forward_result_leaves_the_secondary_informative(self):
        message = secondary_expectation({"headline": HEADLINE_FORWARD_EXPLAINS})
        self.assertIn("carries information", message)

    def test_neither_result_leaves_the_secondary_informative(self):
        message = secondary_expectation({"headline": HEADLINE_NEITHER})
        self.assertIn("carries information", message)


# --------------------------------------------------------------------------
# Sub-diagnostic: softmax shape vs scaling order.
#
# X0 and X1 differ in two ways at once -- the softmax shape and where the * 100
# happens. These tests pin that the two are separated, so the 2x2 result cannot
# be over-claimed as "scaling order" when the shape might also matter.
# --------------------------------------------------------------------------

class TestShapeAndScalingProbe(unittest.TestCase):
    def setUp(self):
        self.torch = HalfTorch(PROBE_PROBABILITIES)
        self.logits = HalfTensor([PROBE_PROBABILITIES])
        self.probe = shape_and_scaling_probe(self.logits, self.torch)

    def test_softmax_shape_does_not_change_the_probability(self):
        self.assertEqual(self.probe["p_wang_shape"], self.probe["p_repo_shape"])
        self.assertTrue(self.probe["probabilities_bit_identical"])
        self.assertEqual(self.probe["probability_delta"], 0.0)

    def test_probability_is_reported_before_any_scaling(self):
        self.assertEqual(self.probe["p_wang_shape"], PROBE_PROBABILITY)

    def test_scaling_order_alone_spans_the_two_recorded_endpoints(self):
        # One shared probability tensor; the only difference is where * 100 goes.
        self.assertEqual(self.probe["scale_after"], WANG_169)
        self.assertEqual(self.probe["scale_inside"], REPO_169)
        self.assertEqual(
            self.probe["scaling_only_delta"], REFERENCE_OBSERVATION["raw_delta"]
        )
        self.assertFalse(self.probe["scaling_order_bit_identical"])

    def test_summary_aggregates_across_pairs(self):
        summary = summarize_shape_and_scaling([self.probe] * 5)
        self.assertEqual(summary["pairs"], 5)
        self.assertTrue(summary["softmax_shape"]["all_bit_identical"])
        self.assertEqual(summary["softmax_shape"]["max_absolute_probability_delta"], 0.0)
        self.assertFalse(summary["scaling_order"]["all_bit_identical"])
        self.assertEqual(summary["scaling_order"]["bit_identical_pairs"], 0)

    def test_interpretation_when_scaling_order_is_specifically_sufficient(self):
        summary = summarize_shape_and_scaling([self.probe])
        result = interpret_shape_and_scaling(self.probe, summary)
        self.assertTrue(result["scaling_order_specifically_sufficient"])
        self.assertFalse(result["softmax_shape_material"])
        self.assertIn("scaling ORDER is specifically sufficient", result["message"])

    def test_interpretation_when_the_softmax_shape_also_differs(self):
        # A material shape difference must block attribution to scaling alone.
        probe = dict(self.probe, probability_delta=0.5, probabilities_bit_identical=False)
        summary = summarize_shape_and_scaling([probe])
        result = interpret_shape_and_scaling(probe, summary)
        self.assertFalse(result["scaling_order_specifically_sufficient"])
        self.assertTrue(result["softmax_shape_material"])
        self.assertIn("combined extraction path", result["message"])

    def test_interpretation_when_scaling_does_not_span_the_endpoints(self):
        probe = dict(self.probe, scale_after=33.0, scale_inside=33.0000004)
        summary = summarize_shape_and_scaling([probe])
        result = interpret_shape_and_scaling(probe, summary)
        self.assertFalse(result["scaling_order_specifically_sufficient"])
        self.assertFalse(result["softmax_shape_material"])
        self.assertIn("not established as sufficient", result["message"])

    def test_the_probe_costs_no_forward_call(self):
        calls = []

        class Model:
            def __call__(self, input_ids, **kwargs):
                calls.append(1)
                return {"logits": HalfTensor([PROBE_PROBABILITIES])}

        result = score_pair_both_extractions(
            Model(), {"input_ids": "IDS"}, FORWARD_SPECS["F0"], self.torch
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("shape_scaling", result)
        self.assertEqual(result["shape_scaling"]["scale_after"], WANG_169)


# --------------------------------------------------------------------------
# The tightened strong verdict.
# --------------------------------------------------------------------------

class TestTightenedExtractionVerdict(unittest.TestCase):
    def test_off_diagonal_endpoint_fields_are_reported(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        verdict = build(wang, repo, wang, repo, controls=clean_controls(wang, wang))
        self.assertIn("D01_vs_step1_a3", verdict["endpoints"])
        self.assertIn("D10_vs_step1_a1", verdict["endpoints"])
        self.assertTrue(verdict["endpoints"]["D01_vs_step1_a3"]["reproduces"])
        self.assertTrue(verdict["endpoints"]["D10_vs_step1_a1"]["reproduces"])

    def test_strong_verdict_requires_d01_to_reach_the_repository_endpoint(self):
        # D01 changes extraction but lands somewhere other than A3.
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        elsewhere = vector(25.0)
        verdict = build(wang, elsewhere, wang, repo, controls=clean_controls(wang, wang))
        self.assertFalse(verdict["endpoints"]["D01_vs_step1_a3"]["reproduces"])
        self.assertNotEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)

    def test_strong_verdict_requires_d10_to_stay_at_the_wang_endpoint(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        drifted = vector(25.0)
        verdict = build(wang, repo, drifted, repo, controls=clean_controls(wang, drifted))
        self.assertFalse(verdict["endpoints"]["D10_vs_step1_a1"]["reproduces"])
        self.assertNotEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)

    def test_a_sub_threshold_forward_delta_that_crosses_a_bucket_is_an_effect(self):
        # The failure mode the review named: a forward raw delta below 1e-4 that
        # still moves an NBC bucket must not be treated as "no forward effect".
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        # 9.94999 -> 9.9 (bucket 0); 9.95001 -> 10.0 (bucket 1). Delta 2e-5.
        d00 = list(wang)
        d10 = list(wang)
        d00[0] = 9.94999
        d10[0] = 9.95001
        d01 = list(repo)
        d11 = list(repo)
        d01[0] = 9.94999
        d11[0] = 9.95001
        verdict = build(d00, d01, d10, d11, controls=clean_controls(d00, d10))

        forward = verdict["forward_path"]
        self.assertLess(forward["max_absolute_delta"], 1e-4)
        self.assertEqual(forward["moves_raw_score"], NO)
        self.assertEqual(forward["changes_nbc_bucket"], YES)
        self.assertEqual(forward["has_effect"], YES)
        self.assertGreater(forward["bucket_disagreements"], 0)
        self.assertNotEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)

    def test_both_paths_changing_buckets_gives_both_contribute(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        d00, d01, d10, d11 = list(wang), list(repo), list(wang), list(repo)
        d00[0] = d01[0] = 9.94999
        d10[0] = d11[0] = 9.95001
        verdict = build(d00, d01, d10, d11, controls=clean_controls(d00, d10))
        self.assertEqual(verdict["forward_path"]["changes_nbc_bucket"], YES)
        self.assertEqual(verdict["extraction_path"]["changes_nbc_bucket"], YES)
        self.assertEqual(verdict["headline"], HEADLINE_BOTH_CONTRIBUTE)

    def test_strong_verdict_requires_the_probe_bucket_flip(self):
        # Endpoints reconstruct, but no pair actually flips bucket.
        wang = vector(WANG_169)
        verdict = build(wang, wang, wang, wang, controls=clean_controls(wang, wang))
        self.assertFalse(verdict["probe_bucket_flip_reproduced"])
        self.assertNotEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)

    def test_extraction_implicated_is_a_defensive_branch(self):
        """Extraction has an effect, the forward has none, yet a strong link fails.

        Through a real 2x2 this is unreachable: if both endpoints reconstruct
        and the forward has no effect, then D01 ~ D11 ~ A3 and D10 ~ D00 ~ A1
        follow arithmetically, so links 4, 5 and 7 cannot fail. The branch is
        kept as a defensive fallback in case the endpoint bound and the
        difference bound ever diverge, and is exercised here by doctoring the
        probe directly rather than by contriving impossible score vectors.
        """
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        matrix, scores = build_matrix(wang, repo, wang, repo)
        probe = probe_cell_report("positive", 169, scores, POLARITIES)
        probe["cells"]["D01"] = {
            "raw": 25.0,
            "rounded_one_decimal": 25.0,
            "nbc_bucket": 2,
        }
        verdict = build_primary_verdict(
            matrix, probe, scores, POLARITIES, controls=clean_controls(wang, wang)
        )
        self.assertEqual(verdict["headline"], HEADLINE_EXTRACTION_IMPLICATED)
        self.assertFalse(verdict["endpoints"]["D01_vs_step1_a3"]["reproduces"])
        self.assertIn("not established because", verdict["causal_candidate"])
        self.assertFalse(verdict["strong_verdict_links_passed"])

    def test_causal_text_does_not_claim_bit_identity_unless_true(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        nudged_wang = perturb(wang, 5e-5)
        nudged_repo = perturb(repo, 5e-5)
        verdict = build(
            wang, repo, nudged_wang, nudged_repo, controls=clean_controls(wang, nudged_wang)
        )
        self.assertEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)
        self.assertFalse(verdict["forward_path"]["bit_identical"])
        self.assertNotIn("bit-identical", verdict["causal_candidate"])
        self.assertIn("perturbs the raw score", verdict["causal_candidate"])

    def test_causal_text_claims_bit_identity_when_it_holds(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        verdict = build(wang, repo, wang, repo, controls=clean_controls(wang, wang))
        self.assertEqual(verdict["headline"], HEADLINE_EXTRACTION_EXPLAINS)
        self.assertTrue(verdict["forward_path"]["bit_identical"])
        self.assertIn("bit-identical", verdict["causal_candidate"])

    def test_strong_verdict_links_are_tracked_separately_from_guards(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        verdict = build(wang, repo, wang, repo, controls=clean_controls(wang, wang))
        self.assertTrue(verdict["required_links_passed"])
        self.assertTrue(verdict["strong_verdict_links_passed"])
        guards = [l["link"] for l in verdict["causal_chain"] if l["required"]]
        strong = [l["link"] for l in verdict["causal_chain"] if l["strong_verdict"]]
        self.assertEqual(guards, [1, 2, 3])
        self.assertEqual(strong, [1, 2, 3, 4, 5, 6, 7])

    def test_shape_scaling_is_carried_into_the_verdict(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        matrix, scores = build_matrix(wang, repo, wang, repo)
        probe = probe_cell_report("positive", 169, scores, POLARITIES)
        verdict = build_primary_verdict(
            matrix, probe, scores, POLARITIES,
            controls=clean_controls(wang, wang),
            shape_scaling={"F0": {"marker": True}},
        )
        self.assertEqual(verdict["shape_and_scaling"], {"F0": {"marker": True}})


# --------------------------------------------------------------------------
# Report / printing structure.
#
# The secondary histogram printer read a top-level key that no longer existed
# after the report grew a primary/secondary split, which crashed a full run
# with KeyError. These tests exercise the printers against the report shape the
# runner actually builds.
# --------------------------------------------------------------------------

def load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diagnose_forward_path_under_test", "scripts/diagnose_forward_path.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunnerReportStructure(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def silent(self, fn, *args):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fn(*args)
        return buffer.getvalue()

    def primary_report(self):
        wang = vector(WANG_169)
        repo = vector(REPO_169)
        matrix, scores = build_matrix(wang, repo, wang, repo)
        probe = probe_cell_report("positive", 169, scores, POLARITIES)
        controls = clean_controls(wang, wang)
        verdict = build_primary_verdict(
            matrix, probe, scores, POLARITIES, controls=controls
        )
        return {
            "comparisons": matrix,
            "determinism_controls": controls,
            "probe": probe,
            "histograms_by_cell": {
                name: histograms_by_polarity(s, POLARITIES)
                for name, s in scores.items()
            },
            "shape_and_scaling": None,
            "verdict": verdict,
        }

    def test_print_primary_runs_against_the_runner_report_shape(self):
        output = self.silent(self.runner.print_primary, self.primary_report())
        self.assertIn("PRIMARY VERDICT", output)
        self.assertIn("D01_vs_step1_a3", output)
        self.assertIn("D10_vs_step1_a1", output)

    def test_print_secondary_histograms_reads_the_nested_key(self):
        # The exact regression: this must read report["secondary"][...], not a
        # top-level key.
        report = {
            "secondary": {
                "nbc_laplace_histograms_by_arm": {
                    "C0": {"positive": [1] * 10, "negative": [1] * 10},
                    "C4": {"positive": [2] * 10, "negative": [2] * 10},
                }
            }
        }
        output = self.silent(self.runner.print_secondary_histograms, report)
        self.assertIn("C0", output)
        self.assertIn("C4", output)

    def test_print_secondary_histograms_fails_loudly_on_the_old_shape(self):
        # A report carrying only the pre-restructure top-level key must not
        # silently print nothing; it must raise.
        with self.assertRaises(KeyError):
            self.silent(
                self.runner.print_secondary_histograms,
                {"histograms_by_arm": {"C0": {"positive": [], "negative": []}}},
            )

    def test_runner_imports_without_torch(self):
        # The printers must stay importable in a CI environment with numpy only.
        self.assertTrue(hasattr(self.runner, "print_primary"))
        self.assertTrue(hasattr(self.runner, "print_secondary_histograms"))
        self.assertTrue(hasattr(self.runner, "run_primary_decomposition"))


if __name__ == "__main__":
    unittest.main()
