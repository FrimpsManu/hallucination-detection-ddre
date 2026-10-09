"""Held-out ablation replay rules (preregistration §6-§8).

The frozen cells load exactly from the validation freeze; the A/D gate is
exact; the bootstrap is paired across cells at the passage level; the ratio
guard behaves as preregistered; and every interpretation threshold equals the
number written in the preregistration. Synthetic observations only.
"""

import copy
import math
import re
import unittest
from pathlib import Path
from unittest import mock

from src import ablation_replay as R
from src.baseline_core import DetectionResult
from src.evaluation import EvaluatedSentence
from src.paired_bootstrap import PairedInputMismatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREG = (PROJECT_ROOT / "docs" / "ablation_preregistration.md").read_text()

FREEZE = {
    "B": {"outcome": "selected", "selected": {"cell": "B", "lower": 0.2, "upper": 0.85}},
    "C": {"outcome": "selected",
          "selected": {"cell": "C", "kappa": 2.5, "lower": 0.05, "upper": 0.95}},
    "D0": {"outcome": "no_eligible_configuration", "selected": None,
           "eligible": 0, "candidates": 32},
    "candidate_counts": {"B": 32, "C": 224, "D0": 32},
    "held_out_records_evaluated": 0,
    "held_out_cache_opened": False,
}


def obs(passage, sentence, label, p, docs, nli=None):
    return EvaluatedSentence(
        passage_index=passage, sentence_index=sentence, gold_label=label,
        result=DetectionResult(p_factual=p, prediction=int(p > 0.2258),
                               documents_used=docs, nli_calls=nli if nli is not None else 3 * docs),
    )


def cell_obs(docs_by_passage, shift=0.0):
    """12 passages x 3 sentences, both labels present everywhere."""
    out = []
    for passage in range(12):
        for s in range(3):
            label = (passage + s) % 2
            p = min(max(0.15 + 0.6 * label + 0.02 * s + shift, 0.0), 1.0)
            out.append(obs(passage, s, label, p, docs_by_passage(passage)))
    return out


def four_cells(d_saving=2):
    return {
        "A": cell_obs(lambda p: 6),
        "B": cell_obs(lambda p: 5 + (p % 2)),
        "C": cell_obs(lambda p: 5),
        "D": cell_obs(lambda p: 6 - d_saving),
    }


class TestFrozenCells(unittest.TestCase):
    def test_loads_exactly_the_frozen_configuration(self):
        cells = R.frozen_cells(FREEZE)
        self.assertEqual(cells["B"], {"kappa": 1.0, "lower": 0.2, "upper": 0.85})
        self.assertEqual(cells["C"], {"kappa": 2.5, "lower": 0.05, "upper": 0.95})
        self.assertEqual(cells["D"], {"sigma": 0.11346435546875, "lambda": 1.0,
                                      "lower": 0.2, "upper": 0.8})
        self.assertFalse(cells["D0"]["evaluated"])
        self.assertIn("no_eligible_configuration", cells["D0"]["reason"])

    def test_any_other_configuration_is_refused(self):
        for path, value in ((("B", "selected", "upper"), 0.9),
                            (("C", "selected", "kappa"), 3.0),
                            (("C", "outcome"), "no_eligible_configuration"),
                            (("D0", "outcome"), "selected"),
                            (("candidate_counts",), {"B": 32, "C": 223, "D0": 32}),
                            (("held_out_cache_opened",), True)):
            freeze = copy.deepcopy(FREEZE)
            node = freeze
            for part in path[:-1]:
                node = node[part]
            node[path[-1]] = value
            with self.assertRaises(R.ReplayRefused, msg=str(path)):
                R.frozen_cells(freeze)

    def test_d0_with_a_selection_is_refused(self):
        freeze = copy.deepcopy(FREEZE)
        freeze["D0"]["selected"] = {"lower": 0.2, "upper": 0.8}
        with self.assertRaises(R.ReplayRefused):
            R.frozen_cells(freeze)


class TestConsistencyGate(unittest.TestCase):
    ROW = {"passage_index": "1", "sentence_index": "0", "p_factual": "0.1348",
           "prediction": "0", "retrieved_documents": "13", "nli_span_calls": "66",
           "subclaim_documents_used": "[3,6,4]", "subclaim_nli_calls": "[19,25,22]"}

    def test_exact_reproduction_passes(self):
        self.assertTrue(R.consistency_gate("A", [self.ROW], [dict(self.ROW)])["exact"])

    def test_every_field_and_one_ulp_fail(self):
        tiny = repr(math.nextafter(0.1348, 1.0))
        for field, value in (("p_factual", tiny), ("prediction", "1"),
                             ("retrieved_documents", "12"), ("nli_span_calls", "65"),
                             ("subclaim_documents_used", "[3,6,3]")):
            with self.assertRaises(R.ConsistencyGateFailed, msg=field):
                R.consistency_gate("D", [self.ROW], [dict(self.ROW, **{field: value})])

    def test_missing_sentence_fails(self):
        with self.assertRaises(R.ConsistencyGateFailed):
            R.consistency_gate("A", [self.ROW], [])


class TestBootstrap(unittest.TestCase):
    def test_frozen_settings(self):
        self.assertEqual((R.N_RESAMPLES, R.SEED, R.CI_LEVEL), (10_000, 42, 0.95))
        self.assertEqual(R.RATIO_DENOMINATOR_GUARD, 0.05)

    def test_identical_cells_give_exactly_zero_in_every_replicate(self):
        cells = {c: cell_obs(lambda p: 6) for c in R.CELLS}
        out = R.multi_cell_bootstrap(cells, n_resamples=200)
        for key in ("docs_S_B", "docs_B_minus_C", "nli_C_minus_D", "balanced_D_minus_C"):
            e = out["endpoints"][key]
            self.assertEqual((e["observed"], e["ci_lower"], e["ci_upper"]), (0.0, 0.0, 0.0))

    def test_resampling_unit_is_the_passage_shared_by_all_cells(self):
        cells = four_cells()
        seen = []
        real = R.replicate_indices

        def spy(blocks, rng):
            seen.append(blocks)
            return real(blocks, rng)

        with mock.patch.object(R, "replicate_indices", side_effect=spy):
            out = R.multi_cell_bootstrap(cells, n_resamples=50)
        self.assertEqual(len(seen), 50)
        self.assertEqual(len(seen[0]), 12)  # 12 passages, not 36 sentences
        self.assertEqual(seen[0][0], (0, (0, 1, 2)))
        self.assertEqual(out["unit"], "passage")
        self.assertEqual(out["paired_across"], ["A", "B", "C", "D"])

    def test_paired_differences_within_a_replicate(self):
        # B saves exactly 1 doc on odd passages, 0 on even: S(B) is a passage mix,
        # and S(B) + docs(B) - docs(C) = S(C) holds in every replicate.
        out = R.multi_cell_bootstrap(four_cells(), n_resamples=300)
        e = out["endpoints"]
        self.assertAlmostEqual(e["docs_S_B"]["observed"] + e["docs_B_minus_C"]["observed"],
                               e["docs_S_C"]["observed"])
        self.assertLess(e["docs_S_B"]["ci_lower"], e["docs_S_B"]["ci_upper"])
        self.assertEqual(e["docs_S_C"]["ci_lower"], 1.0)

    def test_misaligned_cells_are_refused(self):
        cells = four_cells()
        cells["C"] = list(reversed(cells["C"]))
        with self.assertRaises(PairedInputMismatch):
            R.multi_cell_bootstrap(cells, n_resamples=10)

    def test_seed_makes_it_deterministic(self):
        a = R.multi_cell_bootstrap(four_cells(), n_resamples=100)
        b = R.multi_cell_bootstrap(four_cells(), n_resamples=100)
        self.assertEqual(a["endpoints"], b["endpoints"])


class TestRatioGuard(unittest.TestCase):
    def test_reported_when_every_replicate_clears_the_guard(self):
        out = R.multi_cell_bootstrap(four_cells(d_saving=2), n_resamples=200)
        r = out["endpoints"]["ratio_S_B_over_S_D"]
        self.assertEqual(r["ci_status"], "reported")
        self.assertEqual(r["replicates_at_or_below_guard"], 0)
        self.assertAlmostEqual(r["observed"], 0.25)

    def test_unstable_when_any_replicate_is_at_or_below_the_guard(self):
        cells = four_cells()
        # D saves 1 doc on passage 0 only: many replicates have S_D <= 0.05.
        cells["D"] = cell_obs(lambda p: 5 if p == 0 else 6)
        out = R.multi_cell_bootstrap(cells, n_resamples=200)
        for name in ("ratio_S_B_over_S_D", "ratio_S_C_over_S_D"):
            r = out["endpoints"][name]
            self.assertEqual(r["ci_status"], "unstable")
            self.assertIsNone(r["ci_lower"])
            self.assertIsNone(r["ci_upper"])
            self.assertGreater(r["replicates_at_or_below_guard"], 0)
        # Component savings are still reported with intervals.
        self.assertIsNotNone(out["endpoints"]["docs_S_D"]["ci_lower"])
        self.assertIsNotNone(out["endpoints"]["docs_S_B"]["ci_lower"])

    def test_observed_ratio_withheld_when_observed_S_D_fails_the_guard(self):
        cells = four_cells(d_saving=0)
        out = R.multi_cell_bootstrap(cells, n_resamples=20)
        self.assertIsNone(out["endpoints"]["ratio_S_B_over_S_D"]["observed"])


def endpoints(s_b, s_c, s_d, cd_lo, cd_hi):
    ratio = lambda x: x / s_d if s_d > R.RATIO_DENOMINATOR_GUARD else None
    return {
        "docs_S_B": {"observed": s_b}, "docs_S_C": {"observed": s_c},
        "docs_S_D": {"observed": s_d},
        "docs_B_minus_C": {"observed": s_b - s_c},
        "docs_C_minus_D": {"observed": s_c - s_d, "ci_lower": cd_lo, "ci_upper": cd_hi},
        "ratio_S_B_over_S_D": {"observed": ratio(s_b)},
        "ratio_S_C_over_S_D": {"observed": ratio(s_c)},
    }


BAL = {"A": 0.73, "B": 0.73, "C": 0.735, "D": 0.74}


class TestInterpretation(unittest.TestCase):
    def test_thresholds_match_the_preregistration_text(self):
        self.assertIn("If S(B) ≥ 0.5 × S(D)", PREREG)
        self.assertIn("If S(C) ≥ 0.8 × S(D) **and** C's balanced PR-AUC is within 0.01 of D's",
                      PREREG)
        self.assertIn("D's balanced PR-AUC is not lower than\nC's by more than 0.01", PREREG)
        self.assertIn("S_D(b) ≤ 0.05 documents per sentence", PREREG)
        self.assertIn("Stated as a number: S(B) / S(D) < 0.25.", PREREG)
        self.assertTrue(re.search(r"passage unit, 10,000 resamples, seed 42, 95% percentile",
                                  PREREG.replace("\n", " ")))
        self.assertEqual((R.A_TO_B_SHARE, R.B_TO_C_SHARE, R.BALANCED_MARGIN,
                          R.PREDICTION_SHARE), (0.5, 0.8, 0.01, 0.25))
        self.assertIn("(A 0, B 32, C 224, D 640)", PREREG)
        self.assertEqual(R.TUNING_BUDGETS, {"A": 0, "B": 32, "C": 224, "D": 640})

    def test_a_to_b_boundary_is_inclusive(self):
        self.assertTrue(R.interpret(endpoints(0.5, 1.0, 1.0, 0.1, 0.2), BAL)
                        ["A_to_B"]["condition_met"])
        self.assertFalse(R.interpret(endpoints(math.nextafter(0.5, 0), 1.0, 1.0, 0.1, 0.2), BAL)
                         ["A_to_B"]["condition_met"])

    def test_b_to_c_needs_share_and_balanced_margin(self):
        ok = R.interpret(endpoints(0.1, 0.8, 1.0, 0.1, 0.2), dict(BAL, C=0.735, D=0.74))
        self.assertTrue(ok["B_to_C"]["condition_met"])
        far = R.interpret(endpoints(0.1, 0.8, 1.0, 0.1, 0.2), dict(BAL, C=0.72, D=0.74))
        self.assertFalse(far["B_to_C"]["condition_met"])
        short = R.interpret(endpoints(0.1, 0.79, 1.0, 0.1, 0.2), BAL)
        self.assertFalse(short["B_to_C"]["condition_met"])

    def test_margin_boundary_is_closed_in_floating_point(self):
        # Numerical implementation of the inclusive preregistered boundary.
        self.assertEqual(R.BALANCED_MARGIN, 0.01)
        self.assertGreater(0.74 - 0.73, 0.01)  # the binary-float artefact
        self.assertTrue(R.within_closed_margin(0.74 - 0.73, 0.74, 0.73))
        self.assertFalse(R.within_closed_margin(0.7401 - 0.73, 0.7401, 0.73))
        self.assertFalse(R.within_closed_margin(0.01 + 1e-12, 0.5, 0.49))

    def test_decimal_boundary_passes_both_margin_rules(self):
        at = dict(BAL, C=0.73, D=0.74)
        self.assertTrue(R.interpret(endpoints(0.1, 0.8, 1.0, 0.1, 0.2), at)
                        ["B_to_C"]["condition_met"])
        lower = dict(BAL, C=0.74, D=0.73)  # D lower than C by exactly 0.01
        self.assertTrue(R.interpret(endpoints(0.1, 0.5, 1.0, -0.8, -0.2), lower)
                        ["C_to_D"]["condition_met"])
        beyond = dict(BAL, C=0.7401, D=0.73)
        self.assertFalse(R.interpret(endpoints(0.1, 0.5, 1.0, -0.8, -0.2), beyond)
                         ["C_to_D"]["condition_met"])
        self.assertFalse(R.interpret(endpoints(0.1, 0.8, 1.0, 0.1, 0.2), dict(BAL, C=0.7299, D=0.74))
                         ["B_to_C"]["condition_met"])

    def test_c_to_d_needs_ci_excluding_zero(self):
        self.assertTrue(R.interpret(endpoints(0.1, 0.5, 1.0, -0.8, -0.2), BAL)
                        ["C_to_D"]["condition_met"])
        touching = R.interpret(endpoints(0.1, 0.5, 1.0, -0.8, 0.0), BAL)
        self.assertFalse(touching["C_to_D"]["ci_excludes_zero"])
        self.assertFalse(touching["C_to_D"]["condition_met"])
        worse = R.interpret(endpoints(0.1, 0.5, 1.0, -0.8, -0.2), dict(BAL, C=0.76, D=0.74))
        self.assertFalse(worse["C_to_D"]["condition_met"])

    def test_prediction_and_unmet_wording(self):
        out = R.interpret(endpoints(0.2, 0.5, 1.0, -0.1, 0.1), BAL)
        self.assertTrue(out["prediction"]["held"])
        self.assertEqual(out["A_to_B"]["statement"],
                         "The preregistered A -> B condition was not met.")
        self.assertEqual(out["status"], "exploratory / post-hoc held-out ablation")

    def test_no_causal_language(self):
        import json
        for case in (endpoints(0.9, 0.9, 1.0, 0.1, 0.2), endpoints(0.1, 0.1, 1.0, -0.1, 0.1)):
            R.check_language(json.dumps(R.interpret(case, BAL)))
        with self.assertRaises(ValueError):
            R.check_language("this proves the mechanism")


if __name__ == "__main__":
    unittest.main()
