"""Validation-selection rules for ablation cells B, C and D0 (prereg §4).

Pure logic on synthetic metrics: the histogram evidence, the grids, the exact
reference gate, eligibility at its boundary, the ranking order, no fallback,
and the two consistency checks. No torch, no cache, no Wang data.
"""

import math
import unittest
from pathlib import Path

from src import ablation_selection as S
from src.baseline_core import bayes_update, discretize_document_score
from src.cache_completion import sha256_file
from src.ddre_core import (
    CANDIDATE_LOWER_GRID,
    CANDIDATE_UPPER_GRID,
    cost_consistent_thresholds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POS = [1, 78, 19, 13, 6, 5, 28, 57, 1, 1]
NEG = [1, 141, 38, 15, 4, 1, 3, 4, 1, 1]
REF = {"nonfactual_auc_pr": 0.88, "factual_auc_pr": 0.69, "balanced_pr_auc": 0.785}


def metrics(nonfactual=0.88, factual=0.69, docs=5.0, nli=15.0, balanced=None):
    return {
        "nonfactual": {"auc_pr": nonfactual, "precision": 0.8, "recall": 0.9},
        "factual": {"auc_pr": factual, "precision": 0.7, "recall": 0.5},
        "balanced_pr_auc": (nonfactual + factual) / 2 if balanced is None else balanced,
        "accuracy": 0.8, "macro_f1": 0.7,
        "efficiency": {"avg_retrieved_documents_per_sentence": docs,
                       "avg_retrieved_documents_per_subclaim": docs / 1.5,
                       "avg_nli_span_calls_per_sentence": nli},
    }


def rows(cell, eligible_docs=None, **kw):
    out = []
    for i, params in enumerate(S.cell_parameters(cell)):
        docs = eligible_docs(i) if eligible_docs else 5.0
        out.append(S.candidate_row(cell, params, metrics(docs=docs, **kw), REF))
    return out


class TestHistogramRatio(unittest.TestCase):
    def test_kappa_one_is_bse_evidence(self):
        est = S.HistogramRatio(POS, NEG)
        for score in (0.0, 15.0, 35.3, 62.0, 99.0):
            b = discretize_document_score(score)
            p1, p0 = POS[b] / sum(POS), NEG[b] / sum(NEG)
            self.assertAlmostEqual(est.ratio(score), p1 / p0, places=12)

    def test_accumulation_equals_bse_bayes_update(self):
        # The band cell's log-odds update reproduces BSE's own Bayes update.
        est = S.HistogramRatio(POS, NEG)
        p, log_odds = 0.5, 0.0
        for score in (35.0, 22.0, 61.0, 35.5):
            b = discretize_document_score(score)
            p = bayes_update(p, POS[b] / sum(POS), NEG[b] / sum(NEG))
            log_odds += est.log_ratio(score)
            self.assertAlmostEqual(1 / (1 + math.exp(-log_odds)), p, places=12)

    def test_kappa_scales_log_evidence(self):
        one, three = S.HistogramRatio(POS, NEG), S.HistogramRatio(POS, NEG, kappa=3.0)
        self.assertAlmostEqual(three.log_ratio(35.0), 3 * one.log_ratio(35.0))
        self.assertAlmostEqual(one.log_ratio(35.0), -0.1431008436406733, places=12)

    def test_refuses_bad_inputs(self):
        with self.assertRaises(ValueError):
            S.HistogramRatio([0] + POS[1:], NEG)
        with self.assertRaises(ValueError):
            S.HistogramRatio(POS, NEG, kappa=0.0)
        with self.assertRaises(ValueError):
            S.HistogramRatio(POS, NEG).ratio(float("nan"))


class TestGrids(unittest.TestCase):
    def test_bands_are_the_repository_cost_consistent_pairs(self):
        space = cost_consistent_thresholds(28.0, 96.0, CANDIDATE_LOWER_GRID, CANDIDATE_UPPER_GRID)
        self.assertEqual(len(S.verify_band_grid(space)), 32)
        bad = dict(space, effective_lower_grid=list(space["effective_lower_grid"])[:-1])
        with self.assertRaises(S.SelectionRefused):
            S.verify_band_grid(bad)

    def test_preregistered_counts_and_kappa_grid(self):
        self.assertEqual(S.KAPPA_GRID, (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0))
        self.assertEqual({c: len(S.cell_parameters(c)) for c in ("B", "C", "D0")},
                         {"B": 32, "C": 224, "D0": 32})
        self.assertEqual(S.EXPECTED_CANDIDATES, {"B": 32, "C": 224, "D0": 32})
        self.assertEqual((S.D0_SIGMA, S.D0_LAMBDA), (0.2269287109375, 1.0))

    def test_preregistration_document_is_the_pinned_one(self):
        self.assertEqual(sha256_file(PROJECT_ROOT / S.PREREGISTRATION_PATH),
                         S.PREREGISTRATION_SHA256)


class TestReference(unittest.TestCase):
    ARTIFACT = {"bse_official_validation": {
        "nonfactual": {"auc_pr": 0.8827013410907382},
        "factual": {"auc_pr": 0.6928579387788498},
        "balanced_pr_auc": 0.787779639934794}}

    def test_reads_full_precision_values_from_the_artifact(self):
        self.assertEqual(S.stored_reference(self.ARTIFACT),
                         {"nonfactual_auc_pr": 0.8827013410907382,
                          "factual_auc_pr": 0.6928579387788498,
                          "balanced_pr_auc": 0.787779639934794})

    def test_missing_reference_is_refused(self):
        with self.assertRaises(S.SelectionRefused):
            S.stored_reference({"bse_official_validation": {"nonfactual": {}}})

    def test_exact_equality_one_ulp_fails(self):
        ref = S.stored_reference(self.ARTIFACT)
        replay = metrics(0.8827013410907382, 0.6928579387788498, balanced=0.787779639934794)
        S.verify_reference(replay, ref)
        off = metrics(math.nextafter(0.8827013410907382, 1.0), 0.6928579387788498,
                      balanced=0.787779639934794)
        with self.assertRaises(S.SelectionRefused):
            S.verify_reference(off, ref)


class TestEligibilityAndRanking(unittest.TestCase):
    def test_eligibility_boundary_is_inclusive(self):
        edge = REF["nonfactual_auc_pr"] - S.QUALITY_TOLERANCE
        at = S.candidate_row("B", {"lower": 0.2, "upper": 0.8},
                             metrics(nonfactual=edge, balanced=1.0), REF)
        below = S.candidate_row("B", {"lower": 0.2, "upper": 0.8},
                                metrics(nonfactual=math.nextafter(edge, 0.0), balanced=1.0), REF)
        self.assertTrue(at["eligible"])
        self.assertFalse(below["eligible"])

    def test_all_three_pr_aucs_are_required(self):
        for kw in ({"nonfactual": 0.80}, {"factual": 0.60}, {"balanced": 0.70}):
            row = S.candidate_row("B", {"lower": 0.2, "upper": 0.8}, metrics(**kw), REF)
            self.assertFalse(row["eligible"], kw)

    def test_rank_order(self):
        def row(docs, bal, nf, f, nli, lower, upper, kappa=None):
            params = {"lower": lower, "upper": upper}
            if kappa is not None:
                params = {"kappa": kappa, **params}
            r = S.candidate_row("C" if kappa else "B", params,
                                metrics(nonfactual=nf, factual=f, docs=docs, nli=nli,
                                        balanced=bal), REF)
            return r
        base = dict(docs=5, bal=0.8, nf=0.9, f=0.7, nli=15, lower=0.2, upper=0.8)
        ranked = sorted([
            row(**dict(base, docs=6)),
            row(**dict(base, bal=0.79)),
            row(**dict(base, nf=0.89)),
            row(**dict(base, f=0.695)),
            row(**dict(base, nli=16)),
            row(**dict(base, upper=0.85)),
            row(**dict(base, lower=0.15, upper=0.85)),
            row(**base),
        ], key=S.order_key)
        # Exact ties on every metric fall back to ascending lower, then upper.
        self.assertEqual([(r["lower"], r["upper"]) for r in ranked[:3]],
                         [(0.15, 0.85), (0.2, 0.8), (0.2, 0.85)])
        # Then fewer NLI spans, higher factual, nonfactual, balanced; docs last.
        self.assertEqual(ranked[3]["avg_nli_span_calls_per_sentence"], 16)
        self.assertEqual(ranked[4]["factual_auc_pr"], 0.695)
        self.assertEqual(ranked[5]["nonfactual_auc_pr"], 0.89)
        self.assertEqual(ranked[6]["balanced_pr_auc"], 0.79)
        self.assertEqual(ranked[7]["avg_retrieved_documents_per_sentence"], 6)
        c = sorted([row(**dict(base, kappa=2.0, lower=0.05)), row(**dict(base, kappa=1.5))],
                   key=S.order_key)
        self.assertEqual(c[0]["kappa"], 1.5)  # kappa before lower

    def test_select_picks_minimum_documents(self):
        out = S.select("B", rows("B", eligible_docs=lambda i: 10 - i * 0.1))
        self.assertEqual(out["outcome"], S.SELECTED)
        self.assertEqual(out["eligible"], 32)
        self.assertEqual((out["selected"]["lower"], out["selected"]["upper"]), (0.2, 0.95))

    def test_no_eligible_configuration_has_no_fallback(self):
        out = S.select("D0", rows("D0", nonfactual=REF["nonfactual_auc_pr"] - 0.0051))
        self.assertEqual(out, {"cell": "D0", "candidates": 32, "eligible": 0,
                               "outcome": S.NO_ELIGIBLE, "selected": None})

    def test_candidate_counts_are_enforced(self):
        with self.assertRaises(S.SelectionRefused):
            S.select("C", rows("B"))
        with self.assertRaises(S.SelectionRefused):
            S.select("B", rows("B")[:31])


class TestConsistency(unittest.TestCase):
    def recorded(self, d0_rows):
        return [{"lower": r["lower"], "upper": r["upper"],
                 **{theirs: r[mine] for mine, theirs in S.COMPARED_FIELDS},
                 "preserves_nonfactual": r["eligible"], "preserves_factual": True,
                 "preserves_balanced": True} for r in d0_rows]

    def test_d0_must_match_the_recorded_rows(self):
        d0 = rows("D0")
        self.assertEqual(S.d0_matches_record(d0, self.recorded(d0)), [])
        recorded = self.recorded(d0)
        recorded[3]["balanced_pr_auc"] += 1e-12
        self.assertEqual(len(S.d0_matches_record(d0, recorded)), 1)
        recorded = self.recorded(d0)
        recorded[0]["preserves_nonfactual"] = not recorded[0]["preserves_nonfactual"]
        self.assertTrue(S.d0_matches_record(d0, recorded))

    def test_recorded_production_row_must_be_the_d0_pair(self):
        artifact = {"hyperparameter_rows": [
            {"is_production_pair": True, "sigma": 0.11346435546875, "lambda": 1.0,
             "threshold_candidates": []}]}
        with self.assertRaises(S.SelectionRefused):
            S.recorded_production_rows(artifact)

    def test_c_kappa_one_must_equal_b(self):
        b, c = rows("B"), rows("C")
        self.assertEqual(S.c_kappa_one_matches_b(b, c), [])
        c[0]["avg_retrieved_documents_per_sentence"] += 0.1
        self.assertEqual(len(S.c_kappa_one_matches_b(b, c)), 1)


if __name__ == "__main__":
    unittest.main()
