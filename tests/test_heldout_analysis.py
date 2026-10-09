"""Descriptive analysis of the frozen held-out run.

The arithmetic in ``src/heldout_analysis.py`` is checked on hand-checkable
inputs, and the rendered summary is checked for descriptive labelling and the
absence of causal language.

No model, no torch, no Wang data. Everything runs on synthetic values.
"""

import math
import tempfile
import unittest

import numpy as np

from src import heldout_analysis as A

C = {"c_miss": 28.0, "c_false_alarm": 96.0}
CUT = 28.0 / (28.0 + 96.0)


def ddre_steps(log_ratios, p0=0.5):
    log_odds, steps = math.log(p0 / (1 - p0)), []
    for lr in log_ratios:
        log_odds += lr
        steps.append({"score": 35.0, "log_ratio": lr, "cum_log_odds": log_odds,
                      "p": 1 / (1 + math.exp(-log_odds)), "flip_region": True})
    return steps


def bse_steps(posteriors):
    return [{"score": 35.0, "bucket": 3, "llr": -0.14, "p": p} for p in posteriors]


def subclaim(bse_p_seq, ddre_lr, stop="lower"):
    d = ddre_steps(ddre_lr)
    return {"text": "x", "available": 10, "bse": bse_steps(bse_p_seq),
            "bse_p": bse_p_seq[-1] if bse_p_seq else 0.5,
            "ddre": d, "ddre_p": d[-1]["p"], "ddre_stop": stop}


class TestSentenceLevel(unittest.TestCase):
    def test_class_metrics_matches_sklearn(self):
        from sklearn.metrics import f1_score, precision_score, recall_score

        rng = np.random.default_rng(0)
        gold = rng.integers(0, 2, 200)
        pred = rng.integers(0, 2, 200)
        m = A.class_metrics(gold, pred)
        for c, name in ((0, "nonfactual"), (1, "factual")):
            self.assertAlmostEqual(m[name]["precision"], precision_score(gold, pred, pos_label=c))
            self.assertAlmostEqual(m[name]["recall"], recall_score(gold, pred, pos_label=c))
        self.assertAlmostEqual(m["macro_f1"], f1_score(gold, pred, average="macro"))
        cm = m["confusion_rows_gold_cols_pred_0_hallucinated_1_factual"]
        self.assertEqual(sum(map(sum, cm)), 200)

    def test_paired_correctness_and_one_way_changes(self):
        gold = [0, 0, 1, 1, 0]
        base = [1, 0, 1, 1, 1]
        comp = [0, 0, 0, 1, 1]
        out = A.paired_correctness(gold, base, comp)
        self.assertEqual(out["all"], {"both_right": 2, "baseline_only_right": 1,
                                      "comparison_only_right": 1, "both_wrong": 1})
        self.assertEqual(out["prediction_changes"],
                         {"gold=0 baseline=1 comparison=0": 1,
                          "gold=1 baseline=1 comparison=0": 1})

    def test_efficiency_by_label_shares_sum_to_one(self):
        out = A.efficiency_by_label([0, 0, 1], [10, 6, 4], [5, 6, 3], [30, 20, 10], [15, 20, 9])
        self.assertAlmostEqual(out["gold_hallucinated"]["share_of_total_doc_saving"]
                               + out["gold_factual"]["share_of_total_doc_saving"], 1.0)
        self.assertAlmostEqual(out["gold_hallucinated"]["doc_reduction_fraction"], 5 / 16)

    def test_saving_concentration(self):
        out = A.saving_concentration([10, 2, 2, 2], [0, 2, 2, 3])
        self.assertEqual(out["net_saving"], 9.0)
        self.assertEqual(out["sentences_for_fraction"]["0.50"], 1)
        self.assertEqual((out["cheaper"], out["equal"], out["more_expensive"]), (1, 2, 1))

    def test_posterior_placement_boundaries_follow_the_cost_rule(self):
        # p == cut classifies hallucinated, so it belongs to the (lower, cut] bin.
        out = A.posterior_placement([0, 0, 0, 0], [0.2, CUT, 0.5, 0.8], 0.2, CUT, 0.8)
        g = out["gold_hallucinated"]
        self.assertEqual(list(g.values()), [0.25, 0.25, 0.25, 0.25])

    def test_recall_by_subclaim_count(self):
        out = A.recall_by_subclaim_count([1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 2, 5, 1])
        self.assertEqual(out["1"]["factual_recall_baseline"], 1.0)
        self.assertEqual(out["2"]["factual_recall_comparison"], 0.0)
        self.assertEqual(out[">=3"]["gold_factual_sentences"], 1)


class TestDepth(unittest.TestCase):
    def test_joint_table_and_saving_by_row(self):
        out = A.depth_joint_table([10, 10, 3, 1], [4, 10, 3, 2], max_docs=10)
        self.assertEqual(out["total_doc_saving"], 5)
        self.assertEqual(out["saving_by_baseline_depth"]["10"]["doc_saving"], 6)
        self.assertEqual(out["saving_by_baseline_depth"]["1"]["doc_saving"], -1)
        self.assertEqual(out["joint_counts_rows_baseline_cols_comparison"][10][4], 1)
        self.assertEqual((out["budget_hits_baseline"], out["budget_hits_comparison"]), (2, 1))

    def test_long_chain_share(self):
        out = A.long_chain_share([9, 8, 2, 2], [4, 4, 2, 3], min_depth=8)
        self.assertEqual(out["subclaims"], 2)
        self.assertAlmostEqual(out["share_of_saving"], 9 / 8)


class TestEvidence(unittest.TestCase):
    def test_flip_region_requires_unanimous_strict_sign(self):
        signs = np.array([[1, 1, -1, 1, -1],
                          [1, -1, -1, 0, -1]])
        self.assertEqual(A.flip_region(signs).tolist(), [False, True, False, True, False])

    def test_region_boundaries(self):
        self.assertEqual(A.region_boundaries([0, 1, 2, 3], [True, True, False, True]), [2.0, 3.0])

    def test_evidence_table_bins(self):
        grid = np.arange(0, 10, 1.0)
        rows = A.evidence_table(grid, (0, 5, 10), {"id": lambda x: x}, np.zeros(10, bool))
        self.assertEqual([r["id"] for r in rows], [2.0, 7.0])
        self.assertEqual(rows[0]["flip_region_fraction"], 0.0)


class TestTrajectories(unittest.TestCase):
    def test_responsible_subclaims_use_the_cost_rule(self):
        hit = subclaim([0.6, 0.5], [-0.8, -0.8])          # DDRE ends ~0.17, BSE 0.5
        miss = subclaim([0.6, 0.5], [0.5])                # DDRE ends factual
        self.assertEqual(A.responsible_subclaims([hit, miss], **C), [hit])

    def test_decomposition_three_cases(self):
        lr = [-0.8, -0.8]
        self.assertEqual(A.stop_depth_decomposition(subclaim([0.6], lr), **C),
                         "baseline_stopped_first")
        self.assertEqual(A.stop_depth_decomposition(subclaim([0.6, 0.3, 0.9], lr), **C),
                         "opposite_side_at_k")
        self.assertEqual(A.stop_depth_decomposition(subclaim([0.6, 0.1, 0.9], lr), **C),
                         "same_side_at_k")

    def test_evidence_drivers(self):
        one = A.evidence_drivers(subclaim([0.5], [-2.0]), lower=0.2)
        self.assertTrue(one["first_document_alone_crosses_lower"])
        self.assertEqual(one["largest_single_share_of_negative"], 1.0)
        acc = A.evidence_drivers(subclaim([0.5], [-0.5, 0.2, -0.5, -0.5]), lower=0.2)
        self.assertFalse(acc["first_document_alone_crosses_lower"])
        self.assertAlmostEqual(acc["largest_single_share_of_negative"], 1 / 3)
        self.assertEqual((acc["documents_toward_factual"], acc["documents_toward_hallucinated"]),
                         (1, 3))

    def test_continuation_outcome(self):
        s = subclaim([0.5, 0.45, 0.6, 0.9], [-0.8, -0.8])  # BSE went 2 docs further
        self.assertIsNone(A.continuation_outcome(subclaim([0.5], [-0.8, -0.8]),
                                                 lambda x: 5.0, **C))
        self.assertTrue(A.continuation_outcome(s, lambda x: 2.0, **C))
        self.assertFalse(A.continuation_outcome(s, lambda x: -0.1, **C))

    def test_group_summary_counts(self):
        sentence = {"key": (0, 0), "gold": 1,
                    "subclaims": [subclaim([0.6, 0.5, 0.9], [-0.8, -0.8]),
                                  subclaim([0.9], [2.0], stop="upper")]}
        out = A.group_trajectory_summary([sentence], lower=0.2, log_ratio=lambda x: 3.0, **C)
        self.assertEqual(out["responsible_subclaims"], 1)
        self.assertEqual(out["stop_depth_decomposition"], {"opposite_side_at_k": 1})
        self.assertEqual(out["continuation"]["fraction_continued_posterior_classifies_factual"], 1.0)


class TestIntegrity(unittest.TestCase):
    def test_verify_digests_lists_every_failure(self):
        with self.assertRaises(A.ArtifactDigestMismatch) as ctx:
            A.verify_digests({"a": "1", "b": "x"}, {"a": "1", "b": "2", "c": "3"})
        self.assertIn("b:", str(ctx.exception))
        self.assertIn("c:", str(ctx.exception))
        A.verify_digests({"a": "1"}, {"a": "1"})

    def test_sha256_path(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"abc")
            f.flush()
            self.assertEqual(A.sha256_path(f.name),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_replay_row_is_exact(self):
        saved = {"p_factual": "0.1", "prediction": "0", "retrieved_documents": "3",
                 "nli_span_calls": "9", "subclaim_documents_used": "[3]",
                 "subclaim_nli_calls": "[9]"}
        replayed = {"p_factual": 0.1, "prediction": 0, "documents_used": 3, "nli_calls": 9,
                    "subclaim_documents_used": [3], "subclaim_nli_calls": [9]}
        self.assertEqual(A.verify_replay_row(saved, replayed), [])
        off = dict(replayed, p_factual=0.1 + 1e-15)
        self.assertEqual(len(A.verify_replay_row(saved, off)), 1)
        off = dict(replayed, subclaim_documents_used=[2])
        self.assertIn("subclaim_documents_used", A.verify_replay_row(saved, off)[0])

    def test_pair_rows_refuses_disagreeing_gold(self):
        row = {"passage_index": "0", "sentence_index": "0", "gold_label": "1",
               "subclaim_documents_available": "[10]", "n_subclaims": "1",
               "prediction": "1", "p_factual": "0.9", "retrieved_documents": "1",
               "nli_span_calls": "3", "subclaim_documents_used": "[1]"}
        rows = [dict(row, method="bse_official"), dict(row, method="frozen_ddre", gold_label="0")]
        with self.assertRaises(ValueError):
            A.pair_prediction_rows(rows)


def synthetic_report():
    gold, base, comp = [0, 1, 1], [1, 1, 0], [0, 1, 0]
    sentence = {"key": (0, 0), "gold": 1, "subclaims": [subclaim([0.6, 0.5, 0.9], [-0.8, -0.8])]}
    group = A.group_trajectory_summary([sentence], lower=0.2, log_ratio=lambda x: 0.1, **C)
    return {
        "replay": {"sentences": 3},
        "sentence_level": {
            "bse": A.class_metrics(gold, base), "ddre": A.class_metrics(gold, comp),
            "paired": A.paired_correctness(gold, base, comp),
            "efficiency_by_label": A.efficiency_by_label(gold, [5, 5, 5], [3, 4, 5],
                                                         [9, 9, 9], [6, 8, 9]),
        },
        "evidence": {"modal_score_weights": [{
            "score": 35.0, "bse_histogram_llr": -0.143, "frozen_ddre_log_ratio": -0.441,
            "production_log_ratio": 0.924, "pairs_positive": 17, "pairs_total": 20}]},
        "trajectories": {"regressions": group},
        "long_chains": A.long_chain_summary([sentence], **C),
    }


class TestRenderedSummary(unittest.TestCase):
    def test_summary_is_labelled_and_associational(self):
        text = A.render_summary(synthetic_report())
        self.assertIn("descriptive / post-hoc / exploratory", text)
        self.assertIn("strongly associated with", text)
        for phrase in A.FORBIDDEN_CAUSAL_PHRASES:
            self.assertNotIn(phrase, text.lower())

    def test_causal_phrasing_is_refused(self):
        report = synthetic_report()
        report["trajectories"] = {"savings come from evidence": report["trajectories"]["regressions"]}
        with self.assertRaises(ValueError):
            A.render_summary(report)


if __name__ == "__main__":
    unittest.main()
