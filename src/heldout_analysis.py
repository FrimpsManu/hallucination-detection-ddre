"""Descriptive analysis of the frozen held-out run. Pure functions, numpy only.

Everything here describes ONE completed experiment (the frozen held-out
comparison of BSE official against the frozen DDRE configuration). It is
post-hoc and exploratory: the held-out set had already been evaluated when this
analysis was designed. Nothing here selects a configuration, scores evidence,
or establishes a mechanism. Association is reported; causation is left to the
pre-registered ablation.

The I/O, the cache replay and the artifact checks live in
``scripts/analyze_frozen_heldout.py``. This module takes plain Python values so
every computation is unit-testable without torch, a model, or the Wang data.
"""

import hashlib
import json
import math
from collections import Counter

import numpy as np

from src.baseline_core import cost_based_prediction

ANALYSIS_NAME = "frozen-heldout-descriptive-analysis"
ANALYSIS_VERSION = "v1"
ANALYSIS_STATUS = "descriptive / post-hoc / exploratory"
INTERPRETATION_RULES = (
    "Every quantity describes the single completed frozen held-out run.",
    "The analysis was designed after the held-out results were seen. It is "
    "post-hoc and exploratory, not confirmatory.",
    "Reported patterns are associations. No statement here establishes why "
    "DDRE uses fewer documents; the pre-registered ablation is the test.",
    "No intervals are attached to these breakdowns. Small cells are small.",
    "The continuation counterfactual only uses documents BSE happened to "
    "retrieve, so it covers a selected subset of subclaims.",
)

# The rendered summary must not slip into causal language. Tested.
FORBIDDEN_CAUSAL_PHRASES = (
    "come from", "comes from", "caused by", "causes", "because of",
    "due to", "proves", "proof that", "explains why", "is responsible for",
)

LONG_CHAIN_MIN_BSE_DEPTH = 8
SCORE_BINS = (0.0, 1.0, 5.0, 20.0, 50.0, 80.0, 95.0, 100.0001)


class ReplayMismatch(AssertionError):
    """The replay did not reproduce the canonical artifact. Analysis refused."""


class ArtifactDigestMismatch(AssertionError):
    """An input artifact is not the one this analysis is pinned to."""


# ------------------------------------------------------------------ integrity

def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_digests(observed, expected):
    """Every expected digest must be present and equal. Lists all failures."""
    problems = [
        f"{name}: expected {want}, observed {observed.get(name)}"
        for name, want in sorted(expected.items())
        if observed.get(name) != want
    ]
    if problems:
        raise ArtifactDigestMismatch(
            "input artifacts do not match the pinned digests:\n  "
            + "\n  ".join(problems)
        )


def verify_replay_row(saved, replayed):
    """Exact comparison of one saved CSV row against a replayed result.

    ``saved`` is the CSV row dict; ``replayed`` carries p_factual, prediction,
    documents_used, nli_calls and the per-subclaim document/NLI vectors.
    Floats are compared for exact equality: the replay reads the same cached
    scores through the same code, so any difference is a real difference.
    """
    problems = []
    checks = (
        ("p_factual", float(saved["p_factual"]), float(replayed["p_factual"])),
        ("prediction", int(saved["prediction"]), int(replayed["prediction"])),
        ("retrieved_documents", int(saved["retrieved_documents"]),
         int(replayed["documents_used"])),
        ("nli_span_calls", int(saved["nli_span_calls"]), int(replayed["nli_calls"])),
        ("subclaim_documents_used", json.loads(saved["subclaim_documents_used"]),
         list(replayed["subclaim_documents_used"])),
        ("subclaim_nli_calls", json.loads(saved["subclaim_nli_calls"]),
         list(replayed["subclaim_nli_calls"])),
    )
    for field, want, got in checks:
        if want != got:
            problems.append(f"{field}: saved {want!r}, replayed {got!r}")
    return problems


# ------------------------------------------------------------ predictions CSV

def pair_prediction_rows(rows, methods=("bse_official", "frozen_ddre")):
    """Group CSV rows into per-sentence pairs, in (passage, sentence) order."""
    grouped = {}
    for row in rows:
        key = (int(row["passage_index"]), int(row["sentence_index"]))
        grouped.setdefault(key, {})[row["method"]] = row
    paired = []
    for key in sorted(grouped):
        entry = grouped[key]
        missing = [m for m in methods if m not in entry]
        if missing:
            raise ValueError(f"sentence {key} lacks method rows {missing}")
        a, b = (entry[m] for m in methods)
        if int(a["gold_label"]) != int(b["gold_label"]):
            raise ValueError(f"sentence {key}: gold labels disagree between methods")
        if a["subclaim_documents_available"] != b["subclaim_documents_available"]:
            raise ValueError(f"sentence {key}: available documents disagree")
        paired.append({
            "key": key,
            "gold": int(a["gold_label"]),
            "n_subclaims": int(a["n_subclaims"]),
            **{
                m: {
                    "prediction": int(r["prediction"]),
                    "p_factual": float(r["p_factual"]),
                    "documents": int(r["retrieved_documents"]),
                    "nli_calls": int(r["nli_span_calls"]),
                    "subclaim_depths": json.loads(r["subclaim_documents_used"]),
                    "subclaim_available": json.loads(r["subclaim_documents_available"]),
                }
                for m, r in zip(methods, (a, b))
            },
        })
    return paired


# ------------------------------------------------------------ sentence-level

def class_metrics(gold, pred):
    """Confusion matrix and per-class precision/recall/F1. 0 = hallucinated."""
    gold = np.asarray(gold, dtype=int)
    pred = np.asarray(pred, dtype=int)

    def per_class(c):
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}

    nonfactual, factual = per_class(0), per_class(1)
    return {
        "confusion_rows_gold_cols_pred_0_hallucinated_1_factual": [
            [int(np.sum((gold == g) & (pred == p))) for p in (0, 1)] for g in (0, 1)
        ],
        "predicted_factual": int(np.sum(pred == 1)),
        "gold_factual": int(np.sum(gold == 1)),
        "nonfactual": nonfactual,
        "factual": factual,
        "macro_f1": 0.5 * (nonfactual["f1"] + factual["f1"]),
    }


def paired_correctness(gold, pred_a, pred_b):
    """Who is right where. ``a`` is the baseline, ``b`` the comparison."""
    gold = np.asarray(gold, dtype=int)
    ra = np.asarray(pred_a, dtype=int) == gold
    rb = np.asarray(pred_b, dtype=int) == gold
    out = {}
    for name, mask in (("all", np.ones_like(gold, bool)),
                       ("gold_hallucinated", gold == 0),
                       ("gold_factual", gold == 1)):
        out[name] = {
            "both_right": int(np.sum(ra & rb & mask)),
            "baseline_only_right": int(np.sum(ra & ~rb & mask)),
            "comparison_only_right": int(np.sum(~ra & rb & mask)),
            "both_wrong": int(np.sum(~ra & ~rb & mask)),
        }
    flips = Counter(
        f"gold={g} baseline={a} comparison={b}"
        for g, a, b in zip(gold, pred_a, pred_b) if a != b
    )
    out["prediction_changes"] = dict(sorted(flips.items()))
    return out


def efficiency_by_label(gold, docs_a, docs_b, nli_a, nli_b):
    gold = np.asarray(gold, dtype=int)
    docs_a, docs_b = np.asarray(docs_a, float), np.asarray(docs_b, float)
    nli_a, nli_b = np.asarray(nli_a, float), np.asarray(nli_b, float)
    total_saving = float(np.sum(docs_a - docs_b))
    out = {}
    for name, g in (("gold_hallucinated", 0), ("gold_factual", 1)):
        m = gold == g
        out[name] = {
            "sentences": int(m.sum()),
            "docs_per_sentence_baseline": float(docs_a[m].mean()),
            "docs_per_sentence_comparison": float(docs_b[m].mean()),
            "doc_reduction_fraction": float(1 - docs_b[m].sum() / docs_a[m].sum()),
            "nli_reduction_fraction": float(1 - nli_b[m].sum() / nli_a[m].sum()),
            "share_of_total_doc_saving": (
                float(np.sum(docs_a[m] - docs_b[m]) / total_saving) if total_saving else None
            ),
        }
    return out


def saving_concentration(docs_a, docs_b, fractions=(0.10, 0.25, 0.50)):
    """How many sentences carry a given fraction of the net document saving."""
    diff = np.asarray(docs_a, float) - np.asarray(docs_b, float)
    total = diff.sum()
    if total <= 0:
        return {"net_saving": float(total), "sentences_for_fraction": None}
    cum = np.cumsum(np.sort(diff)[::-1]) / total
    return {
        "net_saving": float(total),
        "sentences": int(diff.size),
        "cheaper": int(np.sum(diff > 0)),
        "equal": int(np.sum(diff == 0)),
        "more_expensive": int(np.sum(diff < 0)),
        "sentences_for_fraction": {
            f"{f:.2f}": int(np.searchsorted(cum, f) + 1) for f in fractions
        },
    }


def saving_by_correctness(gold, pred_a, pred_b, docs_a, docs_b):
    gold = np.asarray(gold, int)
    ra = np.asarray(pred_a, int) == gold
    rb = np.asarray(pred_b, int) == gold
    diff = np.asarray(docs_a, float) - np.asarray(docs_b, float)
    total = diff.sum()
    out = {}
    for name, m in (("both_right", ra & rb), ("baseline_only_right", ra & ~rb),
                    ("comparison_only_right", ~ra & rb), ("both_wrong", ~ra & ~rb)):
        out[name] = {
            "sentences": int(m.sum()),
            "mean_doc_saving": float(diff[m].mean()) if m.any() else None,
            "share_of_net_saving": float(diff[m].sum() / total) if total else None,
        }
    return out


def posterior_placement(gold, p_factual, lower, cut, upper):
    """Where sentence posteriors land relative to the band and the cost cut."""
    gold = np.asarray(gold, int)
    p = np.asarray(p_factual, float)
    out = {}
    for name, g in (("gold_hallucinated", 0), ("gold_factual", 1)):
        x = p[gold == g]
        if not x.size:
            out[name] = None
            continue
        out[name] = {
            f"p<={lower}": float(np.mean(x <= lower)),
            f"{lower}<p<={cut:.6f}": float(np.mean((x > lower) & (x <= cut))),
            f"{cut:.6f}<p<{upper}": float(np.mean((x > cut) & (x < upper))),
            f"p>={upper}": float(np.mean(x >= upper)),
        }
    return out


def recall_by_subclaim_count(gold, pred_a, pred_b, n_subclaims, cap=3):
    gold = np.asarray(gold, int)
    n = np.asarray(n_subclaims, int)
    pa, pb = np.asarray(pred_a, int), np.asarray(pred_b, int)
    out = {}
    for k in range(1, cap + 1):
        sel = (gold == 1) & ((n == k) if k < cap else (n >= cap))
        out[f"{k}" if k < cap else f">={cap}"] = {
            "gold_factual_sentences": int(sel.sum()),
            "factual_recall_baseline": float(np.mean(pa[sel] == 1)) if sel.any() else None,
            "factual_recall_comparison": float(np.mean(pb[sel] == 1)) if sel.any() else None,
        }
    return out


# ------------------------------------------------------------ subclaim depth

def depth_joint_table(depth_a, depth_b, max_docs=10):
    """Joint stopping depths on the same subclaims, and saving per baseline depth."""
    a = np.asarray(depth_a, int)
    b = np.asarray(depth_b, int)
    table = np.zeros((max_docs + 1, max_docs + 1), int)
    for x, y in zip(a, b):
        table[x, y] += 1
    total = int(np.sum(a - b))
    by_row = {}
    for d in range(max_docs + 1):
        m = a == d
        if m.any():
            saving = int(np.sum(a[m] - b[m]))
            by_row[str(d)] = {
                "subclaims": int(m.sum()),
                "doc_saving": saving,
                "share_of_saving": saving / total if total else None,
            }
    return {
        "subclaims": int(a.size),
        "comparison_earlier": float(np.mean(b < a)),
        "same_depth": float(np.mean(b == a)),
        "comparison_later": float(np.mean(b > a)),
        "total_doc_saving": total,
        "joint_counts_rows_baseline_cols_comparison": table.tolist(),
        "saving_by_baseline_depth": by_row,
        "budget_hits_baseline": int(np.sum(a == max_docs)),
        "budget_hits_comparison": int(np.sum(b == max_docs)),
    }


def long_chain_share(depth_a, depth_b, min_depth=LONG_CHAIN_MIN_BSE_DEPTH):
    a = np.asarray(depth_a, int)
    b = np.asarray(depth_b, int)
    m = a >= min_depth
    total = np.sum(a - b)
    return {
        "min_baseline_depth": min_depth,
        "subclaims": int(m.sum()),
        "fraction_of_subclaims": float(m.mean()),
        "share_of_saving": float(np.sum(a[m] - b[m]) / total) if total else None,
        "comparison_depth_mean": float(b[m].mean()) if m.any() else None,
        "comparison_depth_median": float(np.median(b[m])) if m.any() else None,
    }


# ------------------------------------------------------------ evidence functions

def flip_region(sign_matrix):
    """True where the sign of log r is not unanimous across the surface.

    ``sign_matrix`` is (n_hyperparameter_pairs, n_grid_points) of log-ratio
    signs. A point is stable only if every pair is strictly positive or every
    pair is strictly negative; a zero anywhere counts as not unanimous.
    """
    s = np.asarray(sign_matrix)
    return ~(np.all(s > 0, axis=0) | np.all(s < 0, axis=0))


def region_boundaries(grid, mask):
    grid = np.asarray(grid, float)
    mask = np.asarray(mask, bool)
    return [float(grid[i]) for i in range(1, len(grid)) if mask[i] != mask[i - 1]]


def evidence_table(grid, bins, functions, flip_mask):
    """Mean per-document evidence (log factual/hallucinated) in score bins.

    ``functions`` maps a name to a callable score -> log-evidence.
    """
    grid = np.asarray(grid, float)
    flip_mask = np.asarray(flip_mask, bool)
    rows = []
    for lo, hi in zip(bins, bins[1:]):
        m = (grid >= lo) & (grid < hi)
        if not m.any():
            continue
        xs = grid[m]
        row = {"score_bin": [float(lo), float(min(hi, 100.0))]}
        for name, fn in functions.items():
            row[name] = float(np.mean([fn(float(x)) for x in xs]))
        row["flip_region_fraction"] = float(np.mean(flip_mask[m]))
        rows.append(row)
    return rows


def score_bin_fractions(scores, bins=SCORE_BINS):
    scores = np.asarray(scores, float)
    counts = np.histogram(scores, bins=bins)[0]
    return {
        "bins": [[float(a), float(min(b, 100.0))] for a, b in zip(bins, bins[1:])],
        "fractions": (counts / max(scores.size, 1)).tolist(),
        "n": int(scores.size),
        "median": float(np.median(scores)) if scores.size else None,
        "quartiles": (
            [float(np.percentile(scores, 25)), float(np.percentile(scores, 75))]
            if scores.size else None
        ),
    }


# ------------------------------------------------------------ trajectories
#
# A subclaim trace is a dict:
#   bse:        [{"score", "bucket", "llr", "p"}, ...]   documents BSE consumed
#   bse_p:      final BSE posterior
#   ddre:       [{"score", "log_ratio", "cum_log_odds", "p", "flip_region"}, ...]
#   ddre_p:     final DDRE posterior
#   ddre_stop:  "lower" | "upper" | "budget" | "exhausted"

def classifies_factual(p, c_miss, c_false_alarm):
    return cost_based_prediction(p, c_miss, c_false_alarm) == 1


def responsible_subclaims(subclaims, c_miss, c_false_alarm):
    """Subclaims whose DDRE posterior classifies hallucinated while BSE's does not.

    In a sentence BSE calls factual and DDRE calls hallucinated, these are the
    subclaims whose minimum moved the sentence across the cost cut.
    """
    return [
        s for s in subclaims
        if classifies_factual(s["bse_p"], c_miss, c_false_alarm)
        and not classifies_factual(s["ddre_p"], c_miss, c_false_alarm)
    ]


def stop_depth_decomposition(subclaim, c_miss, c_false_alarm):
    """Compare BSE's posterior after the same documents at which DDRE stopped.

    k = DDRE's stopping depth.
      ``baseline_stopped_first``: BSE stopped before k (on the factual side,
          for a responsible subclaim). DDRE continued and ended hallucinated.
      ``same_side_at_k``: after the same k documents BSE also classified
          hallucinated; BSE then kept retrieving and finished factual. The two
          evidence models agree at k; only the continuation rule differs.
      ``opposite_side_at_k``: after the same k documents BSE still classified
          factual. The two evidence models disagree on the same documents.
    Descriptive. It locates where the two runs diverged; it does not say why.
    """
    k = len(subclaim["ddre"])
    if len(subclaim["bse"]) < k:
        return "baseline_stopped_first"
    p_bse_at_k = subclaim["bse"][k - 1]["p"]
    if classifies_factual(p_bse_at_k, c_miss, c_false_alarm):
        return "opposite_side_at_k"
    return "same_side_at_k"


def evidence_drivers(subclaim, lower):
    """Was a DDRE stop driven by one document or by accumulation?"""
    lr = [step["log_ratio"] for step in subclaim["ddre"]]
    negative_total = sum(x for x in lr if x < 0)
    crossing = math.log(lower / (1.0 - lower))
    return {
        "depth": len(lr),
        "first_document_alone_crosses_lower": bool(lr and lr[0] <= crossing),
        "largest_single_share_of_negative": (
            min(lr) / negative_total if negative_total < 0 else 0.0
        ),
        "documents_toward_factual": sum(x > 0 for x in lr),
        "documents_toward_hallucinated": sum(x < 0 for x in lr),
    }


def continuation_outcome(subclaim, log_ratio, c_miss, c_false_alarm):
    """Continue DDRE's log-odds over documents BSE retrieved beyond DDRE's stop.

    Returns None when BSE did not go further (no cached evidence to continue
    over). Otherwise whether the continued posterior would classify factual.
    Uses cached scores only; it is a counterfactual over a selected subset.
    """
    k = len(subclaim["ddre"])
    extra = subclaim["bse"][k:]
    if not extra:
        return None
    log_odds = subclaim["ddre"][-1]["cum_log_odds"]
    for step in extra:
        log_odds += log_ratio(step["score"])
    p = 1.0 / (1.0 + math.exp(-log_odds))
    return classifies_factual(p, c_miss, c_false_alarm)


def group_trajectory_summary(sentences, *, lower, c_miss, c_false_alarm, log_ratio):
    subs = [s for sentence in sentences
            for s in responsible_subclaims(sentence["subclaims"], c_miss, c_false_alarm)]
    drivers = [evidence_drivers(s, lower) for s in subs]
    scores = [step["score"] for s in subs for step in s["ddre"]]
    flips = [step["flip_region"] for s in subs for step in s["ddre"]]
    final_p = [s["ddre_p"] for s in subs]
    cont = [c for c in (continuation_outcome(s, log_ratio, c_miss, c_false_alarm)
                        for s in subs) if c is not None]
    per_sentence = Counter(
        len(responsible_subclaims(x["subclaims"], c_miss, c_false_alarm)) for x in sentences
    )
    return {
        "sentences": len(sentences),
        "responsible_subclaims": len(subs),
        "responsible_per_sentence": {str(k): v for k, v in sorted(per_sentence.items())},
        "ddre_stop_reason": dict(Counter(s["ddre_stop"] for s in subs)),
        "stop_depth_decomposition": dict(
            Counter(stop_depth_decomposition(s, c_miss, c_false_alarm) for s in subs)
        ),
        "ddre_depth_mean": float(np.mean([d["depth"] for d in drivers])) if drivers else None,
        "ddre_depth_distribution": {
            str(k): v for k, v in sorted(Counter(d["depth"] for d in drivers).items())
        },
        "first_document_alone_crosses_lower_fraction": (
            float(np.mean([d["first_document_alone_crosses_lower"] for d in drivers]))
            if drivers else None
        ),
        "largest_single_share_of_negative_median": (
            float(np.median([d["largest_single_share_of_negative"] for d in drivers]))
            if drivers else None
        ),
        "fraction_with_any_document_toward_factual": (
            float(np.mean([d["documents_toward_factual"] > 0 for d in drivers]))
            if drivers else None
        ),
        "consumed_scores": score_bin_fractions(scores),
        "consumed_in_flip_region_fraction": float(np.mean(flips)) if flips else None,
        "final_ddre_p_median": float(np.median(final_p)) if final_p else None,
        "final_ddre_p_within_0.15_to_lower": (
            float(np.mean([(0.15 < p <= lower) for p in final_p])) if final_p else None
        ),
        "continuation": {
            "subclaims_where_bse_retrieved_further": len(cont),
            "fraction_continued_posterior_classifies_factual": (
                float(np.mean(cont)) if cont else None
            ),
            "caveat": INTERPRETATION_RULES[4],
        },
    }


def long_chain_summary(sentences, *, c_miss, c_false_alarm,
                       min_depth=LONG_CHAIN_MIN_BSE_DEPTH, upper=0.8):
    pairs = [(x, s) for x in sentences for s in x["subclaims"] if len(s["bse"]) >= min_depth]
    cut_side = lambda p: classifies_factual(p, c_miss, c_false_alarm)
    cut_short = [s for _, s in pairs if len(s["ddre"]) < len(s["bse"])]
    hesitation = [
        np.mean([cut_side(st["p"]) and st["p"] < upper for st in s["bse"]]) for _, s in pairs
    ]
    return {
        "min_bse_depth": min_depth,
        "subclaims": len(pairs),
        "gold_hallucinated_fraction": (
            float(np.mean([x["gold"] == 0 for x, _ in pairs])) if pairs else None
        ),
        "ddre_stop_reason": dict(Counter(s["ddre_stop"] for _, s in pairs)),
        "final_side_counts": {
            f"bse_factual={a} ddre_factual={b}": n
            for (a, b), n in sorted(Counter(
                (cut_side(s["bse_p"]), cut_side(s["ddre_p"])) for _, s in pairs
            ).items())
        },
        "bse_steps_between_cut_and_upper_fraction": (
            float(np.mean(hesitation)) if hesitation else None
        ),
        "bse_posterior_at_ddre_stop": dict(Counter(
            "classifies_factual" if cut_side(s["bse"][len(s["ddre"]) - 1]["p"])
            else "classifies_hallucinated" for s in cut_short
        )),
        "most_common_bse_bucket_prefixes": [
            [list(prefix), n] for prefix, n in Counter(
                tuple(st["bucket"] for st in s["bse"][:4]) for _, s in pairs
            ).most_common(5)
        ],
    }


# ------------------------------------------------------------ rendering

def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_summary(report):
    """Human-readable summary. Descriptive wording only; tested for that."""
    s = report["sentence_level"]
    t = report["trajectories"]
    w = report["evidence"]["modal_score_weights"]
    lc = report["long_chains"]
    eff = s["efficiency_by_label"]
    lines = [
        "# Frozen held-out run: descriptive analysis",
        "",
        f"**Status: {ANALYSIS_STATUS}.** "
        "Every number describes the single completed frozen held-out run. "
        "Patterns are associations; none establishes a mechanism.",
        "",
        "## Replay",
        "",
        f"- {report['replay']['sentences']} sentences x 2 methods replayed from the "
        "read-only cache reproduce the canonical predictions CSV exactly.",
        "- Input artifacts were verified against pinned SHA-256 digests and were "
        "byte-identical before and after the analysis.",
        "",
        "## Sentence level",
        "",
        f"- Predicted factual: BSE {s['bse']['predicted_factual']}, DDRE "
        f"{s['ddre']['predicted_factual']} (gold {s['bse']['gold_factual']}).",
        f"- Factual recall: BSE {_pct(s['bse']['factual']['recall'])}, DDRE "
        f"{_pct(s['ddre']['factual']['recall'])}. Nonfactual recall: BSE "
        f"{_pct(s['bse']['nonfactual']['recall'])}, DDRE "
        f"{_pct(s['ddre']['nonfactual']['recall'])}.",
        f"- Prediction changes between the methods: {s['paired']['prediction_changes']}.",
        f"- Document reduction on gold-hallucinated sentences "
        f"{_pct(eff['gold_hallucinated']['doc_reduction_fraction'])}, on gold-factual "
        f"{_pct(eff['gold_factual']['doc_reduction_fraction'])}.",
        "",
        "## Evidence weights at the dominant score region",
        "",
    ]
    for row in w:
        lines.append(
            f"- score {row['score']:g}: BSE histogram {row['bse_histogram_llr']:+.3f}, "
            f"frozen DDRE {row['frozen_ddre_log_ratio']:+.3f}, CV-selected production "
            f"{row['production_log_ratio']:+.3f}; {row['pairs_positive']}/"
            f"{row['pairs_total']} D-03 pairs positive."
        )
    lines += [
        "",
        "The observed savings, prediction shift, and factual-recall loss are "
        "strongly associated with how the frozen DDRE weights the low-to-moderate "
        "NLI-score region that dominates this benchmark. Whether the evidence "
        "weighting, the stopping band, or their combination is responsible is "
        "the question the pre-registered ablation addresses.",
        "",
        "## Flipped sentences",
        "",
    ]
    for name, g in t.items():
        lines.append(
            f"- {name}: {g['sentences']} sentences, {g['responsible_subclaims']} "
            f"responsible subclaims; stop-depth decomposition "
            f"{g['stop_depth_decomposition']}; first document alone crosses the lower "
            f"boundary in {_pct(g['first_document_alone_crosses_lower_fraction'])}; "
            f"consumed-score median {g['consumed_scores']['median']:.2f}; "
            f"continued posterior classifies factual in "
            f"{_pct(g['continuation']['fraction_continued_posterior_classifies_factual'])} "
            f"of {g['continuation']['subclaims_where_bse_retrieved_further']}."
        )
    lines += [
        "",
        "## Long chains",
        "",
        f"- {lc['subclaims']} subclaims where BSE used >= {lc['min_bse_depth']} "
        f"documents; DDRE stop reasons {lc['ddre_stop_reason']}; final sides "
        f"{lc['final_side_counts']}.",
        "",
        "## Interpretation rules",
        "",
    ]
    lines += [f"- {rule}" for rule in INTERPRETATION_RULES]
    text = "\n".join(lines) + "\n"
    lowered = text.lower()
    found = [p for p in FORBIDDEN_CAUSAL_PHRASES if p in lowered]
    if found:
        raise ValueError(f"summary contains causal phrasing: {found}")
    return text
