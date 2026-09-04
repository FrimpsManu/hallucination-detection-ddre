"""Sensitivity analysis: could the two unreleased NBC examples close the gap?

Wang et al. report sampling s = 200 factual and s = 200 nonfactual examples to
estimate the NBC feature likelihoods. The released ``NBC_positive.json`` and
``NBC_negative.json`` contain 199 each. Two examples -- one per class -- are
therefore described in the paper but absent from the released artifacts, and
every histogram this repository builds is missing them.

This module asks one bounded question: **could those two missing examples,
whatever bins they fall in, plausibly account for the remaining Gate 1
CM=14/CFA=24 reproduction gap?**

It is a sensitivity analysis and nothing else. It enumerates all 10 x 10 = 100
ways one extra positive example and one extra negative example could be
distributed across the ten discretized bins, and reports how the frozen Gate 1
verdict responds. It does **not** identify the true bins, and it must never be
used to select a favourable histogram: the released data remain the baseline.

Interpretation, fixed in advance so the result cannot be re-read after the fact:

* **Zero combinations reproduce CM=14/CFA=24** -> the missing 200th examples
  cannot explain the remaining discrepancy under this model. The gap has
  another source.
* **Some combinations reproduce it** -> the unreleased examples are a
  *plausible* explanation. That is all. It does not identify their true bins,
  and it is not evidence that any particular combination is the real one.

Standard library only, so the enumeration and summary logic are testable
without torch, a model, or the released data.
"""

# The Laplace-smoothed NBC histograms produced by the corrected Wang-fidelity
# scorer on the released 199 + 199 examples. Recorded here as the fixed starting
# point of the sensitivity analysis; this module never recomputes them.
RELEASED_POSITIVE_HISTOGRAM = (1, 78, 19, 13, 6, 5, 28, 57, 1, 1)
RELEASED_NEGATIVE_HISTOGRAM = (1, 141, 38, 15, 4, 1, 3, 4, 1, 1)

N_BINS = 10

# Wang et al.'s stated protocol, and what the released files actually hold.
PAPER_EXAMPLES_PER_CLASS = 200
RELEASED_EXAMPLES_PER_CLASS = 199
MISSING_EXAMPLES_PER_CLASS = PAPER_EXAMPLES_PER_CLASS - RELEASED_EXAMPLES_PER_CLASS

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_INCOMPLETE = "INCOMPLETE"


def raw_count(histogram):
    """Undo Laplace smoothing: the number of real examples behind a histogram."""
    return sum(histogram) - N_BINS


def add_one_to_bin(histogram, bin_index):
    """Place one additional observed example in ``bin_index``.

    The released histograms are already Laplace-smoothed (+1 per bin), and
    smoothing is additive, so one extra *observed* example raises exactly one
    smoothed bin by exactly one. No re-smoothing is applied, and the input is
    never mutated.
    """
    if not 0 <= bin_index < N_BINS:
        raise IndexError(f"bin_index must be in [0, {N_BINS}), got {bin_index}")
    updated = list(histogram)
    updated[bin_index] += 1
    return updated


def enumerate_combinations(n_bins=N_BINS):
    """All (positive bin, negative bin) placements for the two missing examples."""
    return [(p, n) for p in range(n_bins) for n in range(n_bins)]


def combination_histograms(positive_bin, negative_bin):
    """The two histograms implied by one placement of the missing examples."""
    return (
        add_one_to_bin(RELEASED_POSITIVE_HISTOGRAM, positive_bin),
        add_one_to_bin(RELEASED_NEGATIVE_HISTOGRAM, negative_bin),
    )


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _verdict(row, configuration):
    return (row.get("configurations") or {}).get(configuration, {}).get("verdict")


def summarize(rows, primary="CM_14_CFA_24", secondary="CM_28_CFA_96"):
    """Count how the frozen Gate 1 verdicts respond across the grid.

    ``rows`` are per-combination records carrying a ``configurations`` mapping
    of configuration name to the block returned by
    ``src.reproduction_gate.evaluate_configuration``.

    Combinations whose evaluation could not be completed -- typically because a
    required document score was absent from the cache -- are counted separately
    and never silently folded into a PASS or FAIL tally.
    """
    complete = [row for row in rows if not row.get("incomplete")]
    incomplete = [row for row in rows if row.get("incomplete")]

    primary_pass = [r for r in complete if _verdict(r, primary) == STATUS_PASS]
    primary_warn = [r for r in complete if _verdict(r, primary) == STATUS_WARN]
    primary_fail = [r for r in complete if _verdict(r, primary) == STATUS_FAIL]
    secondary_pass = [r for r in complete if _verdict(r, secondary) == STATUS_PASS]
    both_pass = [
        r
        for r in complete
        if _verdict(r, primary) == STATUS_PASS
        and _verdict(r, secondary) == STATUS_PASS
    ]

    return {
        "combinations_evaluated": len(rows),
        "combinations_complete": len(complete),
        "combinations_incomplete": len(incomplete),
        "incomplete_combinations": [
            {
                "positive_bin": r["positive_bin"],
                "negative_bin": r["negative_bin"],
                "reason": r.get("incomplete_reason"),
            }
            for r in incomplete
        ],
        "primary_configuration": primary,
        "secondary_configuration": secondary,
        f"{primary}_pass": len(primary_pass),
        f"{primary}_warn_not_fail": len(primary_warn),
        f"{primary}_fail": len(primary_fail),
        f"{secondary}_pass": len(secondary_pass),
        "both_configurations_pass": len(both_pass),
        "bins_that_make_primary_pass": sorted(
            (r["positive_bin"], r["negative_bin"]) for r in primary_pass
        ),
        "bins_where_both_pass": sorted(
            (r["positive_bin"], r["negative_bin"]) for r in both_pass
        ),
    }


def metric_distance(row, configuration):
    """Sum of |signed_delta| over the six compared metrics, or None.

    A crude scalar for ranking only. Metrics are on different scales, so this
    orders candidates for inspection; it is never used to decide a verdict, and
    the frozen per-metric tolerances remain the only pass/fail authority.
    """
    block = (row.get("configurations") or {}).get(configuration)
    if not block:
        return None
    total = 0.0
    for metric_row in block["metrics"]:
        delta = metric_row.get("absolute_delta")
        if delta is None:
            return None
        total += float(delta)
    return total


def closest_combinations(rows, configuration="CM_14_CFA_24", limit=5):
    """The combinations landing nearest the published values, for inspection."""
    scored = []
    for row in rows:
        if row.get("incomplete"):
            continue
        distance = metric_distance(row, configuration)
        if distance is None:
            continue
        scored.append(
            {
                "positive_bin": row["positive_bin"],
                "negative_bin": row["negative_bin"],
                "summed_absolute_metric_delta": distance,
                "verdict": _verdict(row, configuration),
                "metrics": {
                    m["metric"]: {
                        "published": m["published"],
                        "reproduced": m["reproduced"],
                        "signed_delta": m["signed_delta"],
                        "status": m["status"],
                    }
                    for m in row["configurations"][configuration]["metrics"]
                },
            }
        )
    scored.sort(key=lambda entry: entry["summed_absolute_metric_delta"])
    return scored[:limit]


def interpretation(summary, primary="CM_14_CFA_24"):
    """The predeclared reading of the result."""
    passes = summary.get(f"{primary}_pass", 0)
    incomplete = summary.get("combinations_incomplete", 0)

    if incomplete:
        caveat = (
            f" {incomplete} of {summary['combinations_evaluated']} combinations "
            "could not be evaluated (a required document score was not "
            "available), so this conclusion covers only the completed ones."
        )
    else:
        caveat = ""

    if summary.get("combinations_complete", 0) == 0:
        # No completed combination means nothing was measured. Reporting
        # "cannot explain the gap" here would be a conclusion drawn from an
        # empty sample.
        return {
            "headline": "INCONCLUSIVE_INSUFFICIENT_CACHE_COVERAGE",
            "message": (
                "No combination could be evaluated: every one required a "
                "document score that was absent from the cache. Nothing has "
                "been established about the missing 200th examples either way. "
                "Replay against a cache that covers the documents these "
                "histograms cause the policy to retrieve."
            ),
            "combinations_reproducing_primary": 0,
            "is_sensitivity_analysis_only": True,
            "does_not_identify_true_bins": True,
            "released_data_remain_the_baseline": True,
        }

    if passes == 0:
        headline = "MISSING_EXAMPLES_CANNOT_EXPLAIN_THE_GAP"
        message = (
            "Zero combinations reproduce the published CM=14/CFA=24 values within "
            "the frozen tolerances. Under this model -- one additional observed "
            "example per class, placed in any bin -- the missing 200th examples "
            "cannot explain the remaining discrepancy. The gap has another "
            "source." + caveat
        )
    else:
        headline = "MISSING_EXAMPLES_ARE_A_PLAUSIBLE_EXPLANATION"
        message = (
            f"{passes} of {summary['combinations_complete']} completed "
            "combinations reproduce the published CM=14/CFA=24 values within the "
            "frozen tolerances. This establishes only that the unreleased "
            "examples are a PLAUSIBLE explanation of the remaining discrepancy. "
            "It does NOT identify their true bins, and no combination here may "
            "be adopted as the baseline: the released 199+199 data remain the "
            "experiment's NBC input." + caveat
        )
    return {
        "headline": headline,
        "message": message,
        "combinations_reproducing_primary": passes,
        "is_sensitivity_analysis_only": True,
        "does_not_identify_true_bins": True,
        "released_data_remain_the_baseline": True,
    }
