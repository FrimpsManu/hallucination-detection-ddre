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

# Every combination is evaluated once per cost configuration, and the two are
# tracked independently. A completed CM_14_CFA_24 result is a real measurement
# and must survive the secondary configuration hitting a cache miss afterwards.

def configuration_block(row, configuration):
    return (row.get("configurations") or {}).get(configuration) or {}


def is_complete(row, configuration):
    return bool(configuration_block(row, configuration).get("complete"))


def _verdict(row, configuration):
    block = configuration_block(row, configuration)
    return block.get("verdict") if block.get("complete") else None


def completed_rows(rows, configuration):
    return [row for row in rows if is_complete(row, configuration)]


def incomplete_rows(rows, configuration):
    return [row for row in rows if not is_complete(row, configuration)]


def _incomplete_entries(rows, configuration):
    return [
        {
            "positive_bin": row["positive_bin"],
            "negative_bin": row["negative_bin"],
            "reason": configuration_block(row, configuration).get("incomplete_reason"),
        }
        for row in incomplete_rows(rows, configuration)
    ]


def summarize(
    rows,
    primary="CM_14_CFA_24",
    secondary="CM_28_CFA_96",
    expected_combinations=N_BINS * N_BINS,
):
    """Count how the frozen Gate 1 verdicts respond across the grid.

    ``rows`` are per-combination records whose ``configurations`` mapping holds,
    for each cost configuration, the block returned by
    ``src.reproduction_gate.evaluate_configuration`` plus a ``complete`` flag.

    Completeness is tracked **per configuration**. A combination whose primary
    configuration evaluated cleanly counts toward the primary tallies even if the
    secondary later hit a cache miss, and vice versa. ``both_configurations_pass``
    is the only count requiring both to have completed.
    """
    primary_complete = completed_rows(rows, primary)
    secondary_complete = completed_rows(rows, secondary)

    primary_pass = [r for r in primary_complete if _verdict(r, primary) == STATUS_PASS]
    primary_warn = [r for r in primary_complete if _verdict(r, primary) == STATUS_WARN]
    primary_fail = [r for r in primary_complete if _verdict(r, primary) == STATUS_FAIL]
    secondary_pass = [
        r for r in secondary_complete if _verdict(r, secondary) == STATUS_PASS
    ]
    both_pass = [
        r
        for r in rows
        if is_complete(r, primary)
        and is_complete(r, secondary)
        and _verdict(r, primary) == STATUS_PASS
        and _verdict(r, secondary) == STATUS_PASS
    ]

    return {
        "combinations_evaluated": len(rows),
        "expected_combinations": expected_combinations,
        "grid_fully_enumerated": len(rows) == expected_combinations,
        "primary_configuration": primary,
        "secondary_configuration": secondary,
        f"{primary}_complete": len(primary_complete),
        f"{primary}_incomplete": len(rows) - len(primary_complete),
        f"{secondary}_complete": len(secondary_complete),
        f"{secondary}_incomplete": len(rows) - len(secondary_complete),
        f"{primary}_pass": len(primary_pass),
        f"{primary}_warn_not_fail": len(primary_warn),
        f"{primary}_fail": len(primary_fail),
        f"{secondary}_pass": len(secondary_pass),
        "both_configurations_pass": len(both_pass),
        "both_configurations_complete": len(
            [r for r in rows if is_complete(r, primary) and is_complete(r, secondary)]
        ),
        "incomplete_combinations": {
            primary: _incomplete_entries(rows, primary),
            secondary: _incomplete_entries(rows, secondary),
        },
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
    block = configuration_block(row, configuration)
    if not block or not block.get("complete") or not block.get("metrics"):
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


HEADLINE_PLAUSIBLE = "MISSING_EXAMPLES_ARE_A_PLAUSIBLE_EXPLANATION"
HEADLINE_CANNOT_EXPLAIN = "MISSING_EXAMPLES_CANNOT_EXPLAIN_THE_GAP"
HEADLINE_PARTIAL_COVERAGE = "INCONCLUSIVE_PARTIAL_CACHE_COVERAGE"
HEADLINE_NO_COVERAGE = "INCONCLUSIVE_INSUFFICIENT_CACHE_COVERAGE"


def interpretation(summary, primary="CM_14_CFA_24"):
    """The predeclared reading of the primary CM=14/CFA=24 result.

    Four branches, in this order:

    * **D** no primary combination completed -> nothing was measured;
    * **A** at least one completed primary PASS -> plausible. This holds even
      under partial coverage: a single reproducing placement is enough to
      establish plausibility, and unevaluated placements cannot take it away;
    * **B** zero primary PASS but some primary combinations unevaluated ->
      inconclusive. A negative claim needs the whole grid: an unevaluated
      placement could still pass, so "cannot explain" would not be supported;
    * **C** every primary combination in the full grid completed and none
      passed -> the missing examples cannot explain the gap.

    Branch C additionally requires the full grid to have been enumerated, so a
    ``--limit-combinations`` debugging run can never produce a negative
    conclusion from a truncated grid.
    """
    passes = summary.get(f"{primary}_pass", 0)
    complete = summary.get(f"{primary}_complete", 0)
    incomplete = summary.get(f"{primary}_incomplete", 0)
    evaluated = summary.get("combinations_evaluated", 0)
    expected = summary.get("expected_combinations", N_BINS * N_BINS)
    fully_enumerated = summary.get("grid_fully_enumerated", evaluated == expected)

    base = {
        "combinations_reproducing_primary": passes,
        "primary_complete": complete,
        "primary_incomplete": incomplete,
        "is_sensitivity_analysis_only": True,
        "does_not_identify_true_bins": True,
        "released_data_remain_the_baseline": True,
    }

    # D -- nothing measured at all.
    if complete == 0:
        return dict(
            base,
            headline=HEADLINE_NO_COVERAGE,
            message=(
                f"No {primary} combination could be evaluated: every one required "
                "a document score that was absent from the cache. Nothing has "
                "been established about the missing 200th examples either way. "
                "Replay against a cache that covers the documents these "
                "histograms cause the policy to retrieve."
            ),
        )

    # A -- a single reproducing placement is enough, whatever the coverage.
    if passes > 0:
        coverage = ""
        if incomplete or not fully_enumerated:
            coverage = (
                f" Coverage was partial ({complete} of {expected} placements "
                "evaluated), but that does not weaken this finding: one "
                "reproducing placement is sufficient to establish plausibility."
            )
        return dict(
            base,
            headline=HEADLINE_PLAUSIBLE,
            message=(
                f"{passes} of {complete} completed {primary} combinations "
                "reproduce the published values within the frozen tolerances. "
                "This establishes only that the unreleased examples are a "
                "PLAUSIBLE explanation of the remaining discrepancy. It does NOT "
                "identify their true bins, and no combination here may be "
                "adopted as the baseline: the released 199+199 data remain the "
                "experiment's NBC input." + coverage
            ),
        )

    # B -- zero passes, but the grid was not fully evaluated.
    if incomplete or not fully_enumerated:
        return dict(
            base,
            headline=HEADLINE_PARTIAL_COVERAGE,
            message=(
                f"None of the {complete} completed {primary} combinations "
                f"reproduces the published values, but {expected - complete} of "
                f"{expected} placements were not evaluated. A negative claim "
                "needs the whole grid: one of the unevaluated placements could "
                "still pass, so this run does NOT establish that the missing "
                "200th examples cannot explain the gap. Replay against a cache "
                "covering the remaining placements."
            ),
        )

    # C -- the full grid completed and nothing passed.
    return dict(
        base,
        headline=HEADLINE_CANNOT_EXPLAIN,
        message=(
            f"All {expected} {primary} combinations were evaluated and none "
            "reproduces the published values within the frozen tolerances. Under "
            "this model -- one additional observed example per class, placed in "
            "any bin -- the missing 200th examples cannot explain the remaining "
            "discrepancy. The gap has another source."
        ),
    )
