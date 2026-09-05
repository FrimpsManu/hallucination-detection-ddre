# Confirmatory statistical protocol

**This is a prospective protocol, not a results document.** It was written and
frozen **before any held-out DDRE result was inspected**. No number in it was
chosen after seeing a test-set outcome, and nothing here reports one.

Everything below is implemented in `src/paired_bootstrap.py` and
`src/threshold_selection.py`, and pinned by `tests/test_confirmatory_statistics.py`.
Changing any frozen constant after the held-out run would turn a confirmatory
analysis into an exploratory one; the code therefore reports such a run as
`NOT_CONFIRMATORY` rather than silently adjusting the claim rule.

---

## 1. What is being claimed

> Retrieval-aware direct density-ratio estimation (DDRE/uLSIF) **reduces the
> retrieval cost** of hallucination detection **while preserving detection
> performance**, relative to Wang et al.'s Bayesian sequential estimation.

The comparator is **published BSE official**: `CM = 28`, `CFA = 96`,
`c_retrieve = 1`, `P0 = 0.5`, `max_docs = 10`. Not a tuned variant. Both methods
consume the identical sentences, subclaims, retrieved documents in the same
order, segmentation, NLI scores, labels, aggregation rule and held-out split.

## 2. Frozen constants

| Constant | Value |
| --- | --- |
| `CONFIRMATORY_PR_AUC_MARGIN` | **0.005** (absolute PR-AUC difference) |
| `CONFIRMATORY_BOOTSTRAP_RESAMPLES` | **10,000** |
| `CONFIRMATORY_BOOTSTRAP_SEED` | **42** |
| `CONFIRMATORY_CI_LEVEL` | **0.95** |
| `CONFIRMATORY_BOOTSTRAP_UNIT` | **`"passage"`** |
| CI method | **percentile** (2.5th, 97.5th) |

The margin is an **absolute** PR-AUC difference. `DDRE − BSE ≥ −0.005` means
detection performance is non-inferior within the predeclared margin — any loss
is smaller than half a PR-AUC point.

It equals the default validation quality tolerance deliberately, so a
configuration chosen on validation is judged on test by the same yardstick. A
run whose validation tolerance differs is **`NOT_CONFIRMATORY`**; alternative
tolerances remain available for exploratory work, but they cannot quietly change
the claim rule.

## 3. Validation selection (D-07)

A threshold pair is **confirmatorily feasible** only if it preserves, each
within the validation tolerance, all three of:

* **nonfactual PR-AUC** — the hallucination-detection metric, and Wang's headline
* **factual PR-AUC**
* **balanced PR-AUC**

Nonfactual is the one that must not be omitted. Balanced PR-AUC is the *mean* of
the two class metrics, so a large factual gain can offset a nonfactual loss and
clear a balanced-only floor while the method has become worse at the task it
exists for. All three booleans and all three deltas are recorded per candidate,
so the trade-off is visible in the saved validation table rather than implied.

Among feasible candidates the objective stays **minimum retrieved documents per
sentence** — that is the hypothesis under test. Quality breaks only **exact**
document-count ties, in the order balanced → nonfactual → factual, with a final
deterministic ordering on the thresholds themselves.

**If nothing is feasible**, the predeclared fallback objective still selects a
configuration for exploratory analysis, but it is marked
`confirmatory_validation_selection = false` and **can never support the
confirmatory claim, however large the held-out effect.**

Tuning uses validation data only. The held-out split is never inspected before
the final evaluation.

## 4. Paired passage-level cluster bootstrap (D-08)

**Unit: the passage.** The held-out split has **190 passages** and 1,525
sentences. Sentences from one passage share a topic, a source document set and
an annotator pass, so they are not independent; resampling sentences would
understate the variance and narrow every interval. Each replicate draws **190
passages with replacement** and carries each drawn passage's **complete sentence
block**. A passage drawn twice contributes its block **twice** — collapsing the
duplicate would shrink the resample and destroy the variance the cluster
bootstrap exists to capture.

**Paired.** DDRE and BSE are evaluated on the identical sentence set in the
identical order, so each replicate applies the **same** passage draw to both
methods and the difference is taken within the replicate. That removes the
between-passage variance the two methods share, which is exactly the variance
irrelevant to which method is better.

**PR-AUC** is `src.evaluation.wang_pr_auc` — `precision_recall_curve` followed by
`auc(recall, precision)` — the same function the normal evaluation calls. Not
`average_precision_score`, and not a second implementation: an interval computed
on a different estimand from the point estimate it brackets is not an interval
for that point estimate.

### Sign conventions

```
performance delta = DDRE − BSE      (positive favours DDRE)
efficiency saving = BSE − DDRE      (positive favours DDRE)
```

Both are recorded on every endpoint in the output.

Efficiency is a **difference**, not a fraction. A fractional reduction divides by
a bootstrap denominator that can be very small, which makes the estimand behave
badly at exactly the replicates that matter. Fractional reductions remain useful
descriptive effect sizes and are reported separately as such.

### Intervals

Fixed **95% percentile** intervals: the 2.5th and 97.5th percentiles of the
10,000 replicate values. No other CI method is computed, before or after seeing
results.

Recorded for every endpoint: `observed`, `bootstrap_mean`, `ci_lower`,
`ci_upper`, `ci_level`, `ci_method`, `n_resamples`, `seed`, `bootstrap_unit`,
`sign_convention` — plus the number of unique held-out passages.

### Invalid replicates — fail closed

A replicate used for PR-AUC must contain **both** classes. If one does not,
`BootstrapUnavailable` is raised and the confirmatory analysis becomes
unavailable. Replicates are **never** dropped, **never** redrawn, and **never**
replaced by a substitute metric — each of those silently changes the estimand.
With 190 passages this should be extremely unlikely; the behaviour is
deterministic and predeclared regardless.

## 5. The claim rule

### Primary endpoints

**Performance non-inferiority** — all three required:

```
lower 95% CI( nonfactual PR-AUC delta ) ≥ −0.005
lower 95% CI( factual    PR-AUC delta ) ≥ −0.005
lower 95% CI( balanced   PR-AUC delta ) ≥ −0.005
```

**Retrieval-efficiency superiority** — required:

```
lower 95% CI( retrieved documents per sentence saved ) > 0
```

### The conjunction

`primary_claim_supported` is true only if **all** of:

1. `confirmatory_validation_selection == true`
2. validation quality tolerance `== 0.005`
3. all three performance non-inferiority gates pass
4. retrieval-efficiency superiority passes

### Secondary endpoint

```
nli_efficiency_supported = lower 95% CI( NLI span calls per sentence saved ) > 0
```

**Not required** for the primary Wang-compatible retrieval claim, and it changes
only the wording:

* primary passes **and** NLI passes → *the evidence supports lower retrieval AND
  lower NLI-evaluation cost with preserved detection performance*
* primary passes, NLI does not → *the evidence supports lower retrieval cost with
  preserved detection performance, but **not** a statistically supported
  reduction in NLI span calls*

## 6. Claim statuses

| Status | Meaning |
| --- | --- |
| `SUPPORTED` | Every primary gate passed. |
| `NOT_SUPPORTED` | The run **addressed** the claim and the evidence did not support it. A negative result. |
| `NOT_CONFIRMATORY` | The run **cannot address** the claim. Not a negative result, and must never be reported as one. |

`NOT_CONFIRMATORY` applies if the threshold selection used the fallback, the
validation tolerance differs from the frozen margin, the bootstrap is
unavailable or invalid, or the run is a smoke/debug run. Every disqualifier is
listed by name in the output.

**`NOT_CONFIRMATORY` is never converted into `NOT_SUPPORTED`.** They are
different scientific statements and the code keeps them apart.

## 7. Multiplicity

The primary claim is **conjunctive** — an intersection-union rule requiring
*every* primary condition to hold simultaneously. There is no selection among
endpoints to correct for: we do not report whichever endpoint happened to be
significant. Fixed 95% intervals are used throughout and **no ad-hoc
multiple-comparison correction is applied**.

**No p-values are produced.**

## 8. What point estimates are for

Observed factual, nonfactual and balanced deltas, and the fractional retrieval
and NLI reductions, are still reported — as **descriptive effect sizes**,
labelled as such. They carry no uncertainty and **cannot set
`primary_claim_supported`**. The old point-estimate-only boolean
(audit finding D-08) is gone.

## 9. Where it lands in the results

The experiment summary carries three explicit sections:

* `confirmatory_statistical_protocol` — this protocol, machine-readable
* `confirmatory_bootstrap` — the actual intervals, once a formal run is performed
* `claim_assessment` — every gate, recorded separately, plus `claim_status`

## 10. What this protocol does not address

It does not resolve audit findings **D-02** (retrieval-floor asymmetry),
**D-03** (density-ratio support and stability), **D-09** (tuning-budget
asymmetry), **D-10**, **D-11** or **D-12**. Those remain open, and D-09 in
particular bounds what the comparison may claim regardless of the outcome here.
