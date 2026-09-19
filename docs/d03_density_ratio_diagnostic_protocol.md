# D-03 density-ratio support and stability — diagnostic protocol

**Written before the measurement.** This document and the code implementing it
were frozen before any real D-03 number was produced. Nothing here selects a
cap, a calibration or any method change; the diagnostic measures, and a separate
reviewed decision follows.

Protocol version: `d03-support-stability-v1`
Implementation: `src/density_ratio_diagnostics.py` (pure, testable without a GPU)
Runner: `scripts/diagnose_ddre_ratio_support.py`

---

## 1. The question

Audit finding D-03 is **not**:

> large density ratios are automatically wrong

A large ratio is exactly what a density-ratio estimator *should* produce where
factual support is strong and hallucinated support is weak. Treating magnitude
as a defect would discard the method's signal.

The answerable question is narrower:

> Are the evidence values DDRE actually consumes **stable** under the production
> uLSIF hyperparameter surface, and **supported** by the score regions the
> training data represents?

That distinguishes **large-but-stable** evidence from **weak-support /
tail-driven** evidence. Only the second is a problem, and only the second can be
detected by measuring support and stability rather than size.

## 2. What is deliberately not done

* No cap is selected. `selected_cap` is `null` in the report, always.
* No calibration is selected. `selected_calibration` is `null`.
* `method_change_made` is `false`.
* The `[1e-6, 1e6]` production clip is unchanged.
* The sigma and lambda grids, the uLSIF fit, the stopping thresholds, the
  confirmatory statistics and every other production behaviour are unchanged.
* **D-03 is not resolved by this PR**, and is not resolved by running the
  diagnostic either. It reads `DIAGNOSTIC PROTOCOL FROZEN / INSTRUMENTED —
  empirical run still required before tuning` until the real measurement has
  been run *and reviewed*.

## 3. Scorer provenance — the existing machinery, reused

No second provenance system is introduced. The run is gated by
`src/provenance_guard.py` and `src/score_compatibility.py`, exactly as the NBC
cache-completion work is:

| pinned | value |
| --- | --- |
| model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| score version | `wang-emnlp23-temp5-seg400-overlap100-hostscale-v2` |
| Wang source commit | `3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea` |
| batch size | **1** |

Batch size 1 is deliberate: Wang's released scorer evaluates one pair at a time,
and the formal v2 provenance machinery already treats batch 1 as the canonical
Wang-fidelity path. **GPU batch 8 is not substituted**, and a test asserts it.

Required before any scoring: the formal provenance artifact, the
checkpoint-provenance supplement, a resolved revision present *before*
`from_pretrained`, the runtime provenance checks, the fixed **398-pair**
compatibility probe against the formal v2 cache with **exact** equality, and the
formal source digest measured on **both sides** of that probe.

### The historical limitation is preserved

`checkpoint_identity_established` may remain **false**, because the historical v2
Gate report did not record its Hub revision. That is a real limitation and is
**not rewritten as if it were solved**.

What D-03 relies on is the weaker but *directly measured* statement:

```
score_compatibility_established == true      (all 398 released NBC pairs, exact)
```

## 4. The formal cache is read-only

The formal v2 source cache is opened read-only and hashed. New validation
document scores go to a **derived diagnostic cache**, created by
`prepare_derived_cache` and bound to the exact certified source digest by
`bind_compatibility_to_source`. The source is never opened for writing, and no
main experiment artifact is overwritten.

## 5. Population — validation only, threshold-independent

Held-out test evidence is **never scored**.

The split is the frozen one (`validation_fraction = 0.20`, `seed = 42`) and is
verified by **passage identity**, not by count: a different 48 passages would be
a different diagnostic.

The population is **all candidate documents for all validation subclaims, up to
`max_docs = 10`**:

| quantity | value |
| --- | --- |
| validation passages | 48 |
| sentences | 383 |
| subclaims | 603 |
| document occurrences | 5,969 |
| first-document occurrences | 603 |

> **Why not the documents DDRE actually retrieves.** Running DDRE first and
> inspecting only what it fetched would condition the diagnostic on the very
> stopping policy that has not been tuned yet — and first-document behaviour,
> the most direct D-03 concern, would be selected on rather than measured.

Every occurrence keeps its identity (`passage_index`, `sentence_index`,
`subclaim_index`, `document_index`). Occurrences are **not deduplicated**: the
same page reached from two subclaims is two pieces of evidence in the
application, and collapsing them would silently reweight the population.
First-document scores are additionally identified as their own population.

Documents are scored by the ordinary production scorer — first 4,000 words,
400-word spans, overlap 100, document score = maximum entailment span score. No
second document score is invented.

## 6. The fit is the production fit

`ULSIFDensityRatio(max_centers=100, random_state=42)` on the exact 199 factual +
199 hallucinated NBC scores from the canonical scoring path. Not histogram
midpoints, not scores synthesised from released counts, and nothing from the
held-out split. Normalisation, kernel, centre selection, 5-fold CV, sigma
generation, lambda grid, non-negative truncation, numerical guards and the
`[1e-6, 1e6]` clip are all the production ones.

Recorded: `fit_diagnostics`, `cv_table`, selected sigma and lambda, centre count,
alpha summary. Coefficients are summarised rather than dumped.

## 7. What is measured

### Ratios, raw and returned

For each of four populations — NBC factual training, NBC hallucinated training,
all validation document-max scores, validation first-document scores — the
selected fit's `raw_ratio`, `returned_ratio = clip(raw, 1e-6, 1e6)` and
`log(returned_ratio)`, each reported at min / p01 / p05 / p25 / p50 / p75 / p95 /
p99 / max. Raw and returned are **never conflated**: the difference between them
is the clip's contribution.

Clip activity is counted on raw values, with the **lower and upper bounds
separate**, plus counts of `raw <= 0`, `raw < 1e-6` and `raw > 1e6`.

`alpha_sum` is recorded. For finite non-negative coefficients and Gaussian
kernels bounded by 1, `raw_ratio <= sum(alpha)` on ordinary fitted evaluation:
the estimator is **bounded**, and this document does not describe it as
mathematically unbounded.

### Local support

For each evaluation score, the number of factual and hallucinated training
scores within **one selected bandwidth**, measured in the **normalized `[0, 1]`
space the kernel uses** — sigma is a distance in that space, so measuring on the
0-100 scale would be wrong by a factor of 100.

`min_class_support = min(factual_within_sigma, hallucinated_within_sigma)`.

**No single support cutoff is declared "weak".** Pre-registered strata instead:

```
0    1-2    3-4    5-9    >=10
```

Each stratum reports its count and fraction, ratio and log-ratio quantiles,
log-ratio span, direction crossings, one-document stopping rate, and
hyperparameter unanimity — together, so large-but-stable separates visibly from
tail-driven.

Range status (inside / below / above the combined training range, and
descriptively against each class's range) is recorded. **Being outside one
class's range is not an error** and does not by itself indicate an unsupported
estimate.

### Hyperparameter-surface sensitivity

The evidence is re-derived at **every `(sigma, lambda)` pair present in the
production estimator's own `cv_table`** — taken from the estimator rather than
regenerated, so the surface cannot drift from the grid the production fit
searched.

Every refit uses the **same full training sets and the same final kernel centre
set as the selected model**, refitting only alpha. Letting each refit draw its
own centres would confound hyperparameter sensitivity with centre-sampling
noise, and the former is the question.

The production estimator's own `_as_column`, `_kernel` and `_solve` are used —
deliberately reaching for private helpers — so the diagnostic measures the
deployed mathematics rather than a second implementation that could drift. This
is diagnostic-only use; production fitting is untouched, and a test asserts the
estimator's state is unchanged afterwards.

Per evaluation score: minimum / maximum / median returned ratio, the same for
log ratio, `log_ratio_span = max_log - min_log`, and a direction verdict — all
above 1, all below 1, or **crosses 1**.

### One-document stopping

The **general log-odds rule** already frozen in the audit:

```
state_after_one = logit(P0) + log(returned_ratio)

low stop  iff  state_after_one <= logit(lower)
high stop iff  state_after_one >= logit(upper)
```

with `P0 = 0.5` for the formal primary protocol, and the **actual
cost-consistent CM=28 / CFA=96 search space** produced by the code (32 pairs at
the current grids). `low`, `high` and `either` fractions are reported
**separately** — concluding hallucination on one document and concluding factual
on one document are different events.

`|log r|` is **not** used as the criterion. It is only the special case of a band
symmetric about `P0 = 0.5`, and using it for an asymmetric band gives the wrong
answer; a test exhibits a concrete disagreement.

Evaluated on validation first-document scores, for the selected fit and for
every sigma/lambda refit.

### Stop-decision stability

For each first-document score and each threshold pair, the selected fit's
decision (`NONE` / `LOW` / `HIGH`) is compared with every refit's, counting:

```
unanimous_across_hyperparameters   any_decision_flip
low_to_none    high_to_none    none_to_low    none_to_high
direction_flip_low_to_high      direction_flip_high_to_low
```

Flip kinds are kept apart. A LOW-to-HIGH reversal — the same evidence concluding
hallucination under one setting and factual under another — is far more serious
than a stop softening to no decision, and a single mean delta would hide it.

## 8. Escalation triggers — frozen in advance

`additional_sensitivity_required = true` if **any** of the following occurs on
validation document evidence:

1. **The selected production fit hits either existing ratio clip** `[1e-6, 1e6]`.
2. **A validation first-document score that produces a one-document stop under
   the selected fit has a ratio direction (>1 vs <1) that flips** somewhere over
   the production sigma/lambda surface.
3. **A validation first-document score that produces a one-document stop under
   the selected fit has `min_class_support == 0` AND its stop decision is not
   unanimous** across the production sigma/lambda surface.

These trigger **"run a separate cap / calibration / evidence-scale sensitivity
study"**. They do **not** trigger "apply a cap", and the diagnostic never emits
one.

If nothing fires, `additional_sensitivity_required = false`, which means **the
predeclared D-03 escalation criteria did not fire** — *not* "uLSIF is proven
correct".

**No percentage threshold is introduced after seeing the data.** There is no
magnitude criterion anywhere in the triggers.

### Explicitly not triggers

None of the following, on its own, indicates a defect or causes a method change:

* a ratio above 10, or above 100
* a large `|log ratio|`
* one-document stopping
* low hallucinated support alone
* a score outside one class's observed range

A large ratio with coherent local support that is stable across the
hyperparameter surface is a **legitimate estimate**.

## 9. D-13, descriptively

The diagnostic already holds both the raw NBC pair scores and the validation
document-max scores, so it reports the known application shift between them:
quantiles, fixed 0–10, 10–20 … 90–100 histograms, mean/median, min/max.

The shift is **not corrected**, nothing is retrained on document-max scores, and
**D-13 is not marked resolved**. Both BSE and DDRE inherit this pair-score →
document-max-score shift, so it does not by itself break the controlled
comparison.

## 10. Fail closed

The real run aborts — before or during scoring, without a verdict — if the
provenance guard fails, the 398-pair compatibility probe fails, the source cache
changes, the split identity fails, the NBC training counts are not exactly
199/199, any required validation document score is non-finite, any required
document cannot be scored, the uLSIF selected fit fails its existing numerical
checks, any diagnostic hyperparameter refit is non-finite or degenerate, or
document identity/count accounting disagrees.

**Problematic rows are never dropped.** A partial scoring cache may be resumable,
but the JSON verdict must not claim `COMPLETE` until every required validation
document occurrence is accounted for — `require_complete` raises otherwise.

## 11. Dry run

`--dry-run` performs no model download, no inference and no cache write. It
reports the expected scorer and provenance inputs, the exact frozen validation
split identity (including a passage-ID SHA-256), the passage / sentence /
subclaim counts, the candidate document-occurrence count up to `max_docs = 10`,
the first-document count, the cost-consistent threshold-pair count, the 398
expected compatibility forward passes, and the output and derived-cache paths —
so the measurement population is reviewable before any GPU time is spent.

## 12. What running this will and will not settle

Running the diagnostic produces a measurement and, at most, a pre-registered
statement that a further sensitivity study is warranted. It does not resolve
D-03, does not authorise a cap, and does not permit threshold tuning to begin
before the result has been reviewed.
