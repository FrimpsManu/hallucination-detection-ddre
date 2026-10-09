# Evidence/stopping ablation — preregistration

**Status: frozen when this document is merged to `main`.** It was written on
2026-10-09, before any ablation code, any validation selection for cells B, C
or D0, any held-out cache completion, and any ablation result.

**Classification: exploratory.** The held-out split was already used once, for
the frozen BSE-official vs frozen-DDRE comparison (result SHA-256
`b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67`). Nothing
here is confirmatory, nothing here changes that run's `NOT_CONFIRMATORY`
status, and nothing here reruns it.

---

## 1. Question

The frozen DDRE used about 21% fewer documents per sentence than published BSE
on the held-out split. It differs from BSE in three ways at once: the evidence
model (continuous uLSIF ratio vs discretized histogram), the stopping rule
(posterior band vs Bayes-risk look-ahead), and the selection history (640
validation configurations after D-03 vs none). This ablation separates:

1. the effect of replacing Bayes-risk stopping with a band (A → B);
2. the effect of strengthening the magnitude of histogram evidence (B → C);
3. the effect of the learned continuous uLSIF shape beyond a global rescaling
   (C → D);
4. the effect of the post-D-03 selection history (D0 → D).

### Context carried in, not re-derived

The Phase 1/1b analysis (`scripts/analyze_frozen_heldout.py`) is post-hoc and
descriptive. It found that the frozen DDRE's savings, its shift toward
hallucinated predictions, and its factual-recall loss are **strongly associated
with** how it weights the low-to-moderate NLI-score region that dominates this
benchmark. It did not establish a mechanism; this ablation is the test.

**Known warning, preserved.** In that dominant region the sign of the evidence
depends on the estimator. At NLI score 35:

| Estimator | Per-document log-evidence |
|---|---|
| BSE histogram (bucket 3) | −0.143 |
| Frozen DDRE (σ = 0.11346435546875, λ = 1.0) | −0.441 |
| Original CV-selected DDRE (σ = 0.2269287109375, λ = 1.0) | +0.924 |
| D-03 (σ, λ) surface | 17 of 20 pairs positive |

Only 26 of the 398 NBC training scores lie in [30, 40). The final frozen
estimator assigns the opposite sign to most of the D-03 surface, including the
CV-selected estimator, in the region where most held-out evidence sits.

---

## 2. Cells

| Cell | Evidence per document | Stopping | Selected on validation | Search budget |
|---|---|---|---|---|
| **A** | BSE histogram log-likelihood ratio | Bayes-risk look-ahead (`BSEDetector`, `mode="official"`) | nothing | 0 |
| **B** | the same histogram log-likelihood ratio | posterior band | band only | 32 |
| **C** | κ × the histogram log-likelihood ratio | posterior band | κ and band | 7 × 32 = 224 |
| **D0** | uLSIF, σ = 0.2269287109375, λ = 1.0 (original CV-selected fit) | posterior band | band only | 32 |
| **D** | uLSIF, σ = 0.11346435546875, λ = 1.0 (frozen) | band [0.20, 0.80] | nothing now; historically 640 | 640 (historical) |

### Definitions

**Histogram log-likelihood ratio.** For a document score `s`, the bucket is
`b = discretize_document_score(s)` (released `main.py` discretizer), and
`LLR(s) = log( (pos_hist[b] / Σ pos_hist) / (neg_hist[b] / Σ neg_hist) )`, with
`pos_hist`/`neg_hist` the Laplace-smoothed released NBC histograms exactly as
`build_nbc_histograms` produces them and BSE uses them. This is the per-document
evidence BSE's own Bayes update applies.

**Band cells (B, C, D0, D)** use the existing `DDREDetector` accumulation
unchanged: start at `logit(P0)`, add the per-document log-evidence, clip the
running log-odds to ±40, stop at `P ≤ lower` or `P ≥ upper`, otherwise continue
until `max_docs` or the evidence list ends. Cell B is `ratio(s) = exp(LLR(s))`;
cell C is `ratio(s) = exp(κ · LLR(s))`; D0 and D use the uLSIF estimator. The
classification of the final posterior is `cost_based_prediction` at
CM = 28, CFA = 96 in every cell.

**κ grid (C):** {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}. κ = 1.0 is exactly cell B,
so C's search space contains B's.

**D0 estimator:** the production fit reproduced by `verify_production_fit`, at
its own σ and λ with its own centres, constructed the way the sensitivity study
constructed its rows (`candidate_estimator`), so the re-derived validation
outcome is comparable with the recorded one. **D estimator:** the frozen reconstruction used by
the frozen run (`candidate_estimator`, production centres held fixed, α
re-solved at σ = 0.11346435546875, λ = 1.0).

### Band grid (B, C, D0)

The 32 cost-consistent pairs used for the frozen DDRE selection
(t = CM / (CM + CFA) = 0.22580645161290322):

- lower ∈ {0.05, 0.10, 0.15, 0.20}
- upper ∈ {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}

All 4 × 8 combinations. The grid is not extended or narrowed for any cell.

---

## 3. Held fixed across all cells

- sentences, subclaims, and the frozen split (validation fraction 0.20, seed 42;
  48 validation / 190 held-out passages, verified by passage identity);
- retrieved documents and their order;
- P0 = 0.5; CM = 28; CFA = 96; C_retrieve = 1; max_docs = 10;
- NLI model `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` at
  revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, its tokenizer, score
  version `wang-emnlp23-temp5-seg400-overlap100-hostscale-v2`, 400/100-word
  segmentation, max-over-spans document score;
- NBC training pairs (the released 199 + 199) for the histograms and both uLSIF
  fits;
- minimum-over-subclaims sentence aggregation;
- the PR-AUC implementation (`src.evaluation.wang_pr_auc`).

---

## 4. Validation selection (B, C, D0)

Run on the 48 validation passages only, replayed from the D-03 derived cache
(SHA-256 `66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776`),
which already scores every validation document up to max_docs = 10. **No
inference is needed for validation selection.**

**Reference:** BSE official on the same validation passages (balanced PR-AUC
0.78778, nonfactual 0.88270, factual 0.69286, as recorded in the sensitivity
artifact `fc5604854b18…`). The replay must reproduce those values exactly before
any cell is selected.

**Eligibility** (the rule that froze D): a configuration is eligible iff its
validation nonfactual, factual and balanced PR-AUC are each ≥ the BSE value
− 0.005.

**Ranking among eligible configurations** (the rule that froze D):

1. minimum documents per sentence;
2. higher balanced PR-AUC;
3. higher nonfactual PR-AUC;
4. higher factual PR-AUC;
5. fewer NLI span calls per sentence;
6. deterministic ascending parameters: κ (C only), then lower, then upper.

**No eligible configuration → no fallback.** If a cell has no eligible
configuration, that is recorded as its outcome, the cell is **not** evaluated on
held-out, and the comparisons that need it are reported as not evaluable, with
the validation result stated directly. No relaxed tolerance, alternative
objective or hand-picked configuration may substitute.

**D0's validation outcome is already known.** The existing sensitivity artifact
(`fc5604854b18…`) records all 32 bands for the CV-selected pair, and **none is
eligible**. D0 is therefore not a blind prediction: the ablation re-derives this
result under the rule above and must agree with the recorded one. If it does,
D0 is not evaluated on held-out. If it does not agree, the discrepancy is
reported and investigated before anything proceeds.

**Freeze.** The selected B and C configurations (or "no eligible
configuration"), the D0 outcome, all candidate rows, and the BSE validation
reference are written to one freeze artifact and hashed **before** any held-out
ablation evaluation.

---

## 5. Held-out cache completion

Approved as one new inference pass, to run **only after this document is
merged**. Its sole purpose is a complete replay surface for the ablation.

- Source: the frozen-run held-out cache (SHA-256
  `dce6c49f813cb0ebecaf15d813e35ef940d259f97ad246d42ab8208c7a85be25` at the time
  of writing). It is never modified; it is copied to a new derived cache.
- Coverage: **every** span of **every** document in the first max_docs = 10
  positions of **every** held-out subclaim's evidence list, regardless of which
  cell would request it. Cell behaviour plays no part in what gets scored.
- Protocol: identical to the frozen run: model, revision, tokenizer, score
  version, segmentation, max-over-spans, batch size 1, device `mps`, model dtype
  `torch.float16`.
- Rows already in the source cache are kept as they are and never re-scored into
  the derived cache. Before any new row is written, the existing 398-pair exact
  score-compatibility probe must pass in the completion environment.
- Recorded: source-cache SHA-256, resulting-cache SHA-256, model revision,
  pre-existing row count, added row count, environment and device, and
  completion status (complete or not, with any missing pairs listed).
- No detector runs and no ablation metric is computed during completion.
- The completed cache is hashed and frozen before cells A–D are replayed on
  held-out.

---

## 6. Held-out evaluation (exploratory)

After the validation freeze and the cache freeze, each evaluable cell (A, B, C,
D, and D0 only if it had an eligible configuration) is replayed **once** on the
190 held-out passages from the completed cache.

**Consistency gate.** A and D are the two methods of the frozen run. Their
replay on the completed cache must reproduce the canonical predictions CSV
(`a78b09b7…`) exactly (posterior, prediction, documents, NLI calls). If it does
not, nothing is reported until the difference is explained.

**Required metrics for every evaluated cell:** nonfactual PR-AUC, factual
PR-AUC, balanced PR-AUC, accuracy, macro-F1, factual precision, factual
recall, nonfactual precision, nonfactual recall, documents per sentence,
documents per subclaim, NLI spans per sentence, and the per-subclaim
stopping-depth distribution.

**Intervals.** Paired passage-level cluster bootstrap with the frozen settings
(passage unit, 10,000 resamples, seed 42, 95% percentile intervals), for:

- docs(A) − docs(X) for X ∈ {B, C, D}: the savings S(X);
- docs(C) − docs(D);
- the ratios S(B)/S(D) and S(C)/S(D), computed within each bootstrap replicate;
- balanced PR-AUC differences B − A, C − B, D − C;
- NLI spans per sentence for the same pairs as documents.

No multiplicity correction is applied, because no confirmatory claim is made.
Point estimates, ratios and intervals are always reported, whatever the
interpretation below says.

---

## 7. Interpretation rules

These are **descriptive interpretation thresholds, fixed now**. They are not
significance tests and do not license causal or proof language. The continuous
quantities and their intervals are always reported alongside.

`S(X)` = documents-per-sentence saving of cell X relative to A on held-out.

**A → B (stopping rule).** Primary quantity: S(B) / S(D).
If S(B) ≥ 0.5 × S(D): *the stopping-rule change accounts for a substantial
share of DDRE's observed saving, and most of the saving cannot be attributed
solely to the DDRE evidence model.*

**B → C (evidence magnitude).** Primary quantity: S(C) / S(D).
If S(C) ≥ 0.8 × S(D) **and** C's balanced PR-AUC is within 0.01 of D's:
*evidence that the magnitude assigned to weak evidence accounts for most of the
efficiency behaviour, making the detailed uLSIF shape secondary.*

**C → D (continuous shape).** Report the bootstrap interval for
docs(C) − docs(D).
If that interval excludes zero **and** D's balanced PR-AUC is not lower than
C's by more than 0.01: *in this exploratory analysis, the learned continuous
density-ratio shape contributes beyond a global rescaling.*

**D0 → D (selection history).** No binary threshold. Report the validation
outcome directly. If D0 has no eligible configuration, that is reported as a
finding in its own right: the original CV-selected estimator could not meet the
validation quality constraint at any of the 32 bands, and the evaluated DDRE is
the product of the post-D-03 selection.

**When a cell is not evaluable** (no eligible validation configuration), every
rule that needs it is reported as not evaluable, with the validation outcome.

**Tuning budgets differ** (A 0, B 32, C 224, D 640). This favours the cells
further down the table and is disclosed next to every comparison.

**Language.** Results are described as observed differences in this
exploratory replay. Words like "proves", "causes", "is caused by", "comes from"
and "explains why" are not used for any outcome.

---

## 8. Prediction (frozen; not to be revised after results)

> I expect B to recover well under one quarter of D's document saving, because
> the modal BSE histogram bucket contributes only about −0.14 log-odds per
> weak-support document, whereas frozen DDRE contributes roughly −0.44.

Stated as a number: S(B) / S(D) < 0.25. Whether it holds is reported either way.

---

## 9. Execution order

1. Merge this document. From then on it is frozen.
2. Separate PR: held-out cache-completion script (§5).
3. Complete, record and hash the held-out cache.
4. Implement validation selection for B, C and D0 (§4).
5. Freeze the B, C and D0 outcomes (§4), hashed.
6. Replay A–D on held-out once (§6), with the consistency gate.
7. Report against §6–§8.

## 10. Amendments

Any change after merge goes in a dated amendment section below, with its
reason, and only before step 6. After step 6 starts, this document is not
changed. A run that deviates from it is reported as a deviation, not as this
experiment.

## 11. Out of scope

Quality–cost frontier curves, new datasets, runtime measurement, a cell C with
a different transform (shift, per-bucket weights), and any uLSIF + Bayes-risk
cell. None of these may be added to this experiment after the fact.
