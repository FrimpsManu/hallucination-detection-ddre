# DDRE scientific audit — before held-out evaluation

Read-only audit of the DDRE/uLSIF implementation against the scientific contract
for comparison with Wang et al.'s (EMNLP 2023) Bayesian sequential estimation
baseline.

**Nothing in this audit changes research behaviour.** No formal experiment was
run, no GPU was used, no model was downloaded, no threshold was tuned, and the
held-out test split was never evaluated. The only code added is
`tests/test_ddre_audit.py`, 48 tests that pin current behaviour — including
behaviour this audit judges to be wrong, so that a later fix has to change a
test visibly.

Files audited: `src/ddre_core.py`, `main.py`, `src/evaluation.py`,
`src/wang_data.py` (splitting), `tests/test_core_math.py`.
`src/baseline_core.py` was read only to establish the comparison contract and
was not modified.

---

## 1. Executive verdict

### NOT READY

**One blocker** must be resolved before any tuning run:

* **D-01** — the stopping-threshold grid contains four values that contradict
  the CM/CFA classification rule, and the tuner's lexicographic objective
  optimises average documents first, so it searches that region freely. This is
  established by exact arithmetic on the released costs and by an end-to-end
  reproduction, not by simulation.

**One finding requires an empirical check before tuning** and cannot be settled
by this audit:

* **D-03** — density-ratio estimates in regions with weak numerator or
  denominator support are poorly constrained, so a single document may carry
  very large log-evidence. The mechanism is real and demonstrated; its
  *prevalence on the actual formal NBC fit is unmeasured*, because this
  repository contains no fixed raw NBC scores and measuring them needs the NLI
  model. See §4-A for the measurement to run first.

Three further findings (**D-02**, **D-07/D-08**, **D-09**) bear on what the
DDRE-vs-BSE comparison is entitled to claim. They do not corrupt the numbers,
but they change the boundary of the claim, and all three are cheaper to settle
now than to caveat later.

Two things are in good shape and should be stated plainly:

* **There is no test-set leakage.** See §5.
* **The uLSIF mathematics are correctly oriented.** `H` is built from the
  denominator (hallucinated) sample and `h` from the numerator (factual) sample,
  centres sit on the numerator sample, the regulariser is `λ/2 αᵀα`, and the
  cross-validation criterion is the genuine held-out uLSIF objective. There is
  no numerator/denominator inversion and no sign error. See §4-A.

---

## 2. Findings

| ID | Severity | Location | Scientific issue | Why it matters | Recommended action |
| --- | --- | --- | --- | --- | --- |
| D-01 | **BLOCKER** | `main.py::tune_ddre_thresholds`, `ddre_core.py::DDREDetector.detect_subclaim` | The lower stopping grid is `{0.05 … 0.40}`; the CM=28/CFA=96 classification threshold is `0.225806`. For `lower ∈ {0.25, 0.30, 0.35, 0.40}` the detector can stop *because it is confident of hallucination* and the cost rule then labels the claim **factual**. | The two rules disagree on the same posterior. The selector is lexicographic — average documents first, PR-AUC only on exact ties — and a larger `lower` stops earlier, so within the feasible set nothing but the document count pushes back on choosing a contradictory threshold. | Constrain `lower ≤ CM/(CM+CFA)` (§7 option 1, recommended). Do not silently re-grid. |
| D-03 | **MAJOR / NEEDS EMPIRICAL CHECK** | `ddre_core.py::ULSIFDensityRatio.ratio` | Poorly constrained extrapolation: `r̂` is well determined only where both samples have support. `H` constrains magnitude on the hallucinated sample and the centres carry mass near the factual sample, so between and beyond them the estimate is weakly identified and can be very large or very small. | A weakly identified region can contribute large log-evidence from one document. Whether that happens often enough on the **actual** NBC fit to shorten retrieval materially is **unmeasured** — the demonstrations below use synthetic scores and establish only the mechanism. | Before tuning, measure `r̂` over the fitted score support and report the diagnostics in §4-A. Pre-register a per-document log-evidence sensitivity analysis if the measurement warrants it. **Do not choose a cap value yet.** |
| D-02 | MAJOR | `ddre_core.py::detect_subclaim` vs `baseline_core.py::detect_subclaim` | Algorithm/protocol asymmetry. BSE's stop/continue rule is decision-theoretic (expected cost of retrieving vs stopping) and can decline the **first** fetch, retrieving **0** documents. DDRE's rule is a fixed probability band evaluated after an update, so it has a floor of **1 document per subclaim**. | Retrieval counts have different origins. The floor is ~2,990 documents DDRE must spend before it can win on cost, and where BSE retrieves 0, DDRE cannot win at all. **This currently disadvantages DDRE.** | Report BSE's zero-retrieval frequency and the retrieval-floor difference alongside the efficiency numbers. A threshold pre-check does **not** fix this (see §6); giving DDRE a Bayes-risk pre-retrieval decision would be a **stopping-rule redesign**, not a small fairness fix. |
| D-07 | MAJOR | `main.py::tune_ddre_thresholds` | The feasibility test constrains only `factual_auc_pr` and `balanced_pr_auc`. Nonfactual AUC-PR — Wang's headline metric — is unconstrained. | Balanced PR-AUC is the mean of the two, so a large factual gain can mask a nonfactual loss and still qualify. The tuner may select a configuration that is worse at detecting hallucination. | Add an explicit nonfactual AUC-PR floor to the feasibility test. |
| D-08 | MAJOR | `main.py::hypothesis_comparison` | `hypothesis_supported_on_test` requires `factual_delta > 0` and `balanced_delta ≥ 0`, with no nonfactual floor and **no uncertainty quantification of any kind**. | A single point estimate is written into the summary as a scientific conclusion, over three methods and several metrics, with no interval and no multiplicity control. | Require a nonfactual floor, and gate the claim on the paired passage-level bootstrap (§4-H). |
| D-09 | MAJOR (claim boundary / robustness) | `main.py`, `ddre_core.py` | DDRE selects its stopping band from **64** validation configurations. Published BSE has **no tunable counterpart** — its stopping rule follows from the fixed published costs. | Part of any DDRE advantage may be a model-selection advantage. This bounds what the comparison may claim; it does **not** invalidate it. | Keep **published BSE (CM=28, CFA=96, c_retrieve=1) as the primary comparator** for comparability with Wang et al. Optionally add a validation-tuned BSE variant as a clearly labelled **secondary** robustness comparator, never as the primary. |
| D-04 | MAJOR | `ULSIFDensityRatio._solve` / `.ratio` | Post-hoc non-negativity truncation could in principle drive `α → 0`. Then `r̂ ≡ 0`, clipped to `1e-6`, i.e. `log r = −13.8` **per document**. Nothing in `fit()` detects or reports it. | A silently degenerate fit would look like a spectacularly confident detector. **Not observed in the synthetic audit fixture, and not established on the actual formal fit** — no fixed raw NBC scores exist in this repository to check it against. | Add a post-fit sanity check: `α` not all zero, and `r̂` finite and non-degenerate across the observed score range. |
| D-06 | MAJOR | `.ratio`, `detect_subclaim` | A NaN score propagates silently: `log(NaN)` → `clip` → `sigmoid` all yield NaN; every stopping comparison is False, so the **full retrieval budget is spent**, and NaN reaches the metrics. `+inf` is clipped to a perfect score of 100. | Silent corruption of both the quality and the efficiency numbers, with no warning anywhere. | Validate the score at the detector boundary and fail loudly. |
| D-05 | MINOR | `detect_subclaim` | The running log-odds are clipped to `±40` **after each update**, so accumulation is non-associative and evidence beyond the bound is discarded. | Harmless under any stopping band in the current grid (`sigmoid(±40)` is far outside it), but it is a silent modification of the stated update rule. | Document it, or clip only at the sigmoid. |
| D-10 | MINOR | `evaluation.py::prediction_rows`, `detect_sentence` | Per-subclaim results are collapsed into one `DetectionResult` with summed counters. Per-subclaim stopping depth is unrecoverable, and the CSV omits the subclaim count. | §4-G asks for stopping depth; §4-H needs the subclaim count to express documents-per-subclaim on a bootstrap resample. | Add `n_subclaims` to `prediction_rows` and retain per-subclaim depths. |
| D-11 | MINOR | `main.py::tune_ddre_thresholds` | `normalized_docs = avg_docs / max_docs` divides a **per-sentence** document count by a **per-subclaim** budget. With ~1.57 subclaims/sentence it can exceed 1.0. | The fallback penalty's exchange rate against balanced PR-AUC is therefore not the declared one. | Normalise by `max_docs × subclaims_per_sentence`, or state the scale explicitly. |
| D-12 | MINOR | `main.py::auto_push_results` | Result artifacts are committed and pushed automatically at the end of a full run unless `--no-push-results`. | A crashed or partial run can publish artifacts; the default direction is toward publishing, not toward review. | Invert the default. |
| D-13 | NOTE | `main.py`, `ddre_core.py` | uLSIF is fit on the **NBC sentence-pair** scores but applied to **document** scores (max over 400-word spans). | A real distribution shift — but BSE's histograms inherit exactly the same one, so the comparison is fair. Worth one sentence in the paper. | Disclose. |
| D-14 | NOTE | `detect_subclaim` | A subclaim with no documents returns `P = P0 = 0.5`, which is above `0.225806`, so it is classified **factual** by default. | Not reachable in the released data (0 of 2,990 subclaims), so it is a latent rather than an active issue. | Assert non-empty at load, or document the default. |
| D-15 | NOTE | `ULSIFDensityRatio.fit` | CV folds choose centres from each fold's training half; the final model chooses centres from all factual scores. The selected `(σ, λ)` is therefore optimal for a centre set the final model does not use. | Standard practice, small effect, but it means the CV objective is not exactly the final model's objective. | Disclose. |
| D-16 | NOTE | `evaluation.py` | Pearson/Spearman are computed at the **passage** level. | Confirmed correct: this is the granularity `scripts/reproduce_wang_baseline.py` compares against Wang's published 0.7137 / 0.6455. | None. |
| D-17 | NOTE | `main.py` vs `scripts/reproduce_wang_baseline.py` | Gate 1 evaluates BSE on all 1,908 sentences; `main.py` evaluates on the 1,525-sentence test split. | Internally fair (both methods use the test split), but `main.py`'s BSE numbers are **not** comparable to Wang's Table 1. | Label clearly in the results. |

---

## 3. Dataset and split facts established by this audit

| Quantity | Value |
| --- | --- |
| Passages | 238 |
| Sentences | 1,908 |
| Subclaims | 2,990 (1.567 per sentence) |
| Documents per subclaim | min 1, max 10, mean 9.832 |
| Subclaims with zero documents | 0 |
| Validation / test passages | 48 / 190 |
| Validation / test sentences | 383 / 1,525 |
| Classification threshold, CM=28/CFA=96 | 0.2258064516 |
| Classification threshold, CM=14/CFA=24 | 0.3684210526 |

---

## 4. Detailed audit

### A. uLSIF mathematics — correct orientation, unsafe extrapolation

The intended objective is

    ½ αᵀHα − hᵀα + (λ/2) αᵀα,   solved by   (H + λI) α = h

with `H = E_de[φφᵀ]` and `h = E_nu[φ]`, and `r(s) = p_nu(s)/p_de(s)`.

For `r(s) = p(s|F)/p(s|H)` the numerator is factual and the denominator
hallucinated. `_solve` builds `h_matrix` from `phi_h` (hallucinated) and
`h_vector` from `phi_f` (factual). **This is the correct orientation.**
`test_H_is_built_from_the_denominator_and_h_from_the_numerator` reproduces the
normal equations by hand and asserts equality with `_solve`, so a future
inversion cannot pass silently.

| Element | Implementation | Verdict |
| --- | --- | --- |
| Which distribution builds `H` | hallucinated (denominator) | correct |
| Which distribution builds `h` | factual (numerator) | correct |
| Kernel | Gaussian, `exp(−(x−c)²/2σ²)` | correct |
| Centre selection | up to 100, sampled from the **numerator** (factual) | correct, standard |
| Normalisation | `s/100`, clipped to `[0,1]` | correct |
| Regularisation | `+ λI`, i.e. `λ/2 αᵀα` | correct |
| Non-negativity | post-hoc `max(α, 0)` | standard (Kanamori et al. 2009); see **D-04** |
| Numerical stability | `solve` with `pinv` fallback | adequate |
| Clipping of `r` | `[1e-6, 1e6]`, symmetric in log | see **D-03** |
| Sign / inversion errors | none found | — |

#### D-03: poorly constrained extrapolation — mechanism established, prevalence unmeasured

**Correction to an earlier draft of this audit.** `r̂` is **not** unbounded. It
is a finite non-negative combination of Gaussian kernels evaluated on an input
clipped to `[0,1]` with finite coefficients, so it is bounded above by `Σ_l α_l`
— a finite, exactly computable quantity. Any claim of unboundedness was wrong
and is withdrawn.

The real issue is weaker and still worth acting on: **`r̂` is well determined
only where both samples have support.** `H` constrains magnitude on the
hallucinated sample and the centres carry mass near the factual sample, so
between and beyond those regions the estimate is weakly identified. Its value
there is decided by kernel tails and the regulariser rather than by data, and
the bound `Σ_l α_l` is itself data-dependent and can be large.

Illustrative measurements on **synthetic** scores:

| Fixture | `Σ α` (exact bound) | observed `r̂` range | reached the `[1e-6, 1e6]` clip? |
| --- | --- | --- | --- |
| Scores shaped to the released NBC histograms | 17.1 | `[0.37, 14.6]` | no |
| Well-separated Gaussians (μ=78 vs μ=22) | 16 434 | up to `1.6×10⁴` | no |

**Correction to a second earlier claim.** An earlier draft said the clip "maps
both to `log r = ±13.8`". That magnitude arises **only** when the raw ratio
reaches the `[1e-6, 1e6]` bounds, and **neither** synthetic fit came close: the
NBC-shaped fit spans `log r ∈ [−0.98, +2.68]` and the well-separated fit peaks
at `|log r| ≈ 9.7`. The clip is a backstop, not the operating regime observed
here.

**What the synthetic evidence does and does not establish.** On the NBC-shaped
fixture, 48 of 101 points on the 0–100 grid satisfy
`|log r̂| ≥ log(0.8/0.2) = 1.386`, i.e. one document at such a score would end
the sequence under the default `[0.20, 0.80]` band. **This establishes a
mechanism, not a rate.** Those scores were drawn uniformly inside the released
histogram buckets; the real fit is on raw continuous NBC scores with a different
within-bucket shape, and the observed *document* scores are a different
distribution again. Specifically, this audit does **not** claim that DDRE is
"routinely one-shot", that the estimator is unbounded, that this can manufacture
the headline result, or any prevalence figure for the actual formal fit.

**Why it cannot be settled here.** The repository contains no fixed raw NBC
scores — `results/diagnostics/sample.json` records counts (199 + 199 = 398), not
values — and the only NLI caches are the formal Gate 1 artifacts, which this
audit must not depend on. Measuring the real fit needs the model and a GPU, both
out of scope.

**Measurement to run before final tuning** (read-only, one fit, no tuning):

1. Fit `ULSIFDensityRatio` on the actual formal raw NBC scores.
2. Evaluate `r̂` over the **fitted score support** — the observed NBC scores and
   the observed retrieved-document scores, not a uniform grid.
3. Report **quantiles of `log r̂`** (min, 1%, 5%, 25%, 50%, 75%, 95%, 99%, max)
   separately for the NBC scores and the document scores.
4. Report the **fraction of observed document scores whose `|log r̂|` alone
   crosses a candidate stopping boundary after one document**, for each
   `(lower, upper)` pair in the grid.
5. Report `Σ α`, the fraction of observed scores falling outside the range of
   each training sample, and whether the `[1e-6, 1e6]` clip is ever reached.
6. **Only then**, if the measurement warrants it, pre-register a per-document
   log-evidence clipping or calibration together with a sensitivity analysis
   over it. **No cap value is selected in this audit.**

#### D-04: degenerate fit — not observed, not established

At λ = 10¹² the solved `α` is ~4.7×10⁻¹³ and `r̂ ≈ 0` everywhere, giving
`log r = −13.8` per document. On the synthetic fixture at the grid maximum
λ = 1.0 all 100 coefficients remain positive.

That is the *only* evidence available, and it is a synthetic fixture, so the
correct statement is: **not observed in the synthetic audit fixture, and not
established on the actual formal fit.** `fit()` performs no post-fit sanity
check, so a degenerate solution would be reported as an ordinary one. The
recommendation stands regardless of prevalence: check that `α` is not all zero
and that `r̂` is finite and non-degenerate across the observed score range.

### B. Hyperparameter selection — sound, no leakage

* σ grid: `{0.5, 1, 2, 4} × max(0.02, median pairwise distance)`, floored at
  0.01. Standard median heuristic.
* λ grid: `{1e-4, 1e-3, 1e-2, 1e-1, 1.0}`.
* CV: k-fold (default 5) over factual and hallucinated indices independently,
  with **centres fixed per fold** across all `(σ, λ)` candidates — so candidates
  are compared on identical bases. Good practice.
* Criterion: the genuine held-out uLSIF objective
  `½ E_de[r̂²] − E_nu[r̂]`, verified against a hand computation.
* Selection: minimum mean CV objective; refit on all data. Deterministic for a
  fixed seed.

**Training/validation boundary.** The estimator's entire interface is
`fit(factual_scores, hallucinated_scores)`. It has no access to sentence
records, labels, passages or splits. The scores it receives come from Wang's
**separate NBC pair files**, which are disjoint from the SelfCheckGPT sentences
that form the validation/test splits. **No test data can enter uLSIF fitting or
model selection.**

Caveat D-15: the final model's centres differ from every fold's, so the selected
`(σ, λ)` is optimal for a basis the final model does not use.

### C. Sequential DDRE detector

| Property | Behaviour | Verdict |
| --- | --- | --- |
| Log-odds initialisation | `logit(P0)`, `= 0` at `P0 = 0.5` | correct |
| Accumulation | `log O_n = log O_0 + Σ log r(s_i)` | correct, verified numerically |
| Probability conversion | overflow-safe two-branch sigmoid | correct |
| Stopping | after each update, `p ≤ lower` or `p ≥ upper` | see D-01, D-02 |
| `max_docs` | `documents[:max_docs]` — identical to BSE | correct |
| Zero documents | returns `P0`; predicted factual | D-14 |
| Sentence aggregation | `min` over subclaims — identical to BSE | correct |
| Retrieval counting | `+1` per document scored | correct |
| NLI-evaluation counting | sums the span count returned by the scorer | correct |

Numerical problems found:

* `log(0)` — **not reachable**: `ratio()` floors at `1e-6`. But the floor is
  itself the D-03 problem.
* Overflow — **not reachable**: the two-branch sigmoid plus the `±40` clip.
* Underflow — `sigmoid(−40) ≈ 4.2×10⁻¹⁸`, finite.
* **NaN/inf — reachable and silent (D-06).** A NaN score yields NaN log-odds and
  NaN `p_factual`; every stopping comparison is False, so the detector consumes
  all 10 documents and returns NaN, which then flows into PR-AUC and the
  correlation metrics. `+inf` is clipped to a perfect score of 100.
* Clipping — `±40` on the **running** total (D-05), so accumulation is
  non-associative. Immaterial under the current grid.

### D. Stopping thresholds — see §7 for the full treatment.

### E. Validation-only tuning

**What it optimises.** Feasible set: `factual_auc_pr ≥ baseline − 0.005` **and**
`balanced_pr_auc ≥ baseline − 0.005`, both against BSE-official measured on the
same validation records. Among feasible candidates it minimises
`avg_retrieved_documents_per_sentence`, breaking ties on higher balanced PR-AUC
then higher factual PR-AUC.

* **Performance-preservation constraint — implemented, but incomplete (D-07).**
  Nonfactual AUC-PR is not constrained. Because `balanced = ½(nonfactual +
  factual)`, a factual gain can offset a nonfactual loss and still qualify. For
  a hallucination-detection paper, that is the wrong metric to leave free.
* **Retrieval-efficiency tie-break — correct in form, but strictly secondary.**
  The selector is **lexicographic**: average documents per sentence is optimised
  first; balanced and factual PR-AUC only affect **exact retrieval-count ties**.
  This audit did not run threshold tuning, so it makes no claim about how often
  such ties arise. The ordering alone is enough to establish the D-01 concern:
  within the feasible set, nothing but the document count drives the choice
  unless two configurations retrieve exactly the same average number.
* **Test split — never touched.** See §5.
* **Fallback — declared in advance, but two problems.** It is predeclared and
  recorded in the summary (`selection_rule`), which is good. But (i) the
  normalisation is dimensionally wrong (D-11), and (ii) the fallback maximises
  `balanced_pr_auc − 0.05·normalized_docs` with no quality floor at all, so it
  can select a configuration far below baseline quality. It is defensible only
  because it is explicitly recorded as "no threshold pair preserved BSE-official
  validation quality" — a reader can see the constraint failed.

### F. DDRE / BSE fairness — see §6.

### G. Metrics and compute accounting

| Required | Available | Where |
| --- | --- | --- |
| Nonfactual PR-AUC | yes | `nonfactual.auc_pr` (Wang's `auc(recall, precision)`) |
| Factual PR-AUC | yes | `factual.auc_pr` |
| Accuracy | yes | `accuracy` |
| Pearson | yes | `passage_level.pearson` |
| Spearman | yes | `passage_level.spearman` |
| Documents per sentence (Wang-compatible) | yes | `efficiency.avg_retrieved_documents_per_sentence` |
| Documents per subclaim | yes | `efficiency.avg_retrieved_documents_per_subclaim` |
| NLI span evaluations | yes | `efficiency.total_nli_span_calls`, `avg_..._per_sentence` |
| **Stopping depth** | **no** | only p50/p95 documents **per sentence**, which sum across subclaims (D-10) |

Runtime is recorded (`wall_clock_seconds`) and correctly labelled as
non-comparable when the NLI cache is used. It does not replace the
model-independent counts. Good.

### H. Statistical evaluation readiness

Not implemented, as instructed. Assessed for readiness:

**What is preserved.** `prediction_rows` emits one row per (method, sentence)
with `passage_index`, `sentence_index`, `gold_label`, `p_factual`,
`prediction`, `retrieved_documents`, `nli_span_calls`. All three methods are
evaluated on the same `test_records` in the same order, so rows are **paired**.
Resampling the 190 test passages with replacement carries whole passages —
hence all their sentences and subclaims — which respects the nesting, and every
sentence-level metric (PR-AUC, accuracy, documents/sentence, span calls) can be
recomputed from raw `p_factual` on each resample. **This is sufficient for the
intended passage-level paired bootstrap.**

Note that the bootstrap population is the **190 test passages**, not 238; the
other 48 are the validation split.

**What would need to change first:**

1. Add `n_subclaims` to `prediction_rows` (D-10) — otherwise
   documents-per-subclaim cannot be formed on a resample.
2. Retain per-subclaim depths if subclaim-level efficiency claims are wanted
   (D-10); passage-level sentence metrics do not require this.
3. Write the predictions CSV before the auto-push (D-12), so a failed push
   cannot leave the bootstrap input unwritten.
4. Decide and predeclare the bootstrap protocol — number of resamples, interval
   type, which metrics, and the multiplicity correction across methods and
   metrics — **before** the test split is evaluated.

---

## 5. Potential test-set leakage

### VERDICT: NO

Evidence, established by reading every use and pinned by
`TestNoTestSetAccessDuringTuning`:

1. **uLSIF fitting.** `ULSIFDensityRatio.fit(factual_scores,
   hallucinated_scores)` is called (`main.py:340`) with scores from
   `build_nbc_histograms`, i.e. Wang's separate `NBC_positive.json` /
   `NBC_negative.json` pair files. These are disjoint from the SelfCheckGPT
   sentences that form the splits. The estimator's signature admits no records,
   labels or splits.
2. **Hyperparameter selection.** σ, λ and centres are chosen inside `fit` by
   cross-validation on those same NBC scores. No split data is in scope.
3. **Stopping-threshold tuning.** `tune_ddre_thresholds` is called with
   `validation_records`; its AST contains no reference to `test_records` and it
   accepts no test argument.
4. **The tuner's quality constraint** uses `bse_val_metrics`, computed on
   `validation_records`.
5. **All uses of `test_records`.** In `main.py` the name is loaded at lines 307
   (smoke-test subsetting only, writing to separate files explicitly marked "not
   paper results"), 411/419/427 (the three final evaluations), 479 (a count),
   and 511–513 (prediction rows). There is **no** load of `test_records` before
   the tuner returns, other than the smoke-test path.
6. **The split itself** is by passage, seeded, disjoint by assertion, and its
   membership is recorded in the summary (`validation_passage_ids`,
   `test_passage_ids`).

One non-issue worth naming: validation tuning warms the shared NLI cache with
document scores that the test evaluation later reads. This is not leakage — the
cached scores are deterministic functions of (premise, hypothesis) and carry no
label information.

---

## 6. DDRE / BSE fairness

### VERDICT: FAIR ON EVIDENCE; TWO PROTOCOL ASYMMETRIES TO DISCLOSE

**The evidence stream is shared.** Both detectors receive the same
`SentenceRecord`s, the same subclaims, the same `subclaim.documents` in the same
order, the same `max_docs` slice, the same `EntailmentScorer.score_document`
(hence the same 400-word/100-overlap segmentation, the same 4,000-word
truncation, the same NLI model, the same max-over-spans aggregation), the same
labels, the same `min`-over-subclaims sentence rule, and the same
`cost_based_prediction`. Verified by a recording scorer asserting identical
document order.

**The one permitted difference** is representation: DDRE sees the continuous
score; BSE sees `discretize_document_score(score)`. Quantisation is part of
BSE's algorithm, so this is allowed by the contract. **No other hidden evidence
advantage was found.**

**Two asymmetries that are not about evidence:**

**D-02 — an algorithm/protocol asymmetry in the retrieval floor.** BSE evaluates
`should_continue` before the first fetch and can retrieve zero documents; DDRE
evaluates only after an update and always retrieves at least one. The two
efficiency numbers therefore have different origins, and **this currently
disadvantages DDRE**: it must spend one document per subclaim (~2,990 in total)
before it can win on cost, and where BSE retrieves nothing, DDRE cannot win by
construction.

*A threshold pre-check does not fix this.* With `P0 = 0.5`, every `lower` in the
grid is ≤ 0.40 and every `upper` is ≥ 0.60, so `P0` lies strictly inside the
band and a band evaluated before the first document **can never stop**. Adding
one would change nothing. BSE's zero-retrieval behaviour comes from its
decision-theoretic rule — it compares the expected cost of retrieving against
the cost of stopping now — not from the mere fact that it checks early. Giving
DDRE an equivalent Bayes-risk pre-retrieval decision would be a **stopping-rule
redesign**, not a small fairness fix, and it should be treated and reviewed as
such.

Recommended for now: **report BSE's zero-retrieval frequency** (subclaims and
sentences at which it declines the first fetch) and the retrieval-floor
difference alongside every efficiency comparison, so a reader can see which part
of any gap is method and which is protocol.

**D-09 — a claim boundary, not an invalid comparison.** DDRE selects its
stopping band from 64 validation configurations; published BSE has no tunable
counterpart, because its stopping rule follows from the published costs. Part of
any DDRE advantage may therefore be a model-selection advantage.

This bounds what the comparison may claim; it does not invalidate it. **The
primary comparator stays published BSE — official mode, CM = 28, CFA = 96,
c_retrieve = 1** — because that is what preserves comparability with Wang et al.
If the tuning-budget question is worth studying, the right shape is a
**secondary robustness analysis**: published BSE remains primary, and a
validation-tuned BSE variant is reported alongside it, clearly labelled as *not*
the published Wang configuration. Replacing the published baseline with a tuned
one would trade the paper's comparability for a fairness argument that a
labelled secondary analysis already answers.

---

## 7. Stopping-threshold consistency with the CM/CFA classification threshold

### The inconsistency exists in the current code. CONFIRMED.

**Exactly when it occurs.** The final rule is factual iff
`(1−P)·C_M < P·C_FA`, i.e. `P > C_M/(C_M+C_FA)`. For CM=28/CFA=96 that is
**0.2258064516**. The tuner searches
`lower ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}` — of which
**`{0.25, 0.30, 0.35, 0.40}` exceed the cost threshold**. Whenever such a
`lower` is selected and the posterior stops in the half-open window

    (0.225806, lower]

the detector has stopped *because it is confident the claim is hallucinated*,
and the cost rule then labels it **factual**. Confirmed end to end: with
`lower = 0.30` and a single document driving `P` to 0.28, the detector stops
after one document and returns `prediction = 1` (factual).

The upper side has no analogous defect: every value in `{0.60 … 0.95}` is above
the cost threshold, so stopping high always yields "factual", which is
consistent.

**Why this is not merely cosmetic.** The selector is lexicographic: average
documents per sentence is optimised first, and balanced and factual PR-AUC only
break **exact retrieval-count ties**. A larger `lower` stops earlier and so
retrieves fewer documents. Within the feasible set, therefore, nothing but the
document count pushes back on choosing a contradictory threshold. (This audit
did not run tuning and makes no claim about how often exact ties occur — the
ordering alone establishes the concern.)

Note the same arithmetic for the Gate-1 primary configuration: CM=14/CFA=24 puts
the threshold at **0.368421**, where only `lower = 0.40` is inconsistent.

### Scientifically valid resolutions

1. **Constrain the grid to `lower ≤ C_M/(C_M+C_FA)`.** The stopping rule then
   means "confident enough that the cost rule will say nonfactual", which is
   what a sequential stopping rule is *for*. Costs the search four grid points
   and no expressiveness that the decision rule can actually use.
2. **Derive the band from the costs entirely** — e.g. stop when the Bayes stop
   risk `min((1−P)·C_M, P·C_FA)` falls below a tuned budget. This is closest to
   Wang's decision-theoretic framing and would make DDRE and BSE stop on
   comparable grounds. It is a redesign of the stopping rule, not a repair.
3. **Anchor the band on the threshold**, parameterising by a single confidence
   margin `δ`: stop low at `t − δ`, high at `t + δ` with `t = C_M/(C_M+C_FA)`.
   Consistent by construction and reduces the search to one dimension.
4. **Make the final classification agree with the stopping rule** — i.e. predict
   nonfactual on a stop-low. **Rejected**: it abandons Wang's cost rule, which
   the contract fixes, and would break comparability with the baseline.
5. **Keep the grid and report the contradiction rate.** **Rejected as a primary
   fix**: it documents an incoherence instead of removing it, and the tuner
   would still be biased toward it.

### Recommendation: option 1, with option 3 as the pre-registered alternative

Option 1 is the minimal change that removes the contradiction, keeps the search
space two-dimensional and interpretable, and does not touch Wang's cost rule or
BSE. It costs only configurations that were incoherent anyway. Option 3 is
worth adopting if the threshold grid turns out to be the dominant free parameter
in the final tuning, since it halves the search dimension and cannot produce the
contradiction under any cost setting — but it changes the meaning of the tuned
quantity, so it should be chosen *before* tuning, not after seeing results.

Option 2 is the most principled and the most work; it is the right direction for
a follow-up, not for this comparison.

**Whichever option is chosen, it must be fixed before tuning**, because the
tuner's own objective prefers the region the fix removes.

---

## 8. Changes required before held-out evaluation

**Blocking — must be resolved before any tuning run:**

1. **D-01** — remove the stopping/classification contradiction (§7, option 1
   recommended). Fix before tuning, not after, because the selector optimises
   average documents first and a larger `lower` retrieves fewer.

**Measure before tuning, then decide:**

2. **D-03** — run the six-step measurement in §4-A on the actual formal NBC fit:
   quantiles of `log r̂` over the fitted score support, the fraction of observed
   document scores able to cross a stopping boundary after one document, `Σ α`,
   out-of-range fractions, and whether the clip is ever reached. **Only then**
   decide whether a per-document log-evidence cap is warranted, and pre-register
   it with a sensitivity analysis. **No cap value is chosen in this audit.**

**Required for the comparison to state its own boundaries:**

3. **D-02** — report BSE's zero-retrieval frequency and the retrieval-floor
   difference alongside the efficiency numbers. Do **not** treat a threshold
   pre-check as an equaliser; a Bayes-risk pre-retrieval decision for DDRE is a
   stopping-rule redesign and needs its own review.
4. **D-07** — add a nonfactual AUC-PR floor to the tuner's feasibility test.
5. **D-08** — add the same floor to `hypothesis_comparison`, and gate the
   supported/not-supported claim on the bootstrap rather than on a point
   estimate.
6. **D-09** — keep published BSE (CM = 28, CFA = 96, c_retrieve = 1) as the
   primary comparator; optionally add a validation-tuned BSE variant as a
   clearly labelled **secondary** robustness comparator, and state the
   tuning-budget asymmetry as a claim boundary.

**Required for defensible reporting:**

7. **D-04, D-06** — post-fit sanity check on the estimator; loud failure on
   non-finite scores.
8. **D-10** — add `n_subclaims` to `prediction_rows`; retain per-subclaim
   stopping depth.
9. **D-11** — fix the fallback's cost normalisation, or state its scale.
10. **D-12** — make result auto-push opt-in.
11. Pre-register the bootstrap protocol (resamples, interval, metrics,
    multiplicity) **before** the test split is evaluated.

**Disclose in the write-up (no code change):** D-05, D-13, D-14, D-15, D-17.

---

## 9. Tests added

`tests/test_ddre_audit.py` — 52 tests, numpy + stdlib only, no model, no GPU, no
test-split access. They pin current behaviour across the ratio orientation and
normal equations, centre selection, the held-out objective, positivity and
clipping, hyperparameter selection and determinism, log-odds accumulation and
conversion, stopping order, budget handling, span-call counting, the
zero-document case, sentence aggregation, the cost threshold and its strict
boundary, evidence-stream equality between BSE and DDRE, and the reporting
structures.

**Centre selection is proved on disjoint supports** — factual scores in
`[60, 95]`, hallucinated in `[5, 40]` — so every selected centre is asserted to
be a member of the normalised factual set and absent from the hallucinated-only
support. A centre drawn from the wrong sample would fail.

**Three tests record the corrections made in this revision:**

* `test_the_ratio_is_bounded_above_by_the_sum_of_the_coefficients` — the
  estimator is bounded by `Σ α`; the earlier "unbounded" claim is withdrawn.
* `test_neither_synthetic_fixture_reaches_the_clip_bounds` — `log r = ±13.8`
  arises only at the `[1e-6, 1e6]` bounds, which neither synthetic fit reaches.
* `test_a_threshold_pre_check_could_never_stop_at_the_prior` — with `P0 = 0.5`
  no `(lower, upper)` pair in the grid can fire before the first document, so a
  pre-check does not address D-02.

`test_the_published_primary_comparator_configuration_is_unchanged` pins BSE
official at CM = 28, CFA = 96, c_retrieve = 1, P0 = 0.5, so the published
primary comparator cannot be swapped for a tuned one silently.

Eleven tests are named `test_AUDIT_…` and assert behaviour this audit judges to be
wrong — the D-01 contradiction, the D-02 retrieval floor, the D-03 weakly
constrained extrapolation, the D-04 degenerate fit, the D-06 NaN propagation,
the D-07/D-08 missing nonfactual constraints, the D-09 tuning asymmetry and the
D-10 discarded subclaim detail. **They must be updated, not deleted, when each
fix lands**, so that no fix can land without visibly changing the recorded
behaviour.

The D-03 and D-04 tests use **synthetic** fixtures and are labelled as such in
the source: they establish mechanisms, not rates on the actual formal fit.

No production file was modified by this audit.
