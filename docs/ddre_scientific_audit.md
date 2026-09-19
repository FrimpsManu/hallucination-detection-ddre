# DDRE scientific audit — before held-out evaluation

Read-only audit of the DDRE/uLSIF implementation against the scientific contract
for comparison with Wang et al.'s (EMNLP 2023) Bayesian sequential estimation
baseline.

**Nothing in this audit changes research behaviour.** No formal experiment was
run, no GPU was used, no model was downloaded, no threshold was tuned, and the
held-out test split was never evaluated. The only code added is
`tests/test_ddre_audit.py`, 57 tests that pin current behaviour — including
behaviour this audit judges to be wrong, so that a later fix has to change a
test visibly.

Files audited: `src/ddre_core.py`, `main.py`, `src/evaluation.py`,
`src/wang_data.py` (splitting), `tests/test_core_math.py`.
`src/baseline_core.py` was read only to establish the comparison contract and
was not modified.

---

## 1. Executive verdict

### NOT READY

> **Status update.** The single blocker, **D-01, is RESOLVED by PR #8**
> ("research: enforce cost-consistent DDRE stopping thresholds"). Everything
> below is preserved as the historical record of what the defect was and how it
> was found; §10 records the resolution. No other finding is addressed by that
> PR, so the verdict stands until they are worked through.

**One blocker** must be resolved before any tuning run:

* **D-01 — RESOLVED (PR #8).** The stopping-threshold grid contained four values
  that contradict the CM/CFA classification rule, and the tuner's lexicographic
  objective optimises average documents first, so it searched that region
  freely. This was established by exact arithmetic on the released costs and by
  an end-to-end reproduction, not by simulation.

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
| D-01 | **BLOCKER — RESOLVED by PR #8** | `main.py::tune_ddre_thresholds`, `ddre_core.py::DDREDetector.detect_subclaim` | The lower stopping grid is `{0.05 … 0.40}`; the CM=28/CFA=96 classification threshold is `0.225806`. For `lower ∈ {0.25, 0.30, 0.35, 0.40}` the detector can stop *because it is confident of hallucination* and the cost rule then labels the claim **factual**. | The two rules disagree on the same posterior. The selector is lexicographic — average documents first, PR-AUC only on exact ties — and a larger `lower` stops earlier, so within the feasible set nothing but the document count pushes back on choosing a contradictory threshold. | **Done (PR #8):** §7 option 1 — `cost_decision_threshold` is the single source of the arithmetic, `DDREDetector` refuses `lower > t` or `upper ≤ t` at construction, and `cost_consistent_thresholds` filters the tuner's grid from the *configured* costs. See §10. |
| D-03 | **MAJOR / NEEDS EMPIRICAL CHECK** — **DIAGNOSTIC PROTOCOL FROZEN / INSTRUMENTED by PR #13** (empirical run still required) | `ddre_core.py::ULSIFDensityRatio.ratio` | The estimate is most empirically constrained where the training data provide local support. Sparse and tail regions are more sensitive to bandwidth, regularisation and kernel extrapolation, so the ratio there can be large or small for reasons that are not evidential. | A large ratio is **not** by itself a defect — where factual support is strong and hallucinated support is weak, a large ratio is exactly what the estimator should report. The question is whether large values come from stable local support or from weak-support instability, and that is **unmeasured** on the actual NBC fit; the demonstrations below use synthetic scores and establish only the mechanism. | Before tuning, run the measurement in §4-A over the fitted score support, **distinguishing large-but-stable ratios from weak-support instability**. Pre-register a per-document log-evidence sensitivity analysis only if the measurement warrants it. **Do not choose a cap value yet.** |
| D-02 | MAJOR — **REPORTED by PR #12** (not corrected) | `ddre_core.py::detect_subclaim` vs `baseline_core.py::detect_subclaim` | Algorithm/protocol asymmetry. BSE's stop/continue rule is decision-theoretic (expected cost of retrieving vs stopping) and can decline the **first** fetch, retrieving **0** documents. DDRE's rule is a fixed probability band evaluated after an update, so it has a floor of **one retrieval per non-empty evaluated subclaim**. | Retrieval counts have different origins. On the 190-passage held-out split that floor is **2,387 documents** (all test subclaims are non-empty); DDRE must spend that before it can win on cost, and where BSE retrieves 0, DDRE cannot win at all. **This currently disadvantages DDRE.** | Report BSE's zero-retrieval frequency and the retrieval-floor difference alongside the efficiency numbers. A threshold pre-check does **not** fix this (see §6); giving DDRE a Bayes-risk pre-retrieval decision would be a **stopping-rule redesign**, not a small fairness fix. **Done (PR #12), as reporting:** the summary now carries `retrieval_protocol_asymmetry` with BSE's observed zero-retrieval frequency, DDRE's structural floor and the documents above it. Both stopping protocols are unchanged, and no floor adjustment enters the confirmatory endpoint. See §14. |
| D-07 | MAJOR — **RESOLVED by PR #10** | `main.py::tune_ddre_thresholds` → `src/threshold_selection.py` | The feasibility test constrains only `factual_auc_pr` and `balanced_pr_auc`. Nonfactual AUC-PR — Wang's headline metric — is unconstrained. | Balanced PR-AUC is the mean of the two, so a large factual gain can mask a nonfactual loss and still qualify. The tuner may select a configuration that is worse at detecting hallucination. | **Done (PR #10):** feasibility now requires nonfactual AND factual AND balanced PR-AUC, each within the validation tolerance; all three booleans and all three deltas are recorded per candidate. The rule moved into `src/threshold_selection.py` so it is reviewable on its own. See §12. |
| D-08 | MAJOR — **RESOLVED by PR #10** | `main.py::hypothesis_comparison` → `src/paired_bootstrap.py` | `hypothesis_supported_on_test` requires `factual_delta > 0` and `balanced_delta ≥ 0`, with no nonfactual floor and **no uncertainty quantification of any kind**. | A single point estimate is written into the summary as a scientific conclusion, over three methods and several metrics, with no interval and no multiplicity control. | **Done (PR #10):** the point-estimate boolean is gone. The claim now comes from a pre-registered paired passage-level cluster bootstrap with a conjunctive rule and three claim statuses, and the pre-registration is **enforced**: bootstrap provenance, the actual run configuration and the split's passage IDs are each verified before a claim can be confirmatory, and the pairing is checked by per-sentence identity attached at evaluation time. See §12 and `docs/confirmatory_statistical_protocol.md`. |
| D-09 | MAJOR (claim boundary / robustness) | `main.py`, `ddre_core.py` | DDRE selects its stopping band from **64** validation configurations. Published BSE has **no tunable counterpart** — its stopping rule follows from the fixed published costs. | Part of any DDRE advantage may be a model-selection advantage. This bounds what the comparison may claim; it does **not** invalidate it. | Keep **published BSE (CM=28, CFA=96, c_retrieve=1) as the primary comparator** for comparability with Wang et al. Optionally add a validation-tuned BSE variant as a clearly labelled **secondary** robustness comparator, never as the primary. |
| D-04 | MAJOR — **RESOLVED by PR #9** | `ULSIFDensityRatio._solve` / `.ratio` | Post-hoc non-negativity truncation could in principle drive `α → 0`. Then `r̂ ≡ 0`, clipped to `1e-6`, i.e. `log r = −13.8` **per document**. Nothing in `fit()` detects or reports it. | A silently degenerate fit would look like a spectacularly confident detector. **Not observed in the synthetic audit fixture, and not established on the actual formal fit** — no fixed raw NBC scores exist in this repository to check it against. | **Done (PR #9):** `_validate_final_fit` rejects non-finite parameters, an all-zero `α`, and fitted ratios that are non-finite or identically zero on the training support — before the estimator is usable. Rejection clears the fitted state. See §11. |
| D-06 | MAJOR — **RESOLVED by PR #9** | `.ratio`, `detect_subclaim` | A NaN score propagates silently: `log(NaN)` → `clip` → `sigmoid` all yield NaN; every stopping comparison is False, so the **full retrieval budget is spent**, and NaN reaches the metrics. `+inf` is clipped to a perfect score of 100. | Silent corruption of both the quality and the efficiency numbers, with no warning anywhere. | **Done (PR #9):** `ratio()` rejects a non-finite score before normalization and a non-finite raw ratio before clipping; `detect_subclaim` validates the score, then the ratio (finite and strictly positive), then the posterior. See §11. |
| D-05 | MINOR | `detect_subclaim` | The running log-odds are clipped to `±40` **after each update**, so accumulation is non-associative and evidence beyond the bound is discarded. | Harmless under any stopping band in the current grid (`sigmoid(±40)` is far outside it), but it is a silent modification of the stated update rule. | Document it, or clip only at the sigmoid. |
| D-10 | MINOR — **RESOLVED by PR #12** | `evaluation.py::prediction_rows`, `detect_sentence` | Per-subclaim results are collapsed into one `DetectionResult` with summed counters. Per-subclaim stopping depth is unrecoverable, and the CSV omits the subclaim count. | §4-G asks for stopping depth; §4-H needs the subclaim count to express documents-per-subclaim on a bootstrap resample. | **Done (PR #12):** `detect_sentence_with_trace` returns the exact subclaim results from the same pass, `EvaluatedSentence` carries `n_subclaims` and `subclaim_traces`, sentence totals are checked against trace totals, and the CSV exports the per-subclaim depth, availability and NLI-call vectors as JSON. See §14. |
| D-11 | MINOR — **RESOLVED by PR #11** | `main.py::tune_ddre_thresholds` | `normalized_docs = avg_docs / max_docs` divides a **per-sentence** document count by a **per-subclaim** budget. With ~1.57 subclaims/sentence it can exceed 1.0. | The fallback penalty's exchange rate against balanced PR-AUC is therefore not the declared one. | **Done (PR #11):** the fallback cost is now `avg_retrieved_documents_per_subclaim / max_documents_per_subclaim`, so numerator and denominator share a unit and the value is a genuine fraction in `[0, 1]`. A value outside that range raises rather than being clamped. Feasible selection is unchanged and still minimises documents *per sentence*. See §13. |
| D-12 | MINOR — **RESOLVED by PR #11** | `main.py::auto_push_results` | Result artifacts are committed and pushed automatically at the end of a full run unless `--no-push-results`. | A crashed or partial run can publish artifacts; the default direction is toward publishing, not toward review. | **Done (PR #11):** publication is opt-in behind `--push-results` (default off); `auto_push_results` refuses `main`, `master` and a detached HEAD **before staging anything**; the summary records the intent. See §13. |
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
| Validation / test subclaims | 603 / 2,387 (all non-empty) |
| Held-out DDRE retrieval floor (D-02) | 2,387 documents — **not** the full-dataset 2,990 |
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

The real issue is weaker and still worth acting on: **the estimate is most
empirically constrained where the training data provide local support.** `H`
constrains magnitude on the hallucinated sample and the centres carry mass near
the factual sample, so in sparse and tail regions the fitted value is decided
more by bandwidth, regularisation and kernel extrapolation than by nearby data,
and the bound `Σ_l α_l` is itself data-dependent and can be large.

**A large ratio is not by itself a defect.** Where factual support is strong and
hallucinated support is genuinely weak, a large `r̂` is exactly what the
estimator should report, and the resulting large log-evidence is legitimate. The
question the measurement below must answer is therefore *which of the two* is
happening: **large but stable density ratios**, or **large ratios produced by
weak-support instability**. Only the second is a problem, and only the second
would justify a cap.

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
the sequence under the default `[0.20, 0.80]` band. Note that `|log r̂|` is a
valid criterion **only for this symmetric special case** — a band symmetric in
log-odds together with `P0 = 0.5`, which makes `logit(P0) = 0` and
`logit(upper) = −logit(lower)`. The general rule is in the measurement plan
below and must be used for every other `(lower, upper)` pair. **This establishes
a mechanism, not a rate.** Those scores were drawn uniformly inside the released
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

**Measurement to run before final tuning** (read-only, one fit, no tuning, and
using the configured `P0` throughout):

1. Fit `ULSIFDensityRatio` on the actual formal raw NBC scores.
2. Evaluate `r̂` over the **fitted score support** — the observed NBC scores and
   the observed retrieved-document scores, not a uniform grid.
3. Report **quantiles of `log r̂`** (min, 1%, 5%, 25%, 50%, 75%, 95%, 99%, max)
   separately for the NBC scores and the document scores.
4. Report **one-document stopping fractions using the general log-odds rule**,
   not `|log r̂|`. After the first document,

       updated_log_odds = logit(P0) + log(r̂)

   and, with the **configured** `P0` rather than an implicit 0.5,

   * a one-document **LOW** stop occurs iff `updated_log_odds ≤ logit(lower)`;
   * a one-document **HIGH** stop occurs iff `updated_log_odds ≥ logit(upper)`.

   For **every** candidate `(lower, upper)` pair, report three fractions of the
   observed document scores separately:

   | | |
   | --- | --- |
   | low-stop fraction | `logit(P0) + log r̂ ≤ logit(lower)` |
   | high-stop fraction | `logit(P0) + log r̂ ≥ logit(upper)` |
   | either-stop fraction | the union of the two |

   Reporting them separately matters: a low stop and a high stop have opposite
   consequences for the D-01 contradiction and for the cost rule, so a combined
   figure hides which one is driving any retrieval saving.

   The `|log r̂|` shorthand collapses to this rule **only** when `P0 = 0.5` and
   the band is symmetric in log-odds (e.g. `[0.20, 0.80]`). Use it nowhere else.
5. Report `Σ α`, the fraction of observed scores falling outside the range of
   each training sample, and whether the `[1e-6, 1e6]` clip is ever reached.
6. **Distinguish large-but-stable ratios from weak-support instability.** For
   the scores with the largest `|log r̂|`, report local support (how many
   factual and hallucinated training scores lie within, say, one bandwidth) and
   the sensitivity of `r̂` to σ and λ across the selected grid. A large ratio
   with dense local support on one side and genuinely sparse support on the
   other is a legitimate estimate; a large ratio that moves substantially with
   bandwidth or regularisation is instability.
7. **Only then**, if the measurement warrants it, pre-register a per-document
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
  normalisation is dimensionally wrong (D-11 — **fixed in PR #11**; see §13),
  and (ii) the fallback maximises
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
3. Write the predictions CSV before the auto-push (D-12). ~~Largely moot as of
   PR #11: a normal run no longer pushes at all unless `--push-results` is
   given, so nothing is published behind the bootstrap input.~~
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
disadvantages DDRE**: its floor is **one retrieval per non-empty evaluated
subclaim**, and where BSE retrieves nothing, DDRE cannot win by construction.

The floor for the held-out comparison is **2,387 documents** — the number of
non-empty subclaims in the 190-passage test split, computed from the released
data without running the experiment (all 2,387 test subclaims have at least one
document). **The full-dataset figure of 2,990 subclaims is not the held-out
retrieval floor**; it spans all 238 passages, of which 48 are the validation
split (603 subclaims). The test-split floor should be restated alongside the
final evaluation, since it moves with any change to the split.

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

### The inconsistency existed in the code. CONFIRMED — and RESOLVED by PR #8.

*The analysis below is preserved unchanged as the record of the defect. The
resolution is in §10.*

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

1. ~~**D-01** — remove the stopping/classification contradiction (§7, option 1
   recommended). Fix before tuning, not after, because the selector optimises
   average documents first and a larger `lower` retrieves fewer.~~
   **DONE — PR #8.** See §10.

**Measure before tuning, then decide:**

2. **D-03** — run the seven-step measurement in §4-A on the actual formal NBC
   fit: quantiles of `log r̂` over the fitted score support; **low-stop,
   high-stop and either-stop fractions reported separately** for every
   `(lower, upper)` pair using the general rule
   `logit(P0) + log r̂` against `logit(lower)` / `logit(upper)` with the
   configured `P0`; `Σ α`, out-of-range fractions, and whether the clip is ever
   reached; and a local-support and σ/λ-sensitivity check that separates
   **large-but-stable** ratios from **weak-support instability**. **Only then**
   decide whether a per-document log-evidence cap is warranted, and pre-register
   it with a sensitivity analysis. **No cap value is chosen in this audit.**

**Required for the comparison to state its own boundaries:**

3. **D-02** — report BSE's zero-retrieval frequency and the retrieval-floor
   difference alongside the efficiency numbers. Do **not** treat a threshold
   pre-check as an equaliser; a Bayes-risk pre-retrieval decision for DDRE is a
   stopping-rule redesign and needs its own review.
4. ~~**D-07** — add a nonfactual AUC-PR floor to the tuner's feasibility test.~~
   **DONE — PR #10.** See §12.
5. ~~**D-08** — add the same floor to `hypothesis_comparison`, and gate the
   supported/not-supported claim on the bootstrap rather than on a point
   estimate.~~ **DONE — PR #10.** See §12.
6. **D-09** — keep published BSE (CM = 28, CFA = 96, c_retrieve = 1) as the
   primary comparator; optionally add a validation-tuned BSE variant as a
   clearly labelled **secondary** robustness comparator, and state the
   tuning-budget asymmetry as a claim boundary.

**Required for defensible reporting:**

7. ~~**D-04, D-06** — post-fit sanity check on the estimator; loud failure on
   non-finite scores.~~ **DONE — PR #9.** See §11.
8. **D-10** — add `n_subclaims` to `prediction_rows`; retain per-subclaim
   stopping depth.
9. ~~**D-11** — fix the fallback's cost normalisation, or state its scale.~~
   **DONE — PR #11.** See §13.
10. ~~**D-12** — make result auto-push opt-in.~~ **DONE — PR #11.** See §13.
11. Pre-register the bootstrap protocol (resamples, interval, metrics,
    multiplicity) **before** the test split is evaluated.

**Disclose in the write-up (no code change):** D-05, D-13, D-14, D-15, D-17.

---

## 9. Tests added

`tests/test_ddre_audit.py` — 57 tests, numpy + stdlib only, no model, no GPU, no
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

**Tests recording the corrections made across the review rounds:**

* `test_the_ratio_is_bounded_above_by_the_sum_of_the_coefficients` — the
  estimator is bounded by `Σ α`; the earlier "unbounded" claim is withdrawn.
* `test_neither_synthetic_fixture_reaches_the_clip_bounds` — `log r = ±13.8`
  arises only at the `[1e-6, 1e6]` bounds, which neither synthetic fit reaches.
* `test_a_threshold_pre_check_could_never_stop_at_the_prior` — with `P0 = 0.5`
  no `(lower, upper)` pair in the grid can fire before the first document, so a
  pre-check does not address D-02.

`TestOneDocumentStoppingRule` pins the **general** one-document rule that the
D-03 measurement plan must use. `test_the_state_after_one_document_is_logit_p0_
plus_log_r` verifies `logit(P0) + log r̂` against the detector for
`P0 ∈ {0.3, 0.5, 0.7}`;
`test_the_shorthand_is_exact_for_a_symmetric_band_at_p0_one_half` shows the
`|log r̂|` shorthand agrees **only** in that special case; and two
`test_AUDIT_the_shorthand_is_wrong_…` tests exhibit concrete disagreements — an
asymmetric band `[0.05, 0.80]` where the shorthand counts a stop that does not
happen, and `P0 = 0.35` where it misses one that does.
`test_low_and_high_stops_must_be_counted_separately` pins that the two stop
directions are disjoint and must be reported apart.

`test_the_published_primary_comparator_configuration_is_unchanged` pins BSE
official at CM = 28, CFA = 96, c_retrieve = 1, P0 = 0.5, so the published
primary comparator cannot be swapped for a tuned one silently.

Thirteen tests are named `test_AUDIT_…`. Eleven assert behaviour this audit
judges to be wrong — the D-01 contradiction, the D-02 retrieval floor, the D-03
sparse-region sensitivity, the D-04 degenerate fit, the D-06 NaN propagation,
the D-07/D-08 missing nonfactual constraints, the D-09 tuning asymmetry and the
D-10 discarded subclaim detail. The other two record a defect in an **earlier
draft of this audit** rather than in the code: the two cases where the `|log r̂|`
shorthand disagrees with the general one-document stopping rule. **They must be updated, not deleted, when each
fix lands**, so that no fix can land without visibly changing the recorded
behaviour.

The D-03 and D-04 tests use **synthetic** fixtures and are labelled as such in
the source: they establish mechanisms, not rates on the actual formal fit.

No production file was modified by this audit.

---

## 10. Resolution log

### D-01 — RESOLVED by PR #8, *research: enforce cost-consistent DDRE stopping thresholds*

Resolved by §7 **option 1**: constrain the search space so that stopping is
cost-consistent **both analytically and operationally**. Wang's cost rule is
unchanged.

**Analytic clause.** `lower ≤ t < upper`, where `t = C_M/(C_M + C_FA)`. The
invariant is asymmetric, and deliberately so: stopping LOW asserts the posterior
will classify **nonfactual**, and `P == t` does classify nonfactual under the
strict rule `(1−P)·C_M < P·C_FA`, so `lower ≤ t` admits equality; stopping HIGH
asserts **factual**, which `P == t` does *not* give, so `upper > t` is strict.

**Operational clause.** The analytic rule reasons about real numbers, but claims
are classified by `cost_based_prediction`, which compares two *rounded floats*.
Those disagree at the boundary for some cost pairs, in both directions:

| Cost pair | Edge | Analytically | Operationally |
| --- | --- | --- | --- |
| C_M=3 / C_FA=7 | `lower == t` | admissible | `cost_based_prediction(t) == 1` — a low stop there is classified **factual** |
| C_M=2 / C_FA=7 | `upper == nextafter(t, 1)` | admissible | still `== 0` — a high stop there is classified **nonfactual** (it takes 2 ULPs above `t`) |

So the boundaries are checked against the classifier itself:

```
cost_based_prediction(lower, C_M, C_FA) == 0
cost_based_prediction(upper, C_M, C_FA) == 1
```

`cost_based_prediction` is **monotone non-decreasing in `P`** — as `P` rises
`(1−P)·C_M` cannot increase and `P·C_FA` cannot decrease, and correctly-rounded
arithmetic preserves that ordering — so these two boundary checks are enough:
*every* posterior at or below `lower` classifies nonfactual and *every*
posterior at or above `upper` classifies factual. Those are exactly the two
claims the stopping rule makes. Monotonicity is itself pinned by test.

The classifier is the source of operational truth; the cost formula is not
restated. **Both experiment cost pairs (28/96 and 14/24) tie exactly at `t`**,
so `lower == t` remains admissible for them and neither search space shrinks —
verified by test.

Three changes, none of which touch `cost_based_prediction`, `BSEDetector`, the
costs, the histograms, the scorer, the uLSIF mathematics or the split:

1. **`ddre_core.cost_decision_threshold(c_miss, c_false_alarm)`** — one shared
   helper, with defensive validation (finite, non-negative, positive total). The
   detector guard, the tuner's search space and the recorded provenance all read
   the threshold from here, so the arithmetic exists once. It lives in
   `ddre_core.py` rather than beside `cost_based_prediction` so that
   `src/baseline_core.py` stays byte-identical.
2. **`threshold_consistency(lower, upper, c_miss, c_false_alarm)`** evaluates
   both clauses and returns the full verdict, including which clause failed and
   how each boundary was classified. `thresholds_are_cost_consistent` is the
   predicate form.
3. **`DDREDetector.__init__` refuses an inconsistent configuration**, naming the
   failed clause, `lower`, `upper`, the threshold, both costs and both boundary
   classifications. An incoherent detector cannot be built by hand, not merely
   avoided by the tuner. `BSEDetector` is deliberately untouched — its stopping
   rule is derived from the costs and must stay exactly as published.
4. **`cost_consistent_thresholds` derives the tuner's grid from the configured
   costs**, applying the analytic filter and the operational check to the grids
   and then confirming each pair with the *same predicate the constructor uses*,
   so the search-space provenance, the tuner and the constructor cannot
   disagree. Nothing is hardcoded: at CM=28/CFA=96 the lower grid becomes
   `[0.05, 0.10, 0.15, 0.20]` (32 of 64 pairs survive), while at CM=14/CFA=24
   the threshold is 0.3684 and `0.35` survives but `0.40` does not.

The experiment summary now carries `ddre.threshold_search_space` with the
decision threshold, the candidate and effective grids, the excluded values, the
surviving pairs and the counts — so the restriction is auditable prospectively
rather than inferred from a log line.

**Tests.** `tests/test_cost_consistent_thresholds.py` (53 tests) covers the
threshold value, the strict/non-strict asymmetry in both directions, the
constructor guard, both cost configurations, both floating-point edge cost
pairs, classifier monotonicity, the agreement between the reported grids and the
admissible pairs, and the property that a low stop now always classifies
nonfactual and a high stop always factual across every surviving pair. The D-01 tests in `tests/test_ddre_audit.py` were **updated,
not deleted**: they still assert that `cost_based_prediction` calls 0.25–0.40
factual, and now record that no detector can stop there. Ten mutations are
caught, including `lower <= t` → `lower < t`, `upper > t` → `upper >= t`,
dropping either half of the operational check, replacing the operational check
with an analytic comparison, removing either grid filter, hardcoding
`0.225806`, using `C_FA/(C_M+C_FA)`, having the constructor consult only the
analytic clause, and pointing the tuner at the unfiltered grid.

`cost_based_prediction` itself is **not** modified. The floating-point edge it
exhibits is handled by consulting it rather than by changing it, which is what
makes the invariant operational as well as analytic.

### D-04 and D-06 — RESOLVED by PR #9, *research: fail closed on invalid DDRE numerical states*

One principle: **an invalid numerical state must never be converted into
evidence.** A NaN is not a weak signal, an infinite score is not a perfect one,
and an identically-zero fitted ratio is not proof of hallucination.

**D-04 — degenerate fit.** `_validate_final_fit` runs after the final fit and
before the estimator is usable. It rejects: non-finite or empty centres; a
non-positive or non-finite σ; a negative or non-finite λ; an empty, non-finite
or negative `α`; an `α` with no strictly positive coefficient; and fitted ratios
on the observed training support that are non-finite, negative, or identically
zero. On rejection the fitted state is **cleared**, so `ratio()` reports "not
fit" rather than serving an invalid model — the 1e-6 floor can no longer turn
a broken fit into `log r = −13.82` per document.

The claim boundary is respected and pinned by test: **no fit is rejected for the
scale or spread of its ratios.** Large ratios, tiny-but-positive ratios, a narrow
range and heavy class overlap all still pass. Those are D-03 questions, and **no
log-evidence cap is chosen here**.

A read-only `fit_diagnostics` record (JSON-serialisable, reaching the summary as
`ddre.ulsif_fit_diagnostics`) carries the sample sizes, σ, λ, the selected CV
objective, the centre and coefficient counts, `α` sum/min/max, and the raw
fitted-ratio range on the factual, hallucinated and combined training support.
It is provenance only and selects nothing.

Model selection is also protected: a non-finite σ, λ or CV objective now fails
loudly rather than being skipped. Skipping would silently change the effective
search space — a scientific behaviour change, not a safety fix. This matters
because NaN loses every comparison, so an unchecked NaN objective could survive
as `best` purely through comparison semantics.

**D-06 — non-finite values.** `ratio()` rejects a non-finite score **before**
normalization (so `+inf` can no longer become a perfect 100) and rejects a
non-finite raw ratio **before** the clip, rather than clipping it into range.
The existing `[1e-6, 1e6]` clip is unchanged. `fit()` rejects non-finite training
scores with a bounded message naming the input set, the count and the first few
offending indices.

`DDREDetector.detect_subclaim` validates at three points: the document score
immediately after scoring, the returned ratio (finite **and strictly positive**)
before `log()`, and the posterior after the update. The score check duplicates
the estimator's own guard deliberately — the detector accepts any ratio-estimator
implementation, and the sequential accumulator must not depend on one of them
being careful. Invalid values are **not** repaired with an epsilon. Failure stops
retrieval immediately: later documents are never scored, `ratio()` is never
invoked for a bad score, and no `DetectionResult` is produced.

Errors are `NonFiniteScoreError` and `DegenerateULSIFFit`, both subclasses of
`DDRENumericalError(ValueError)`.

**Worth recording:** D-06 was DDRE-specific. BSE already fails on a non-finite
score, because its bucket discretiser calls `int()` on it — `ValueError` for NaN,
`OverflowError` for an infinity. That is incidental rather than a designed guard,
and its message names neither the subclaim nor the document, but it does mean BSE
never silently converted NaN into evidence. `baseline_core.py` is untouched.

**Failed and repeated fits.** A fit attempt must leave the object representing
*that* attempt, or no usable fit at all. `_clear_fit_state()` discards centres,
`α`, σ, λ, the diagnostics and the CV table, and it runs **at the start of every
fit attempt — before input validation and before model selection** — as well as
on a final-fit failure. Without it, a second fit that failed early (non-finite
training input, a non-finite CV objective) left the previous successful model in
place: `ratio()` went on serving evidence from a fit the caller believed was
replaced, and `fit_diagnostics` described that older fit, making provenance
ambiguous exactly when something had gone wrong. The final-fit failure path now
clears the *whole* state; previously it left σ and λ behind, describing a model
that no longer existed. After any failed attempt `ratio()` reports "must be fit
before use".

**Tests.** `tests/test_ddre_numerical_safety.py` (67 tests) covers training-input
validation, model-selection safety, every final-fit rejection reason, the
diagnostics record, `ratio()` input and output validation, all seven detector
failure modes, failed and repeated fit state management, and the claim
boundary. The D-04 and D-06 tests in
`tests/test_ddre_audit.py` were **updated, not deleted**, and still explain the
historical behaviour. Twenty-five mutations are caught, including removing any
individual guard, turning any failure into silent clipping, and failing to clear
any single element of the fitted state on a failed or repeated fit.

Three of those mutations initially escaped, which sharpened the tests: the
finite-`α` check was masked by the later fitted-ratio check (now pinned by
asserting *which* check fires), the fitted-ratio check needed a case with finite
coefficients whose sum overflows, and the posterior check needed the accumulator
itself faulted, since the score and ratio guards make it unreachable through
them.

**D-03 is NOT resolved.** Nothing in this PR bounds, caps or calibrates the
log-evidence a single document may carry.

### D-07 and D-08 — RESOLVED by PR #10, *research: freeze DDRE performance-preservation and confirmatory statistics*

Both were fixed **before any held-out DDRE result was inspected**. The full
prospective protocol is `docs/confirmatory_statistical_protocol.md`.

**D-07 — validation selection now protects nonfactual PR-AUC.** A threshold pair
is confirmatorily feasible only if it preserves BSE-official **nonfactual AND
factual AND balanced** PR-AUC, each within the validation tolerance. All three
booleans and all three deltas (`nonfactual_auc_pr_delta_vs_bse` and siblings) are
recorded per candidate, so the trade-off is visible in the saved validation table
rather than implied. The primary objective stays retrieval efficiency; quality
breaks only exact document-count ties, deterministically.

If nothing is feasible the predeclared fallback still selects a configuration for
exploratory work, but it is marked `confirmatory_validation_selection = false`
and **can never support the confirmatory claim**, however large the held-out
effect.

The rule moved out of `main.py` into `src/threshold_selection.py`: a decision
rule that determines what the paper may claim should be reviewable and
unit-testable on its own, not buried in a driver that needs a GPU to import.

*One honest note.* Given the same tolerance, the balanced clause is
**mathematically implied** by the other two — if `n ≥ bn − t` and `f ≥ bf − t`
then `(n+f)/2 ≥ (bn+bf)/2 − t`. It therefore cannot change any feasibility
verdict, and no behavioural test can detect its removal. It is kept because the
requirement is explicit auditability, and the implication is now stated by test
so the redundancy is documented rather than accidental.

**D-08 — the claim now comes from a pre-registered bootstrap.** The
point-estimate boolean is gone. `src/paired_bootstrap.py` freezes the protocol:
**10,000** paired resamples, unit = **passage** (190 held out), seed **42**,
**95% percentile** intervals, PR-AUC non-inferiority margin **0.005** absolute.

Sentences within a passage are not independent, so the bootstrap is a **cluster**
bootstrap: each replicate draws passages with replacement and carries each drawn
passage's complete sentence block, duplicated if drawn twice. It is **paired** —
the same draw is applied to both methods within each replicate, so the shared
between-passage variance cancels. Sign conventions are explicit and recorded on
every endpoint: performance is `DDRE − BSE`, efficiency is `BSE − DDRE`.
Efficiency is a difference, not a fraction, because a fractional estimand behaves
badly when the bootstrap denominator is small.

PR-AUC is `src.evaluation.wang_pr_auc`, now public, so the normal evaluation and
the bootstrap call **one** function — an interval computed on a different
estimand from the point estimate it brackets is not an interval for that
estimate.

The claim rule is **conjunctive**: confirmatory validation selection, validation
tolerance equal to the frozen margin, all three PR-AUC lower bounds at or above
−0.005, and the retrieved-documents lower bound strictly above 0. NLI span calls
are a **secondary** endpoint that changes only the wording. Three statuses —
`SUPPORTED`, `NOT_SUPPORTED`, `NOT_CONFIRMATORY` — and the last is **never**
collapsed into the second: "cannot address the claim" and "addressed it and the
evidence was against" are different scientific statements.

Invalid replicates **fail closed**: a one-class replicate raises
`BootstrapUnavailable` and the confirmatory analysis becomes unavailable.
Replicates are never dropped, redrawn, or given a substitute metric.

No p-values, and no multiplicity correction: the rule requires every primary gate
simultaneously, so there is no selection among endpoints to correct for.

**The pre-registration is enforced, not merely documented.** A frozen protocol
that nothing checks is a comment, so a run must *be* the pre-registered run
before any claim can be called confirmatory. Three verifications sit in front of
`assess_claim`, each listing every deviation it finds rather than repairing
anything:

* **Bootstrap provenance.** The record and *every endpoint inside it* must carry
  the frozen `bootstrap_unit`, `n_resamples`, `seed`, `ci_level`, `ci_method` and
  the correct sign convention, with a two-sided interval present. An endpoint
  that disagrees with its own header describes a different analysis from the one
  the header claims. Exploratory bootstraps — 200 resamples, seed 7, an 80%
  interval — remain perfectly legal and can never produce a confirmatory claim,
  however good their bounds look.
* **The metric implementation.** `paired_passage_bootstrap` accepts an injected
  `pr_auc`, so a run could otherwise carry every frozen setting and still be
  measuring a different quantity at high precision. The hook stays open for unit
  tests and exploratory work, but only the default path (`pr_auc=None` →
  `src.evaluation.wang_pr_auc`) may record the canonical
  `CONFIRMATORY_PR_AUC_DEFINITION`; an injected function is recorded as
  `custom:<module>.<qualname>` and is `NOT_CONFIRMATORY`. Equivalence is never
  inferred from a name — a function *called* `wang_pr_auc` that is not this
  repository's is still custom.
* **Interval validity.** Every required endpoint's bound must be present,
  numeric, finite and correctly ordered. `float("nan") >= -0.005` is False, so a
  NaN bound would otherwise fail its non-inferiority gate quietly and be
  reported as `NOT_SUPPORTED` — a negative result manufactured out of a broken
  computation. Corrupted intervals make the analysis unavailable, and no bound
  is repaired, clipped or reordered.
* **Run configuration.** The **actual** CLI values (`c_miss`, `c_false_alarm`,
  `c_retrieve`, `p0`, `max_docs`, `validation_fraction`, `split_seed`) are
  compared with `FROZEN_RUN_CONFIGURATION`. The frozen comparator is BSE official
  at C_M = 28, C_FA = 96, c_retrieve = 1. The secondary C_M = 14 / C_FA = 24 pair
  stays a legitimate analysis; it is simply not the comparator this protocol
  pre-registered.
* **Split identity.** The **actual passage IDs** of the validation and held-out
  sets are compared against the split re-derived from the released records at
  `validation_fraction = 0.20`, `random_state = 42`, and a SHA-256 fingerprint of
  each is recorded. Size is not identity: a *different* 190 passages is a
  different held-out set. A split that was never checked fails closed rather than
  being assumed correct.

**And the pairing itself is verified.** `DetectionResult` carries no identity, so
two result lists of equal length look paired even when one is permuted — and
nothing downstream could see it, because passage counts, sentence counts and
lengths all still agree while method A's sentence *i* is differenced against
method B's sentence *j*. Each result is therefore wrapped in an
`EvaluatedSentence` **inside the evaluation loop**, built from the same record
that produced it, and `validate_paired_inputs` compares
`(passage_index, sentence_index, gold_label)` at every position before any
resampling begins. Duplicate identities are rejected too: the released data has
one row per `(passage, sentence)`, so a duplicate means the evaluated set was
built wrongly.

**Not resolved by PR #10:** D-02, D-03, D-09, D-10, D-11, D-12 remain open.

---

## 13. Resolution log — PR #11

### D-11 — RESOLVED by PR #11, *research: correct fallback cost scale and make result pushes opt-in*

The defect, unchanged from the table above: the fallback objective computed
`normalized_docs = avg_retrieved_documents_per_sentence / max_docs`, dividing a
**per-sentence** document count by a **per-subclaim** budget. That is not a
fraction of anything, and with roughly 1.57 subclaims per sentence in this
dataset it can exceed 1.0.

**What the defect actually does, stated precisely.** Every threshold candidate in
one tuning run is evaluated on the *same* `validation_records`, so the sentence
and subclaim counts are fixed across candidates:

```
avg_docs_per_sentence = total_docs / N_sentences
avg_docs_per_subclaim = total_docs / N_subclaims

  =>  avg_docs_per_sentence
        = avg_docs_per_subclaim × (N_subclaims / N_sentences)
```

and `N_subclaims / N_sentences ≈ 1.57` is a **constant of the split**, identical
for every candidate. So no two real candidates differ in that factor, and the
defect is *not* that some configuration's sentences carried more subclaims than
another's — they cannot.

The defect is the **exchange rate**. The old cost was the correct cost multiplied
by that constant, so the fallback objective was effectively

```
balanced_pr_auc − (retrieval_penalty × subclaims_per_sentence) × correct_cost
```

i.e. it traded balanced PR-AUC against retrieval cost at ~1.57× the declared
`retrieval_penalty`. Balanced PR-AUC is not scaled alongside the cost, so the two
objectives are **not order-equivalent** and the old one could select a different
fallback configuration. `TestRealisticFallbackRegression` exhibits exactly that,
on a candidate pair that obeys the fixed-record invariant above (asserted
directly: both candidates share one `docs_per_sentence / docs_per_subclaim`
ratio).

The corrected cost is

```
normalized_document_cost = avg_retrieved_documents_per_subclaim
                         / max_documents_per_subclaim
```

equivalently `total_documents / (total_subclaims × max_docs)`. Both quantities
are documents per subclaim, so the ratio is dimensionless and lies in `[0, 1]`:
5 documents per subclaim against a budget of 10 scores 0.5, and the penalty's
exchange rate against balanced PR-AUC is the declared `retrieval_penalty` rather
than that value inflated by the split's subclaims-per-sentence constant. The
repository already computed `avg_retrieved_documents_per_subclaim`; it simply was
not the quantity being used.

Nothing is clamped. `max_docs ≤ 0` raises, and so does a normalized cost below 0
or above 1 — a subclaim cannot consume more than the budget, so such a value is
either a unit inconsistency or a corrupted count, and pulling it silently back
into range would hide exactly the error this check exists to catch. The
`retrieval_penalty` value itself is unchanged and is recorded, not rescaled.

Each candidate now records `avg_documents` (per sentence),
`avg_documents_per_subclaim`, `normalized_document_cost`,
`max_documents_per_subclaim`, `fallback_retrieval_penalty`, `fallback_objective`,
and the normalization in words, so a reviewer never has to infer the units.

**Scope.** This corrects the **fallback objective only**. Selection among
confirmatorily feasible configurations is untouched: minimum documents *per
sentence*, then balanced, nonfactual and factual PR-AUC tie-breaks, then a
deterministic threshold ordering. The corrected fallback may choose a different
*exploratory* configuration when nothing preserves baseline quality — that is
what the fix is for — and a fallback selection remains
`confirmatory_validation_selection = false`, unable to support the confirmatory
claim however large the held-out effect.

### D-12 — RESOLVED by PR #11

The defect, unchanged: a full run committed and pushed result artifacts unless
the user remembered `--no-push-results`, so the default direction was toward
publishing rather than toward review.

Publication is now **opt-in**. `--push-results` (default `False`) is required;
without it a run writes its artifacts to disk and nothing is staged, committed or
pushed. `--no-push-results` is retained as a hidden no-op so existing invocations
keep working. The two are mutually exclusive, and neither may default to a value
that resolves to pushing — asserted against `main.py`'s own argparse call rather
than a copy of it. A smoke test never auto-pushes even when the flag is given:
its outputs are debugging-only. Declining to push is reported as an ordinary
outcome, not an error.

`auto_push_results` itself now refuses `main`, `master` and a detached or
ambiguous HEAD, **before `git add`** — staging first and refusing afterwards
would still leave the index dirty on a protected branch. This repository's
workflow is branch → PR → review → merge, and an experiment helper must not
bypass it. There is no override flag.

It also refuses a **pre-existing dirty index**, again before `git add`. A plain
`git commit -m` commits everything already staged, so a contributor who had run
`git add src/unrelated_work.py` before starting the experiment would find that
file swept into a commit labelled "update full experiment results" — precisely
what this function's own docstring says it never does. The refusal is total:
nothing is unstaged, no partial commit is attempted, and the existing index is
left exactly as it was. The staged-changes probe fails closed on any answer but
a clean index — `git diff --cached --quiet` returning 0 continues, 1 refuses as
dirty, and any other code refuses because the check itself did not work. The
full order is: resolve branch → protected/detached → pre-existing index → `git
add` → post-add no-change → commit → push.

The summary records `result_publication.automatic_push_requested` and the policy.
That is the **intent** expressed on the command line, not whether a remote later
accepted the push; the already-written summary is never rewritten to insert a
push result.

**Not resolved by PR #11:** D-02, D-03, D-09 and D-10 remain open.

---

## 14. Resolution log — PR #12

### D-10 — RESOLVED by PR #12, *research: retain subclaim retrieval traces and report stopping asymmetry*

The defect, unchanged from the table above: `detect_sentence` collapsed the
per-subclaim results into one `DetectionResult` with summed counters and
discarded them, so the distribution of stopping depths could not be recovered,
and the predictions CSV omitted the subclaim count needed to express
documents-per-subclaim on a resample.

**Exact per-subclaim retrieval depths are now retained.** Both detectors gained
`detect_sentence_with_trace(record, scorer, *, use_cache=True)` returning
`(sentence_result, subclaim_results)`, where the subclaim results are an
immutable tuple in `record.subclaims` order. `detect_sentence` **delegates** to
it, so there is exactly one aggregation implementation per detector — two copies
would be free to drift, and a traced run could then disagree with an untraced
one on the same input.

The trace comes from the **same pass** that produced the result. It is
deliberately not obtainable by re-running `detect_subclaim` afterwards: that
would duplicate NLI work, distort the wall-clock accounting, touch the cache a
second time, and produce a trace of a *different* computation from the one that
produced the sentence result. A source-level test asserts each held-out method is
evaluated exactly once and that every held-out pass is the trace-bearing one.

**`n_subclaims` is exported.** `EvaluatedSentence` carries `n_subclaims` and
`subclaim_traces`; both are `None` — not empty — when an evaluation did not
record them, so a trace-less observation can never be mistaken for a sentence
that genuinely had no subclaims. The pairing identity stays
`(passage_index, sentence_index, gold_label)`, so PR #10's paired-bootstrap check
is unaffected. The predictions CSV gains `n_subclaims` plus
`subclaim_documents_used`, `subclaim_documents_available` and
`subclaim_nli_calls` as compact JSON arrays whose lengths equal `n_subclaims`.

**Sentence totals are checked against trace totals.** Before an observation is
kept, the retrieval depths must sum to the sentence's `documents_used`, the NLI
calls to its `nli_calls`, and the minimum subclaim posterior must equal the
sentence `p_factual` exactly. This is an integrity check, never a
recomputation — the detector's result is what is kept — and a mismatch raises
`SubclaimTraceMismatch`, because a trace that does not account for its result
would make every subclaim-level number wrong in a way nothing downstream could
see. `documents_available` is `min(len(subclaim.documents), max_docs)`: the raw
corpus length would overstate what the protocol could ever have used and make a
full-budget run look like early stopping.

`summarize_subclaim_efficiency` reports the depth distribution — histogram,
median, 95th percentile, zero- and one-retrieval counts — from the individual
traces. None of it is inferred by dividing sentence totals by a subclaim count,
which would reconstruct a mean and invent the distribution around it.

### D-02 — REPORTED AS A PROTOCOL ASYMMETRY by PR #12 (not corrected)

The defect description above is unchanged and still stands. What PR #12 adds is
**measurement, not a fix**, and the distinction matters:

* **No algorithmic fairness redesign was made.** Giving DDRE a Bayes-risk
  pre-retrieval decision would be a new stopping algorithm, not a small fairness
  adjustment, and §6 already shows a threshold pre-check cannot substitute for
  one (at P0 = 0.5 the prior lies strictly inside every candidate band).
* **BSE's pre-first-fetch stop is preserved.** `should_continue` is evaluated
  before document 1 exactly as before, and BSE may still retrieve zero documents
  for a non-empty subclaim.
* **DDRE's one-document floor on non-empty subclaims is preserved.** Its band is
  still evaluated only after an evidence update.
* **Raw retrieval counts remain the confirmatory efficiency quantity.** The
  frozen primary endpoint is still BSE minus DDRE retrieved documents *per
  sentence*, with no floor adjustment. `BOOTSTRAP_ENDPOINTS`, the resample count,
  seed, interval and margin are untouched.
* **The summary now reports the asymmetry alongside those counts:**
  `retrieval_protocol_asymmetry` gives BSE's observed zero-retrieval non-empty
  subclaim count and fraction, DDRE's structural floor, and
  `ddre_documents_above_first_retrieval_floor` — the last **descriptively only**,
  never as an adjusted metric.

The floor is defined **prospectively from the protocol**: the number of evaluated
subclaims with `documents_available > 0`, because the current DDRE protocol
consumes at least one document for each. Deriving it from observed depths would
make it a description of the data rather than a property of the protocol, and it
could then never be violated. Accordingly, a DDRE trace with
`documents_available > 0` and `documents_used == 0` **fails loudly**: the
implementation would no longer match the protocol whose floor the report states.
The identical pattern is *accepted and counted* for BSE, where it is precisely
the pre-first-fetch stop being measured.

**Not resolved by PR #12:** D-03 and D-09 remain open. D-03 — the density-ratio
support and stability question — is the next major pre-tuning empirical task.

---

## 15. Status update — PR #13

### D-03 — DIAGNOSTIC PROTOCOL FROZEN / INSTRUMENTED. **Empirical run still required before tuning.**

**D-03 is NOT resolved.** PR #13 pre-registers and implements the measurement;
it does not run it. The full protocol is
`docs/d03_density_ratio_diagnostic_protocol.md`; the methodology lives in
`src/density_ratio_diagnostics.py` (pure, GPU-free, unit-tested) and the runner
in `scripts/diagnose_ddre_ratio_support.py`.

The finding is often misread as "large density ratios are wrong". It is not: a
large ratio is what a density-ratio estimator should produce where the classes
separate. The measurable question is whether the evidence DDRE consumes is
**supported** by the training score regions and **stable** across the production
uLSIF hyperparameter surface — separating large-but-stable evidence from
tail-driven evidence.

Frozen by this PR, before any real number exists:

* **Validation only**, threshold-independent: all candidate documents for all
  validation subclaims up to `max_docs = 10` (48 passages, 383 sentences, 603
  subclaims, 5,969 document occurrences, 603 first documents). The held-out
  split is never scored, and the validation split is verified by passage
  **identity**, not count. Running DDRE first and inspecting only what it
  retrieved would condition the diagnostic on the untuned stopping policy.
* **Canonical batch-1 Wang-fidelity scorer**, gated by the existing
  `provenance_guard` / `score_compatibility` machinery, with the fixed 398-pair
  exact-equality probe and the source digest measured on both sides of it. The
  formal cache is read-only; new scores go to a derived cache bound to the
  certified digest.
* The historical limitation is **preserved**: `checkpoint_identity_established`
  may remain false because the v2 Gate report did not record its Hub revision.
  D-03 relies instead on the directly measured
  `score_compatibility_established`.
* **Support strata** `0 / 1-2 / 3-4 / 5-9 / >=10`, measured in the kernel's own
  normalized space at a radius of the selected sigma. No single "weak support"
  cutoff is declared.
* **Full sigma/lambda sensitivity** over exactly the production `cv_table`
  pairs, with the selected fit's final centre set held fixed so the surface
  isolates the hyperparameters.
* **The general log-odds one-document rule** with the actual cost-consistent
  CM=28/CFA=96 grid, reporting low / high / either separately. `|log r|` is not
  used as the criterion.
* **Three escalation triggers**, and no magnitude criterion anywhere. They can
  conclude only that a *separate* cap/calibration study is required; the
  diagnostic never emits a cap (`selected_cap` and `selected_calibration` are
  always `null`, `method_change_made` always `false`). A false result means the
  predeclared criteria did not fire, **not** that uLSIF is proven correct.
* **D-13** is measured descriptively from the two score populations the
  diagnostic already holds. It is not corrected and **not marked resolved**.

`--dry-run` reports the exact measurement population without loading a model,
so it can be reviewed before any GPU time is spent.

**Not resolved by PR #13:** D-03 (protocol only), D-09 and D-13 remain open.
