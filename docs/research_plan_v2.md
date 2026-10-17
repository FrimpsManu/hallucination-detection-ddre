# Research plan v2 — DDRE as a drop-in likelihood ratio for BSE

**Status: decisions recorded in §9; awaiting merge. Nothing in it has been run.** It follows the
repository audit of 2026-10-17, which found that no experiment so far compares
DDRE with BSE with step 5 as the only difference. This plan defines that
experiment and everything the paper needs around it. Each phase below is its
own reviewed PR, run once from clean `main` after merge, as before.

---

## 0. The paper in one paragraph

**Working title:** *Toward Efficient Retriever-aware Hallucination Detection in
Large Language Models via Direct Density Ratio Estimation.*

**Setting:**
- Wang et al. (EMNLP 2023), *Hallucination Detection for Generative Large
  Language Models by Bayesian Sequential Estimation* (BSE), reads retrieved
  documents one at a time.
- After each document, BSE updates P(factual) with a likelihood ratio
  r(f) = P(f | factual) / P(f | hallucinated).
- It stops when a Bayes-risk look-ahead says another document is not worth its
  cost.
- BSE gets r by counting two 10-bin histograms of entailment and dividing them.
  The density-ratio literature warns against exactly that two-step approach.

**Method:** the identical BSE pipeline, with **only** r replaced by a
**directly estimated density ratio** (uLSIF):
- **DDRE-1D** uses entailment alone.
- **DDRE-M** also uses contradiction and the retriever's rank, which is what
  makes it "retriever-aware".

**Goals:**

| Goal | Question | Status |
|---|---|---|
| **G1, cost** | At equal detection quality, does DDRE need fewer NLI calls and documents than BSE? | settled |
| **G2, detection** | At equal cost, does DDRE separate factual from hallucinated claims better than BSE? | settled |
| **G3, calibration** | Are DDRE's posteriors better calibrated than BSE's? This tests the premise directly: a better ratio should give truer probabilities. | approved 2026-10-17 |

**Language:**
- Say "hallucination-detection performance".
- No causal or "proof" wording.
- Results are "supported" or "not supported" by a stated test.

## 1. What existing results remain, and their status

All frozen artifacts stay as they are, and every number stays true as
reported. What changes is how each result is described.

| Earlier stage | What it actually tested | Status in the paper |
|---|---|---|
| Gate 1 reproduction | BSE vs Wang Table 1 | Reproduction. PASS at 28/96; FAIL at 14/24 (§6.4) |
| Frozen held-out 1-D DDRE (−20.7% documents) | Tuned band and σ (640 configurations) vs untuned BSE; the ratio is unnormalised (E_h[r] = 0.736) | Prior exploratory work; disclosed as tuned and biased |
| FActScore transfer | Same detector, new data | Prior exploratory work (+9.4% documents) |
| Multivariate study | Data, features and estimator changed together, under BSE's band | Prior exploratory work |
| Confirmatory test (ExpertQA, BingCheck) | DDRE-M under BSE's band | **A valid preregistered test of that configuration. Not confirmed.** Reported in full |

**Data already used:**
- The SelfCheckGPT held-out split, FActScore, Factcheck-Bench, ExpertQA and
  BingCheck have all had detector results examined.
- Any new analysis on them is **exploratory by definition**.
- A confirmatory claim in v2 needs a dataset no detector has touched (§5.3).

## 2. Design principles

1. **One factor at a time.** Every headline contrast changes exactly one thing:
   the ratio estimator, the features, or the training data.
2. **Wang's own stopping rule for every method.** Each method is stopped by its
   own Bayes-risk look-ahead (§3.2). Nobody borrows BSE's band.
3. **Each method's own cost knob.** Cost–quality curves come from sweeping
   Wang's retrieval cost C_retrieve, not from rescaling a band.
4. **Equal tuning budget.** DDRE hyperparameters come only from the uLSIF
   cross-validation objective. No detection metric selects anything for any
   method.
5. **Cost in NLI calls first.** Documents per sentence is reported second.
   Wall-clock is measured on cold caches.
6. **Metrics robust to ties.** Average precision is reported alongside Wang's
   trapezoidal PR-AUC.
7. **Same habits as before:** preregister, freeze, run once from clean `main`,
   pin by SHA-256, and report every outcome.

## 3. Method

### 3.1 The ratio is the only thing that differs

Wang's Eq. 4 depends on the two likelihoods only through r:

  π' = π·r / (π·r + 1 − π)

So the update, the prior (0.5), the documents and their order, the
min-over-subclaims aggregation and the final decision rule
(1 − P)·C_M < P·C_FA are shared by every method, and are implemented once.

### 3.2 Wang's Bayes-risk look-ahead for any ratio

Wang's Eq. 8 continues if

  C_retrieve + E_next[ R_stop(π') ] < R_stop(π),  where R_stop(π) = min((1 − π)·C_M, π·C_FA).

The expectation is over the next document under the predictive mixture
π·p_f + (1 − π)·p_h. Since p_f = r·p_h,

  E_next[g] = E_h[ (π·r(x) + 1 − π) · g(x) ]

This needs only:
- the ratio r;
- a sample from the hallucinated-claim feature distribution, which is the
  estimator's own training sample.

**For BSE**, with E_h taken over its histogram, this is exactly Eq. 8. So:
- **BSE-eq8 and DDRE-eq8 differ only in r**, plus the representation of p_h
  that each estimator naturally has (bins vs the empirical sample).
- This pair is the **primary contrast**.

**Details:**
- **DDRE-M:** the next document's rank is known (n + 1), so E_h is taken over
  hallucinated training documents **at that rank**.
- **Decision regions:** for a fixed estimator and fixed costs, the decision
  depends only on π. It is computed exactly per step and cached on a fine
  π-grid. A gate checks that the grid and the exact computation agree.
- **BSE-official** (Wang's released code) averages the 10 next posteriors
  uniformly instead of using Eq. 8. It is reported as the reproduction
  reference next to BSE-eq8, so readers can see whether the two BSE variants
  differ.

### 3.3 Cells

**Tier 1 — Wang's setting, one feature, Wang's training data (398 NBC pairs):**

| Cell | Ratio | Feature | Stop |
|---|---|---|---|
| BSE-official | 10-bin histogram, counted (Wang) | entailment | Wang's code |
| **BSE** | 10-bin histogram, counted | entailment | Eq. 8 |
| **DDRE-1D** | uLSIF, self-normalised, CV-only | entailment | Eq. 8 |

**Tier 2 — retriever-aware, larger training data** (FActScore entity-disjoint
split, BM25 Wikipedia, 10 ranked documents per fact):

| Cell | Ratio | Features | Stop |
|---|---|---|---|
| Hist-1D | 10-bin histogram | e | Eq. 8 |
| DDRE-1D′ | uLSIF | e | Eq. 8 |
| **NB-M** | Naive-Bayes histograms (per-feature bins) | e, k, log rank | Eq. 8 |
| **DDRE-M** | uLSIF | e, k, log rank | Eq. 8 |
| Logit-M | logistic-regression ratio | e, k, log rank | Eq. 8 |

**Single-factor contrasts:**

| Contrast | Factor isolated | Status |
|---|---|---|
| BSE → DDRE-1D (Tier 1) | ratio estimator, Wang's own data and setting | **primary** |
| NB-M → DDRE-M (Tier 2) | ratio estimator with three features | primary |
| Hist-1D → DDRE-1D′ | ratio estimator, larger data | secondary |
| Hist-1D → NB-M, DDRE-1D′ → DDRE-M | features | secondary |
| BSE → Hist-1D | training data | secondary |
| Logit-M → DDRE-M | direct-ratio family: uLSIF vs probabilistic classification | secondary |

**Baselines from Wang's Figure 2:** fixed-k with k = 1…K, aggregated by
Maximum, Average and NBC. Fixed-k is the obvious cheap baseline for any cost
claim.

### 3.4 Estimator fixes found by the audit

- **Always self-normalise:** E_h[r] = 1 on the training sample.
- **Cross-validation folds grouped by claim (or entity)**, not by document, so
  the 10 documents of one fact never straddle folds.
- **σ grid extended below its current edge.** If the selection lands on an edge
  again, it is reported.
- **Ratio clipping:** the ±13.8-nat clip is kept, and the number of clipped
  updates is reported.
- **Rank feature:** reported per retriever. An ablation without rank shows how
  much "retriever-aware" contributes.
- **NBC training features:** Wang scores NBC premises whole, truncated at 512
  tokens, while test documents are scored as the max over spans.
  - Tier 1 keeps Wang's choice, for fidelity.
  - A sensitivity analysis re-scores the NBC premises with span-max for both BSE
    and DDRE-1D.
  - The 199 + 199 pair count is disclosed.

## 4. Evaluation

### 4.1 Cost–quality curves (G1, G2)

- **Sweep:** C_retrieve on a log grid (e.g. 0.125 … 16, 13 points) at
  C_M/C_FA = 28/96, and again at Wang's default 14/24.
- **Point per method and setting:**
  - x = mean NLI calls per sentence (secondary x: documents);
  - y = quality (§4.2).
- **Envelope:** each method's curve is replaced by its monotone (running-max)
  envelope. This avoids the first-crossing fragility found in the audit.

**Endpoints, per contrast and dataset:**

| Endpoint | Goal | Definition |
|---|---|---|
| E-cost | G1 | NLI calls saved at the reference method's quality, at Wang's operating point 28/96, C_retrieve = 1, on the envelopes |
| E-quality | G2 | quality gain at the reference method's NLI calls, same point |
| AUC-gap | G1 and G2 | area between the two envelopes over their shared cost range: one number for the whole curve |

**When the target is out of range:**
- If a method never reaches the target quality, E-cost is "not reached".
- It is reported as a **fraction of bootstrap replicates**, plus a censored
  estimate. No −1e9 sentinel goes into percentiles.

### 4.2 Detection metrics (G2)

- **Primary quality:** balanced average precision, i.e. the mean of factual and
  hallucinated AP.
- **Wang's full Table 1 for every method and dataset:** nonfactual and factual
  AUC-PR (trapezoidal, as Wang), accuracy, documents per sentence and NLI calls.
  Passage-level Pearson and Spearman are added on SelfCheckGPT, the only
  dataset with passage scores.
- **Also reported:** macro-F1 and ROC-AUC.
- **Ties:** posteriors are rounded to 1e-9 before ranking, so floating-point
  noise cannot reorder ties.

### 4.3 Calibration (G3)

- **Metrics:** Brier score, ECE (15 equal-mass bins) and reliability diagrams,
  for P(factual) at the operating point.
- **Prior:** calibration is judged on each dataset's own label prior, so the
  difference between the prior and the 0.5 prior is reported, not hidden.

### 4.4 Runtime (G1)

- **Cold-cache wall-clock** per sentence on a fixed random subset (e.g. 200
  sentences per dataset), the same hardware for every method, split into:
  - NLI;
  - estimator evaluation;
  - decision logic.
- **Fit time** of each estimator.
- **Statement:** whether estimator overhead is negligible next to DeBERTa,
  backed by these numbers.

### 4.5 Statistics

- Paired cluster bootstrap: 10,000 resamples, seed 42, the same draws for every
  cell. Clusters as before.
- Two-sided 95% percentile CIs for estimation.
- Confirmatory tests (§5.3) are **one-sided** in the hypothesised direction,
  with Holm over the preregistered family.
- p-values reported as p < 1e-4 rather than 0.

## 5. Data

### 5.1 Training

- **Tier 1:** Wang's NBC pairs (199 + 199), with Wang's preprocessing.
- **Tier 2:** the existing FActScore entity-disjoint training split (91
  entities, 7,382 facts, 73,820 documents). Its features are already cached.

### 5.2 Exploratory evaluation (all five used sets)

| Dataset | Comparison with Wang's setting |
|---|---|
| SelfCheckGPT held-out | Wang's own data |
| FActScore test entities | |
| Factcheck-Bench | |
| ExpertQA | flagged as low-signal: BSE is at chance |
| BingCheck | at most 3 pages, so little cost to save |

Everything is computed from cached features: no new NLI except the runtime
subset.

### 5.3 Fresh confirmatory dataset (G1 and G2 confirmation)

**Criteria, all required:**
- LLM-generated text;
- human claim- or sentence-level labels;
- **saved ranked evidence with many documents per claim (ideally ≥ 10)**, so
  that cost can differ;
- a usable licence;
- no detector results examined by us.

**Candidates to survey:**
- FELM;
- RAGTruth (fixed context, which may not fit);
- additional FActScore-style generator sets, with our BM25 index;
- re-retrieving evidence for a labelled LLM dataset with our own retriever,
  which is fully controlled and allows K = 10–50.

AVeriTeC has rich saved web evidence, but its claims are written by people, not
LLMs. It is a candidate only for a secondary cost-scaling analysis, not for the
hallucination claim.

**Commitment:** if no dataset meets the criteria, the paper says that v2 is
exploratory and does not claim confirmation.

## 6. Fidelity and disclosure items

1. **Deviations table, paper vs Wang's released code:**
   - spans: 400 words with overlap 100 (step 300), first 4,000 words;
   - 3-class NLI with temperature 5;
   - rounding plus two discretisers;
   - uniform-average look-ahead;
   - default costs 28/96;
   - 199 + 199 NBC pairs;
   - whole-premise NBC scoring.

   We follow the code, and say so.
2. **The 1-D history in full:** the protocol, the 0.005 margin, D-03, the
   640-configuration sweep, the NOT_CONFIRMATORY status and the unnormalised
   ratio. Prerequisite: a written **D-03 outcome and decision record**, which no
   document currently has.
3. **The six BingCheck labels printed during the survey**, disclosed in the
   confirmatory summary and the paper.
4. **The 14/24 reproduction failure:** run `scripts/reproduce_wang_original.py`
   (Wang's own code) to see whether the gap is ours or theirs. Needs a CUDA
   machine or a CPU/MPS patch (question Q1).
5. **BingCheck licence:** email sent 2026-10-16. Record the reply, or the lack
   of one.

## 7. Repository and paper

- **README rewrite:**
  - top section: title, the method, the goals, a results table, run commands,
    licences and citation;
  - move the reproduction forensics (about 75% of the file) to
    `docs/reproduction_gate1.md`;
  - fix the stale statements (DDRE "drops the look-ahead", the float32 rerun,
    auto-push).
- **One naming scheme everywhere:**
  - methods: BSE, DDRE-1D, DDRE-M, NB-M, Logit-M;
  - goals: G1, G2, G3;
  - gates: "R-gates" (reproduction) vs "E-gates" (engine);
  - retire the reused letters A–D and the RQ numbering that differs from doc
    to doc.
- **Paper, rebuilt on this plan:**
  - related work;
  - BSE recap;
  - DDRE and uLSIF formulation, including the §3.2 look-ahead;
  - pipeline figure;
  - deviations table;
  - Tier 1 and Tier 2 results with Wang's metrics;
  - cost–quality curves;
  - calibration;
  - runtime;
  - prior exploratory work and the earlier confirmatory test, reported
    honestly;
  - limitations.

## 8. Phases

Each phase is one PR. Code PRs carry tests. Runs happen once, after merge.

| # | Phase | Output | New compute |
|---|---|---|---|
| 0 | This plan approved; merge PR #51 (it records the frozen confirmatory result) | plan merged | none |
| 1 | Records: D-03 outcome, deviations table, label-slip disclosure, naming map | docs PR | none |
| 2 | Engine: generic Eq. 8 look-ahead for any ratio estimator (§3.2); fixed-k baselines; gates (BSE-eq8 vs official on SelfCheckGPT; π-grid vs exact; DDRE-eq8 sanity on synthetic data with a known ratio) | code PR | none |
| 3 | Estimators: Tier 1 DDRE-1D on NBC; Tier 2 NB-M; grouped CV, σ grid, self-normalisation; refit Tier 2 cells | code PR + fit run | minutes |
| 4 | Metrics: balanced AP, Wang's table, calibration, censored "not reached", envelopes, AUC-gap | code PR | none |
| 5 | Exploratory v2 replay on the five used sets, all cells, C_retrieve sweeps at 28/96 and 14/24 | run + results summary | hours, from cache |
| 6 | Runtime measurement on cold caches | run | about 1–2 h |
| 7 | Fresh dataset survey, then a preregistration of the confirmatory v2 test, written from Phase 5 results | docs PRs | none |
| 8 | Scoring and confirmatory run on the fresh dataset | run + summary | depends on the dataset |
| 9 | README rewrite; paper rebuilt | docs PRs | none |
| — | Wang original-code reproduction (§6.4), in parallel | run + note | CUDA |

**Time:**
- Phases 0–6 and 9 fit before early November if each PR is reviewed promptly.
- Phases 7–8 depend on finding a suitable fresh dataset.
- The preprint can go out with v2 labelled exploratory and the confirmatory test
  marked as in progress. If it does, it says so.
- The deadline sets the pace, not the standard. Anything not done properly is
  left out and stated as future work.

## 9. Decisions (2026-10-17)

- **Q1, Wang's original code:** no CUDA machine. Patch Wang's code to run on
  CPU/MPS and disclose the patch.
- **Q2, live retrieval:** none. Use saved evidence plus our own BM25 retriever,
  which also allows K = 10–50.
- **Q3, PR #50:** closed. The paper is rebuilt in Phase 9.
- **Q4, G3 calibration:** approved as a third goal.

## 10. Amendment 1 (2026-10-17): the stopping rule

Recorded after Phase 3's fit and before any v2 detector was run on any
evaluation data. It changes §3.2's primary stopping rule. Everything above
this section is unchanged.

**Finding:**
- Wang's paper defines the look-ahead as Eq. 8, a one-step rule. Wang's
  released code uses a different heuristic: the stop risk of the unweighted
  mean of the ten next posteriors (`docs/wang_deviations.md` row 16).
  Our BSE reproduction matches Wang's numbers because it follows the code.
- Implemented exactly, Eq. 8 **never reads a document at Wang's costs**
  (C_M 28, C_FA 96), for any C_retrieve > 0. Two independent implementations
  agree (`src/baseline_core.py` eq8 and `src/sequential.py`):
  - At P0 = 0.5 the decision is "factual" (threshold 0.226).
  - No single document's histogram ratio can carry P below 0.226: the strongest
    negative bin has log r = −0.69, and logit(0.226) = −1.23.
  - The one-step value of a document is therefore exactly 0.

  This is the known myopia of one-step look-ahead. At 14/24 the one-step
  value at P0 is 0.43, so Eq. 8 reads only when C_retrieve < 0.43.
- Continue regions at 28/96, C_retrieve = 1, with Wang's histogram:

  | Rule | Continue while P is in |
  |---|---|
  | BSE-official (code) | (0.204, 0.769) |
  | Eq. 8 | (0.048, 0.335): excludes P0 |
  | backward induction over 10 documents | (0.034, 0.813) |

**Decision (approved by the author, 2026-10-17):**
- **Primary rule: backward induction**, the Bayes-optimal finite-horizon rule.
  Wang's §3.4 names it as optimal and approximates it because exact induction
  over discrete features grows exponentially. Our state is the scalar
  posterior, so it is a small dynamic programme on a logit grid
  (`BackwardInductionPolicy`).
- **Same structure as Eq. 8:** it uses the same predictive expectation (§3.2)
  over each estimator's own support, at the next document's rank. With one
  document left it is Eq. 8 exactly.
- **Horizon:** the subclaim's available documents, capped at the dataset's
  maximum.
- **Fairness:** every method (BSE, DDRE-1D, Hist-1D, DDRE-1D′, NB-M, DDRE-M,
  Logit-M) is stopped by this rule, so step 5 remains the only difference.
- **Also reported, for transparency:**
  - BSE-official, Wang's code: the reproduction reference;
  - the one-step Eq. 8 for every method, including where it reads nothing.
- **Unchanged:**
  - the operating point (28/96, C_retrieve = 1);
  - the C_retrieve sweeps at 28/96 and 14/24 (§4.1);
  - the metrics, the statistics and the data.

**Names (`docs/naming.md`):**
- **BSE** now means Wang's histogram under backward induction.
- **BSE-eq8** means Wang's histogram under the one-step Eq. 8.
- **BSE-official** means Wang's code.

**Verification** (`tests/test_backward_induction.py`):
- horizon 1 equals Eq. 8 to 1e-9;
- values and decisions match full enumeration on small supports;
- regions shrink as documents run out and as C_retrieve rises;
- rank conditioning;
- Wang's case kept as a regression test.

Five of six deliberate mutants are caught. The sixth removes a clip that
`np.interp` already applies, so its behaviour is identical.

**Phase table:** Phase 2b (this amendment and its code) is inserted before
Phase 4.

## 11. Amendment 2 (2026-10-17): a budget knob (exploratory)

Recorded after Phase 5's results (`docs/v2_results_summary.md`) had been
seen. Everything it adds is therefore **exploratory**. It changes no Phase 5
result.

**Why:**
- Under backward induction the C_retrieve sweep is nearly all-or-nothing.
  Between C_retrieve ≈ 1.4 and 2, most methods jump from reading nearly every
  document to reading none.
- The curves therefore have no middle-budget points, and the envelope
  endpoints compare against gaps.
- Phase 5 also showed that Wang's fixed-k "Maximum" baseline is competitive,
  but did not bootstrap it.

**What is added (`src/v2_budget.py`, `scripts/run_v2_budget.py`):**
- **Knob K:** the maximum number of documents per subclaim, from 1 to the
  dataset's maximum.
  - It is crossed with C_retrieve ∈ {0.125, 0.25, 0.5, 1}, the values below the
    cliff plus the operating point.
  - Backward induction plans over min(available documents, K).
- **Curves:** a method's curve is the step envelope over all its (K,
  C_retrieve) points. Point (K = maximum, C_retrieve = 1) is Phase 5's
  operating point.
- **Methods:** BSE, DDRE-1D, NB-M, DDRE-M, at C_M/C_FA 28/96 and 14/24. Also
  fixed-k max entailment for K = 1 … maximum.
- **Contrasts and endpoints (bootstrapped):**
  - **BSE → DDRE-1D and NB-M → DDRE-M,** anchored at the reference's
    operating point: net E-cost in NLI calls and in documents, net E-quality,
    AUC-gap.
  - **Fixed-k → each of the four methods,** anchored at the method's own
    operating point:
    - the method's quality minus the best fixed-k quality within the
      method's cost;
    - the extra cost fixed-k needs to reach the method's quality;
    - the fraction of replicates in which fixed-k never reaches it;
    - AUC-gap.
- **Statistics:** paired cluster bootstrap, 10,000 replicates, seed 42, as in
  §4.5.
- **Gates:**
  - X0: pins, now including this amendment and the Phase 5 result;
  - X2: BSE-official reproduces the frozen replays, as before;
  - **X4:** the (K = maximum, C_retrieve = 1) points reproduce Phase 5's
    operating-point predictions exactly, for every method.

**Not changed:** Phase 5's frozen result, the estimators, the metrics and the
data.
