# Multivariate DDRE: design (draft for review)

**Status: DRAFT. Not frozen.** Written on 2026-10-14 before any multivariate
feature has been computed, any multivariate estimator has been fit, or any new
method has been run on any dataset. It becomes the preregistration once the
open decisions in §10 are settled and it is merged. Until then nothing in §5–§8
is run.

## 0. Why a new design

The SelfCheckGPT ablation, the FActScore transfer test and the audits that
followed established four facts:

1. **BSE's stopping rule is a posterior band.** At CM 28 / CFA 96 / C_retrieve 1,
   Wang's released one-step look-ahead continues exactly while
   P(factual) ∈ (0.204, 0.769). The frozen DDRE stopped on [0.20, 0.80]. The two
   detectors differed almost only in the per-document evidence curve log r(s).
2. **On one scalar score, uLSIF and the histogram estimate the same ratio**,
   r(s) = p(s | factual) / p(s | hallucinated). Which of the two reads fewer
   documents depends on where a dataset's scores fall: −20.7% documents on
   SelfCheckGPT, +9.4% on FActScore.
3. **The frozen uLSIF estimator was not normalised.** E_hallucinated[r] = 0.736
   on its own training data, and the 20 (σ, λ) fits range from 0.63 to 74.8
   (`ulsif_normalisation.json`, `d421015d…`). After normalising, frozen D's
   evidence at the dominant score (35) falls from −0.44 to −0.13, close to BSE's
   −0.14.
4. **398 training pairs cannot identify the ratio where most documents fall.**
   After normalising, the sign of the evidence at score 35 still splits
   15 positive and 5 negative across the 20 fits.

So a one-dimensional DDRE has nothing to offer over BSE's histogram. Direct
density-ratio estimation is useful where histograms fail, on **several
features at once**. This design tests that.

## 1. Research questions

All three concern **hallucination-detection performance** and its
**computational cost** (documents retrieved and NLI calls). None concerns the
factual accuracy of the text an LLM generates.

- **RQ1 (cost).** At the same detection quality as BSE, does multivariate DDRE
  read fewer documents and make fewer NLI calls?
- **RQ2 (quality).** At the same cost as BSE, does multivariate DDRE detect
  hallucinations better?
- **RQ3 (why).** Which components account for any gain: more training data,
  normalisation, or the additional features?

## 2. Evidence features

Each retrieved document *d* for subclaim *c* gets a feature vector **x**(c, d).
Every feature below comes from computation the pipeline already does: the NLI
forward pass that BSE runs anyway, or the retrieval order. **No feature adds a
model call**, so the cost comparison stays fair.

| Feature | Definition |
|---|---|
| e | max over spans of P(entailment), softmax at temperature 5 (Wang's score / 100) |
| k | max over spans of P(contradiction), same softmax |
| ρ | retrieval rank of *d*, 1 … K, used as log ρ |

The NLI scorer is extended to **store the three raw logits per span**, so any
temperature or feature is derivable later without rescoring. Its
entailment-at-temperature-5 output must equal the existing caches' scores
exactly on overlapping pairs; this check runs before anything else.

Candidate features **not** in the primary set (exploratory only): neutral
probability, span count, BM25 score, entity-title match. The last two exist
only for some datasets.

## 3. Estimators (cells)

All cells share the same stopping family (§5), the same prior P0 = 0.5, the
same sentence aggregation (minimum over subclaims) and the same documents in
the same order.

| Cell | Evidence model | Training data | Purpose |
|---|---|---|---|
| **A** | BSE histogram of e (Wang's 10 bins, Laplace +1) | Wang's 398 NBC pairs | published baseline |
| **A′** | BSE histogram of e | new training set (§4) | effect of more training data |
| **U** | 1-D uLSIF on e, normalised | new training set | effect of normalised DDRE on one feature |
| **M** | multivariate uLSIF on (e, k, log ρ), normalised | new training set | **the proposed method** |
| **L** | logistic regression on (e, k, log ρ), converted to a ratio | new training set | the standard classifier-based ratio estimator |

**Fitting rules (all estimators), fixed now:**
- Features are standardised on the training set.
- Gaussian kernel. Centres are 100 points drawn from the factual training
  samples, as in the existing uLSIF code.
- σ and λ are chosen **only by the uLSIF cross-validation objective** on the
  training set, on a grid fixed in the code before fitting. No detector cost or
  detection metric is ever used to choose any hyperparameter. This removes the
  640-configuration selection that made frozen D post hoc.
- **Self-normalisation:** after fitting, r is divided by its mean over the
  hallucinated training samples, so E_hallucinated[r] = 1 by construction.
- L uses an L2-regularised logistic regression with C chosen by 5-fold
  log-loss. Its ratio is r = (p / (1 − p)) · (n_h / n_f).
- The normalisation diagnostic from PR #38 is reported for every fitted
  estimator.

**Training pairs.** One pair per (subclaim, retrieved document), labelled by
the subclaim's own S/NS label. This is exactly the quantity the sequential
update multiplies: p(**x** | claim factual) / p(**x** | claim hallucinated).
Like BSE, the sequential update treats documents as independent given the
label. That assumption is shared and is disclosed.

## 4. Data and splits

**Training (new).** FActScore human-annotated facts. Fact-level S/NS labels are
clean, unlike SelfCheckGPT's sentence-level labels. The three generators share
almost every entity: InstructGPT 182, ChatGPT ∪ PerplexityAI 178, overlap 177.
So the split must be **by entity**:

- The 183 FActScore entities are split 50/50 by a hash of the entity name
  (SHA-256, fixed now).
- **Training entities:** all three generators' facts (InstructGPT, ChatGPT,
  PerplexityAI).
- **Test entities:** ChatGPT and PerplexityAI facts only, with the exclusions of
  the transfer preregistration (§1 there).
- "Bridget Moynahan", which also appears in SelfCheckGPT, is excluded from both.
- Evidence comes from the existing BM25 pipeline, unchanged (top 10, query =
  entity name + fact). InstructGPT facts need one retrieval run.

**Test sets.**

| Test set | Size | Status for this design |
|---|---|---|
| SelfCheckGPT held-out | 190 passages, 1,525 sentences | used before with A–D; M, L, U, A′ never run on it |
| FActScore test entities | about half of the transfer set | used before with A–D; new methods never run on it |
| **Factcheck-Bench** | 94 responses, 661 claims, ≤ 5 ranked documents | **never used: the untouched test** |

Factcheck-Bench is the only dataset no analysis has touched, and the only
confirmatory test (§7). It is small, so its intervals will be wide; that is
reported, not hidden. Claims labelled "not enough evidence" are excluded. A
sentence is hallucinated if any of its claims is false.

## 5. Stopping rule and cost–quality curves

Every cell stops on a **posterior band** [l, u] around the cost rule's decision
threshold. §0.1 shows this is BSE's own rule.

- **Operating point.** l = 0.204, u = 0.769, BSE's own band at
  CM 28 / CFA 96. Labels use the cost rule (factual iff P > 0.2258), as before.
- **Curve.** The band is swept over a grid fixed in the code (half-widths in
  log-odds from 0.25 to 4.0 in steps of 0.25, around the decision threshold).
  Each point gives (documents per sentence, balanced PR-AUC) for each cell.
- A's official look-ahead result is also reported, and should lie on A's curve.

## 6. Endpoints

On each test set, with one paired cluster bootstrap across all cells
(10,000 resamples, seed 42, 95% percentile intervals). Clusters are passages
(SelfCheckGPT), entities (FActScore) and responses (Factcheck-Bench).

| RQ | Endpoint | Definition |
|---|---|---|
| RQ1 | **matched-quality saving** | docs(A at the operating point) − docs(M at the same balanced PR-AUC), interpolated on M's curve |
| RQ2 | **matched-cost gain** | balanced PR-AUC(M at A's documents per sentence) − balanced PR-AUC(A), interpolated on M's curve |
| both | same-band differences | docs and balanced PR-AUC of M − A at the operating point |
| RQ3 | component chain | the same two endpoints for A → A′ → U → M, and M vs L |

Curves and interpolation are recomputed inside every bootstrap replicate. If
M's curve never reaches A's quality, the RQ1 endpoint is "not reached" and is
not extrapolated.

## 7. Decision rules (fixed now)

- **Confirmatory, Factcheck-Bench only:**
  - RQ1 is supported if the 95% CI of the matched-quality saving lies above 0.
  - RQ2 is supported if the 95% CI of the matched-cost gain lies above 0.
  - The two are tested separately at 0.05 each, with Holm correction across
    the two.
- **SelfCheckGPT held-out and FActScore test entities:** the same endpoints and
  rules, reported as **pre-specified replications on previously used data**,
  not as confirmatory tests.
- **RQ3:** descriptive and exploratory. No causal or proof language.
- A result counts as "lower cost and better detection" only if both endpoints
  meet their rule.

## 8. Execution order (each step a reviewed PR, run once after merge)

1. Merge this design as the preregistration.
2. **Logit-storing scorer:**
   - code, plus the exact-equality check against the existing caches;
   - one scoring run over the training set, the SelfCheckGPT held-out documents,
     the FActScore documents and Factcheck-Bench.
3. Factcheck-Bench loader and InstructGPT retrieval (code, then run).
4. Estimators A′, U, M, L, fit once on the training set, with the normalisation
   diagnostic. No test data is read.
5. Replay of all cells on all test sets: curves, endpoints, one result per
   dataset.
6. Results summary, then the paper.

## 9. Known limitations, stated in advance

- Documents are treated as independent given the label (shared with BSE).
- The training set and the FActScore test share genre, generators and retrieval
  pipeline, though not entities. The SelfCheckGPT and Factcheck-Bench tests do
  not share these.
- Factcheck-Bench is small and capped at 5 documents.
- Cost is counted in documents and NLI calls. Wall-clock time is measured
  separately.

## 10. Open decisions (for review before freezing)

1. **Training data:** the entity-disjoint FActScore split (§4, recommended), or
   another source.
2. **Confirmatory dataset:** Factcheck-Bench (recommended; the only untouched
   option that fits the sequential setup). RAGTruth was considered, but it
   provides at most 3 documents per claim and measures faithfulness to a given
   context rather than factuality, so it has almost no retrieval cost to save.
3. **Cell L (logistic regression):** recommended. Reviewers will ask whether a
   plain classifier does as well. If it does, the contribution is the
   multivariate evidence model, not uLSIF specifically, and the paper should
   say so.
4. **Precision:** float16 as before, or float32. Decided after the float32
   Gate 1 check reports.
