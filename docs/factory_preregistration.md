# Second confirmatory test, on FACTORY — preregistration

**Status: frozen when merged; every later change is a dated amendment at
the end.**
- Written on 2026-10-26, before any detector, NLI model or retrieval touched
  FACTORY.
- It repeats the HELM test (`docs/helm_preregistration.md`, confirmed on
  2026-10-17, `docs/helm_results_summary.md`) on a second dataset. Method,
  hypotheses, statistics and decision rule are HELM's unless stated here.

**Language:**
- All results concern **hallucination-detection performance** and its cost.
- No causal or proof wording. A hypothesis is "confirmed" or "not confirmed"
  by the rule in §5, or the test is "uninformative" by §5.1.

---

## 0. Why this test

On HELM, a preregistered run-once test confirmed both hypotheses:
- **H1:** DDRE-1D's cost–quality envelope lies above BSE's (AUC-gap +0.043 at
  28/96, +0.037 at 14/24).
- **H2:** it also lies above fixed-k max entailment's (+0.008, +0.016).

One dataset is one domain: Wikipedia continuations by seven 2023 open models.
This test asks whether the same result holds on a second untouched dataset
of a different kind: long-form answers to fact-seeking questions, written by
six 2025 retrieval-augmented models and fact-checked by people with web search.

## 1. Hypotheses

Exactly HELM's §1:

| | Hypothesis | Cost setting (C_M/C_FA) |
|---|---|---|
| **H1a** | DDRE-1D's envelope lies above BSE's: AUC-gap > 0 | 28/96 |
| **H1b** | same | 14/24 |
| **H2a** | DDRE-1D's envelope lies above fixed-k max entailment's: AUC-gap > 0 | 28/96 |
| **H2b** | same | 14/24 |

## 2. Data (frozen)

### 2.1 Release

**Source:** FACTORY, from Chen et al., *FACTORY: A Challenging Human-Verified
Prompt Set for Long-Form Factuality*, 2025
([arXiv 2508.00109](https://arxiv.org/abs/2508.00109)).

**Repository:** Hugging Face dataset
[`facebook/FACTORY`](https://huggingface.co/datasets/facebook/FACTORY),
revision `c820453a0c8637ad0eea2692b7e03e79acc591d1`. Licence CC BY-NC 4.0.

**File:** only `fact_checking/human_annotations.jsonl` is used, SHA-256
`4d1904629e6f4570e3d59efb1d674cddeb2a76921bb223937acf9b7f63e0598f`. It is
copied read-only to `~/ddre-artifacts/confirmatory-v3/factory/release/`.

**How the annotations were made** (paper §4.1):
- **Models:** Claude 3.7 Sonnet, Gemini 2.5 Pro, DeepSeek V3, GPT-4o, Qwen3
  and Llama 4 Maverick, each retrieval-augmented (top-20 MassiveDS passages),
  answering LongFact, FactBench Hard and FACTORY prompts.
- **Sampling:** 100 sentences sampled at random from each model's output per
  benchmark. The file holds 2,999 records; the paper does not give their
  breakdown by model or benchmark.
- **Claims:** extracted automatically; annotators could edit them.
- **Labels:** annotators judged each claim with online search as supported,
  unsupported or inconclusive (VeriScore's labels). 25 annotators, about 12
  minutes per sentence.

### 2.2 Units, labels, clusters

**Records:** each line of the file is one annotated sentence (`question`,
`response`, `target_sentence`, and up to five claims, each with a `Tag`,
`Source` and `Snippet`).

**Tags:** a claim's tag is read after stripping whitespace and
case-folding. Empty tags mean no claim in that slot.

| Tag | Meaning |
|---|---|
| `factual` | supported |
| `nonfactual` | unsupported |
| `inconclusive` | no supporting information found |
| `no verifiable facts` | not a factual claim |

Any other tag stops the run (gate G0).

**Label, primary (Rule A):**
- **Hallucinated** (label 0): at least one claim is `nonfactual` or
  `inconclusive`.
- **Factual** (label 1): otherwise, with at least one `factual` claim.
- **Excluded:** sentences with no labelled claim, or only `no verifiable
  facts`.

Rule A is FACTORY's own scoring, where a claim without support counts against
factual precision, and FActScore's, where "not supported" is non-factual.

**Units:** every record that Rule A labels, in file order. The unit ID is the
record's 0-based line number. Duplicate sentence texts in different records
stay separate units.

**Clusters:** the question. Sentences answering the same question share it,
across models and responses.

**Totals for gate G0:**
- 2,999 records;
- 2,983 units: 1,938 factual and 1,045 hallucinated;
- 1,373 clusters;
- 16 excluded: 10 with no labelled claim, and 6 with only `no verifiable
  facts`.

**Not available:** the file does not record which model or benchmark a
sentence came from, so there is no per-generator analysis.

### 2.3 Claims and evidence

**Subclaims:** each sentence (`target_sentence`) is its own single subclaim,
as on HELM. The released claims are not used: they were edited by the
annotators while labelling.

**Not used:** the annotators' `Source` URLs and `Snippet` texts. They were
chosen to justify each label, so they would leak it. The `response` text is
not used either.

**Retrieval:** exactly HELM's (`src/wikipedia_bm25.py`):
- BM25 (SQLite FTS5) over every passage of the enwiki-20230401 snapshot,
  index SHA-256
  `79d0e38df6414235731e4c469b951e9ec884a07bce4746680f28f01f6d89925f`;
- query = `query_terms(question, target_sentence)`, the role HELM's article
  title plays: the topic, then the sentence, stopwords removed, first
  occurrence kept;
- top 10 passages in rank order, premise = page title + passage.

The retrieved lists are written once, read-only, and pinned before any NLI
scoring.

## 3. Method (frozen)

Exactly HELM's §3:
- **NLI scoring:** Wang's checkpoint, float16 on MPS, batch size 1,
  `FEATURE_VERSION mddre-nli3-temp5-seg400-overlap100-v1`. Gate F as on
  HELM.
- **Estimators:** the frozen plan-v2 estimators,
  `~/ddre-artifacts/v2/estimators_v2_float16.json`, SHA-256
  `6a007e379a3fc84fabb6d367ede41a3c81c53accc4f7f86b0a29e2ec546fadec`.
  Tier 1: BSE and DDRE-1D. Tier 2, secondary: NB-M and DDRE-M. Nothing is
  refitted.
- **Stopping:** backward induction over min(available documents, K).
- **Operating points:** C_M/C_FA ∈ {28/96, 14/24}; K = 1 … 10 × C_retrieve ∈
  {0.125, 0.25, 0.5, 1}; operating point K = 10, C_retrieve = 1.
- **Fixed-k max entailment:** K = 1 … 10.
- **Curves:** step envelopes of balanced AP against NLI calls per sentence.

## 4. Endpoints and statistics

Exactly HELM's §4:
- **Primary endpoint:** the AUC-gap.
- **Bootstrap:** paired cluster bootstrap over questions, 10,000 replicates,
  seed 42, the same draws for every method and point.
- **p-value:** each hypothesis gets the one-sided `p_greater`.

## 5. Decision rule

Exactly HELM's §5: Holm over H1a, H1b, H2a and H2b at family-wise α = 0.05.
A hypothesis is supported if Holm rejects and the 5th percentile of its
replicates is above 0. The statements are HELM's.

### 5.1 Informativeness (new)

**The risk:** the annotators checked against the whole web, but our evidence
is 2023 Wikipedia. If Wikipedia lacks the facts, every detector is near
chance (as on ExpertQA, where BSE's AP was 0.516), and a null result would
say nothing about the ratio estimator.

**Rule, fixed now:** the test is **informative** only if fixed-k max
entailment at K = 10 (the method that reads the most evidence and uses no
estimator) has balanced AP whose 95% bootstrap interval lies above 0.5.

**If not informative:** the result is reported as "uninformative: the
evidence does not separate the labels". It is not reported as "not
confirmed", and H1 and H2 are shown only as descriptive numbers. This rule
cannot turn a failed hypothesis into a success: it applies whatever DDRE-1D
does, and it uses only a baseline that does not involve DDRE.

## 6. Secondary and sensitivity analyses (descriptive, no decision)

**As on HELM:**
- **At the operating point:** E-cost, E-quality, Brier and ECE.
- **Wang's Table 1 metrics** for every method.
- **Other comparisons:** NB-M vs DDRE-M; BSE vs fixed-k; BSE-official.

**Sensitivity:**
- **Rule B labels:** hallucinated only if at least one claim is
  `nonfactual`. Sentences whose only non-supported claims are `inconclusive`
  are excluded. Expected 2,243 units (1,938 factual, 305 hallucinated) in
  1,187 clusters.
- **Pooled HELM + FACTORY:** the AUC-gap with clusters from both datasets,
  descriptive only.

## 7. Gates (any failure stops the run before anything is written)

| Gate | Check |
|---|---|
| P | this file, plan v2 at amendment 2, the estimators, the BM25 index and the FACTORY file match their pins |
| G0 | the totals in §2.2; every tag in the table in §2.2 |
| R | retrieval complete: one top-10 list per unit, written once (units with fewer than 10 hits are counted and reported) |
| F | the 398 NBC pairs score identically to the earlier float16 caches |
| X2 | on SelfCheckGPT held-out, BSE-official reproduces the frozen replays row for row |
| X4 | on SelfCheckGPT held-out, the (K = 10, C_retrieve = 1) points reproduce Phase 5's operating points |

## 8. Execution order

Each step is a reviewed PR, run once from clean `main`:

1. This preregistration.
2. The FACTORY loader with gate G0 and the retrieval script (dry run: counts
   only). Then retrieval is run once.
3. Feature scoring of the retrieved passages (gate F): about 30,000 pairs,
   roughly 1.5 hours.
4. The confirmatory replay.
5. The results summary.

The code reuses HELM's scripts. Only the loader and the query differ.

## 9. Known limitations, stated in advance

- **Evidence:**
  - Labels rest on web search, while our evidence is 2023 Wikipedia (§5.1).
  - Some 2025 facts postdate the snapshot.
  - The generators' own retrieval (MassiveDS) differs from ours.
  - **Queries:** FACTORY questions are long, so their terms can dominate the
    query. Sentences of one answer may then retrieve similar passages.
- **Labels:**
  - One annotator per sentence. No agreement statistic is published.
  - Annotators could edit claims before labelling. Our unit is the
    unedited sentence.
- **Generators:** frontier 2025 chat models with retrieval, unlike HELM's 2023
  open base and chat models. Their errors may be harder to detect.
- **Training mismatch:** the estimators were fitted on Wang's NBC data and on
  FActScore, not on FACTORY.
- **Licence:** CC BY-NC 4.0. Research use only; the data are not
  redistributed.
- **Power:** 2,983 units in 1,373 clusters, against HELM's 3,736 in 229.
  Clusters are small (at most 11 sentences), so the effective sample size
  should be similar to HELM's or larger. No formal power calculation is
  possible without running a detector.

## 10. What was seen before writing this

Before this preregistration was written, the following were seen on FACTORY:
- the README, the file listing and the paper;
- the record keys;
- counts of claim tags, of labelled claims per sentence and of sentences per
  question;
- the sentence-level counts under Rules A and B;
- the first four words of the questions, counted, to rule out FActScore-style
  biography prompts.

**No sentence or claim was viewed together with its label. No retrieval,
NLI score or detector output exists for FACTORY.**

---

## Amendments

None.
