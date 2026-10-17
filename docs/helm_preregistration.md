# Confirmatory test on HELM — preregistration

**Status: frozen when merged; every later change is a dated amendment at
the end.**
- Written on 2026-10-17, before any detector, NLI model or retrieval touched
  HELM.
- The hypotheses come from the exploratory results of plan v2:
  - `docs/v2_results_summary.md`;
  - `docs/v2_budget_results.md`.

**Language:**
- All results concern **hallucination-detection performance** and its cost.
- No causal or proof wording. A hypothesis is "confirmed" or "not confirmed"
  by the rule in §5.

---

## 0. Why this test

**What the exploratory work suggests.** Across five used datasets:
- **DDRE-1D vs BSE.** DDRE-1D (BSE with only the likelihood ratio replaced
  by direct density-ratio estimation, both fitted on Wang's NBC data) had a
  cost–quality envelope above BSE's on every dataset at both cost settings.
  All ten AUC-gap intervals exclude 0, but the effects are small (+0.003 to
  +0.041 AP).
- **Against fixed-k.** BSE's envelope lay below Wang's fixed-k
  maximum-entailment baseline on most datasets. DDRE-1D's lay at or above it.

**Why it is not yet a claim.** These patterns were found after repeated use
of those datasets, so they are exploratory. This test checks them once on a
dataset no detector has touched.

## 1. Hypotheses

| | Hypothesis | Cost setting (C_M/C_FA) |
|---|---|---|
| **H1a** | DDRE-1D's envelope lies above BSE's: AUC-gap > 0 | 28/96 |
| **H1b** | same | 14/24 |
| **H2a** | DDRE-1D's envelope lies above fixed-k max entailment's: AUC-gap > 0 | 28/96 |
| **H2b** | same | 14/24 |

**Mapping to the goals:**
- H1 is goals G1 and G2 together: better quality at equal cost, or the same
  quality at lower cost, averaged over the shared cost range.
- H2 asks whether DDRE-1D is worth running instead of the simple baseline.

## 2. Data (frozen)

### 2.1 Release

**Source:** HELM, from Su et al., *Unsupervised Real-Time Hallucination
Detection based on the Internal States of Large Language Models*, Findings of
ACL 2024 ([arXiv 2403.06448](https://arxiv.org/abs/2403.06448)).

**Repository:** [oneal2000/MIND](https://github.com/oneal2000/MIND), commit
`3573342d20db0feb6658a26347337ea8a4217526`, cloned to
`~/ddre-artifacts/confirmatory-v2/helm/release`.

**Pins:** the nine files under `helm/` are listed with their SHA-256 in
`~/ddre-artifacts/confirmatory-v2/helm/file_sha256.txt`, and that listing's
own SHA-256 is
`3a315449f46e2e21feff715308c21c48b9dc50ac2a05e0dedbe8f078f9fe33bc`.

### 2.2 Units, labels, clusters

**Units:**
- A unit is every released sentence of every generator (7 data files), each
  with its own label.
- **Totals for gate G0:** 3,736 units, 1,978 with label 1, 229 clusters, 7
  generators.

**Labels:**
- The released `label` gives the hallucination flag: 1 means hallucination
  (repository README).
- Our factual label is 1 − `label`.
- No unit is excluded. How the paper's "Unverifiable" category maps to the
  binary label is not documented; the authors were asked on 2026-10-17
  (§9). The released labels are used as they are.

**Clusters:** the cluster is the Wikipedia article (the passage ID, which is
shared across generators).

**Generators:** falcon40b, gptj7b, llamabase7b, llamachat13b, llamachat7b,
mpt7b and opt7b. The paper describes six generators; mpt7b is in the
release but not in the paper (§6, sensitivity).

### 2.3 Claims and evidence

**Subclaims:** each sentence is its own single subclaim. There is no
decomposition: HELM releases none, and an LLM decomposition would add an
uncontrolled step. Every method sees the same units. Wang decomposes;
this is a stated protocol difference.

**Article title:** taken from the prompt "This is a Wikipedia passage about
{title}. {opening}". The title ends at the first ". " whose preceding token
- is not a single capital letter (an initial),
- is not "v",
- and does not itself contain ".".

This handles the six titles containing abbreviations: U.S. Route 491, Davis
v. Ayala, Fort Bayard ( Washington , D.C. ), Haywood S. Hansell,
Voltaire P. Twombly and Arthur W. Radford. Gate G0 checks that all 229
titles parse and that these six come out as listed.

**Retrieval:** exactly the FActScore protocol (`src/wikipedia_bm25.py`).
- BM25 (SQLite FTS5, k1 1.2, b 0.75) over every passage of the
  enwiki-20230401 snapshot.
- Index SHA-256 `79d0e38df6414235731e4c469b951e9ec884a07bce4746680f28f01f6d89925f`.
- Query = title + " " + sentence, with stopwords removed.
- Top 10 passages, in rank order.
- Premise = page title + passage.

The retrieved lists are written once, read-only, and pinned before any NLI
scoring.

**Source article:** passages from the source article itself are kept in the
primary analysis. They are legitimate Wikipedia evidence, and the annotators
checked against Wikipedia. Their exclusion is a sensitivity analysis (§6).

## 3. Method (frozen)

**NLI scoring:**
- Model: Wang's checkpoint, in float16 on MPS, with the feature scorer of the
  earlier studies (`FEATURE_VERSION mddre-nli3-temp5-seg400-overlap100-v1`,
  Wang's segmentation).
- **Gate F:** before bulk scoring, the 398 NBC pairs must equal the earlier
  float16 caches bit for bit.

**Estimators:** the frozen plan-v2 estimators,
`~/ddre-artifacts/v2/estimators_v2_float16.json`, SHA-256
`6a007e379a3fc84fabb6d367ede41a3c81c53accc4f7f86b0a29e2ec546fadec`.
- Tier 1: BSE and DDRE-1D.
- Tier 2, secondary: NB-M and DDRE-M.
- Nothing is refitted.

**Stopping:** backward induction over min(available documents, K)
(plan v2, amendment 1).

**Operating points:**
- C_M/C_FA ∈ {28/96, 14/24}.
- K = 1 … 10 × C_retrieve ∈ {0.125, 0.25, 0.5, 1}, exactly the amendment-2
  grid.
- The operating point is K = 10, C_retrieve = 1.

**Fixed-k max entailment:** K = 1 … 10, with the sentence score the minimum
over its (single) subclaim.

**Curves:** step envelopes of balanced average precision against NLI calls
per sentence over each method's points (`src/metrics_v2.py`,
`src/v2_budget.py`).

## 4. Endpoints and statistics

**Primary endpoint:** the AUC-gap, the mean balanced-AP difference between
the two envelopes over their shared NLI-cost range. If the ranges do not
overlap, the AUC-gap is undefined and counts against the hypothesis.

**Bootstrap:**
- Paired cluster bootstrap over articles: 10,000 replicates, seed 42.
- Every method and point is evaluated on the same draws.
- Each hypothesis gets the one-sided bootstrap p-value
  `p_greater` = (replicates with AUC-gap ≤ 0 or undefined) / 10,000.

## 5. Decision rule

- **Multiplicity:** Holm's step-down procedure over the four one-sided tests
  H1a, H1b, H2a and H2b, at family-wise α = 0.05.
- **Support:** a hypothesis is supported if Holm rejects and the 5th
  percentile of the replicates is above 0.
- **Statements:**

  | Outcome | Statement |
  |---|---|
  | H1a and H1b supported | **Confirmed:** DDRE-1D's cost–quality curve lies above BSE's |
  | exactly one of H1a, H1b supported | partial support, naming the cost setting |
  | neither supported | not confirmed |
  | (likewise for H2a/H2b) | against fixed-k |

- **Direction:** if the 95th percentile of the replicates is below 0, the
  result is reported as the method being *below* its comparator.

## 6. Secondary and sensitivity analyses (descriptive, no decision)

**At the operating point and along the curve:**
- DDRE-1D vs BSE at the operating point:
  - E-cost: NLI calls saved at BSE's AP (G1);
  - E-quality: AP gained at BSE's cost (G2);
  - Brier and ECE reductions (G3).
- Wang's Table 1 for every method at both operating points: nonfactual and
  factual PR-AUC, accuracy, documents and NLI calls.
- Passage-level Pearson and Spearman by generator × article.

**Other comparisons:**
- NB-M vs DDRE-M; BSE vs fixed-k; BSE-official (Wang's code rule) as a
  reference.

**Sensitivity analyses:**
- the six generators described in the paper (excluding mpt7b);
- source-article passages excluded from retrieval (passages whose page
  title equals the article title after removing whitespace and
  case-folding), with retrieval otherwise unchanged;
- a per-generator breakdown;
- if the authors' answer (§9) changes the label mapping, the corrected
  mapping, as a separate analysis. The primary analysis stays as in §2.2.

## 7. Gates (any failure stops the run before anything is written)

| Gate | Check |
|---|---|
| P | this file, plan v2 at amendment 2, the estimators, the BM25 index and the HELM listing match their pins |
| G0 | 3,736 units, 1,978 label 1, 229 clusters, 7 generators; 229 / 229 titles parsed; the six abbreviation titles as in §2.3 |
| R | retrieval complete: one top-10 list per unit, written once (units with fewer than 10 hits are counted and reported) |
| F | the 398 NBC pairs score identically to the earlier float16 caches |
| X2 | on SelfCheckGPT held-out, BSE-official reproduces the frozen replays row for row (the engine is unchanged) |
| X4 | on SelfCheckGPT held-out, the (K = 10, C_retrieve = 1) points reproduce Phase 5's operating points |

## 8. Execution order

Each step is a reviewed PR, run once from clean `main`:

1. This preregistration.
2. The HELM loader with gate G0 and the retrieval script (dry run: counts
   only). Then retrieval is run once.
3. Feature scoring of the retrieved passages (gate F). About 37,000 pairs,
   roughly 1.5 hours.
4. The confirmatory replay.
5. The results summary.

## 9. Known limitations, stated in advance

- **Retrieval:** we retrieve evidence ourselves from a 2023 Wikipedia
  snapshot. Wang used live Bing pages, so documents and ranks differ from
  Wang's setting. The source article is in the index.
- **Labels:**
  - Annotators also used Google. Some hallucination labels may rest on
    evidence outside Wikipedia.
  - Only agreed labels were kept. No agreement statistic is published.
  - The handling of "Unverifiable" is unknown (asked).
  - The released counts (3,736 units, 1,978 label 1) differ from the
    paper's Table 1 (3,582; 1,494). Asked.
- **Licence:** none is stated. The authors were emailed on 2026-10-17. The
  data are used for research only and are not redistributed.
- **Text:** the generators are base and chat models continuing Wikipedia
  openings. Some sentences may not be checkable claims.
- **Training mismatch:** the estimators were fitted on Wang's NBC data
  (Tier 1) and on FActScore (Tier 2), not on HELM.
- **Clusters:** 229 articles. All seven generators share the same articles.
- **Power:** HELM has 2.4× SelfCheckGPT held-out's units (3,736 vs 1,525) and 1.2× its
  clusters. The exploratory AUC-gap intervals there were about ±0.007 wide
  for effects of about +0.02. No formal power calculation is possible
  without running a detector.

## 10. What was seen before writing this

Before this preregistration was written, the following were seen on HELM:
- the repository files;
- the data structure (keys and types);
- the per-generator counts of label 0 and label 1;
- the prompts and their article titles.

**No sentence was viewed together with its label. No retrieval, NLI score
or detector output exists for HELM.**
