# Confirmatory test of multivariate DDRE — preregistration

**Status: preregistered, frozen once merged.** This document fixes every choice
of the confirmatory test before any detector, NLI model or feature touches the
two test sets. Changes after merge follow §11.

All results concern **hallucination-detection performance** and its
computational cost. None concerns the factual accuracy of the text an LLM
generates.

---

## 0. Why this test

The multivariate DDRE study
([`multivariate_ddre_design.md`](multivariate_ddre_design.md),
[`mddre_results_summary.md`](mddre_results_summary.md)) ended up exploratory.
Its cost–quality curve was amended after the results were seen. Its
evidence was mixed:

| Test set | M vs BSE: matched-cost balanced PR-AUC gain [95% CI] |
|---|---|
| Factcheck-Bench | +0.096 [0.037, 0.158] |
| SelfCheckGPT held-out | +0.017 [−0.012, 0.045] |
| FActScore test entities | −0.040 [−0.072, −0.010] |

This test asks whether the Factcheck-Bench result holds on two new datasets.
Those datasets were chosen in
[`confirmatory_dataset_survey.md`](confirmatory_dataset_survey.md) for having
the same structure: LLM answers, claim-level human labels and saved, ranked
evidence. Nothing about the method is refit, retuned or reselected.

## 1. Hypotheses

On each dataset, comparing **M** (self-normalised uLSIF on entailment,
contradiction and log rank) with **BSE** (Wang's official look-ahead with the
NBC histogram):

- **H1 (cost, RQ1):** M reaches BSE's balanced PR-AUC with fewer documents per
  unit.
- **H2 (detection, RQ2):** at BSE's documents per unit, M has higher balanced
  PR-AUC.

## 2. Data (frozen)

| | ExpertQA (primary) | BingCheck (second) |
|---|---|---|
| Source | github.com/chaitanyamalaviya/ExpertQA @ `1aa7ba81bd4c083c12ef1aa2fee338f6cb60fa87` | github.com/Miaoranmmm/SelfChecker @ `d33157db37c4475ae1b9a1df4907e7368b35e995`, `bingcheck/` |
| Files (sha256) | `r2_compiled_out_corrected_revised_atomic_w_evidences_factscores_anon.jsonl` `5e6afb0453bc8c5fabfe43fbef0b205ceb3175694640cfd1f5d47a97900d0ac7` (Git LFS oid); `r2_compiled_anon.jsonl` `b5e3070acb1f1df36bd404db720c3961ccd661531886ecc79d416da85d748ac5` | 397 files. The digest of their sorted `shasum -a 256` listing is `0432979f4c7ca6533a624cd8bc425b4b31b777a50f3a9af7800cff9cb606d5f0`, and each file matches its git blob hash |
| Local path | `~/ddre-artifacts/confirmatory/expertqa/release/` | `~/ddre-artifacts/confirmatory/bingcheck/release/` |
| Unit judged | sentence (`claim_string`) | claim |
| Subclaims | the released `atomic_claims` of the sentence | the claim itself (one subclaim) |
| Documents per subclaim | the released `atomic_evidences`, split at `Passage ID i: ` markers, in ID order (embedding-similarity rank); `max_docs = 5` | `search_results[claim]` in stored order (Bing rank); `max_docs = 3` |
| Premise text | the passage text after its marker | the page `content`, list items joined by single spaces; title and URL not used |
| **Factual** | Definitely correct, Probably correct | supported |
| **Hallucinated** | Likely incorrect, Definitely incorrect | refuted, partially supported |
| Excluded | Unsure, missing label; sentences with no atomic claims | not supported (no evidence found) |
| Cluster for the bootstrap | question author (`annotator_id`, 484) | response (396) |
| Expected units (gate G0) | **11,389 sentences, 596 hallucinated**. That is 12,598 minus 1,113 Unsure and 53 missing, minus 43 labelled sentences with no atomic claims | **3,623 claims, 942 hallucinated** |

Further rules:
- A subclaim with no documents is kept. It reads nothing, and its P stays at
  P0.
- For BingCheck, the claims are the keys of `subconclusions`. Documents come
  from `search_results` under the same key, or none if it is absent.
- The human-selected `evidence` sentences in BingCheck, and `revised_*` fields
  in ExpertQA, are **never read**: they encode the annotator's judgement.
- The data loaders fail closed unless the hashes and the totals recorded in the
  survey (§4) match exactly.

**The labels differ in meaning, and the paper will say so.** ExpertQA labels
are experts' judgements of factual correctness. BingCheck labels are crowd
judgements of support by the saved pages, so "partially supported" counts as
hallucinated even if no part of the claim is false.

## 3. Method (frozen)

- **Estimators:** `~/ddre-artifacts/mddre/mddre_estimators_float16.json`, sha256
  `486296a2717b95a4ba661f15fbff8286613f7eeba508dab804614fa98d624014`. Cells A′, U,
  M and L were fit on the FActScore training split. Cell A is Wang's NBC
  histogram. None is refit.
- **Features:** `FEATURE_VERSION = mddre-nli3-temp5-seg400-overlap100-v1`,
  computed with the official DeBERTa NLI checkpoint in float16:
  - e = max entailment over spans;
  - k = max contradiction over spans;
  - rank = the document's position in its list.
  - Before bulk scoring, the 398 NBC pairs must reproduce the earlier caches
    exactly.
  - Features are written to a new cache. Existing caches are read-only.
- **Stopping:** BSE's band at CM 28 / CFA 96, C_retrieve 1, computed for each
  dataset's `max_docs`. Gate G2: cell A under that band must use exactly BSE's
  official documents.
- **Curve:** BSE's band scaled in log-odds about P0 by s = 0.25, 0.50, …, 4.00
  (Amendment 1 of the design). The cost rule labels a unit factual iff
  P > 28/124. Unit P is the minimum over its subclaims.

## 4. Endpoints

Write q_ref and d_ref for BSE official's balanced PR-AUC and mean documents
per unit. On a curve, "first crossing" means the fewest documents at which the
curve reaches q_ref, interpolated linearly as in the design.

- **E1, net matched-quality saving (H1):** D_A − D_M.
  - D_M is the first crossing on M's curve.
  - D_A is the smaller of d_ref and the first crossing on A's curve.
  - This compares M with BSE's best showing on BSE's own curve. It removes the
    optimism found in the exploratory study, where BSE "saved" 2.48 documents
    against itself.
  - If M's curve never reaches q_ref in a replicate, E1 is "not reached" and
    counts against H1.
- **E2, matched-cost gain (H2):** M's balanced PR-AUC at d_ref, interpolated on
  M's curve, minus q_ref.
  - If d_ref lies outside M's curve, E2 is "not reached" and counts against H2.

Both endpoints are recomputed in every bootstrap replicate: a paired cluster
bootstrap with 10,000 resamples, seed 42 and 95% percentile intervals, with all
cells on the same draws.

## 5. Decision rule

- There are four tests: E1 and E2 on ExpertQA and on BingCheck. Each gets a
  two-sided bootstrap p-value. **Holm's correction runs over all four** at
  family-wise α = 0.05.
- A test **supports** its hypothesis if Holm rejects and the 95% CI lies above
  0.
- If the 95% CI lies **below** 0, the result is reported as M being worse than
  BSE on that endpoint.
- Conclusions:

  | Outcome | Statement |
  |---|---|
  | E2 supported on both datasets | **Confirmed: better detection at matched cost** |
  | E1 supported on both datasets | **Confirmed: lower cost at matched quality** |
  | Both of the above | confirmed lower cost and better detection |
  | Supported on one dataset only | partial support, naming the dataset |
  | Supported on neither | not confirmed |

## 6. Descriptive analyses (no decision attached)

- Same-band differences for every cell against BSE (documents, NLI calls,
  balanced PR-AUC).
- The full curves of A, A′, U, M and L.
- The RQ3 chain (A→A′, A′→U, U→M, L→M) on E1 and E2.
- E1 measured in NLI calls instead of documents. This matters for BingCheck,
  whose pages are long.
- Sensitivity checks, reported whatever they show:
  - ExpertQA with Unsure counted as hallucinated.
  - BingCheck with refuted only as hallucinated (42 claims, so expect very wide
    intervals).
  - BingCheck with "not supported" counted as hallucinated.
- ExpertQA by answering system (retrieve-and-read, post-hoc, closed-book, Bing
  Chat).

## 7. Gates (any failure stops the run)

| Gate | Check |
|---|---|
| G0 | data hashes and the recorded totals match |
| F | the 398 NBC pairs score identically to the earlier caches |
| G1 | the replay engine reproduces the canonical SelfCheckGPT ablation predictions of cells A and D |
| G2 | on each new dataset, cell A under BSE's band uses exactly BSE's official documents |
| P | the design file, this file and the estimators file match their pinned digests |

## 8. Execution order

Each step is a reviewed PR, run once from clean `main` after merge:

1. This preregistration.
2. Data loaders and evidence enumeration (dry run only: unit, subclaim and pair
   counts).
3. Feature scoring of all new pairs (about 277k ExpertQA spans and 113k
   BingCheck spans; several hours).
4. The confirmatory replay, writing its result and predictions read-only.
5. The results summary.

## 9. Power

- ExpertQA: 11,389 units, 596 hallucinated, 484 clusters.
- BingCheck: 3,623 units, 942 hallucinated, 396 clusters.
- Both are several times Factcheck-Bench (271 sentences, 92 clusters), whose E2
  standard error was 0.031.
- No further power calculation is possible without running a detector. The test
  will be reported as underpowered if its intervals turn out wide.

## 10. Known limitations, stated in advance

- **ExpertQA:**
  - Each sentence has one annotator, and hallucinations are rare (about 5%).
  - The 5 passages were retrieved per atomic claim from Google and ranked by
    OpenAI embeddings. That ranking is not BM25 and not the order a production
    system would use.
- **BingCheck:**
  - The labels are support against the same pages and come from one crowd
    worker per claim, starting from LLM pre-labels.
  - There are only 3 documents per claim.
  - No licence is stated. The data is used for research only, is not
    redistributed, and the authors will be asked.
- **Training mismatch.** The estimators were trained on FActScore biographies
  with BM25 Wikipedia passages, so both test sets shift domain and retrieval.
- **Cost** is counted in documents and NLI calls. Wall-clock time is reported
  separately.

## 11. Amendments

Any change after merge is a dated amendment at the end of this file. Each states
what changed, why, and whether any result on these datasets had been seen. A
change made after a result has been seen makes the affected endpoint
exploratory.

### Amendment 1 (2026-10-16): ExpertQA cluster count

**No result on either dataset had been seen.** No NLI score, feature or
detector output existed when this was written.

The data loader's gate G0 found that ExpertQA has **483** question-author
clusters after the §2 exclusions, not 484. All 484 authors appear in the
release, but one author's sentences are all excluded (Unsure, missing label,
or no atomic claims). §2 and §9 therefore read 483 clusters for ExpertQA. The
unit and hallucinated totals (11,389 and 596) are unchanged.

A clarification, not a change: the BingCheck digest in §2 is the sha256 of the
pinned `shasum -a 256` listing file `bingcheck/file_sha256.txt`. The loader
checks that this file matches the digest and that every release file matches
its line, in any order.
