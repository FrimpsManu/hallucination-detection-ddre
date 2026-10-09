# Evidence/stopping ablation — results summary

**Status: exploratory / post-hoc. Not confirmatory.**

This document records the outcome of the held-out ablation preregistered in
[`docs/ablation_preregistration.md`](ablation_preregistration.md). It adds no
analysis beyond what the preregistration specified. Every number below is read
from the frozen artifacts listed in §8. Nothing here reruns, retunes or
modifies the SelfCheckGPT A/B/C/D experiment.

All results concern **factuality-detection (hallucination-detection)
performance** and its computational cost: how well each method classifies
generated sentences as factual or hallucinated, and how much retrieval and NLI
work it does to decide. None of them concerns, or supports any claim about, the
factual accuracy of the text an LLM generates.

---

## 1. What was run

| Cell | Evidence per retrieved document | Stopping rule | How it was configured | Validation search budget |
|---|---|---|---|---|
| **A** | BSE histogram log-likelihood ratio | Bayes-risk look-ahead (published BSE, `mode="official"`) | fixed: CM 28, CFA 96, C_retrieve 1 | 0 |
| **B** | the same histogram log-likelihood ratio | posterior band [0.20, 0.85] | band selected on validation | 32 |
| **C** | 2.5 × the histogram log-likelihood ratio | posterior band [0.05, 0.95] | κ and band selected on validation | 224 |
| **D** | uLSIF density ratio, σ 0.11346435546875, λ 1.0 | posterior band [0.20, 0.80] | the frozen DDRE, not retuned | 640 (historical) |
| **D0** | uLSIF, original CV-selected σ 0.2269287109375, λ 1.0 | posterior band | band selection on validation | 32 — **no eligible configuration** |

Held fixed across all cells: P0 0.5, CM 28, CFA 96, max 10 documents per
subclaim, the same 190 held-out passages (1,525 sentences, 2,387 subclaims),
the same retrieved documents in the same order, the same NLI model, revision,
segmentation and max-over-spans score, and minimum-over-subclaims sentence
aggregation.

**Selection (validation, 48 passages).** B, C and D0 were selected with the rule
that froze D: eligible iff nonfactual, factual and balanced validation PR-AUC are
each ≥ the stored BSE reference − 0.005, then fewest documents per sentence.
Eligible configurations: **B 23 / 32, C 35 / 224, D0 0 / 32.**

**D0.** The original CV-selected uLSIF estimator had no eligible configuration
at any of the 32 bands, so it was not evaluated on held-out. This re-derives
the result the D-03 sensitivity artifact had already recorded; it is a
selection-history result, not a new finding. The evaluated DDRE (D) is the
product of the post-D-03 validation selection.

**Consistency gate.** A and D were replayed from the completed held-out cache
and reproduced the canonical frozen predictions exactly on all 1,525 sentences
(posterior, prediction, documents, NLI calls, per-subclaim vectors). All pinned
inputs were unchanged before and after the replay.

---

## 2. Held-out metrics

| Metric | A (BSE) | B (hist + band) | C (2.5 × hist + band) | D (frozen DDRE) |
|---|---|---|---|---|
| Nonfactual PR-AUC | 0.854196 | 0.855839 | 0.836966 | 0.865495 |
| Factual PR-AUC | 0.607024 | 0.613400 | 0.593362 | 0.611819 |
| Balanced PR-AUC | 0.730610 | 0.734620 | 0.715164 | 0.738657 |
| Accuracy | 0.812459 | 0.812459 | 0.813115 | 0.817049 |
| Macro-F1 | 0.754904 | 0.753671 | 0.745510 | 0.736269 |
| Factual precision | 0.677507 | 0.680441 | 0.704969 | 0.761364 |
| Factual recall | 0.599520 | 0.592326 | 0.544365 | 0.482014 |
| Nonfactual precision | 0.855536 | 0.853701 | 0.842062 | 0.828707 |
| Nonfactual recall | 0.892599 | 0.895307 | 0.914260 | 0.943141 |
| Sentences predicted factual (gold: 417) | 369 | 363 | 322 | 264 |
| Documents per sentence | 6.370492 | 6.811148 | 5.614426 | 5.053770 |
| Documents per subclaim | 4.069962 | 4.351487 | 3.586929 | 3.228739 |
| NLI spans per sentence | 18.938361 | 20.272787 | 16.936393 | 14.772459 |
| Documents vs A | — | +6.92% | −11.87% | −20.67% |
| NLI spans vs A | — | +7.05% | −10.57% | −22.00% |

**Stopping depth** (subclaims stopping after *k* documents, *k* = 1…10):

| Cell | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 438 | 197 | 470 | 526 | 153 | 175 | 181 | 73 | 37 | 137 |
| B | 345 | 188 | 475 | 525 | 177 | 182 | 188 | 87 | 48 | 172 |
| C | 438 | 608 | 375 | 230 | 231 | 243 | 66 | 51 | 70 | 75 |
| D | 382 | 230 | 1000 | 495 | 103 | 51 | 43 | 27 | 13 | 43 |

---

## 3. Preregistered bootstrap endpoints

Paired passage-level cluster bootstrap across A, B, C and D (one passage draw
per replicate applied to all four cells), 190 passages, 10,000 resamples,
seed 42, 95% percentile intervals. S(X) = docs(A) − docs(X), documents per
sentence; positive means X uses fewer documents than A.

| Endpoint | Estimate | 95% CI |
|---|---|---|
| S(B) | −0.440656 | [−0.528193, −0.358639] |
| S(C) | 0.756066 | [0.691743, 0.823995] |
| S(D) | 1.316721 | [1.149281, 1.492166] |
| docs(B) − docs(C) — direct B → C effect | 1.196721 | [1.102937, 1.294309] |
| docs(C) − docs(D) — C → D effect | 0.560656 | [0.394499, 0.731354] |
| NLI(A) − NLI(B) | −1.334426 | [−1.661579, −1.034632] |
| NLI(A) − NLI(C) | 2.001967 | [1.793390, 2.216658] |
| NLI(A) − NLI(D) | 4.165902 | [3.541663, 4.829495] |
| NLI(B) − NLI(C) | 3.336393 | [2.975163, 3.727168] |
| NLI(C) − NLI(D) | 2.163934 | [1.579405, 2.791393] |
| Balanced PR-AUC B − A | 0.004009 | [−0.013732, 0.021178] |
| Balanced PR-AUC C − B | −0.019455 | [−0.046456, 0.008467] |
| Balanced PR-AUC D − C | 0.023493 | [−0.006978, 0.054355] |
| S(B) / S(D) | −0.334661 | [−0.437682, −0.252383] |
| S(C) / S(D) | 0.574203 | [0.501202, 0.661638] |

Ratio intervals: both `reported`. Denominator guard (S_D(b) ≤ 0.05 documents per
sentence): 0 replicates.

By construction the document differences add up along the chain A → B → C → D:
S(D) = S(B) + [docs(B) − docs(C)] + [docs(C) − docs(D)], that is
1.316721 = −0.440656 + 1.196721 + 0.560656.

---

## 4. Preregistered interpretation outcomes

These are the mechanical outcomes of the rules in preregistration §7–§8, with
their recorded statements. They are exploratory interpretation rules, not
confirmatory hypothesis tests.

| Rule | Quantities | Outcome | Recorded statement |
|---|---|---|---|
| **A → B**: S(B) ≥ 0.5 × S(D) | S(B)/S(D) = −0.335 | **not met** | "The preregistered A -> B condition was not met." |
| **B → C**: S(C) ≥ 0.8 × S(D) and balanced(C) within 0.01 of balanced(D) | S(C)/S(D) = 0.574; balanced C − D = −0.0235 | **not met** | "The preregistered B -> C condition was not met." |
| **C → D**: CI of docs(C) − docs(D) excludes 0 and balanced(D) not lower than balanced(C) by more than 0.01 | CI [0.394, 0.731]; balanced D − C = +0.0235 | **met** | "In this exploratory analysis, the learned continuous density-ratio shape contributes beyond a global rescaling." |
| **Frozen prediction**: S(B)/S(D) < 0.25 | −0.335 | **held** | — |
| **D0 → D** | 0 / 32 eligible on validation | **not evaluated** | re-derived selection-history result |

Tuning budgets, disclosed with every comparison: **A 0, B 32, C 224, D 640.**

---

## 5. RQ1 — Can DDRE reduce computational overhead while maintaining competitive factuality-detection performance relative to BSE?

From the frozen held-out run (result `b5bc08cc…`) and reproduced exactly as
cells A and D here:

| | Estimate | 95% CI |
|---|---|---|
| Documents per sentence saved (A − D) | 1.316721 (−20.67%) | [1.149281, 1.492166] |
| NLI spans per sentence saved (A − D) | 4.165902 (−22.00%) | [3.541663, 4.829495] |
| Nonfactual PR-AUC, D − A | +0.011299 | [−0.015522, 0.037976] |
| Factual PR-AUC, D − A | +0.004795 | [−0.036683, 0.047054] |
| Balanced PR-AUC, D − A | +0.008047 | [−0.020507, 0.037700] |

**Computational overhead.** On this benchmark the frozen DDRE retrieved fewer
documents and evaluated fewer NLI spans than published BSE, and both intervals
lie entirely above zero.

**Factuality-detection performance.** The PR-AUC point estimates are close to
BSE's and slightly higher, but every interval includes zero. The data are
compatible with PR-AUC losses of up to about 1.6 (nonfactual), 3.7 (factual)
and 2.1 (balanced) points. The preregistered non-inferiority margin of 0.005
was not established. Threshold-based quality moved in both directions:
accuracy +0.46 points, macro-F1 −1.86 points, factual recall 0.600 → 0.482,
nonfactual recall 0.893 → 0.943.

**Answer.** On this benchmark, DDRE reduced retrieval and NLI computation by
about 21–22% while producing PR-AUC point estimates comparable to BSE. Whether
detection quality is preserved is not established: the intervals are wide, and
the threshold-level behaviour shifts toward predicting "hallucinated", which
costs factual recall. The original confirmatory claim remains
`NOT_CONFIRMATORY`, because D was selected after D-03 by a validation sweep
over 640 configurations rather than by the preregistered uLSIF-CV procedure.

---

## 6. RQ2 — Which component accounts for the observed efficiency: stopping policy, evidence magnitude, or continuous evidence shape?

**Stopping policy (A → B).** Changing the stopping policy alone did not explain
the savings, and it actually increased computation. With BSE's own histogram
evidence, the band rule used 0.44 more documents per sentence than BSE
(+6.9%; CI [0.36, 0.53] more) and 1.33 more NLI spans (+7.0%). Balanced
PR-AUC changed by +0.004 (CI [−0.014, 0.021]). The frozen prediction that B
would recover well under a quarter of D's saving held: B recovered none of it.

**Evidence magnitude (B → C).** Strengthening the histogram evidence
substantially reduced computation, so evidence magnitude matters. Scaling the
histogram log-likelihood ratio by κ = 2.5 lowered documents per sentence by
1.20 relative to B (−17.6%; CI [1.10, 1.29]) and NLI spans by 3.34. C
recovered 57% of D's document saving relative to A (S(C)/S(D) = 0.574,
CI [0.50, 0.66]). The preregistered B → C condition was **not met** on either
part: 0.574 is below the 0.8 share, and C's balanced PR-AUC (0.7152) is 0.0235
below D's (0.7387), beyond the 0.01 margin. By the preregistered rule, evidence
magnitude does not account for most of the efficiency behaviour. C's PR-AUC
point estimates were also the lowest of the four cells (balanced C − B −0.019,
CI [−0.046, 0.008]).

**Continuous evidence shape (C → D).** The preregistered C → D condition
passed. D used 0.56 fewer documents per sentence than C (CI [0.39, 0.73],
excluding zero) and 2.16 fewer NLI spans, with balanced PR-AUC 0.0235 higher
(CI [−0.007, 0.054]). In this exploratory ablation, the learned continuous
density-ratio shape contributes beyond global rescaling.

**Selection history (D0 → D).** The CV-selected uLSIF estimator could not meet
the validation quality constraint at any band, so the shape that contributes is
specifically the post-D-03-selected fit (σ = 0.11346435546875), not the
estimator the original procedure chose.

**Taken together.** Along the chain A → B → C → D, the observed document saving
of 1.317 per sentence splits as +0.441 more for the band rule, 1.197 fewer for
scaling the evidence, and 0.561 fewer for the uLSIF shape. The band rule alone
increases work. Along this particular A → B → C → D path, the largest positive
reduction comes from strengthening evidence magnitude, while the learned
continuous shape provides a further reduction beyond global scaling. The same chain shows a
monotone shift in threshold-level behaviour: factual recall 0.600 → 0.592 →
0.544 → 0.482 and nonfactual recall 0.893 → 0.895 → 0.914 → 0.943. Across the
evaluated cells, the shift toward lower retrieval in C and D coincides with
more sentences being classified as hallucinated and lower factual recall.

---

## 7. Limitations

1. **Exploratory and post-hoc.** The held-out split was evaluated once before,
   in the frozen BSE vs DDRE comparison. The Phase 1/1b post-hoc analysis of
   that held-out run motivated this ablation's design, including cell C. B and
   C were frozen on validation before held-out replay, but the held-out set is
   not untouched. Nothing here is confirmatory.
2. **Unequal tuning budgets.** A 0, B 32, C 224, D 640 validation
   configurations. Larger budgets favour the cells further down the chain,
   including in the C → D comparison.
3. **Selection history.** D's estimator was chosen after D-03 showed the sign of
   the evidence at the dominant NLI scores (about 35) is unstable across the 20
   (σ, λ) pairs: 17 of 20, including the CV-selected one, assign the opposite
   sign to frozen D. The CV-selected estimator (D0) had no eligible
   configuration. The C → D outcome describes this particular fit.
4. **One global transform.** C tests a single multiplicative rescaling of the
   histogram evidence. "Beyond global rescaling" does not cover other
   transforms (an additive shift, per-bucket weights, finer bins). The κ grid
   was {1.0 … 4.0}.
5. **One benchmark family.** One dataset (SelfCheckGPT WikiBio, 190 held-out
   passages), one NLI model, one retrieval source (Wang et al.'s released web
   pages), one cost setting (CM 28, CFA 96). Generality is untested.
6. **Small validation set.** Selection used 48 passages (383 sentences), so the
   selected configurations carry selection noise.
7. **Quality not established.** None of the reported PR-AUC differences
   (D − A, B − A, C − B, D − C) has an interval excluding zero. The efficiency results are much more precise
   than the quality results. Macro-F1 and factual recall decline along the
   chain.
8. **No multiplicity correction.** Many endpoints are reported; none is a
   confirmatory test.
9. **Path-dependent decomposition.** The A → B → C → D split depends on the
   chosen order of cells. A different path (for example, scaling the evidence
   under Bayes-risk stopping) was not run and could split differently.
10. **Cost measured as counts.** Documents and NLI span calls, not wall-clock
    time, latency or energy.
11. **Inherited limitations.** The historical NLI checkpoint revision cannot be
    established retrospectively; score compatibility is established
    (398/398 exact). The CM 14/CFA 24 Gate 1 reproduction failed. uLSIF and the
    histograms are fit on NBC sentence-pair scores but applied to document
    max-over-span scores (D-13). DDRE has a one-document floor per subclaim
    that BSE does not (D-02); no held-out subclaim had zero retrieval in any
    cell.
12. **Detection, not generation.** These results are about detecting
    hallucinated sentences. They say nothing about the factual accuracy of
    generated text.

---

## 8. Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| Ablation held-out result (`ablation_heldout_result.json`) | `a18ae5547accdcbb448dbb48699c54f6b4a7d7a5964f757ac419fdaea11fdfec` |
| Ablation predictions (`ablation_heldout_predictions.csv`) | `ebdcf5c2090cffbabbad81a628e0c04d297d0f85193318aa9c0d5903a50e260b` |
| Validation freeze (`ablation_validation_freeze.json`) | `a1824f663653b66df9deec590146de67992bd6a63320ccfc5f18e9a7bd7967ac` |
| Full-coverage held-out cache | `2497487664e2df552ba9bf28947c4ccae7b5f66d2feccbada05ffa5aaea88f28` |
| Cache manifest | `5163c8a2ed8b8730e245964e65ce669d7bb30c6603e40de7be6d25d42d2c12d7` |
| Cache coverage fingerprint | `9084d7e02dadfa6e95b2f5c6e5ab183f00a66ca97982421d934abc284af8b6b5` |
| Preregistration (`docs/ablation_preregistration.md`) | `07fa666e3a5550123e06fefeb905d2057506bb0e7421ea23534485cfd31b587d` |
| Canonical frozen result (`frozen_heldout_result_fae3eee.json`) | `b5bc08ccaed89f42e96194e74df6f81bc4e180272dafbcfdef67670cc100ef67` |
| Canonical frozen predictions | `a78b09b7aa09803a21a761b3bede7600180014b97142a381b509d7a1267211c1` |
| Claim-reasons correction for the frozen run | `a4d982b263011adaafa64332187379e6c23be1446e1b2ef8d95f228365529941` |
| D-03 validation cache | `66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776` |
| D-03 hyperparameter sensitivity artifact | `fc5604854b18f3e3e712fe7ef01859b58dac569e99696041713abb11daa6dcc0` |

| Code | Commit |
|---|---|
| Preregistration merged | `d9238d682c51c39df4540ca1769960cd6bf2653f` |
| Validation selection run | `33f58a36527bf605ac7d67403cc4bf32a50f77cb` (clean tree) |
| Held-out ablation replay run | `ae2f8b0cc90b080bf3dab39dc70b1ce0fe9875d3` (clean tree) |

NLI model `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, revision
`b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, score version
`wang-emnlp23-temp5-seg400-overlap100-hostscale-v2`, MPS, float16, batch size 1.
The artifacts are stored locally under `~/ddre-artifacts/gate1-2026-09-20/` and
are not committed to this repository.
