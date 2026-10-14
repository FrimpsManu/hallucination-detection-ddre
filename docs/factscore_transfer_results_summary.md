# FActScore frozen-transfer test — results summary

**Status: preregistered frozen-transfer test. RQ1 rules are the preregistered
decision rules; RQ2 rules are exploratory.**

This document records the outcome of the test preregistered in
[`docs/factscore_transfer_preregistration.md`](factscore_transfer_preregistration.md).
It adds no analysis beyond what the preregistration specified. Every number
below is read from the frozen artifacts listed in §8. The run happened once,
from clean `main` at `8dbc823`. Nothing here reruns, retunes or modifies the
SelfCheckGPT experiments.

All results concern **factuality-detection (hallucination-detection)
performance** and its computational cost. None of them concerns, or supports
any claim about, the factual accuracy of the text an LLM generates.

---

## 1. What was run

Cells A, B, C and D exactly as frozen on SelfCheckGPT
([`ablation_results_summary.md`](ablation_results_summary.md) §1). Nothing was
selected or tuned on FActScore. D0 was not evaluated.

| Cell | Evidence per retrieved document | Stopping rule |
|---|---|---|
| **A** | BSE histogram log-likelihood ratio | Bayes-risk look-ahead (CM 28, CFA 96, C_retrieve 1) |
| **B** | the same histogram log-likelihood ratio | posterior band [0.20, 0.85] |
| **C** | 2.5 × the histogram log-likelihood ratio | posterior band [0.05, 0.95] |
| **D** | uLSIF density ratio, σ 0.11346435546875, λ 1.0 | posterior band [0.20, 0.80] |

**Analysis set.** FActScore human-annotated ChatGPT and PerplexityAI
biographies (InstructGPT and the entity shared with SelfCheckGPT excluded),
2,573 sentences over 172 entities, 10,386 human atomic facts as subclaims.
Factual prevalence 54.4% (1,401 factual, 1,172 hallucinated); on the
SelfCheckGPT held-out set it was 27.3%.

**Evidence.** Top-10 passages per subclaim from SQLite FTS5 BM25 over the
FActScore `enwiki-20230401` snapshot (19,832,343 passages), query = entity
name + fact. Every subclaim received 10 passages. All 103,860 retrieved
documents were scored once by the frozen NLI model (100,654 unique pairs, 0
missing; compatibility probe 398 / 398 exact). Each passage is a single NLI
span, so NLI calls equal documents for every cell.

---

## 2. Score-support diagnostic (§8)

| Consumed by | Documents | Share > 76.27 | Share > 83.35 | Share ≥ 95 | Max |
|---|---|---|---|---|---|
| A | 21,888 | 0.041% | 0 | 0 | 76.66 |
| B | 26,730 | 0.037% | 0 | 0 | 76.66 |
| C | 18,744 | 0.048% | 0 | 0 | 76.66 |
| D | 23,952 | 0.042% | 0 | 0 | 76.66 |
| All retrieved | 103,860 | 0.100% | 0 | 0 | 76.76 |

No document D consumed scored above 83.35 (limit: 10%). The RQ1 result is
read **within the frozen estimator's support**.

---

## 3. Metrics (pooled)

| Metric | A (BSE) | B | C | D (frozen DDRE) |
|---|---|---|---|---|
| Nonfactual PR-AUC | 0.662735 | 0.615507 | 0.551518 | 0.666701 |
| Factual PR-AUC | 0.784162 | 0.752328 | 0.742035 | 0.777062 |
| Balanced PR-AUC | 0.723449 | 0.683918 | 0.646776 | 0.721881 |
| Accuracy | 0.708900 | 0.675865 | 0.701904 | 0.690634 |
| Macro-F1 | 0.706126 | 0.668676 | 0.698015 | 0.684630 |
| Factual precision | 0.853579 | 0.857503 | 0.860227 | 0.870257 |
| Factual recall | 0.561742 | 0.485368 | 0.540328 | 0.507495 |
| Nonfactual precision | 0.628104 | 0.594944 | 0.619610 | 0.607062 |
| Nonfactual recall | 0.884812 | 0.903584 | 0.895051 | 0.909556 |
| Sentences predicted factual (gold: 1,401) | 922 | 793 | 880 | 817 |
| Documents (= NLI calls) per sentence | 8.506801 | 10.388651 | 7.284881 | 9.308978 |
| Documents per subclaim | 2.107452 | 2.573657 | 1.804737 | 2.306181 |
| Documents vs A | — | +22.12% | −14.36% | +9.43% |

**Stopping depth** (subclaims stopping after *k* documents):

| Cell | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 5,436 | 1,226 | 2,358 | 789 | 182 | 160 | 92 | 76 | 22 | 45 |
| B | 4,643 | 1,179 | 2,446 | 807 | 306 | 300 | 240 | 204 | 89 | 172 |
| C | 5,436 | 3,258 | 887 | 373 | 182 | 140 | 45 | 30 | 16 | 19 |
| D | 5,070 | 1,186 | 2,629 | 497 | 296 | 255 | 223 | 87 | 62 | 81 |

**By generator (descriptive):**

| Generator | Sentences | Balanced PR-AUC A / B / C / D | Macro-F1 A / B / C / D |
|---|---|---|---|
| ChatGPT | 1,202 | 0.7116 / 0.6752 / 0.6252 / 0.7072 | 0.6944 / 0.6770 / 0.6876 / 0.6788 |
| PerplexityAI | 1,371 | 0.6648 / 0.6437 / 0.6130 / 0.6617 | 0.6632 / 0.6154 / 0.6533 / 0.6384 |

---

## 4. Bootstrap endpoints

Paired entity-cluster bootstrap across A, B, C and D (one entity draw per
replicate applied to all four cells), 172 entities, 10,000 resamples, seed 42,
95% percentile intervals. S(X) = docs(A) − docs(X), documents per sentence;
positive means X uses fewer documents than A.

| Endpoint | Pooled | ChatGPT (156 entities) | PerplexityAI (155 entities) |
|---|---|---|---|
| S(B) | −1.8818 [−2.0216, −1.7449] | −1.7903 [−1.9704, −1.6141] | −1.9621 [−2.1645, −1.7609] |
| S(C) | 1.2219 [1.1360, 1.3137] | 1.6614 [1.5343, 1.7929] | 0.8366 [0.7444, 0.9283] |
| S(D) | −0.8022 [−0.9154, −0.6915] | −0.7354 [−0.9044, −0.5693] | −0.8607 [−1.0112, −0.7143] |
| docs(B) − docs(C) | 3.1038 [2.9394, 3.2722] | 3.4517 [3.2473, 3.6616] | 2.7987 [2.5711, 3.0338] |
| docs(C) − docs(D) | −2.0241 [−2.1505, −1.9011] | −2.3968 [−2.5770, −2.2126] | −1.6973 [−1.8666, −1.5328] |
| Nonfactual PR-AUC D − A | 0.0040 [−0.0284, 0.0353] | 0.0064 [−0.0284, 0.0413] | −0.0225 [−0.0701, 0.0240] |
| Factual PR-AUC D − A | −0.0071 [−0.0383, 0.0208] | −0.0153 [−0.0517, 0.0198] | 0.0164 [−0.0318, 0.0289] |
| **Balanced PR-AUC D − A** | **−0.0016 [−0.0279, 0.0226]** | −0.0044 [−0.0339, 0.0247] | −0.0031 [−0.0426, 0.0212] |
| Balanced PR-AUC B − A | −0.0395 [−0.0595, −0.0218] | −0.0364 [−0.0577, −0.0154] | −0.0211 [−0.0546, −0.0047] |
| Balanced PR-AUC C − B | −0.0371 [−0.0602, −0.0104] | −0.0500 [−0.0835, −0.0149] | −0.0307 [−0.0550, 0.0098] |
| Balanced PR-AUC D − C | 0.0751 [0.0479, 0.0983] | 0.0820 [0.0476, 0.1151] | 0.0487 [0.0110, 0.0703] |
| S(B) / S(D), S(C) / S(D) | unstable (not reported) | unstable | unstable |

NLI-call differences equal the document differences. The ratio intervals are
marked unstable by the preregistered guard (S(D) ≤ 0.05 in replicates); S(D)
is negative in the observed data.

---

## 5. RQ1 — preregistered decision (§7)

| Rule | Quantity | Outcome |
|---|---|---|
| Efficiency transfers: 95% CI of S(D) entirely above 0 | S(D) = −0.802 [−0.915, −0.692] | **not met** |
| Quality comparable: lower 95% bound of balanced PR-AUC D − A ≥ −0.03 | −0.0016 [−0.0279, 0.0226] | **met** |
| Quality improvement (reported, not required): lower bound > 0 | — | not shown |
| **RQ1 supported on FActScore** | | **no** |

By generator (descriptive): both subsets fail the efficiency rule, and both
lower quality bounds (−0.034, −0.043) fall just past the −0.03 margin.

**Answer.** On FActScore, the frozen DDRE kept BSE's balanced PR-AUC within the
preregistered margin but used 0.80 *more* documents and NLI calls per sentence
(+9.4%), with the interval entirely below zero. The efficiency saving observed
on SelfCheckGPT (−20.7% documents) did not transfer under the frozen protocol.
Macro-F1 was 2.15 points lower for D than for A (0.6846 vs 0.7061).

---

## 6. RQ2 — SelfCheckGPT rules applied unchanged (exploratory)

| Rule | Quantities | Outcome | Recorded statement |
|---|---|---|---|
| **A → B**: S(B) ≥ 0.5 × S(D) | S(B) = −1.882, S(D) = −0.802; ratio unstable | **not met** | "The preregistered A -> B condition was not met." |
| **B → C**: S(C) ≥ 0.8 × S(D) and balanced(C) within 0.01 of balanced(D) | balanced C − D = −0.0751 | **not met** | "The preregistered B -> C condition was not met." |
| **C → D**: CI of docs(C) − docs(D) excludes 0 and balanced(D) ≥ balanced(C) − 0.01 | docs(C) − docs(D) = −2.024 [−2.151, −1.901]; balanced D − C = +0.0751 | **met** | "In this exploratory analysis, the learned continuous density-ratio shape contributes beyond a global rescaling." |

Tuning budgets, disclosed with every comparison: **A 0, B 32, C 224, D 640.**

The three outcomes match SelfCheckGPT's, but the C → D condition is met in a
different way. On SelfCheckGPT, D used fewer documents than C with
non-lower PR-AUC. On FActScore, D used 2.02 *more* documents per sentence than
C, and its balanced PR-AUC was 0.075 higher. The rule's interval condition
requires only that the difference excludes zero, not its sign.

Along A → B → C → D, the document differences add up:
S(D) = S(B) + [docs(B) − docs(C)] + [docs(C) − docs(D)], that is
−0.8022 = −1.8818 + 3.1038 − 2.0241. C is the only cell that used fewer
documents than A (−14.4%); its balanced PR-AUC was the lowest of the four
(0.6468 vs A's 0.7234; no interval was preregistered for C − A).

---

## 7. Limitations

1. **One frozen operating point.** Each cell was run at the operating point
   frozen on SelfCheckGPT. The result concerns that transfer, not each
   method's best achievable cost–quality trade-off on FActScore.
2. **Different data conditions.** Factual prevalence (54.4% vs 27.3%), human
   atomic facts rather than the SelfCheckGPT subclaims, and Wikipedia BM25
   passages rather than Wang et al.'s released web pages all differ at once.
   This test does not identify which of them relates to the change in
   direction of S(D).
3. **FActScore is now seen.** Any method or configuration chosen after this
   run cannot be confirmed on FActScore; it needs another dataset.
4. **Per-generator results are descriptive.** They were not decision rules.
5. **Unequal tuning budgets.** A 0, B 32, C 224, D 640, all on SelfCheckGPT.
6. **No multiplicity correction.** Many endpoints are reported.
7. **Cost measured as counts.** Documents and NLI calls, not wall-clock time.
8. **Inherited limitations.** Those listed in
   [`ablation_results_summary.md`](ablation_results_summary.md) §7 items 3,
   4, 9 and 11 apply unchanged.
9. **Detection, not generation.** These results are about detecting
   hallucinated sentences, not the factual accuracy of generated text.

---

## 8. Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| Transfer result (`factscore_transfer_result.json`) | `dc68b0f38610d1fd8325d9b0ab170a4460ff6f153b988a8e56f81bf0986d4c03` |
| Transfer predictions (`factscore_transfer_predictions.csv`) | `a439b1c1fd87b1a5c2bf77259a82d43f69206ba0614949d4e1815a8fa21272c3` |
| FActScore NLI cache (`factscore_nli_cache.sqlite`) | `7bd69b0c4a2e809311be9a9748cb1fba53ed09fee1c98d4e5fe7c4c2bb2edff0` |
| NLI cache manifest | `49f1be3adcff3c385c6d9778c2de28aa8cd1df7320d1390d11509b5f33a2a1cd` |
| NLI cache coverage fingerprint | `aeca4d6d2386c50367dd4118924bc1f61dfd90d7c3e4c51d8dd1e65e83576473` |
| Top-10 evidence (`factscore_evidence_top10.jsonl`) | `44f479128d41818fe5a056f125407dbde75a8735669c398a3b945cfeaa310d09` |
| Evidence manifest | `4983050b19eedcee5a3bbd2ac01bbda1a4f08bf3e38e8a6f46d812315eb2ced4` |
| BM25 index (`wikipedia_bm25_index.sqlite`) | `79d0e38df6414235731e4c469b951e9ec884a07bce4746680f28f01f6d89925f` |
| Wikipedia snapshot (`enwiki-20230401.db`) | `31cf7b6b4465459844bb00f3a6ac75560fc7d1525112205a21859323dc5d33d7` |
| FActScore release (`data.zip`) | `404482c9a7d588d581a2f7a78d609fe6bfec4ddc0f729c8560a61b1743dbffab` |
| Source cache (D-03 validation cache, unchanged) | `66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776` |
| Validation freeze (cells B, C) | `a1824f663653b66df9deec590146de67992bd6a63320ccfc5f18e9a7bd7967ac` |
| Preregistration | `5aa2a2a6d3ec07fa6f2ed2fb294ff590510c614efdf67c4d91d231b93f43a385` |

| Step | Commit |
|---|---|
| Preregistration merged | `ac872e6d7f830aac12e6e64654b0f7d7a2a8860c` |
| BM25 index built | `3bfd5b81a5a4163c4343e9248ec2c654ab104de3` (clean tree) |
| Evidence retrieved; NLI scoring run | `b56bebebdb66168df7f6534b9bf20a063f78805c` (clean tree) |
| Transfer replay run | `8dbc823aa4d3ac25a4fa36ae79b73add560e7b58` (clean tree) |

NLI model `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, revision
`b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, score version
`wang-emnlp23-temp5-seg400-overlap100-hostscale-v2`, MPS, float16, batch size 1.
The artifacts are stored locally under `~/ddre-artifacts/factscore-release/`
and are not committed to this repository.
