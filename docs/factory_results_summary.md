# Second confirmatory test, on FACTORY — results

**Status: preregistered confirmatory test, run once.**
- Protocol: [`docs/factory_preregistration.md`](factory_preregistration.md),
  merged before any detector, NLI model or retrieval touched FACTORY.
- **Runs, each once from clean `main`:**
  - retrieval: `79b718b`;
  - scoring: `f37c702`;
  - the confirmatory replay: `78cbe6d`, on 2026-10-26.
- Every number in §§2–6 is generated from the frozen result (§9). §7 is
  exploratory.
- All results concern **hallucination-detection performance** and its
  computational cost.

**Outcome:**
- **The test is informative** (§5.1): fixed-k max entailment at K = 10 has
  balanced AP 0.586, 95% interval [0.568, 0.606], above 0.5.
- **H1 is confirmed:** DDRE-1D's cost–quality curve lies above BSE's at both
  of Wang's cost settings. It is the second independent confirmation after
  HELM.
- **H2 is not confirmed:** DDRE-1D's curve is not above fixed-k max
  entailment's. It is not below it either: both AUC-gaps are within ±0.006 of
  0.

---

## 1. What ran

**Data:** FACTORY human fact-check annotations (Chen et al., 2025), Hugging
Face `facebook/FACTORY` revision `c820453a`.
- 2,983 sentences (1,045 hallucinated by Rule A) from six 2025
  retrieval-augmented models.
- 1,373 questions, which are the bootstrap clusters.
- Each sentence is its own single subclaim.

**Evidence:**
- BM25 top-10 passages from the pinned enwiki-20230401 index; query =
  question + sentence.
- Every sentence received 10 passages.
- NLI features come from one float16 pass (gate F: 396 / 396 NBC pairs
  exact, as on HELM).

**Methods, statistics and gates:** exactly HELM's (see
`docs/helm_results_summary.md` §1). All gates passed:
- P;
- G0, for Rule A and Rule B;
- X2 and X4: 0 rows differing on SelfCheckGPT held-out.

## 2. Preregistered decision

| Test | Comparison | C_M/C_FA | AUC-gap | 5th pct. | One-sided p | Holm threshold | Result |
|---|---|---|---|---|---|---|---|
| H1a | DDRE-1D vs BSE | 28/96 | +0.0105 | +0.0045 | 0.0047 | 0.0125 | **supported** |
| H1b | DDRE-1D vs BSE | 14/24 | +0.0102 | +0.0030 | 0.0096 | 0.0167 | **supported** |
| H2a | DDRE-1D vs fixed-k max | 28/96 | −0.0016 | −0.0052 | 0.6234 | 0.0500 | not supported |
| H2b | DDRE-1D vs fixed-k max | 14/24 | −0.0003 | −0.0046 | 0.5427 | 0.0250 | not supported |

No replicate was undefined. For H2, the 95th percentiles (+0.0034, +0.0040)
are above 0, so neither test reports DDRE-1D as below fixed-k.

**Statements produced by the preregistered rule:**
- H1: **confirmed**, DDRE-1D's cost–quality curve lies above BSE's.
- H2: **not confirmed**, DDRE-1D's curve above fixed-k max entailment's.

## 3. Secondary endpoints (descriptive, 95% intervals)

| Contrast | C_M/C_FA | AUC-gap | E-quality (AP at the reference's cost) | E-cost (NLI calls saved at the reference's AP) |
|---|---|---|---|---|
| DDRE-1D vs BSE | 28/96 | +0.011 [+0.003, +0.021] | +0.003 [−0.006, +0.016] | +1.01 [+0.48, +3.19] (4 not reached) |
| DDRE-1D vs BSE | 14/24 | +0.010 [+0.002, +0.019] | +0.014 [+0.001, +0.024] | 0.00 [0.00, +1.57] |
| BSE vs fixed-k max | 28/96 | **−0.014** [−0.023, −0.006] | −0.006 [−0.017, +0.003] | −1.05 [−3.21, −0.45] |
| BSE vs fixed-k max | 14/24 | **−0.012** [−0.021, −0.003] | −0.013 [−0.023, −0.001] | 0.00 [−1.57, 0.00] |
| DDRE-M vs NB-M | 28/96 | −0.001 [−0.012, +0.011] | −0.002 [−0.013, +0.010] | (not reached in 1,020 of 10,000) |
| DDRE-M vs NB-M | 14/24 | −0.000 [−0.011, +0.012] | −0.002 [−0.014, +0.011] | (not reached in 1,353 of 10,000) |

**Cost at equal quality (G1):**
- At 28/96, BSE reaches its default balanced AP (0.565) at 2.05 NLI calls per
  sentence; DDRE-1D reaches it at 1.04.
- At 14/24, both reach it with one call.

The large HELM saving (2.7 calls) does not recur, because on FACTORY one
document already gives most of the attainable quality (§7).

**Baseline below fixed-k:** BSE lies below fixed-k max entailment at both
settings. DDRE-1D closes that gap.

**Retriever-aware variant:** DDRE-M is indistinguishable from NB-M. This
agrees with the decision to drop "retriever-aware" from the title.

## 4. Operating points (K = 10, C_retrieve = 1)

| Method | C_M/C_FA | Balanced AP | Nonfactual PR-AUC | Factual PR-AUC | ROC-AUC | Accuracy | Macro F1 | Brier | NLI calls |
|---|---|---|---|---|---|---|---|---|---|
| BSE | 28/96 | 0.565 | 0.416 | 0.718 | 0.582 | 0.487 | 0.480 | 0.472 | 6.34 |
| DDRE-1D | 28/96 | 0.576 | 0.418 | 0.734 | 0.596 | 0.521 | 0.521 | 0.419 | 7.35 |
| BSE | 14/24 | 0.552 | 0.386 | 0.718 | 0.568 | 0.487 | 0.480 | 0.412 | 4.15 |
| DDRE-1D | 14/24 | 0.570 | 0.400 | 0.738 | 0.591 | 0.526 | 0.525 | 0.373 | 5.16 |
| NB-M | 28/96 | 0.568 | 0.406 | 0.729 | 0.590 | 0.547 | 0.539 | 0.353 | 6.67 |
| DDRE-M | 28/96 | 0.571 | 0.408 | 0.733 | 0.587 | 0.546 | 0.538 | 0.372 | 3.63 |
| BSE-official | 28/96 | 0.552 | 0.387 | 0.716 | 0.571 | 0.475 | 0.465 | 0.379 | 3.36 |

**At the default budget:** as on HELM, DDRE-1D is better than BSE on every
detection metric and on Brier, and it reads more documents.

**Absolute level:** all detectors are weak on FACTORY (balanced AP 0.55–0.58,
against about 0.76 on HELM). The labels rest on web search, while our
evidence is 2023 Wikipedia (preregistration §9).

## 5. Sensitivity analyses (preregistration §6, descriptive)

| Analysis | Units | Clusters | Informative (fixed-k K10 bAP, lower bound) | H1a | H1b | H2a | H2b |
|---|---|---|---|---|---|---|---|
| Primary (Rule A) | 2,983 | 1,373 | yes (0.586, 0.568) | +0.0105 ✓ | +0.0102 ✓ | −0.0016 | −0.0003 |
| Rule B (nonfactual only) | 2,243 | 1,187 | yes (0.544, 0.528) | +0.0044 | +0.0058 | −0.0012 | +0.0003 |
| HELM + FACTORY pooled | 6,719 | 1,602 | yes (0.695, 0.676) | +0.0555 ✓ | +0.0487 ✓ | +0.0139 ✓ | +0.0160 ✓ |

✓ = would be "supported" by the preregistered rule. Sensitivity rows carry no
decision.

**Rule B:** same direction for H1, smaller, with one-sided p 0.11 and 0.06.
It has only 305 hallucinated sentences.

**Pooled:** the pooled AUC-gaps exceed both datasets' own, which is not an
average of them (§7.3). They are reported, not used as a result.

## 6. What this adds to HELM

| | HELM | FACTORY |
|---|---|---|
| H1, DDRE-1D vs BSE | confirmed (+0.043, +0.037) | **confirmed** (+0.011, +0.010) |
| H2, DDRE-1D vs fixed-k | confirmed (+0.008, +0.016) | not confirmed (−0.002, −0.000) |
| NLI calls saved at BSE's default AP | 2.7 (28/96), 1.4 (14/24) | 1.0 (28/96), 0 (14/24) |
| Fixed-k K10 balanced AP | 0.762 | 0.586 |

**Reading:**
- **Replacing BSE's histogram ratio with direct density-ratio estimation**
  improves the cost–quality curve on both datasets.
- **The size of the gain, and whether sequential stopping beats a fixed
  budget,** depend on how informative the evidence is.

## 7. Exploratory follow-ups (not preregistered)

`scripts/explore_factory_followups.py`, from the frozen features. Intervals
come from 2,000 cluster resamples.

### 7.1 Where DDRE-1D's and fixed-k's curves cross

**FACTORY:**
- The two envelopes differ by less than 0.006 on average within every cost
  band (1–2, 2–4, 4–7 and 7–10 calls).
- Fixed-k gains only +0.015 balanced AP from more documents: 0.576 with 1,
  at most 0.591 with up to 10. There is little for adaptive stopping to save
  or gain.

**HELM:**
- DDRE-1D is above fixed-k over 95% (28/96) and 100% (14/24) of the shared
  cost range.
- At 28/96 it is below fixed-k only below 2 calls per sentence.

### 7.2 Evidence strength

The evidence strength of a sentence is s = the largest entailment or
contradiction score over its ten passages. It uses no label. The DDRE-1D vs
fixed-k AUC-gap is shown within each dataset's tertiles of s:

| Dataset | Tertile | Units | Fixed-k K10 bAP | vs fixed-k 28/96 | vs fixed-k 14/24 |
|---|---|---|---|---|---|
| FACTORY | weak | 996 | 0.549 | −0.013 [−0.027, +0.003] | −0.016 [−0.027, −0.000] |
| FACTORY | middle | 996 | 0.541 | +0.007 [−0.005, +0.017] | +0.009 [−0.000, +0.017] |
| FACTORY | strong | 991 | 0.582 | +0.005 [−0.005, +0.015] | +0.005 [−0.002, +0.019] |
| HELM | weak | 1,247 | 0.680 | +0.020 [+0.004, +0.027] | +0.020 [+0.006, +0.030] |
| HELM | middle | 1,250 | 0.754 | +0.005 [−0.005, +0.016] | +0.010 [+0.001, +0.022] |
| HELM | strong | 1,239 | 0.846 | −0.003 [−0.016, +0.015] | +0.018 [+0.004, +0.034] |

**Reading, with caution:**
- On FACTORY, DDRE-1D trails fixed-k only where the evidence is weakest, and
  is level or slightly ahead elsewhere.
- FACTORY's strongest tertile (fixed-k 0.582) is still weaker than HELM's
  weakest (0.680).
- This fits "sequential stopping pays when documents carry signal". The
  pattern is not monotone, and the intervals overlap, so it is a hypothesis
  for future work, not a finding.

### 7.3 Why the pooled gaps exceed both datasets'

Balanced AP on pooled units also ranks HELM sentences against FACTORY
sentences, which per-dataset AP never does.

| Method, C_M/C_FA | Pooled bAP | Unit-weighted mean of HELM and FACTORY | Difference |
|---|---|---|---|
| BSE, 28/96 | 0.676 | 0.675 | +0.001 |
| DDRE-1D, 28/96 | 0.701 | 0.682 | **+0.020** |
| BSE, 14/24 | 0.629 | 0.646 | **−0.017** |
| DDRE-1D, 14/24 | 0.677 | 0.674 | +0.003 |

**Why:** pooling rewards DDRE-1D's cross-dataset ranking (it is better
calibrated) and penalises BSE's at 14/24. So the pooled AUC-gap measures
something per-dataset AUC-gaps do not, and it is not a summary of the two
tests.

## 8. Limitations

- **Evidence mismatch.** Labels rest on web search, while evidence is 2023
  Wikipedia. Detection is weak for every method, so effects are small.
- **Labels.** One annotator per sentence. Rule A counts "inconclusive" as
  hallucinated. Under Rule B the H1 direction holds, but its intervals reach
  0.
- **One query design.** Long questions dominate BM25 queries
  (preregistration §9). A sentence-focused query might give stronger
  evidence. It was not tried.
- **Unknown source.** The release does not identify the model or benchmark
  of each sentence, so there is no per-generator analysis.
- **Licence.** CC BY-NC 4.0, used for research only.

## 9. Run record and artifacts

| File | SHA-256 |
|---|---|
| `~/ddre-artifacts/confirmatory-v3/factory/factory_evidence_top10.jsonl` | `c9bcf9cf42338b07653025a9f0010f04d30fa18a65d6aa7a25314090337f7ac8` |
| `~/ddre-artifacts/confirmatory-v3/factory/factory_features_float16.sqlite` | `d00eb4ecf0041147e470f5c0e55fe604dbf7a4f810772023461f2f652a97b66b` |
| `~/ddre-artifacts/confirmatory-v3/factory/factory_confirmatory_result_float16.json` | `55efea0501cf78e57bcbca157fa15f03f387ef261b16b471229ba9e2788eaeb6` |
| `~/ddre-artifacts/confirmatory-v3/factory/factory_confirmatory_predictions_float16.csv` | `7e1a5c03811edebfe871940c7e0c307ae9b769a7914841770a72b116f6d21079` |
| `~/ddre-artifacts/confirmatory-v3/factory/exploratory/factory_followups.json` | `7738430a1bead799656f1ee382524fb91fa966f391830626e0dc56d8d39fe30c` (run from clean `main` `1242913`; identical to the draft that produced §7) |

All are read-only.
