# Naming scheme

From 2026-10-17 (research plan v2), new documents, code and the paper use the
names in §1. Frozen historical documents and artifacts keep their own names;
§2 maps those names to the new ones. When citing a historical name, prefix it
with its study (for example, "ablation cell D" or "multivariate G2").

## 1. Names used from now on

### Methods

| Name | Meaning |
|---|---|
| **BSE-official** | Wang's BSE exactly as in the released code: histogram ratio, uniform-average look-ahead (`docs/wang_deviations.md` row 16) |
| **BSE** | Wang's histogram ratio with the paper's Eq. 8 look-ahead |
| **DDRE-1D** | uLSIF ratio on entailment, fitted on Wang's NBC pairs, Eq. 8 look-ahead (plan v2, Tier 1) |
| **Hist-1D**, **DDRE-1D′** | histogram vs uLSIF on entailment, fitted on the FActScore training split (Tier 2) |
| **NB-M**, **DDRE-M**, **Logit-M** | Naive-Bayes histograms, uLSIF and logistic ratio on (entailment, contradiction, log rank), FActScore training split (Tier 2) |
| **Fixed-k (Max / Avg / NBC)** | the first k documents, aggregated as in Wang's Figure 2 |

"DDRE" alone means the method family: BSE with the likelihood ratio estimated
directly.

### Goals

| Name | Meaning |
|---|---|
| **G1, cost** | fewer NLI calls and documents at equal quality |
| **G2, detection** | better hallucination-detection performance at equal cost |
| **G3, calibration** | better-calibrated posteriors |

### Endpoints

- **E-cost:** NLI calls saved at the reference method's quality.
- **E-quality:** quality gain at the reference method's NLI calls.
- **AUC-gap:** area between cost–quality envelopes.

### Gates

- **R1, R2, …** are reproduction gates (Wang's numbers).
- **X1, X2, …** are execution gates of a run (pins, totals, equality).

The letters G and E are not used for gates or endpoints, because they are taken
by goals and by historical endpoints.

### Features

- **e:** max entailment over spans (0–100).
- **k:** max contradiction over spans.
- **rank:** the retriever's position (1 = first).

## 2. Historical names

### Methods and cells

| Historical name | Where | New name or meaning |
|---|---|---|
| BSE official, `A_official`, `bse_official` | everywhere | **BSE-official** |
| Ablation cell **A** | ablation, FActScore transfer | BSE-official |
| Ablation cell **B** | same | BSE histogram ratio under a validation-selected posterior band |
| Ablation cell **C** | same | κ-scaled histogram ratio under a band |
| Ablation cell **D0** | same | uLSIF on e, the CV-selected fit (σ 0.2269) under a band |
| Ablation cell **D**, "frozen DDRE", "1-D DDRE" | frozen held-out run, ablation, FActScore transfer | **DDRE-1D-v1**: uLSIF on e, σ 0.11346, λ 1.0, band [0.20, 0.80], selected from 640 validation configurations, unnormalised. Historical; not the plan-v2 DDRE-1D |
| Multivariate cell **A** | multivariate study, confirmatory test | Wang's histogram under BSE's band, which equals BSE-official's documents (multivariate G2) |
| Multivariate cell **A′** | same | Hist-1D under BSE's band |
| Multivariate cell **U** | same | DDRE-1D′ under BSE's band |
| Multivariate cell **M**, "mDDRE" | same | DDRE-M under BSE's band |
| Multivariate cell **L** | same | Logit-M under BSE's band |

### Research questions

| Historical name | Where | New name or meaning |
|---|---|---|
| RQ1 / RQ2 | README (top) | RQ1: cost while "maintaining" quality; RQ2: which component produces the saving |
| RQ1 / RQ2 | FActScore transfer prereg | RQ1: does cell D use less computation; RQ2: does the A–D pattern recur |
| RQ1 / RQ2 / RQ3 | multivariate design, results, confirmatory | G1 / G2 / component analysis |

### Gates and endpoints

| Historical name | Where | New name or meaning |
|---|---|---|
| Gate 1 (v2) | README, reproduction | R1: our BSE vs Wang Table 1 |
| G0, F, G1, G2, G3, P | multivariate and confirmatory scripts | data totals, NBC equality, engine reproduces the ablation, cell A uses BSE's documents, amended run equals the frozen one, pins |
| E1, E2 | confirmatory prereg | net matched-quality saving (documents), matched-cost gain (balanced PR-AUC) |
| D-01 … D-13 | `docs/ddre_scientific_audit.md` | audit findings; cite as "audit finding D-03" |
