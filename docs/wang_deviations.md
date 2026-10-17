# BSE: Wang et al.'s paper, their released code, and this repository

Wang et al., *Hallucination Detection for Generative Large Language Models by
Bayesian Sequential Estimation*, EMNLP 2023, pp. 15361–15371.

**Code:** [xhwang22/HallucinationDetection](https://github.com/xhwang22/HallucinationDetection)
at `3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea` (2024-01-14, the latest commit).
This is the commit pinned in `src/provenance_guard.py:55`. On 2026-10-17,
`main.py`, `utils.py` and `NBC_feature.py` at that commit were checked
byte-identical to the copy this record was written from.

**Rule:** where the paper and the released code differ, this repository follows
**the code**, because Wang's Table 1 numbers were produced by the code. Every
difference is listed here. The paper cites this table.

| # | Item | Paper (section, equation) | Released code (file:line) | This repository |
|---|---|---|---|---|
| 1 | NLI model | DeBERTa-v3 fine-tuned on NLI (§3.3) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (main.py:41) | same (src/reproduction_gate.py:83) |
| 2 | NLI labels | "two-label classification (entailment and not entailment)" (§3.3) | 3-class model; entailment probability used; neutral + third class summed as "disentailment". The code names the third class `not_entailment`; the model calls it contradiction (utils.py:59-67) | same; entailment probability only |
| 3 | Softmax temperature | not mentioned | logits / 5 before softmax (utils.py:59) | same (src/utils.py:129) |
| 4 | Score scale and rounding | s ∈ (0, 1) (§3.3) | probability × 100, rounded to 1 decimal (utils.py:62) | same |
| 5 | Model input | not specified | `input_ids` only, no attention mask; truncation at the model maximum, 512 tokens (utils.py:56-57) | `**inputs` (with mask). Identical at batch size 1 |
| 6 | Spans | m = 400 **tokens**, step n = 100 tokens (§3.3) | 400 **words**, overlap 100, so the step is **300** words; first 4,000 words only; the last span is the final 400 words (utils.py:71-87) | same; exact duplicate tail spans removed (src/utils.py:19-43). This changes NLI-call counts only |
| 7 | Document score | max over spans (Eq. 1) | max over spans (main.py:238-247) | same |
| 8 | Discretisation of test documents | f = ⌊10·s⌋ (Eq. 2) | `int((s − 0.1)/10)` on the 0–100 scale (main.py:249) | same (src/baseline_core.py:18-21) |
| 9 | Discretisation of NBC pairs | same Eq. 2 | `int(s/10)`, a **different** discretiser (NBC_feature.py:34, 46) | same (src/baseline_core.py:12-15) |
| 10 | NBC training data | s = 200 factual + 200 nonfactual claims, one document each (§3.4) | files as released: **199 + 199** pairs | 199 + 199 (src/wang_data.py:149-164). Sensitivity to the 2 missing pairs: README, "Sensitivity analysis: the two unreleased NBC examples"; 97 of 100 placements were evaluated, so the result is inconclusive |
| 11 | NBC premise scoring | not specified | the **whole premise** in one NLI call, truncated at 512 tokens, **no span split** (NBC_feature.py:24-46) | same. Test documents use the span max, so training and test scores are different quantities; see `docs/d03_decision_record.md` §2 (D-13) |
| 12 | Smoothing | +1 (§4.1) | +1 on every bin count (main.py:160-162) | same |
| 13 | Prior | π(0) = 0.5 (§3.4) | 0.5 (main.py:100) | same |
| 14 | Posterior update | Eq. 4, documents independent (App. B) | Eq. 4 (main.py:249-251) | same |
| 15 | Stop risk | min((1 − π)·C_M, (1 − π₀(n))·C_FA) (Eq. 5) | min((1 − p)·C_M, p·C_FA) (main.py:121-125) | same |
| 16 | Look-ahead | C_retrieve + Σ_f R_stop(n+1)·P(f), weighted by the predictive P(f) (Eq. 8) | C_retrieve + R_stop(**mean of the 10 next posteriors, unweighted**) (main.py:127-140, 230, 255) | BSE-official = the code (src/baseline_core.py:119-124). Eq. 8 is also implemented, as `mode="eq8"` (src/baseline_core.py:126-141) |
| 17 | First decision | not explicit | the stop/continue test runs before the first document, so 0 documents is possible (main.py:229-233) | same (src/baseline_core.py:161-163) |
| 18 | Maximum documents | K = 10 (§4.1) | no cap: every saved page is available (main.py:232) | max_docs = 10; no effect: the released data hold at most 10 pages per subclaim (2,990 subclaims, 29,397 pages) |
| 19 | Aggregation | min over subclaims (Eq. 9) | min (main.py:262) | same |
| 20 | Decision | Bayes rule | factual iff (1 − P)·C_M < P·C_FA (main.py:272) | same |
| 21 | Default costs | C_M = 14, C_FA = 24, C_retrieve = 1 (§4.1); Table 1 also reports 28/96 | defaults C_M = 28, C_FA = 96 (main.py:82, 88; run.sh) | both evaluated; all comparisons use 28/96 (Gate 1 fails at 14/24) |
| 22 | Decomposition | GPT-3.5 `text-davinci-003`, zero-shot (§3.1) | released decompositions (`dataset/decomposed`) | the released decompositions |
| 23 | Retrieval | Bing Search API plus newspaper3k, live (§3.2) | released saved pages (`dataset/webpage`) | the released pages, in saved order, replayed offline |
| 24 | Dataset size | 1,908 sentences, 238 articles (§1); 298 articles (§4.1) | `dataset/selfcheckgpt/dataset.json` | 1,908 sentences, 238 passages |
| 25 | Metrics | AUC-PR nonfactual and factual, accuracy, passage Pearson and Spearman, evidence number | `precision_recall_curve` + trapezoidal `auc`; passage score = mean sentence P (main.py:284-302) | same (src/evaluation.py:30-39, 103-116) |
| 26 | Device and precision | not specified | CUDA if available, else **CPU**, float32 (utils.py:54; main.py:147) | MPS, float16. A float32 check matched to about 0.001 on every metric |

## Reproduction (Gate 1 v2, 2026-09-20)

| Costs | Our BSE vs Wang Table 1 | Verdict |
|---|---|---|
| C_M 28 / C_FA 96 | nonfactual AUC-PR 0.8595 vs 0.8645; factual 0.6253 vs 0.6196; accuracy 0.8218 vs 0.8239; Pearson 0.8080 vs 0.8118; Spearman 0.7259 vs 0.7420; documents 6.47 vs 6.22 | **PASS** |
| C_M 14 / C_FA 24 | documents 3.54 vs 3.05; Pearson 0.7705 vs 0.7137; Spearman 0.7028 vs 0.6455 | **FAIL**, cause open |

Plan v2 §6.4 runs Wang's unmodified code to locate the 14/24 gap. Rows 5, 6
(duplicate removal) and 26 are the only differences on our side. None of them
changes a document's score.
