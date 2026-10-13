# FActScore frozen-transfer test — preregistration

**Status: frozen when merged to `main`.** Written on 2026-10-13, before any
FActScore evidence was retrieved, any FActScore document was scored by the NLI
model, or any detector was run on FActScore. Only label metadata (file
structure, counts, entity names) and page lengths in the Wikipedia snapshot
have been read.

**Purpose.** Test whether the SelfCheckGPT findings carry over to an
independent dataset when every method is used exactly as frozen, with no
retuning:

- **RQ1:** does DDRE (cell D) use less computation than BSE (cell A) while
  keeping comparable factuality-detection performance?
- **RQ2:** does the A/B/C/D pattern recur?

Dataset choice and reasons: [`generalization_dataset_survey.md`](generalization_dataset_survey.md).

---

## 1. Data

**Source.** FActScore human-annotated release, `data.zip`
(SHA-256 `404482c9a7d588d581a2f7a78d609fe6bfec4ddc0f729c8560a61b1743dbffab`):

| File | SHA-256 |
|---|---|
| `data/labeled/ChatGPT.jsonl` | `b58fe7113acc28dee8d3e6ee7e426bfdedaa293bba270f0977f568b9319e92d4` |
| `data/labeled/PerplexityAI.jsonl` | `81ada151f2de8b7c205e3c15491e54b92d89bfb37b05366dca8b7a0486f109ff` |

**Exclusions** (fixed now, for the stated reasons):

| Excluded | Reason |
|---|---|
| The InstructGPT subset | Same generator as SelfCheckGPT (text-davinci-003) |
| Entity "Bridget Moynahan" | Also SelfCheckGPT passage 201 (name check over all 238 passages; no other match) |
| Responses with no annotations | The model declined to answer |
| Sentences with `is-relevant = false` | Not evaluated by the annotators |
| Facts labelled IR (irrelevant) | FActScore removes them without validation |
| Sentences left with no S/NS fact | Nothing to evaluate |

**Analysis set:**

| Generator | Entities | Sentences | Factual | Hallucinated | Facts (S / NS) |
|---|---|---|---|---|---|
| ChatGPT | 156 | 1,202 | 438 | 764 | 4,854 (3,167 / 1,687) |
| PerplexityAI | 155 | 1,371 | 963 | 408 | 5,532 (4,777 / 755) |
| **Pooled** | **172** | **2,573** | **1,401** | **1,172** | **10,386** |

If the implemented loader yields any other count, it stops and the
difference is reported before anything else runs.

---

## 2. Units and labels (mirroring SelfCheckGPT)

- **Unit:** a sentence of a generated biography.
- **Subclaims:** its human-revised atomic facts labelled S or NS, in release
  order.
- **Label:** factual iff every subclaim is S; hallucinated if any is NS.
- **Sentence score:** the minimum over subclaims, unchanged from SelfCheckGPT.
- **Bootstrap cluster:** the entity (person). In the pooled analysis, one entity's
  sentences from both generators form one cluster.

---

## 3. Evidence: ordered top-10 per subclaim

- **Corpus.** FActScore's Wikipedia snapshot `enwiki-20230401.db`
  (SHA-256 `31cf7b6b4465459844bb00f3a6ac75560fc7d1525112205a21859323dc5d33d7`,
  6,187,531 pages), split into passages on FActScore's own separator. Every
  passage of every page is indexed.
- **Retriever.** Okapi BM25 as implemented by SQLite FTS5 (`bm25()`, k1 = 1.2,
  b = 0.75), with two columns, page title and passage text, weighted 1.0 each,
  and tokenizer `unicode61 remove_diacritics 2`.
- **Query.** `entity name + " " + fact text`, FActScore's own query form. It is
  lowercased and tokenized by the same tokenizer, a fixed English stopword list
  (committed with the code) is removed, duplicates are dropped, and the
  remaining terms are joined with OR.
- **Order.** The top 10 passages by BM25 score. Ties are broken by page title,
  then passage index, both ascending. This order is the reading order for
  every cell.
- **Document text given to the NLI scorer:** `title + "\n" + passage`.
- **Freeze.** Retrieval runs once for all 10,386 subclaims, using no labels, and
  is written to one evidence file. That file and the search index are hashed
  before any NLI scoring.

Searching all of Wikipedia was chosen over the entity's own page because
69% of the entities' pages have fewer than 10 passages (median 5). That would
cap the evidence well below SelfCheckGPT's roughly 9.8 documents per subclaim,
and would make the long evidence chains where DDRE saved most impossible.

---

## 4. Scoring

- **Scorer.** The frozen SelfCheckGPT scorer, unchanged: DeBERTa-v3-large NLI
  at revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, score version
  `wang-emnlp23-temp5-seg400-overlap100-hostscale-v2`, 400/100-word segments,
  max over spans, batch size 1, MPS, float16.
- **Coverage.** Every retrieved document (10 per subclaim) is scored in one
  pass, into a new cache. This happens after the existing 398-pair exact-score
  probe passes, and the result is hashed.
- **What may be inspected during scoring.** Only completion status, counts,
  environment and hashes: no score distributions and no detector output.

---

## 5. Methods: all frozen, nothing selected

| Cell | Configuration (as frozen on SelfCheckGPT) |
|---|---|
| A | BSE official, CM 28, CFA 96, C_retrieve 1 |
| B | BSE histogram evidence, band [0.20, 0.85] |
| C | 2.5 × histogram evidence, band [0.05, 0.95] |
| D | uLSIF σ 0.11346435546875, λ 1.0, band [0.20, 0.80] |

- The histograms and the uLSIF fit come from Wang's NBC pairs, as before; there
  is no refit on FActScore. B and C are read from the validation freeze
  `a1824f66…` and D from its constants.
- P0 0.5 and max_docs 10 are unchanged. There is no FActScore validation split
  and no selection of any kind. D0 is not evaluated.
- Each cell runs once on the full analysis set.

---

## 6. Endpoints and statistics

Paired cluster bootstrap over entities, 10,000 resamples, seed 42, 95%
percentile intervals. Every difference is paired across cells within each
replicate.

**Primary (RQ1, pooled analysis set):**

| Endpoint | Definition |
|---|---|
| Efficiency | S(D) = docs(A) − docs(D), documents per sentence |
| Quality | balanced PR-AUC(D) − balanced PR-AUC(A) |

**Secondary:**
- NLI calls saved, A − D;
- nonfactual and factual PR-AUC differences;
- per-generator results;
- all metrics from the SelfCheckGPT ablation (§6 of
  [`ablation_preregistration.md`](ablation_preregistration.md)) for A, B, C and
  D, including S(B), S(C), docs(B) − docs(C), docs(C) − docs(D), the two ratios
  with the same 0.05 denominator guard, and the stopping-depth distributions.

**Threshold metrics.** Accuracy, macro-F1, precision and recall are reported,
but not used for any conclusion. The frozen cost cut (28 / 124) was set for
SelfCheckGPT's 27% factual rate; here the rate is 54% pooled (36% ChatGPT, 70%
PerplexityAI).

---

## 7. Decision rules (fixed now)

**RQ1 on FActScore:**

| Outcome | Condition |
|---|---|
| Efficiency transfers | the 95% CI of S(D) lies entirely above 0 |
| Quality comparable | the lower 95% CI bound of balanced PR-AUC(D) − (A) is ≥ −0.03 |
| RQ1 supported on FActScore | both of the above |
| Quality improvement shown | the lower bound is > 0 (reported, not required) |

**About the 0.03 margin.** It is the largest balanced PR-AUC loss we would still
call comparable for the kind of compute saving being tested. It also matches
the precision this sample can reach: SelfCheckGPT's 190-passage intervals had
half-widths of about 0.03. It is fixed now and not revisited.

**RQ2 (secondary, exploratory).** The A → B, B → C and C → D rules of the
SelfCheckGPT preregistration (§7 there: 0.5, 0.8, 0.01, CI excludes 0) are
applied unchanged and reported as whether the SelfCheckGPT pattern recurs.

**All outcomes are reported, whichever way they fall**, including a failure to
transfer.

---

## 8. Score-support diagnostic (decided before any score is seen)

The frozen evidence models have no training support above about 83: the
histogram buckets 8–9 are empty, and frozen uLSIF turns negative at 95 and
above. Wikipedia evidence may produce stronger entailment than Wang's web
pages did.

- D is run unchanged: no cap, no calibration, no exclusion of documents.
- Reported for every cell: the share of consumed document scores above 76.27
  (the highest SelfCheckGPT held-out score consumed), above 83.35 (edge of the
  D-03 flip region) and at or above 95, plus the same shares over all
  retrieved documents.
- Every RQ1 statement carries this context. If more than 10% of the documents D
  consumes score above 83.35, the result is labelled as **outside the frozen
  estimator's support**, and is not read as evidence for or against the method
  in its supported range.

---

## 9. Execution order

1. Merge this document. From then on it is frozen.
2. PR: corpus index and retrieval code, then build the index, retrieve once,
   and freeze the evidence file and index (hashed).
3. PR: FActScore loader, plus scoring of all retrieved documents after the
   398-pair probe; the cache is frozen (hashed).
4. PR: transfer replay of A–D, run once, and report against §6–§8.

No outcome is inspected before step 4, and step 4 runs once.

## 10. Amendments

After merge, the data, exclusions, units, retrieval specification, methods,
endpoints, margins, decision rules and support rule are frozen. Only clerical
corrections and non-substantive implementation clarifications are allowed,
each dated below with its reason. Any substantive change is a new protocol
version and is reported as a deviation.
