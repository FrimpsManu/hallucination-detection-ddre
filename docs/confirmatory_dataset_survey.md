# Confirmatory dataset survey

**Status: research and planning.** This survey chose the datasets for the
confirmatory test of the multivariate DDRE result
([`mddre_results_summary.md`](mddre_results_summary.md)). Facts come from each
dataset's paper, README and code (October 2026). Anything a primary source does
not confirm is marked *(to verify)*.

**What was looked at in the data.** After the survey, the two chosen datasets
were downloaded and read only for their file structure and label totals (§4).
No model was run on them and no detector output exists.

---

## 1. Requirements

The detector judges each LLM-generated unit as factual or hallucinated. For
each unit it needs:

1. human factuality labels at sentence or claim level;
2. subclaims;
3. an ordered list of evidence documents per subclaim, either released with the
   data or reconstructible offline and deterministically;
4. evidence text for the NLI scorer;
5. a grouping unit for the cluster bootstrap.

The data must be **untouched** by every earlier analysis, and have **no
overlap** with FActScore biographies (the training data), Factcheck-Bench or
SelfCheckGPT WikiBio. The target size is **800–1,500 labelled units in a few
hundred clusters**. The reasoning: on Factcheck-Bench the matched-cost gain had
standard error 0.031 with 271 sentences. Exploratory effects shrink on
retesting, so assume the true gain is half the observed one, about +0.05.
Detecting that with 80% power needs a standard error of about 0.018.

## 2. Candidates

| Dataset | Human labels | Evidence | Size | Overlap | Verdict |
|---|---|---|---|---|---|
| **ExpertQA** (Malaviya et al., NAACL 2024) | experts, sentence-level correctness (5-point) | released: per atomic claim, top-5 passages from Google results, ranked by embedding similarity | 2,177 answers, 12,598 sentences | none | **chosen (primary)** |
| **BingCheck** (Li et al., Findings NAACL 2024) | AMT, claim-level support (4 classes), LLM pre-labels | released: top-3 Bing pages per claim, full text | 396 responses, 3,840 claims | none | **chosen (second)** |
| ANAH, English (Ji et al., ACL 2024) | GPT-4 drafts, human-corrected | one reference document per topic | 6,606 sentences | possible (celebrity topics) | no: labels relative to one document; about 90% non-hallucinated in one half |
| FELM, world knowledge + science (Chen et al., NeurIPS 2023) | experts, 2 per response | annotator URLs only, chosen to justify the label | 1,215 segments | to check (TruthfulQA prompts) | no: small; evidence leaks the label |
| FactBench (Bayat et al., ACL 2025) | 2 annotators, κ 0.57 | live Google, not released | 4,467 units, 40 prompts | none | no: evidence not reproducible; 40 clusters |
| FacTool KB-QA | authors | not released | 233 claims | none | no: too small |
| OpenFactCheck components | mixed | mostly not released | — | **includes Factcheck-Bench and FActScore** | no |
| HaluEval 2.0 (human subset) | 2 labellers + checker | none | 1,000 questions | low | no: evidence would have to be built from scratch |
| REVEAL (Jacovi et al., ACL 2024) | 5 annotators, α 0.49 | ≤ 3 Wikipedia paragraphs | 1,979 steps | none | no: reasoning steps; labels relative to ≤ 3 paragraphs |
| RefChecker NQ subset | experts | 1 reference paragraph | 3,319 triplets | none | no: triplets; one reference |
| FAVA | 2 annotators | not recorded | 460–902 passages *(to verify)* | none | no: no evidence |
| FaStfact-Bench, FactRBench, WildHallucinations, SAFE/LongFact, VeriScore, FactCHD, LLM-Oasis, SciHal25 | — | — | — | some reuse FActScore or Factcheck-Bench | no: LLM labels, planted errors, reused data or too small |

## 3. Why these two

**ExpertQA**
- It matches the detector exactly: sentence → atomic claims → 5 ranked passages
  per atomic claim. That is the same shape as Factcheck-Bench (≤ 5 documents).
- The atomic claims and passages were produced by the authors *after*
  annotation, for FActScore-style estimation. They came from the original claim
  string, not the expert's correction (`break_down_to_atomic_claims.py` reads
  `claim['claim_string']`). So the evidence does not encode the label.
- Labels are expert judgements of factual correctness against world knowledge.
- Weaknesses: a single annotator per sentence, with no inter-annotator agreement
  reported; hallucinations are rare.

**BingCheck**
- It is the closest structural copy of Factcheck-Bench: open-domain chatbot
  answers, claim-level labels, and the search results saved with the data.
- Annotators judged support against the same saved pages the detector reads.
- Weaknesses:
  - The labels are *support against retrieved evidence*, not world-level truth.
  - One crowd worker labelled each claim, correcting LLM pre-labels.
  - Only 3 documents per claim.
  - No licence is stated: research use only, not redistributed, and the authors
    are to be asked.
  - Bing Chat may have written its answers from the same pages.

## 4. What was inspected

Files were downloaded once and pinned by sha256 (see the preregistration, §2).
Only these were printed:

**ExpertQA**
- Structure: field names, types and list lengths.
- Answers per system: rr_gs_gpt4 452, bing_chat 470, post_hoc_gs_gpt4 399,
  gpt4 174, rr_sphere_gpt4 279, post_hoc_sphere_gpt4 403.
- Correctness totals over 12,598 sentences:

  | Label | Count |
  |---|---|
  | Definitely correct | 7,426 |
  | Probably correct | 3,404 |
  | Unsure | 1,113 |
  | Likely incorrect | 255 |
  | Definitely incorrect | 347 |
  | missing | 53 |

- 55,324 atomic claims and 484 question authors.
- 51 sentences have no atomic claims, and 72 atomic claims have no passages.
- After the preregistered rules: 11,389 sentences, of which 596 are
  hallucinated. These were computed as totals only.
- Passage counts: 55,094 atomic claims have 5 passages and 158 have 1–4.
- The two release files agree exactly on every claim string and correctness
  label (counts of mismatches only).

**BingCheck**
- Structure: field names, types and list lengths.
- 396 responses (397 files; `examples.json` is a 134-byte placeholder).
- Claim labels:

  | Label | Count |
  |---|---|
  | supported | 2,681 |
  | partially supported | 900 |
  | not supported | 217 |
  | refuted | 42 |

- Search results per claim: 3,827 claims have 3, 9 have 2, 1 has 1 and 3
  have 0. One response's claim lists differ between fields.
- Page lengths: median 1,120 words, maximum 870,953. That gives about 113k NLI
  spans in total.

**Disclosed slip.** The first structure printout listed one BingCheck
response's six claim texts as dictionary keys. The string lengths of their
labels were shown alongside, which reveals those six labels. Nothing else about
claim content was seen. No detector or NLI score was computed.

Sources: [ExpertQA paper](https://arxiv.org/abs/2309.07852),
[ExpertQA repo](https://github.com/chaitanyamalaviya/ExpertQA),
[Self-Checker/BingCheck paper](https://aclanthology.org/2024.findings-naacl.12),
[BingCheck repo](https://github.com/Miaoranmmm/SelfChecker),
[ANAH](https://arxiv.org/abs/2405.20315), [FELM](https://arxiv.org/abs/2310.00741),
[FactBench](https://aclanthology.org/2025.acl-long.1587),
[FacTool](https://arxiv.org/abs/2307.13528), [OpenFactCheck](https://arxiv.org/abs/2408.11832),
[HaluEval 2.0](https://arxiv.org/abs/2401.03205), [REVEAL](https://arxiv.org/abs/2402.00559),
[RefChecker](https://arxiv.org/abs/2405.14486), [FAVA](https://arxiv.org/abs/2401.06855),
[FaStfact](https://arxiv.org/abs/2510.12839), [WildHallucinations](https://arxiv.org/abs/2407.17468).
