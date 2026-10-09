# External-generalization dataset survey

**Status: research and planning only.** Nothing was downloaded, scored,
trained, tuned or run for this survey. Facts were taken from each dataset's
paper and official repository (October 2026); anything not confirmed from a
primary source is marked *(to verify)*. The SelfCheckGPT A/B/C/D experiment is
frozen and is not touched.

**Goal.** Choose an independent benchmark on which to test whether the frozen
A/B/C/D behaviour seen on SelfCheckGPT carries over: the efficiency of D
relative to A, and the ordering of the ablation cells.

---

## 0. What the frozen pipeline needs from a dataset

The detectors consume, for each unit being judged:

1. a **label**: factual vs hallucinated, at sentence or claim level, from humans;
2. **subclaims**: the unit decomposed into checkable statements (Wang used
   GPT-decomposed subclaims);
3. an **ordered evidence list** per subclaim, up to `max_docs = 10` documents,
   which the policy consumes one at a time;
4. a **premise text** per document, scored by the frozen DeBERTa NLI scorer
   (400-word spans, max over spans);
5. a **grouping unit** for the paired cluster bootstrap (SelfCheckGPT: passage).

**Frozen transfer** here means A, B, C and D are used exactly as frozen:
- the NBC histograms and both uLSIF fits are trained on Wang's released NBC pairs,
  not refit;
- κ = 2.5, the bands, σ/λ, CM 28 / CFA 96, P0 and max_docs are unchanged;
- no validation selection happens on the new data.

What must be built new (and frozen before any labels are inspected) is only
steps 2–3: subclaims and ordered evidence.

**Score-range warning carried over from the ablation.** The frozen evidence
models are supported only where the NBC training scores lie. On the
SelfCheckGPT held-out set the highest document score either method consumed
was **76.3**. Above about 83 both models are effectively unsupported:
- the BSE histogram buckets 8–9 (80–100) have no training examples, so their
  log-likelihood ratio is 0;
- frozen uLSIF falls from +1.27 (80–90) to +0.30 (90–95), then turns negative
  (−0.38 at 95–98, −0.88 at 98–100);
- the D-03 direction-flip region covers everything outside [56.95, 83.35].

A dataset whose evidence entails claims more strongly than Bing web pages did
will push scores into that region. That makes it a sharper test, but it is
also a risk that has to be handled in the preregistration, not afterwards.

---

## 1. Candidates at a glance

| | FActScore (labeled) | Factcheck-Bench | ExpertQA | RAGTruth | FELM | HaluEval |
|---|---|---|---|---|---|---|
| LLM-generated outputs | yes (3 LMs) | yes (ChatGPT) | yes (several systems) | yes (6 LMs) | yes (ChatGPT) | partly synthetic |
| Human labels | yes | yes | yes (experts) | yes | yes | yes / by construction |
| Unit labelled | atomic fact (grouped by sentence) | claim, sentence, document | claim | word span | segment | response |
| Evidence provided | knowledge source (Wikipedia) | top-5 ranked passages per claim | cited passages per claim | the RAG context | annotator reference links | one knowledge paragraph (QA) |
| Ordered list of up to 10 docs | constructible offline | 5, ranked, provided | no (needs live search) | ≤ 3 (QA) or 1 | no (needs live search) | no |
| Subclaims provided | yes (human-revised atomic facts) | yes (atomic claims) | yes (atomic claims, auto) | no | no | no |
| Size (usable units) | ~10–17k facts | 661 claims | thousands of claims *(to verify)* | 17,790 responses | 4,427 segments | 35k responses |
| Licence | MIT code; Wikipedia CC BY-SA | Apache-2.0 | MIT | MIT (sources restricted) | CC BY-NC-SA 4.0 | MIT |
| Independence from SelfCheckGPT | **low–moderate** | high | high | high | high | high |
| Frozen transfer feasible | **yes** | yes (5-doc lists) | only with a new retrieval design | weakly | only with new decomposition and retrieval | no |

---

## 2. Candidate details

### 2.1 FActScore human-annotated data

- **Paper.** Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual
  Precision in Long Form Text Generation*, EMNLP 2023. Repo `shmsw25/FActScore`
  (MIT).
- **Task/domain.** "Tell me a bio of ⟨entity⟩": biographies of 183 people
  sampled from Wikidata across 5 frequency levels × 4 nationality groups.
- **LLM outputs.** Yes: InstructGPT (text-davinci-003), ChatGPT, PerplexityAI.
  Response rates 99.5% / 85.8% / 90.7%.
- **Unit.** Atomic facts, produced per sentence by InstructGPT and then revised
  by human experts (split in 18% of cases, merged in 34%). Facts are grouped
  under their source sentence. Average facts per response: 26.3 / 34.7 / 40.8.
- **Labels.** Supported / Not-supported / Irrelevant against English Wikipedia.
  Label shares (S/NS/IR): InstructGPT 42.3/43.2/14.0%, ChatGPT 50.0/27.5/8.3%,
  PerplexityAI 64.9/11.1/14.8%. "Not-supported" means not supported by
  Wikipedia, which is close to, but not the same as, false.
- **Human annotation.** Yes. Upwork fact-checkers who passed a qualification
  test; 10% double-annotated, agreement 96% / 90% / 88%.
- **Evidence.** The released knowledge source is an English Wikipedia dump
  (2023-04-01) as a passage database. FActScore's own retriever takes passages
  from the entity's page.
- **Fair retrieval reconstruction.** Yes, fully offline and deterministic: a
  fixed retriever (e.g. BM25 with the atomic fact as query) over the fixed
  dump, ranked top-10, with deterministic tie-breaking. No live web, so
  identical for every cell.
- **Size.** About 183 × response rate × facts per response: roughly 4.8k
  InstructGPT, 5.5k ChatGPT and 6.8k PerplexityAI facts, minus irrelevant ones
  *(exact counts to verify from the release)*.
- **Download.** Public (Google Drive link in the repo).
- **Overlap with SelfCheckGPT/WikiBio.** **Substantial.** Same genre (biographies
  of Wikipedia people), and the same generator for the InstructGPT subset
  (SelfCheckGPT also used text-davinci-003). Entity overlap with WikiBio's 238
  people is possible *(to verify)*. ChatGPT and PerplexityAI are new generators;
  the evidence source (one Wikipedia dump instead of Bing pages) is new.
- **Same sequential setup.** Yes: fact → subclaim, retrieved passages → ordered
  documents, with no redesign of the detectors.
- **Same NLI scorer.** Yes, unchanged. Passages are short, so typically one span
  per document.
- **Consistent ordering.** Yes, deterministic from the fixed retriever and dump.
- **Expected cost.** Full coverage at 10 passages × 1 span per fact is roughly
  120k NLI pairs for ChatGPT + PerplexityAI, or roughly 170k with InstructGPT.
  At the about 7 pairs/s observed for batch-1 MPS, that is about 5–7 hours.
- **NLI distribution.** Very likely to have **much stronger entailment** than
  SelfCheckGPT: retrieved passages come from the subject's own Wikipedia page,
  so supported facts will often be directly entailed. This is the strongest
  exercise of the >76 region among the candidates.
- **Risks.**
  - weak independence (genre and one shared generator);
  - "Not-supported" includes unverifiable facts as well as false ones;
  - factual prevalence (about 50–65% supported) differs from SelfCheckGPT's 27%,
    which leaves the fixed cost cut miscalibrated for threshold metrics (PR-AUC
    is unaffected);
  - the high-score support problem above;
  - entity pages can have fewer than 10 passages.
- **Frozen transfer?** **Yes.** A/B/C/D can run unchanged with no validation
  retuning, once the retrieval and unit decisions are frozen.

### 2.2 Factcheck-Bench (Factcheck-GPT)

- **Paper.** Wang et al., *Factcheck-Bench: Fine-Grained Evaluation Benchmark
  for Automatic Fact-checkers*, Findings of EMNLP 2024 (arXiv 2311.09000). Repo
  `yuxiaw/Factcheck-GPT` (Apache-2.0).
- **Task/domain.** Open-domain questions answered by ChatGPT.
- **LLM outputs.** Yes: 94 ChatGPT responses.
- **Unit.** Document → 311 sentences (277 checkworthy) → 678 atomic claims
  (661 checkworthy). The claims are decomposed and decontextualised.
- **Labels.** Claim factuality: true / false / not-enough-evidence. 159 of 661
  claims are false and 30 undetermined. Document level: 61 responses contain an
  error.
- **Human annotation.** Yes: ten in-house annotators (graduate students to
  professors).
- **Evidence.** **Provided and ranked**: top-5 Google-retrieved passages per
  claim, "ranked by semantic relevance degree against the claim" (3,305
  claim–evidence–stance triplets). Annotators added evidence manually when the
  automatic evidence was insufficient.
- **Fair retrieval reconstruction.** No need to reconstruct: the released ranked
  lists can be used as the fixed order.
- **Size.** Small: 661 claims, 94 documents.
- **Download.** Public in the repo.
- **Overlap with SelfCheckGPT.** Low: open-domain questions, a different
  generator, Google evidence.
- **Same sequential setup.** Yes, but with **at most 5 documents** per claim
  instead of 10. Detectors run out of evidence at 5 ("exhausted"), which the
  code supports without change.
- **Same NLI scorer.** Yes.
- **Consistent ordering.** Yes, the released relevance ranking.
- **Expected cost.** About 3.3k NLI pairs: minutes.
- **NLI distribution.** Probably more strong entailment than Bing pages, because
  passages were re-ranked to the claim.
- **Risks.**
  - very small: 94 documents as bootstrap clusters, so quality intervals will be
    very wide;
  - the 5-document cap compresses the efficiency range;
  - manually added evidence may correlate with labels;
  - document-level imbalance (61 of 94 contain errors).
- **Frozen transfer?** **Yes**, with the 5-document cap as a declared property of
  the dataset.

### 2.3 ExpertQA

- **Paper.** Malaviya et al., *ExpertQA: Expert-Curated Questions and Attributed
  Answers*, NAACL 2024. Repo `chaitanyamalaviya/ExpertQA` (MIT).
- **Task/domain.** Expert questions across many professional fields, answered by
  several systems (GPT-4, Bing Chat, retrieve-and-read and post-hoc citation
  systems).
- **Unit.** Claims within answers, plus automatically produced atomic claims.
- **Labels.** Per-claim factual correctness and attribution/support, judged by
  the domain experts *(exact label scale to verify)*.
- **Evidence.** Per claim, the system's cited evidence (URL + passage or URL
  only). That is a small, unranked, system-dependent set, not a top-k list. The
  repo's FActScore-style evidence retrieval uses live Google search (top-10
  results → top-5 passages).
- **Fair retrieval reconstruction.** Not without a new, frozen retrieval design.
  Live Google results are not reproducible, and cited evidence differs by system.
- **Size.** 2,177 validated examples *(claim count to verify)*.
- **Overlap.** Low: expert domains and different generators.
- **Same sequential setup.** Only after building an offline retrieval corpus.
- **Same NLI scorer.** Yes.
- **Risks.**
  - correctness labels partly reflect expert knowledge beyond the shown evidence;
  - heterogeneous systems and citation behaviour;
  - retrieval design is the main open problem.
- **Frozen transfer?** **Only with a new retrieval component**, so not
  "unchanged" in the evidence pipeline.

### 2.4 RAGTruth

- **Paper.** Niu et al., *RAGTruth: A Hallucination Corpus for Developing
  Trustworthy Retrieval-Augmented Language Models*, ACL 2024. Repo
  `ParticleMedia/RAGTruth` (MIT).
- **Task/domain.** RAG outputs for QA (MS MARCO, 989 sources), summarization
  (CNN/DM 628, recent news 315) and data-to-text (Yelp, 1,033). 2,965 sources ×
  6 LLMs (GPT-3.5, GPT-4, Llama-2 7B/13B/70B, Mistral-7B) = 17,790 responses.
- **Unit.** Word-level hallucination spans (types such as evident conflict and
  baseless info). 7,664 responses contain 14,289 hallucination spans.
- **Labels.** Human, high quality. But the target is **faithfulness to the given
  context**, not world factuality, and the `implicit_true` flag marks correct
  but unsupported spans.
- **Evidence.** The provided context only: QA has exactly 3 MS MARCO passages;
  summarization and data-to-text have 1 source.
- **Fair retrieval reconstruction.** Not needed, but the "sequence" is at most 3
  documents, so retrieval efficiency barely has room to differ.
- **Licence.** MIT for the annotations; the source data carry their own terms
  (MS MARCO is research-only and the Yelp data has its own licence) *(to verify
  for redistribution)*.
- **Overlap.** Low.
- **Same sequential setup.** Only degenerately. It also needs a subclaim
  decomposition of response sentences, which is a new component.
- **NLI distribution.** **Strongest entailment of all** for faithful text, since
  the premise is the generation's own source.
- **Risks.**
  - the task definition differs (faithfulness rather than factuality);
  - spans must be converted to claim or sentence labels;
  - there is almost no retrieval budget to save.
- **Frozen transfer?** **Weakly:** the detectors run unchanged, but the
  efficiency question is nearly vacuous.

### 2.5 FELM

- **Paper.** Chen et al., *FELM: Benchmarking Factuality Evaluation of Large
  Language Models*, NeurIPS 2023 Datasets & Benchmarks. Repo `hkust-nlp/felm`.
  Code MIT; **data CC BY-NC-SA 4.0**.
- **Task/domain.** 847 ChatGPT responses across world knowledge, science/tech,
  writing/recommendation, reasoning and math.
- **Unit.** Response segments: 4,427 in total, 785 labelled non-factual. World
  knowledge has 532 segments (147 negative) and science/tech 683 (101 negative).
- **Labels.** Human, binary per segment, with error type, reason and reference
  links.
- **Evidence.** Annotator-chosen reference links only, which are not a
  retrieval list and are tied to the label decision.
- **Fair retrieval reconstruction.** Requires live search. Math, reasoning and
  writing segments are not retrieval-checkable.
- **Overlap.** Low.
- **Same sequential setup.** Only after adding decomposition and retrieval; in
  practice only about 1.2k segments are usable.
- **Risks.**
  - small usable subset;
  - non-commercial share-alike licence;
  - new decomposition component;
  - label-linked references.
- **Frozen transfer?** Not without new decomposition and retrieval components.

### 2.6 HaluEval

- **Paper.** Li et al., *HaluEval: A Large-Scale Hallucination Evaluation
  Benchmark for Large Language Models*, EMNLP 2023. Repo `RUCAIBox/HaluEval`
  (MIT).
- **Task/domain.** 30k task examples (QA from HotpotQA, dialogue from
  OpenDialKG, summarization from CNN/DM) whose hallucinated answers were
  **generated synthetically by ChatGPT on purpose**, plus 5k ChatGPT responses
  to Alpaca queries with human yes/no hallucination labels.
- **Unit.** Whole response.
- **Evidence.** One Wikipedia knowledge paragraph (QA and dialogue). The 5k
  general set has no evidence.
- **Same sequential setup.** No: there are no ranked lists, labels are
  response-level, and the hallucinations are not natural.
- **Frozen transfer?** **No.**

### Considered and excluded

- **LongFact / SAFE.** Labels are produced by an LLM agent, not humans.
- **WildHallucinations.** Labels are automatic.
- **WiCE.** Claims are human-written Wikipedia sentences, not LLM outputs.
- **LLM-AggreFact** (and the grounding sets it aggregates). One grounding
  document per claim, so there is no sequential retrieval.

---

## 3. Ranking

Criteria, in the requested order: (1) independence from SelfCheckGPT;
(2) compatibility with retrieval-aware detection; (3) label quality;
(4) preserving A/B/C/D without redesigning the task; (5) public
reproducibility; (6) computational cost.

| Rank | Dataset | (1) | (2) | (3) | (4) | (5) | (6) | Summary |
|---|---|---|---|---|---|---|---|---|
| 1 | **FActScore** | ◐ | ● | ● | ● | ● | ◐ | The only candidate where the whole protocol transfers unchanged at a useful size; the domain overlaps |
| 2 | **Factcheck-Bench** | ● | ● | ● | ◐ (5 docs) | ● | ● | Independent with native ranked evidence, but tiny |
| 3 | ExpertQA | ● | ◐ | ◐ | ○ | ◐ | ◐ | Independent and larger; retrieval must be invented |
| 4 | RAGTruth | ● | ○ | ● | ◐ | ◐ | ◐ | Excellent labels; wrong task for retrieval efficiency |
| 5 | FELM | ● | ○ | ● | ○ | ◐ | ● | Small usable subset; new components; NC licence |
| 6 | HaluEval | ● | ○ | ◐ | ○ | ● | ● | Synthetic, response-level |

● strong · ◐ partial · ○ weak

**The central trade-off.** The datasets that are most independent of
SelfCheckGPT (ExpertQA, FELM, RAGTruth) are the ones where A/B/C/D cannot run
unchanged: each would need a new retrieval or decomposition component, so a
transfer failure could not be told apart from a pipeline difference. FActScore
transfers cleanly but shares the biography genre. Factcheck-Bench is both
independent and compatible, but small.

---

## 4. Recommendation

**Top two:** FActScore (human-annotated subset) and Factcheck-Bench.

**Recommended primary: FActScore, ChatGPT and PerplexityAI subsets.**
1. It is the only candidate that supplies every input the frozen detectors need
   at a useful size: human-revised atomic facts (subclaims), human labels, and a
   fixed knowledge source. From that source an ordered 10-document list can be
   built offline and deterministically, so every cell sees identical evidence,
   with no live web.
2. A frozen-transfer run needs no redesign of A/B/C/D and no validation
   retuning.
3. It changes three things at once relative to SelfCheckGPT:
   - the generators (ChatGPT, PerplexityAI);
   - the evidence source (a Wikipedia dump instead of Bing pages);
   - the NLI-score distribution, toward strong entailment, which directly
     tests the unsupported >83 region the ablation flagged.
4. Excluding the InstructGPT subset removes the shared generator. Excluding any
   entities that also appear in SelfCheckGPT's WikiBio set removes material
   overlap.
5. **The claim it can support must be stated narrowly**: transfer across
   generators and evidence source within the biography genre, not domain
   generalization.

**Backup: Factcheck-Bench.** It is the strongest independent test the frozen
pipeline can run unchanged (open-domain, ChatGPT, Google evidence ranked by
the dataset authors). Its size (94 documents, 661 claims) suits a secondary,
efficiency-focused replication; the efficiency differences on SelfCheckGPT
were far more precise than the quality differences. If you weight independence
above sample size, the order of these two should be swapped.

---

## 5. Blockers to resolve before preregistration

1. **Release check (FActScore).** Confirm the labeled-data file structure
   (sentence → human atomic facts → S/NS/IR), exact counts per LM, and the
   knowledge-source download. Metadata only; no scoring.
2. **Overlap exclusions.** Compute entity-name overlap between FActScore's 183
   entities and SelfCheckGPT's 238 WikiBio people. Decide in advance to exclude
   overlapping entities and the InstructGPT subset (or to report it separately
   as a non-independent subset).
3. **Retrieval design, frozen before scoring:**
   - corpus: the FActScore 2023-04-01 dump;
   - scope: the entity's page only, or all of Wikipedia;
   - retriever: BM25 or GTR, with the query being the atomic fact alone or the
     fact plus the entity name;
   - top-10 ordering and tie-breaking;
   - handling entities with fewer than 10 passages.
4. **Unit and label mapping:**
   - the unit of analysis: atomic fact as a one-subclaim item, or sentence with
     facts as subclaims under the existing minimum aggregation;
   - Not-supported → hallucinated, and Irrelevant → excluded;
   - the bootstrap cluster: entity, or entity × generator.
5. **Score-support rule (the main scientific blocker).** Decide, before any
   score is seen, how scores in the unsupported region are treated. The options
   are:
   - (a) run frozen as-is and report a support diagnostic: the share of
     consumed scores above 76.3 (the SelfCheckGPT maximum) and above 83.35 (the
     flip-region edge);
   - (b) predefine the result as conditional on support coverage.

   No cap or calibration may be introduced afterwards. Under strict frozen
   transfer, (a) is the only option that keeps D unchanged.
6. **Prevalence shift.** About 50–65% of facts are supported, against 27%
   factual in SelfCheckGPT. With CM/CFA frozen, threshold-based metrics
   (accuracy, macro-F1, recall) will be affected; PR-AUC and the efficiency
   counts will not. Choose the primary endpoints accordingly.
7. **Endpoints and margins.** Size them for the expected number of clusters
   (about 150–180 entities per generator). Reuse the 0.01 interpretation margin
   or choose another, but fix it now.
8. **Factcheck-Bench (if used as backup).** Confirm the release contains the
   ranked top-5 passages per claim, and mark which evidence was added manually.
   Decide whether manually added evidence is excluded.
9. **Licensing.** Wikipedia content is CC BY-SA; Factcheck-Bench is Apache-2.0.
   No redistribution of derived caches is planned, but this should be confirmed.

---

### Sources

- FActScore: Min et al., EMNLP 2023, arXiv 2305.14251; `github.com/shmsw25/FActScore`
- Factcheck-Bench: Wang et al., Findings of EMNLP 2024, arXiv 2311.09000; `github.com/yuxiaw/Factcheck-GPT`
- ExpertQA: Malaviya et al., NAACL 2024; `github.com/chaitanyamalaviya/ExpertQA`
- RAGTruth: Niu et al., ACL 2024; `github.com/ParticleMedia/RAGTruth`
- FELM: Chen et al., NeurIPS 2023 D&B; `github.com/hkust-nlp/felm`
- HaluEval: Li et al., EMNLP 2023; `github.com/RUCAIBox/HaluEval`
- Score-range figures: `scripts/analyze_frozen_heldout.py` output on the canonical frozen held-out artifacts
