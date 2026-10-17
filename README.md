# Direct Density Ratio Estimation for Efficient Retriever-aware Hallucination Detection in Large Language Models

LLMs can state false things fluently and confidently. Retrieval-based
detectors check each claim against evidence documents. Their cost grows with
every document they read and score with an NLI model.

[Wang et al. (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.949/)
introduced **Bayesian Sequential Estimation (BSE)**. It reads documents one at
a time and stops once reading another is not worth its cost. After each
document, BSE updates P(factual) with a likelihood ratio
r = P(evidence | factual) / P(evidence | hallucinated). It obtains r by
counting two 10-bin histograms of entailment scores and dividing them.

**This project keeps BSE's pipeline identical and changes one step.** The
likelihood ratio is estimated **directly**, with density-ratio estimation
(uLSIF; Kanamori, Hido & Sugiyama, JMLR 2009), instead of as a quotient of
two histograms. The question is whether that makes sequential hallucination
detection:

| Goal | Question |
|---|---|
| **G1, cost** | Does it reach BSE's detection quality with fewer NLI calls and documents? |
| **G2, detection** | Does it separate factual from hallucinated claims better at BSE's cost? |
| **G3, calibration** | Are its probabilities better calibrated? |

Here "factuality" means detecting hallucinated claims. The method does not
change what the LLM writes.

## Main result (preregistered, confirmed)

On **HELM** (Su et al., Findings of ACL 2024), the test was preregistered in
[`docs/helm_preregistration.md`](docs/helm_preregistration.md) before any
detector touched the data, and run once. HELM has 3,736 LLM-generated
sentences from seven LLMs, 1,978 of them hallucinated, with human labels.

| Hypothesis | C_M/C_FA | AUC-gap (balanced AP) | 5th percentile | Result |
|---|---|---|---|---|
| DDRE-1D's cost–quality curve lies above BSE's | 28/96 | **+0.043** | +0.036 | **confirmed** |
| | 14/24 | **+0.037** | +0.030 | **confirmed** |
| DDRE-1D's curve lies above fixed-k max entailment's | 28/96 | **+0.008** | +0.002 | **confirmed** |
| | 14/24 | **+0.016** | +0.010 | **confirmed** |

Both comparisons pass Holm's correction over the four one-sided tests
(10,000-replicate cluster bootstrap over 229 articles).

**What the curves mean for the goals:**
- **G1:** at BSE's quality, DDRE-1D needs 2.7 fewer NLI calls per sentence at
  28/96 (95% CI [1.9, 3.5]).
- **G2:** at BSE's cost, its average precision is +0.024 higher [0.016, 0.036].
- **G3:** it has a lower Brier score at both operating points.

**Caveats:**
- The advantage holds along the cost–quality curve. At Wang's exact
  operating point, DDRE-1D reads slightly more documents than BSE.
- Effects are modest.

Full results: [`docs/helm_results_summary.md`](docs/helm_results_summary.md).

### What else we found

- **Wang's Eq. 8 never reads a document at the published costs.**
  Implemented exactly as written in the paper, the one-step look-ahead never
  retrieves at C_M 28 / C_FA 96: no single document can flip the decision from
  the prior. Wang's released code uses a different heuristic. We stop every
  method by **backward induction**, the Bayes-optimal rule that the paper
  names as ideal ([plan amendment 1](docs/research_plan_v2.md)).
- **A simple baseline is strong.** Reading the first k documents and taking
  the maximum entailment beats BSE at equal cost on most datasets, HELM
  included. DDRE-1D beats it (H2 above).
- **The retriever-aware variant is not supported.** DDRE-M is uLSIF on
  entailment, contradiction and retrieval rank. It lies below its own
  histogram control (Naive-Bayes histograms on the same features) on HELM and
  on most exploratory datasets.
- **Earlier configurations did not hold up.** These include a 1-D DDRE tuned
  over 640 configurations, and a multivariate model under BSE's stopping
  band. A preregistered test on ExpertQA and BingCheck did not confirm them.
  Everything is reported in `docs/`.

## Method in one picture

```
sentence ─▶ subclaims ─▶ retrieve documents in rank order ─▶ NLI score per document
                                                                     │
                       likelihood ratio r(evidence)  ◀───────────────┘
              BSE: two counted histograms, divided   |   DDRE: r estimated directly (uLSIF)
                                                                     │
         posterior update  ─▶  stop or read another document (backward induction)
                                                                     │
                      min over subclaims ─▶ factual / hallucinated
```

Every method shares the documents, the prior, the update, the stopping rule
and the aggregation. Only the ratio differs. The differences between the
paper, Wang's released code and this repository are listed in
[`docs/wang_deviations.md`](docs/wang_deviations.md).

## The research record, in reading order

| Document | What it is |
|---|---|
| [`docs/research_plan_v2.md`](docs/research_plan_v2.md) | The plan: framing, single-factor design, goals, amendments 1–2 |
| [`docs/wang_deviations.md`](docs/wang_deviations.md) | BSE in Wang's paper vs Wang's code vs this repository (26 items) |
| [`docs/v2_results_summary.md`](docs/v2_results_summary.md) | Exploratory results on five used datasets (Phase 5) |
| [`docs/v2_budget_results.md`](docs/v2_budget_results.md) | Exploratory budget-knob curves and the fixed-k contrasts (amendment 2) |
| [`docs/helm_preregistration.md`](docs/helm_preregistration.md) | The confirmatory test, frozen before any detector touched HELM |
| [`docs/helm_results_summary.md`](docs/helm_results_summary.md) | The confirmatory results |
| [`docs/naming.md`](docs/naming.md) | Method, goal and gate names; map of historical names |

**Earlier work, kept as it was run:**
- The 1-D DDRE and its D-03 decision: [`docs/d03_decision_record.md`](docs/d03_decision_record.md).
- The ablation: `docs/ablation_*`.
- The FActScore transfer: `docs/factscore_transfer_*`.
- The multivariate study: `docs/multivariate_ddre_design.md` and `docs/mddre_results_summary.md`.
- The ExpertQA/BingCheck confirmatory test: `docs/confirmatory_*`.
- The BSE reproduction forensics, in the previous README:
  [`docs/history/README_v1.md`](docs/history/README_v1.md).

## Reproducing

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests          # ~1,650 tests, under a minute, no data needed
python scripts/prepare_wang_data.py           # Wang et al.'s released data -> data/wang/
```

**Data and artifacts are not in git.** Large inputs and every result live
under `~/ddre-artifacts/` and are identified by SHA-256 in the documents
above. Each script below:
- checks its pinned inputs before running;
- refuses to overwrite an existing output;
- writes its result once, read-only;
- supports `--dry-run` (most scripts), which checks inputs without writing.

**Pipeline for the current results:**

| Step | Script | Produces |
|---|---|---|
| Wikipedia BM25 index (FActScore snapshot) | `scripts/build_wikipedia_index.py` | `wikipedia_bm25_index.sqlite` |
| Training evidence and NLI features | `scripts/retrieve_mddre_training_evidence.py`, `scripts/score_mddre_features.py` | multivariate feature cache |
| Fit estimators and look-ahead supports | `scripts/fit_v2_estimators.py` | `v2/estimators_v2_float16.json` |
| Exploratory replay | `scripts/run_v2_replay.py` | `v2/v2_replay_result_float16.json` |
| Budget-knob curves | `scripts/run_v2_budget.py` | `v2/v2_budget_result_float16.json` |
| HELM retrieval | `scripts/retrieve_helm_evidence.py` | `helm_evidence_top10.jsonl` |
| HELM NLI features | `scripts/score_helm_features.py` | `helm_features_float16.sqlite` |
| HELM confirmatory test | `scripts/run_helm_confirmatory.py` | `helm_confirmatory_result_float16.json` |
| Figures | `scripts/plot_v2_curves.py`, `scripts/plot_v2_budget.py`, `scripts/plot_helm_curves.py` | `docs/figures/*.svg` |

**Runtime:** NLI scoring runs Wang's DeBERTa-v3 checkpoint
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) in float16 on
Apple MPS. Everything after scoring runs from cached features.

## Data

Datasets used: SelfCheckGPT, as released by Wang et al.; FActScore;
Factcheck-Bench; ExpertQA; BingCheck; HELM.

No dataset is redistributed in this repository. Please follow each source's
terms and cite it. BingCheck and HELM state no licence; their authors were
contacted for permission (October 2026).

## How this work was done

- **One factor at a time:** every comparison changes only the likelihood
  ratio.
- **Preregistration:** hypotheses, endpoints and decision rules were written
  and merged before the data they test were touched. Changes are appended as
  dated amendments.
- **Run once:** every run starts from a clean, reviewed commit and is pinned
  by hash.
- **Honest reporting:** exploratory and confirmatory results are labelled as
  such, and every outcome is reported, including the ones that did not
  support the method.

## Citation

```bibtex
@misc{manufrimpong2026ddre,
  author = {Manu-Frimpong, Augustine},
  title  = {Direct Density Ratio Estimation for Efficient Retriever-aware
            Hallucination Detection in Large Language Models},
  year   = {2026},
  note   = {Code and research record: https://github.com/FrimpsManu/hallucination-detection-ddre}
}
```

Please also cite Wang et al. (2023) for BSE and the released data, and the
authors of each dataset used.
