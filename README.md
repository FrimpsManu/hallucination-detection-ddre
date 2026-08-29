# Retrieval-Aware Hallucination Detection with Direct Density-Ratio Estimation

Research code for an in-progress paper extending Wang et al., **“Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation” (EMNLP 2023)**.

## Research objective

The objective is to develop a retrieval-aware hallucination-detection framework that **reduces computational overhead while improving factuality-detection performance** relative to Bayesian sequential estimation (BSE). The application motivation is reliable, lower-latency LLM-based enterprise customer-service systems.

## Research question

**Can retrieval-aware direct density-ratio estimation reduce the computational cost of hallucination detection while improving factuality-detection performance compared with Bayesian sequential estimation?**

## Controlled comparison

The proposed method does **not** receive easier evidence than the BSE baseline. Both methods use the same released experimental pipeline from Wang et al.:

1. SelfCheckGPT generated sentences and human factuality labels;
2. Wang et al.'s released decomposed subclaims;
3. Wang et al.'s ordered retrieved web documents for each subclaim;
4. the same DeBERTa-v3 NLI model and document-entailment scoring rule;
5. the same separate positive/negative NBC evidence pairs used by the published baseline.

The comparison changes only the **statistical evidence-accumulation / stopping mechanism**.

The Wang artifacts are pinned to source commit `3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea` for reproducibility.

## Published BSE baselines

Two BSE variants are retained deliberately:

- **`bse_official`** reproduces the behavior of the authors' released GitHub implementation, including its one-step look-ahead calculation.
- **`bse_equation8`** implements Equation 8 from the EMNLP paper literally by weighting next-step stop risk by the predictive probability of each entailment-feature bin.

This distinction matters because the released code and the written Equation 8 are not mathematically identical. We do not silently replace the published implementation with our preferred interpretation.

The BSE setup follows the published defaults used in the released `run.sh`:

- `P0 = 0.5`
- `C_M = 28`
- `C_FA = 96`
- `C_retrieve = 1`
- maximum retrieved documents per subclaim: `K = 10`
- document segmentation: 400 words with 100-word overlap
- NBC samples: the paper's protocol is s = 200 factual and s = 200 nonfactual
  examples. The released `NBC_positive.json` / `NBC_negative.json` do not
  necessarily contain exactly that many, so the reproduction gate records the
  paper protocol value and the number of pairs actually loaded, separately.

For each external document, DeBERTa scores its text spans and the document score is the **maximum entailment score across spans**, matching Wang et al. The code also preserves the two slightly different discretization formulas used in the authors' released `NBC_feature.py` and `main.py`.

### Deliberate fidelity choices

These look like defects and are not. `tests/test_baseline_fidelity.py` pins each
one against a literal transcription of the released code, so an "improvement"
fails the suite.

- **The stop/continue test runs before the first retrieval.** Released
  `main.py:226-233` initializes `P = P0`, computes `stop_cost` and `search_cost`,
  and only then enters the webpage loop. Because every subclaim starts at `P0`
  and the NBC histograms are global, this first decision is a dataset-wide
  constant: either every subclaim retrieves at least one document or none does.
  `--probe` exists to surface that, not to change it.
- **The one-step look-ahead averages the ten bins uniformly.** Released
  `main.py:127-140` does not weight bins by predictive probability. That is the
  documented divergence from the paper's Equation 8, which is why
  `bse_equation8` is a separate secondary mode.
- **Two different discretizers.** `NBC_feature.py:34` uses `int(score/10)` while
  `main.py:249` uses `int((score-0.1)/10)`. Both are preserved.
- **Documents are truncated to their first 4000 words.** This is Wang's own
  `utils.py:78`, not a choice made here.

The one deliberate deviation is that `split_text` removes the duplicate tail
span the released implementation can emit. Document scores are the maximum over
spans, so deduplication cannot change any score or decision; it only lowers the
reported NLI span count relative to the released implementation. Both detectors
are affected identically.

## Proposed method: uLSIF DDRE

The proposed method now uses a genuine **Direct Density Ratio Estimator**, not a classifier whose posterior is relabeled as a density ratio.

We use unconstrained Least-Squares Importance Fitting (**uLSIF**) to estimate directly

`r(s) = p(s | factual) / p(s | hallucinated)`

from the continuous DeBERTa entailment scores in the same Wang NBC factual/nonfactual evidence data used by BSE. Kernel width and regularization are selected by deterministic cross-validation using the uLSIF held-out objective.

Unlike BSE, DDRE does not:

- discretize the continuous entailment score into ten bins;
- separately estimate both class-conditional densities; or
- compute a Bayesian expected-risk look-ahead after every retrieved document.

For a sequence of retrieved documents, directly estimated evidence ratios are accumulated in log space. The method stops retrieving evidence when its factuality posterior leaves a validation-selected uncertainty interval. The lower and upper stopping thresholds are selected **only on validation passages**.

## What “improving factuality” means here

This experiment measures **factuality detection**, not generation correction. Therefore, an improvement means that the framework more accurately distinguishes factual from hallucinated generated content. It does not yet prove that the underlying LLM itself generates more factual text. A future end-to-end ECSS experiment can test whether rejecting, correcting, or regenerating detected hallucinations increases factual accuracy delivered to users.

## Primary evaluation

To stay comparable with Wang et al., the experiment reports the same trapezoidal PR-AUC convention used by their released code:

- nonfactual sentence-level AUC-PR;
- factual sentence-level AUC-PR;
- sentence-level accuracy;
- passage-level Pearson correlation;
- passage-level Spearman correlation;
- average retrieved documents per sentence;
- average retrieved documents per subclaim.

We additionally report:

- balanced PR-AUC;
- balanced accuracy;
- macro-F1;
- MCC;
- factual/nonfactual precision, recall, and F1;
- scikit-learn average precision as a secondary PR metric;
- total and average NLI span evaluations;
- wall-clock execution time.

The primary computational-overhead measures are **retrieved external documents** and **NLI span evaluations**, because they are hardware-independent.

### Important retrieval interpretation

The released Wang repository already contains the ordered web pages returned by their original retrieval process. Our experiment consumes those pages sequentially instead of issuing new live Bing searches. Therefore, “retrieved documents” means the number of external documents the decision policy chooses to consume from the released retrieval sequence. This makes BSE and DDRE reproducible on identical evidence. A later deployment experiment should additionally measure live search/network latency.

## Hypothesis test

The generated result file explicitly reports whether the primary hypothesis is supported on the held-out test data. The current predeclared rule requires DDRE to:

1. retrieve fewer documents than `bse_official`;
2. improve factual AUC-PR; and
3. not reduce balanced PR-AUC.

This prevents us from declaring success by selecting only favorable metrics after seeing the test results.

## Data preparation

The large Wang et al. artifacts are not duplicated in this repository. Download the authors' released data with:

```bash
python scripts/prepare_wang_data.py
```

This creates:

```text
data/wang/
  NBC/
  decomposed/
  selfcheckgpt/
  webpage/
```

`data/wang/` is ignored by Git. Please cite Wang et al. and follow the source repository's licensing/citation requirements when using those artifacts.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Core regression tests

Run the fast math/syntax checks before expensive NLI experiments:

```bash
python -m unittest discover -s tests -v
```

These cover the core BSE/uLSIF math (`tests/test_core_math.py`), the reproduction
gate's tolerance and verdict logic (`tests/test_reproduction_gate.py`), and
fidelity against literal transcriptions of Wang's released code
(`tests/test_baseline_fidelity.py`). The segmentation tests skip when torch is
not installed, since `src/utils.py` imports it at module scope.

A GitHub Actions workflow also syntax-checks the research pipeline and runs these core tests on changes to `main`.

## Gate 1: baseline reproduction

`bse_official` is only usable as a baseline if this repository reproduces the
released implementation. That is checked, not assumed.

### Cheap probe first

```bash
python scripts/reproduce_wang_baseline.py --probe
```

Scores only the released NBC evidence pairs (a few hundred NLI calls), builds
the Laplace-smoothed histograms, and reports whether `bse_official` retrieves
any documents at all under each published cost setting. Writes
`results/wang_probe.json` and exits non-zero if either configuration would
retrieve nothing. Run this before committing to the full pass.

### Full gate

```bash
python scripts/reproduce_wang_baseline.py
```

Reproduces both published cost settings on the complete released evidence set,
compares every metric against Table 1, and writes `results/wang_reproduction.json`
containing the published value, the reproduced value, the signed delta, the
relative delta, and a PASS/WARN/FAIL status per metric, plus full provenance
(repository commit, Wang data commit, Python and library versions, device, batch
size, model and tokenizer configuration, scoring/cache version). Exits non-zero
on FAIL.

Table 1's evidence count is the **average number of retrieved documents per
sentence**. Average documents per subclaim is reported as a diagnostic and is
never compared against Table 1.

### Protocol preconditions

Tolerances judge how close the reproduction landed. Preconditions judge whether
the run was a reproduction at all. Both output files carry a
`protocol_preconditions` block:

| Check | Requirement |
|---|---|
| `official_model` | The NLI model is exactly `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. |
| `wang_source_commit` | `data/wang/SOURCE.json` records the pinned commit `3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea`. |
| `truncation_equivalence` | `tokenizer.model_max_length` matches the configured `max_length`, so this repository's explicit truncation is equivalent to Wang's `truncation=True`. |

If any check fails, `overall` is FAIL, the run is stamped
`formal_gate1_run: false`, and the gate verdict is forced to FAIL however well
the metrics agree — agreement on the wrong model, the wrong source data, or
under a different truncation regime is not reproduction. A non-official
`--model-name` therefore remains usable for diagnostics but can never produce a
Gate 1 PASS.

A failed `truncation_equivalence` aborts both modes immediately, before any NLI
inference, since every entailment score would otherwise be computed over
different inputs than the released implementation used. An unreadable tokenizer
limit fails the same way as a known mismatch: equivalence is unproven either
way.

These checks detect a protocol mismatch. They never repair one. **Do not edit
`src/utils.py` to make a precondition pass** — investigate the mismatch at its
source.

### Predeclared tolerances

Frozen in `src/reproduction_gate.py` before any run and pinned by a test, so a
run can be shown to have been judged against them rather than the other way
round. They are not sampling-error bands: this reproduces one fixed experiment
on one fixed dataset, so they express how much environmental drift — dependency
versions, batching, device numerics — we are willing to call "reproduced".

| Metric | PASS | WARN | FAIL |
|---|---|---|---|
| Factual AUC-PR | abs delta <= 0.01 | <= 0.03 | > 0.03 |
| Nonfactual AUC-PR | abs delta <= 0.01 | <= 0.03 | > 0.03 |
| Accuracy | abs delta <= 0.01 | <= 0.03 | > 0.03 |
| Passage Pearson | abs delta <= 0.02 | <= 0.05 | > 0.05 |
| Passage Spearman | abs delta <= 0.02 | <= 0.05 | > 0.05 |
| Evidence Num (docs/sentence) | rel delta <= 5% | <= 10% | > 10% |

Zero retrieval is an unconditional FAIL regardless of every other metric: a
baseline that consumes no evidence returns `P = P0` for every sentence and
cannot serve as a comparator. The overall verdict is the worst status across all
metrics and both cost configurations. A WARN exits zero, but Gate 2 must not
begin until the discrepancy has a written explanation. A FAIL is a stop
condition — investigate before running or interpreting any DDRE comparison, and
do not change baseline control flow to make the gate pass.

## Smoke test

After preparing the Wang data:

```bash
python main.py --smoke-test
```

Smoke-test artifacts are debugging-only and ignored by Git. **Do not use them in the paper.** If the official large model is too slow for a pure code-path check, a smaller model can be supplied explicitly, but those numbers remain debugging-only.

## Full paper experiment

```bash
python main.py
```

The full experiment writes:

```text
results/latest_summary.json
results/latest_predictions.csv
results/latest_ddre_model.json
```

By default, a successful full run automatically commits and pushes **only these three result artifacts** to the current GitHub branch. Raw Wang data and the NLI cache are never added. To disable automatic result pushing:

```bash
python main.py --no-push-results
```

## Gate 1 scoring-path diagnostics

Gate 1 has been run and FAILED against the predeclared tolerances. These two
scripts diagnose *why*. They are instruments: they measure and report, and they
change nothing. No baseline behaviour, tolerance, published reference value,
split, or cost parameter is touched by either of them, and nothing in the normal
experiment pipeline imports them.

They also never open `results/wang_nli_cache.sqlite`. Both repository scoring
arms run with `use_cache=False, write_cache=False` against a scratch database in
a temporary directory that is deleted on exit, because a cached score would
measure the cache instead of the model. All output goes to `results/diagnostics/`.

### Step 0: what the formal run already recorded

```bash
python scripts/diagnose_gate1_provenance.py
```

Reads `results/wang_reproduction.json` and reports the environment, checkpoint,
and NBC-histogram facts it contains. Costs no NLI compute.

Five fields the diagnosis needs are not in the formal provenance block -- the
`tokenizers` version, the model's parameter dtype, its training mode, its
attention implementation, and the tokenizer's `is_fast` flag. Those print as
`NOT RECORDED` rather than being guessed, and a live probe fills them in from
the current environment. If the live environment does not match the one that
produced the report, the script says so: the live-only column then describes
*this* machine, not the Gate 1 run.

The live probe also settles the `token_type_ids` question. Wang's released
`utils.py:57` calls `model(inputs["input_ids"])` and never passes
`token_type_ids`; this repository calls `model(**inputs)` and passes whatever
the tokenizer emits. **A non-zero `type_vocab_size` alone proves nothing**, and
neither does the mere presence of the key. Three facts are needed together --
the emitted keys, whether `token_type_ids` appear, and their actual distinct
values -- so all three are reported, and the assessment only concludes that the
difference can matter when the values are non-zero *and* a token-type embedding
exists to consume them.

Useful flags: `--no-live` (report file only), `--no-load-model` (probe the
tokenizer without downloading weights), `--hash-weights` (hash large checkpoint
files too; small files are always hashed).

### Step 1: A/B/C scorer comparison

```bash
python scripts/diagnose_scorer_paths.py
```

Answers one question: inside a single environment, with one loaded model, does
this repository's `EntailmentScorer` make the same decisions as Wang's literal
released scorer?

| Arm | What it is |
| --- | --- |
| `A1` | Literal Wang path, transcribed from released `utils.py:40-68`: batch 1, `tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")`, `model(inputs["input_ids"])`, `softmax(logits[0]/5)`, hardcoded entailment index 0, `round(score * 100, 1)` |
| `A2` | This repository's `EntailmentScorer` at the production batch size of 8 |
| `A3` | This repository's `EntailmentScorer` at batch size 1 |

`A3` is what makes the result actionable. At batch size 1 padding is a no-op, so
**A1 vs A3 isolates the scorer's argument/inference path** (the passed
`attention_mask`/`token_type_ids` and the explicit `max_length`) while
**A3 vs A2 isolates batching and padding numerics**. Without it, a difference
between A1 and A2 would name two suspects at once.

The sample is all 398 released NBC pairs -- they are what the histograms are
built from, so a disagreement there propagates into every stopping decision --
plus a fixed sample of 75 retrieved documents expanded into their text spans.
About 600 pairs per arm, roughly 1,800 forward passes in total, against 85,194
document spans for a full reproduction.

Sampling is by SHA-256 rank over `(seed, document address)`, not by
`random.sample`, whose selection algorithm is a CPython implementation detail
and is not contracted to be stable across interpreter versions. The sample is
therefore byte-identical on any platform, and it is written to
`results/diagnostics/sample.json` so a run elsewhere can be checked against it.

Every comparison reports token `input_ids` exact-match rate and the first ten
mismatches; signed mean, absolute mean, p50/p95/p99 and max of the raw score
delta; the count above 1e-4; one-decimal disagreements; NBC and document bucket
disagreements; each arm's Laplace-smoothed NBC histograms; and per document, the
max-score disagreement, the bucket disagreement, and whether the argmax span
moved.

Materiality is fixed in `src/scoring_diagnostics.py` before any measurement, so
a result cannot be reinterpreted after the fact. A bucket disagreement is
MATERIAL however small the underlying score change was, because the BSE update
consumes only the bucket; conversely a raw delta that never crosses a bucket
edge cannot change a posterior, a stopping decision, or an evidence count.

The verdict follows predeclared rules:

- token ids differ anywhere -> tokenizer invocation semantics come first, and no
  score delta is interpretable until that is resolved;
- A1 differs materially from A3 -> the scorer's argument/inference path;
- A3 differs materially from A2 -> batching and padding numerics;
- all three equivalent -> the scorer is exonerated, and the next investigation is
  dependency/environment drift and checkpoint identity.

`--dry-run` builds and writes the sample and reports its size without loading
the model, so the sample can be checked before spending GPU time.

**These scripts diagnose only. Nothing they find is fixed by them.**

## Persistent NLI cache

NLI scoring is expensive. Deterministic claim/span entailment scores are cached locally in:

```text
results/wang_nli_cache.sqlite
```

The cache is ignored by Git and reused across BSE/DDRE runs. Delete and rebuild it only when intentionally changing the NLI scoring protocol:

```bash
python main.py --rebuild-cache
```

For a hardware-dependent uncached timing run:

```bash
python main.py --live-inference --no-push-results
```

## Model fidelity and development speed

The default model matches the Wang et al. released implementation:

```text
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

For debugging only, a smaller compatible model may be passed with `--model-name`, but **paper results should use the official large model unless the experimental protocol is intentionally changed and reported**.

## Research status

This is active research code. Numerical claims for the paper abstract should come only from a successful Wang-aligned full experiment and should be reported even if the DDRE hypothesis is not supported.
