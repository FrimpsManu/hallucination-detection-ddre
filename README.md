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
plus a fixed sample of 75 retrieved documents expanded into their text spans:
589 pairs per arm.

Two costs are reported separately, because they are not the same number. Every
arm scores every pair, so the run performs `3 x 589 = 1,767` **pair
evaluations**. It does not perform 1,767 **model forward calls**: A1 and A3 run
at batch size 1, but A2 batches, so for a batch size of `B` the total is
`589 + 589 + ceil(589 / B)`, which is **about 1,252 forward calls** at `B = 8`.
A full reproduction scores 85,194 document spans. The script reports the
estimate before the run and the observed call counts after it.

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
MATERIAL however small the underlying score change was, because a BSE update
consumes only a bucket; conversely a raw delta that never crosses a bucket edge
cannot change a posterior, a stopping decision, or an evidence count.

**Which bucket is the whole point.** BSE consumes exactly two:

- the NBC bucket of each evidence pair, which builds the histograms
  (released `NBC_feature.py:34`); and
- the bucket of a document's **maximum** span score (released
  `main.py:237-250`) -- the runtime loop scores every span, keeps the largest
  entailment score, and discretizes only that maximum before the Bayesian
  update.

Individual span buckets are **not** consumed and do not decide materiality. A
non-maximal span can cross a bucket edge and leave the document score, the
posterior, the stopping decision, and the evidence count all untouched.
Span-bucket disagreement is still reported, clearly labelled as a diagnostic
statistic; the verdict is driven by NBC buckets and document-max buckets only.
The same applies to a moved argmax span, which is observable downstream only if
it also changes the maximum score itself.

The verdict follows predeclared rules:

- token ids differ anywhere -> tokenizer invocation semantics come first, and no
  score delta is interpretable until that is resolved;
- A1 differs materially from A3 -> the scorer's argument/inference path;
- A3 differs materially from A2 -> batching and padding numerics;
- all three equivalent -> the scorer is exonerated, and the next investigation is
  dependency/environment drift and checkpoint identity.

`--dry-run` builds and writes the sample and reports its size without loading
the model, so the sample can be checked before spending GPU time.

### Step 2: which forward-path argument explains the Step 1 divergence

```bash
python scripts/diagnose_forward_path.py
```

Step 1 narrowed the question to a single observation. The literal Wang scorer
and this repository's scorer tokenized identically -- 589/589 `input_ids`
matched -- yet disagreed on exactly one BSE decision:

| | raw | rounded | NBC bucket |
| --- | --- | --- | --- |
| literal Wang (A1) | 19.9462890625 | 19.9 | 1 |
| repository (A3) | 19.953125 | 20.0 | 2 |

which moved the positive Laplace-smoothed histogram from
`[1, 78, 19, 13, 6, 5, 28, 57, 1, 1]` to `[1, 77, 20, 13, 6, 5, 28, 57, 1, 1]`.

Tokenization is excluded, and so are `token_type_ids` -- they are emitted with
values 0 and 1, but `type_vocab_size` is 0, so no token-type embedding exists to
consume them. Two candidate differences remain between Wang's released
`model(inputs["input_ids"])` and our `model(**inputs)`:
`torch.inference_mode()`, and passing `attention_mask`.

Step 2 is a controlled factorial over exactly those two factors, on all 398
released NBC pairs:

| Arm | Forward call | Toggles |
| --- | --- | --- |
| `C0` | `model(input_ids)` | reference (literal Wang) |
| `C1` | `with torch.inference_mode(): model(input_ids)` | inference_mode |
| `C2` | `model(input_ids, attention_mask=...)` | attention_mask |
| `C3` | `with torch.inference_mode(): model(input_ids, attention_mask=...)` | both |
| `C4` | `... + token_type_ids=...` | **confirmation only** |
| `C0R` | `model(input_ids)`, run last | **determinism control** |

`C4` is confirmation only: `type_vocab_size` is 0, so the argument is expected
to be inert, and the arm exists to demonstrate that rather than to test a live
hypothesis.

**`C0R` is what makes any of this attributable.** It re-runs `C0` unchanged at
the end. If `C0` and `C0R` are not bit-identical, the forward pass is
nondeterministic on that device and every factor attribution in the report is
unfounded -- so the report says exactly that and withholds a verdict.

Four controls keep the forward call the only variable: the tokenizer and model
are loaded once; every pair is tokenized **once** and the identical tensors are
reused by every arm; the score extraction is byte-identical across arms; and
the batch size is exactly 1 everywhere, so padding is never involved.

Comparisons are computed pairwise, and each is labelled with the factor that
genuinely differs between its two arms rather than with the right-hand arm's
description. This matters: `C1` vs `C3` toggles `attention_mask` alone (both
already run under inference_mode) and `C2` vs `C3` toggles
`torch.inference_mode()` alone (both already pass a mask). Those two are what
separate a dominant factor from an interaction.

#### Reporting discipline

A single word like "MATERIAL" conflates two different findings, so this
diagnostic never emits one. Every comparison reports three separate fields:

- **`numerical_difference`** -- YES when the raw scores are not bit-identical.
  A statement about floating point and nothing else.
- **`bse_decision_impact`** -- YES only when at least one NBC bucket changes.
  The Bayesian update consumes the bucket, so this is the only field that
  licenses a claim about baseline behaviour. **A raw floating-point score change
  alone is NOT evidence that BSE behaviour changed**, and a comparison can
  legitimately be `numerical_difference: YES, bse_decision_impact: NO`.
- **`causal_candidate`** -- which factor the comparison toggles, and whether it
  showed an effect.

The verdict adds `explains_reference_observation`, because the goal is not to
find *a* difference but to find the one that produced the Step 1 result. The
report prints the raw, rounded and bucketed value of positive pair 169 for every
arm, beside the two recorded Step 1 values, so that comparison is direct.

The negligible bound is `1e-4` on a 0-100 score. It is anchored to the effect
being explained: the Step 1 divergence was `0.0068359375`, about 68x the bound.
A threshold used for attribution has to sit well under the effect it attributes,
or it would classify the very difference under investigation as noise.

#### Interpretation

| Observation | Conclusion |
| --- | --- |
| C0 ~ C1 and C2 ~ C3, C0 vs C2 substantial | `attention_mask` is the primary candidate |
| C0 ~ C2 and C1 ~ C3, C0 vs C1 substantial | `torch.inference_mode()` is the primary candidate |
| C0 ~ C1 and C0 ~ C2, but C0 vs C3 substantial | interaction between the two |
| both factors move the score independently | both contribute, neither dominant |
| nothing substantial anywhere | forward-path isolation inconclusive; another difference remains |

Cost: 398 pairs x 6 arms = 2,388 forward calls at batch size 1. `--dry-run`
loads and counts the pairs without loading the model.

### Running the formal follow-up on Colab

The completed Gate 1 artifacts live on Google Drive. Neither script hardcodes a
Drive path -- both take the location as an argument -- but writing their output
back to Drive is what makes the results survive a Colab disconnect.

Mount Drive, then run Step 0 against the completed report:

```bash
python scripts/diagnose_gate1_provenance.py \
  --report /content/drive/MyDrive/ddre-gate1/wang_reproduction.json \
  --output /content/drive/MyDrive/ddre-gate1/diagnostics/step0_provenance.json
```

and Step 1 with its output directory on Drive:

```bash
python scripts/diagnose_scorer_paths.py \
  --output-dir /content/drive/MyDrive/ddre-gate1/diagnostics
```

Step 1 writes both `sample.json` and `step1_scorer_ab.json` into that directory.
Run it from a checkout whose `--data-root` points at the prepared Wang data;
pass `--data-root` explicitly if that is also on Drive.

Step 2 takes an explicit output file:

```bash
python scripts/diagnose_forward_path.py \
  --output /content/drive/MyDrive/ddre-gate1/diagnostics/step2_forward_path.json
```

The formal cache at `/content/drive/MyDrive/ddre-gate1/wang_nli_cache.sqlite` is
**never opened or modified** by any of these scripts, wherever the output goes.
Step 2 does not construct an `EntailmentScorer` at all -- it calls the model
directly -- so it has no cache code path whatsoever. In Step 1 there is no flag
that would make it read the cache: both repository arms are hardcoded to
`use_cache=False, write_cache=False`, and their scratch database is created in a
local `TemporaryDirectory` that is deleted on exit. Reading that cache would
measure the cache instead of the model, and writing to it would contaminate the
formal Gate 1 artifact.

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
