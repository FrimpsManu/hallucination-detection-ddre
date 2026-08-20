# Hallucination Detection with Density-Ratio Estimation

Research code for an in-progress study comparing sequential Bayesian evidence accumulation with a lightweight classifier-based density-ratio estimator for sentence-level hallucination detection.

## Research objective

The objective of this work is to develop a retrieval-aware hallucination-detection framework that **reduces computational cost while improving factuality detection performance** relative to sequential Bayesian evidence accumulation. The broader motivation is to support more reliable, lower-latency LLM-based enterprise customer-service systems.

## Research question

**Can a retrieval-aware density-ratio estimation framework reduce the computational cost of hallucination detection while improving factuality detection performance compared with sequential Bayesian evidence accumulation?**

## Hypothesis

We hypothesize that classifier-based density-ratio estimation can replace repeated sequential probability updates with a learned evidence-consistency decision mechanism, thereby:

1. reducing computational cost, measured through inference latency, NLI evidence evaluations, and evidence-processing steps; and
2. improving factuality detection performance, measured through hallucination precision, recall, F1, macro-F1, ROC-AUC, and PR-AUC.

### Interpretation of "factual accuracy"

The current experiment evaluates **factuality detection**: whether a generated sentence is correctly classified as factual or hallucinated. Therefore, the paper should interpret improved "factual accuracy" at this stage as improved accuracy/reliability of factuality validation, not as direct evidence that the underlying LLM generates more factual responses. Demonstrating that the framework directly increases the factual accuracy of generated responses would require an additional end-to-end experiment in which detected hallucinations are rejected, regenerated, corrected, or otherwise prevented from reaching the user.

## Dataset and labels

The current experiments use the SelfCheckGPT dataset. Each generated sentence is treated as a sentence-level observation with its corresponding Wikipedia biography as evidence.

- `1`: factual (`accurate` in the source annotations)
- `0`: hallucinated (`minor inaccurate` or `major inaccurate`)

Because multiple sentences may share the same biography/evidence, the experiment splits data by `wiki_bio_test_idx`, not by sentence position. This prevents sentences associated with the same biography from appearing in multiple data splits.

## Experimental protocol

The paper experiment uses a reproducible biography-level split:

- 70% train
- 15% validation
- 15% test
- random seed: 42

The training set is used to estimate the Bayesian score distributions and fit the density-ratio model. The validation set is used to select the DDRE decision threshold. The test set remains untouched until final evaluation.

## Methods

### Sequential Bayesian baseline

The baseline evaluates evidence segments sequentially, updates a posterior probability after each NLI observation, and uses expected Bayes risk to determine whether evaluating another evidence segment is worth the additional computation cost.

### Classifier-based density-ratio estimator

The proposed implementation extracts aggregate NLI evidence statistics and fits standardized logistic regression. Posterior odds are converted to the class-conditional density ratio

`r(x) = p(x | factual) / p(x | hallucinated)`

using the empirical class-prior correction

`r(x) = [P(factual | x) / P(hallucinated | x)] * [P(hallucinated) / P(factual)]`.

This repository therefore describes the current method precisely as **classifier-based density-ratio estimation**. A canonical direct estimator such as least-squares density-ratio estimation can be added as a separate method in later experiments rather than conflated with the present implementation.

## Faster, resumable NLI feature computation

NLI inference is the dominant cost in this experiment. The current pipeline therefore:

1. batches train/validation NLI inference instead of running one model call at a time;
2. stores deterministic claim/evidence entailment scores in `results/nli_cache.sqlite`;
3. commits the cache after every completed batch, so an interrupted run resumes rather than starting from zero;
4. reuses the same cached training features for both the Bayesian and DDRE methods;
5. keeps final test inference **uncached**, so reported latency still measures real model execution;
6. reports progress with `tqdm` progress bars.

The SQLite cache is local and ignored by Git because it can become large.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Running the experiment

### Quick smoke test

Use this first to verify that the full pipeline works:

```bash
python main.py --smoke-test
```

It uses small class-stratified subsets of the already leakage-safe train/validation/test splits. Smoke-test outputs are deliberately written to separate files:

- `results/smoke_comparison_results.json`
- `results/smoke_nbc_features.json`

**Do not use smoke-test numbers in the paper.**

### Full publication experiment

```bash
python main.py
```

The full run writes:

- `results/comparison_results.json`
- `results/nbc_features.json`

### Batch-size tuning

The default batch size is 16 on CUDA and 8 on CPU. If memory permits, a larger batch can improve throughput:

```bash
python main.py --batch-size 16
```

or on a GPU with sufficient memory:

```bash
python main.py --batch-size 32
```

If an out-of-memory error occurs, reduce the batch size.

### Resume behavior

If a run is interrupted, rerun the same command. Already-computed train/validation NLI scores are loaded from the persistent SQLite cache and only missing scores are inferred.

To intentionally discard the cache and recompute all train/validation NLI features:

```bash
python main.py --rebuild-cache
```

## Evaluation

The final test-set comparison reports:

- hallucination precision
- hallucination recall
- hallucination F1
- macro F1
- ROC-AUC
- PR-AUC
- 95% bootstrap confidence interval for hallucination F1
- mean, p50, and p95 inference latency
- total and average NLI calls per sample

The positive class for precision, recall, and F1 is **hallucinated (`0`)**.

These metrics directly test the two parts of the hypothesis: **detection quality** and **computational cost**.

## Current status

This is active research code. Numerical claims for the paper or conference abstract should be taken only from results regenerated with the corrected experiment pipeline on `main`.
