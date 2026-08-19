# Hallucination Detection with Density-Ratio Estimation

Research code for an in-progress study comparing sequential Bayesian evidence accumulation with a lightweight classifier-based density-ratio estimator for sentence-level hallucination detection.

## Research question

Can density-ratio estimation provide a computationally efficient alternative to sequential Bayesian evidence accumulation for hallucination detection while preserving or improving detection performance?

## Dataset and labels

The current experiments use the SelfCheckGPT dataset. Each generated sentence is treated as a sentence-level observation with its corresponding Wikipedia biography as evidence.

- `1`: factual (`accurate` in the source annotations)
- `0`: hallucinated (`minor inaccurate` or `major inaccurate`)

Because multiple sentences may share the same biography/evidence, the experiment splits data by `wiki_bio_test_idx`, not by sentence position. This prevents sentences associated with the same biography from appearing in multiple data splits.

## Experimental protocol

The current paper experiment uses a reproducible biography-level split:

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

## Reproducibility

Run the experiment with:

```bash
python main.py
```

The script writes newly generated artifacts to:

- `results/comparison_results.json`
- `results/nbc_features.json`

Old pre-fix result files were removed because they were produced by an earlier implementation containing methodological and baseline errors and should not be used in the paper.

## Current status

This is active research code. Numerical claims for the paper or conference abstract should be taken only from results regenerated with the corrected experiment pipeline on `main`.
