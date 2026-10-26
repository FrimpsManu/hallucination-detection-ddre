# Running time on HELM — protocol

**Status: frozen when merged; every later change is a dated amendment at
the end.**
- Written on 2026-10-26, after the HELM confirmatory test
  (`docs/helm_results_summary.md`) and before any timing was run.
- Research plan v2 §4.4 and Phase 6.

**What this is.** A measurement, not a hypothesis test. The HELM test counted
cost in NLI calls; this protocol turns the same comparison into seconds on
one machine. Nothing here changes the confirmatory decision, and no setting
is chosen after seeing a timing.

---

## 1. Questions

- **T1.** At equal quality, how many seconds per sentence does DDRE-1D save
  over BSE?
- **T2.** At Wang's default budget, how many seconds per sentence does each
  method take? DDRE-1D reads more documents there, so it is expected to be
  slower.
- **T3.** How much of the time is NLI, and how much is the estimator and the
  stopping rule?
- **T4.** How long does each method's offline fit take?

## 2. Inputs

All read-only, checked against their SHA-256 pins before anything runs:

| Input | SHA-256 |
|---|---|
| `estimators_v2_float16.json` | `6a007e37…6fadec` |
| `helm_evidence_top10.jsonl` | `c86f405a…b9c0` |
| `helm_features_float16.sqlite` (verification only) | `a02ef225…a54d` |
| `helm_confirmatory_result_float16.json` | `10bbd730…8151` |
| `helm_confirmatory_predictions_float16.csv` | `994d85f2…a7e8e8` |

Full digests are in the script. The NLI model is Wang's checkpoint
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, revision
`b3546ea6…`), in float16 on Apple MPS, batch size 1, exactly as HELM was
scored.

## 3. Sentences

- **Subset:** 200 of the 3,736 HELM sentences.
- **Draw:** `numpy.random.default_rng(42).choice(3736, 200, replace=False)`
  over the loader order, then sorted.
- The subset is the same for every method and setting.

## 4. Settings

Eight cells. The equal-quality budgets are the ones the HELM result already
fixes: each method's cheapest budget reaching BSE's balanced AP at the
default budget (the `e_cost` endpoint).

| Cell | Costs (C_M, C_FA) | Method | Max documents K | C_retrieve | Why |
|---|---|---|---|---|---|
| D1 | (28, 96) | BSE | 10 | 1 | default budget |
| D2 | (28, 96) | DDRE-1D | 10 | 1 | default budget |
| D3 | (14, 24) | BSE | 10 | 1 | default budget |
| D4 | (14, 24) | DDRE-1D | 10 | 1 | default budget |
| E1 | (28, 96) | BSE | 5 | 0.125 | equal quality |
| E2 | (28, 96) | DDRE-1D | 2 | 0.25 | equal quality |
| E3 | (14, 24) | BSE | 3 | 1 | equal quality |
| E4 | (14, 24) | DDRE-1D | 1 | 0.125 | equal quality |

On all 3,736 sentences these cells read 5.46, 5.87, 3.42, 3.91, 3.85, 1.16,
2.43 and 1.00 NLI calls per sentence.

## 5. What is timed

For each sentence and cell, the detector runs live, document by document,
exactly as in the HELM test (one subclaim, top-10 saved BM25 documents, the
cell's backward-induction policy):

1. **Decide:** the policy says whether to read the next document.
2. **NLI:** split the document into spans and score each span against the
   sentence with the model. Nothing is read from a feature cache: each
   sentence and cell gets a new, empty cache.
3. **Estimator:** the log likelihood ratio of the document's entailment.
4. **Update:** the clipped log-odds update.

**Clock:** `time.perf_counter`, with `torch.mps.synchronize()` before each
NLI reading.

**Recorded per sentence and cell:**
- total time;
- time in NLI, in the estimator, and in decide plus update;
- documents read and NLI calls.

**Not timed:**
- **Retrieval.** Both methods read the same saved BM25 documents, so
  retrieval costs the same per document for both. It is left out, and the
  report says so.
- **Model load,** reported once on its own.

## 6. Order and repeats

- **Warm-up:** 20 NLI calls on Wang's NBC training pairs (not HELM) after the
  model loads, not recorded.
- **Repeats:** three passes over the subset. Each pass visits the sentences
  in a fresh seeded random order (seeds 1, 2, 3), and for each sentence runs
  the eight cells in a seeded random order. Methods are therefore
  interleaved, so drift in machine speed affects all of them alike.
- **Per sentence and cell,** the median over the three passes is used.

**Machine:** Apple M5 Pro, on mains power, other applications closed. The
report records macOS, Python, torch and transformers versions.

## 7. Checks before anything is reported

All must pass; if one fails, the run stops and is reported as failed. Nothing
is retuned.

- **V1. Pins:** every input in §2 matches.
- **V2. Scores:** every live entailment score equals the HELM feature cache
  bit for bit (same model, dtype, device and batch size).
- **V3. Trajectories:**
  - for D1–D4, documents read, NLI calls and final P(factual) equal the frozen
    predictions file on every subset sentence;
  - for E1–E4, they equal a replay from the feature cache through the HELM
    test's own code, and that replay's means over all 3,736 sentences equal
    the frozen curve values in the result file.

## 8. Offline fit (T4)

From the cached features of Wang's 398 NBC pairs, five times each, median
reported:
- BSE: building the histogram;
- DDRE-1D: uLSIF with its cross-validation over the σ grid;
- each method's backward-induction policy for the eight cells.

The refit must equal the stored estimator (BSE exactly; DDRE-1D log ratios
within 1e-9).

## 9. Report

**Per cell:**
- mean seconds per sentence, and its split into NLI, estimator, and
  decide plus update;
- documents and NLI calls per sentence;
- milliseconds per NLI call.

**Contrasts:**
- **T1:** BSE − DDRE-1D seconds per sentence, E1 vs E2 and E3 vs E4, with
  the saving as a percentage;
- **T2:** D2 − D1 and D4 − D3.

Each contrast has a 95% percentile interval from a cluster bootstrap over
the subset's articles: 10,000 resamples, seed 42.

**Wording:**
- A contrast whose interval excludes 0 is called faster or slower. Otherwise
  it is "no measurable difference".
- Timings are for this machine only. Seconds are not compared with any other
  paper's.

## 10. Outputs

Written once and made read-only under `~/ddre-artifacts/runtime/`:
- `runtime_result.json`: summaries, checks, versions, pins and code commit;
- `runtime_timings.csv`: one row per pass, sentence and cell.

A results summary follows in `docs/runtime_results.md`.

## 11. Order of work

1. This protocol, merged.
2. The script with tests and a `--dry-run` (checks V1, the subset and the
   cells; no model), merged.
3. One run from clean `main`.
4. Results summary and paper text.

---

## Amendments

None.
