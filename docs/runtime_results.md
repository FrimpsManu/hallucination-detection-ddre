# Running time on HELM — results

**Status: measurement, run once.**
- Protocol: [`docs/runtime_protocol.md`](runtime_protocol.md), merged before
  any timing.
- The run was made from clean `main` (`cfacc46`) on 2026-10-26, 05:53–06:09
  UTC.
- Every number below comes from the frozen result (§5).
- All checks passed (V1 pins, V2 live scores equal to the HELM feature cache
  bit for bit, V3 every trajectory equal to the frozen results).

**Outcome:**
- **At equal quality,** DDRE-1D is **70% faster** than BSE at 28/96 and
  **59% faster** at 14/24, per sentence on this machine. This matches the
  NLI-call saving.
- **At Wang's default budget,** DDRE-1D is 9% and 14% slower, because it
  reads more documents there.
- **NLI is essentially all the time.** The estimator adds at most 0.32 ms
  per sentence, against about 60 ms per NLI call.

---

## 1. What ran

- **Sentences:** 200 HELM sentences drawn with seed 42 (140 articles).
- **Cells:** eight: both methods at the default budget (K 10, C_retrieve 1)
  and at the equal-quality budgets fixed by the HELM result.
- **Scoring:** live DeBERTa scoring of each document as it is read, float16
  on Apple MPS, batch size 1, with no feature cache.
- **Repeats:** three passes with methods interleaved in seeded random order
  (314, 325 and 334 s). The median per sentence and cell is used.
- **Machine:** Apple M5 Pro on mains power; macOS 27.0.1, Python 3.11.16,
  torch 2.14.0, transformers 5.17.0. Model load took 3.2 s.
- **Not timed:** retrieval. Both methods read the same saved BM25 documents.

## 2. Seconds per sentence

| Cell | Costs | Method | Budget (K, C_retrieve) | Seconds per sentence | NLI calls | ms per NLI call |
|---|---|---|---|---|---|---|
| D1 | 28/96 | BSE | 10, 1 | 0.323 | 5.35 | 60.3 |
| D2 | 28/96 | DDRE-1D | 10, 1 | 0.351 | 5.77 | 60.8 |
| D3 | 14/24 | BSE | 10, 1 | 0.204 | 3.38 | 60.3 |
| D4 | 14/24 | DDRE-1D | 10, 1 | 0.233 | 3.86 | 60.3 |
| E1 | 28/96 | BSE | 5, 0.125 | 0.228 | 3.79 | 60.3 |
| E2 | 28/96 | DDRE-1D | 2, 0.25 | 0.068 | 1.14 | 59.4 |
| E3 | 14/24 | BSE | 3, 1 | 0.145 | 2.42 | 60.0 |
| E4 | 14/24 | DDRE-1D | 1, 0.125 | 0.060 | 1.00 | 59.5 |

On this subset the cells read within 0.12 NLI calls per sentence of their
full-HELM values (5.46, 5.87, 3.42, 3.91, 3.85, 1.16, 2.43, 1.00).

## 3. Contrasts

BSE minus DDRE-1D, seconds per sentence. 95% intervals come from a cluster
bootstrap over the 140 articles (10,000 resamples, seed 42).

| Contrast | Saved per sentence | 95% CI | Saving | Wording (protocol §9) |
|---|---|---|---|---|
| **T1** 28/96, equal quality (E1 − E2) | 0.160 s | [0.146, 0.174] | **70.3%** | DDRE-1D is faster than BSE |
| **T1** 14/24, equal quality (E3 − E4) | 0.085 s | [0.078, 0.093] | **58.9%** | DDRE-1D is faster than BSE |
| **T2** 28/96, default (D1 − D2) | −0.028 s | [−0.042, −0.014] | −8.7% | DDRE-1D is slower than BSE |
| **T2** 14/24, default (D3 − D4) | −0.029 s | [−0.045, −0.013] | −14.2% | DDRE-1D is slower than BSE |

## 4. Where the time goes (T3) and the offline fit (T4)

**Online, per sentence:**
- NLI is at least 99.88% of every cell's time.
- The estimator takes 0.06–0.32 ms per sentence for DDRE-1D (a kernel sum
  over 100 centres), and at most 0.03 ms for BSE's histogram lookup.
- The stopping decision and posterior update take under 0.04 ms.

**Offline, once** (median of five, on Wang's 398 NBC pairs):

| | BSE | DDRE-1D |
|---|---|---|
| Fit estimator and look-ahead support | 0.1 ms | 37 ms (uLSIF with cross-validation over the σ grid) |
| Backward-induction policy, default budget (K 10) | 0.16 s | 0.29 s |
| Backward-induction policy, equal-quality budget | 0.012–0.037 s | 0.002–0.010 s |

Both refits equal the stored estimators exactly.

**Reading:** the estimator's cost is negligible next to NLI scoring, online
and offline. Running time therefore follows the number of NLI calls, and the
70% and 59% NLI-call savings at equal quality carry over to seconds.

## 5. Artifacts

| File | SHA-256 |
|---|---|
| `~/ddre-artifacts/runtime/runtime_result.json` | `6996c03f2bce92c8c5c033a7b2fb11732b82a8b54397e67295a87b2bcd0fe808` |
| `~/ddre-artifacts/runtime/runtime_timings.csv` | `3119c77af06dd538414372a1bbf4d5c4b22e7d861e30cd79651714cccaf0b6e4` |

Both are read-only. The timings file holds one row per pass, sentence and
cell.

## 6. Limitations

- **One machine.** Seconds are for an Apple M5 Pro with MPS. Other hardware
  will differ in absolute terms. The ratio should not, because NLI dominates
  every cell.
- **No retrieval time.** In a live system each document read also costs a
  retrieval call. That adds to both methods per document read, so it would
  widen the absolute saving at equal quality.
- **Background activity.** During the run the machine also did light work:
  editing text files, downloading about 18 MB, and a few loader runs of a
  few seconds each. The interleaved order spreads any effect evenly over the
  methods.
