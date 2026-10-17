# D-03 outcome and decision record

**Written 2026-10-17, retrospectively.** At the time, no document recorded the
D-03 empirical outcome or the decision to select the 1-D DDRE configuration
from a 640-candidate validation sweep. This record is reconstructed from
frozen artifacts and commit messages, all cited below. Where the repository
holds no written reason, this record says so rather than supplying one.

Superseded wording elsewhere:
- `docs/d03_density_ratio_diagnostic_protocol.md` (end) and
  `docs/ddre_scientific_audit.md` (D-03 entry) still say the empirical run is
  required. It was run on 2026-09-30, as recorded below.

---

## 1. What D-03 was

The DDRE methodology audit (PR #7, 2026-09-04,
`docs/ddre_scientific_audit.md`) raised finding D-03. The 1-D uLSIF ratio fitted
on Wang's NBC pairs might be **unsupported or unstable** where DDRE actually
uses it. Large ratios are fine where the two classes genuinely separate; the
question was whether the evidence DDRE consumes is supported by the training
data and stable across the uLSIF hyperparameter surface.

The diagnostic was preregistered and implemented in PR #13
(`cf79b44`, 2026-09-05; `docs/d03_density_ratio_diagnostic_protocol.md`).
It froze three escalation triggers and no magnitude criterion.

- **Population:** every candidate validation document up to 10 per subclaim:
  48 passages, 383 sentences, 603 subclaims, 5,969 document occurrences.
- **Held-out data:** never scored.

## 2. Outcome (run 2026-09-30, main `2661b04`)

Artifact: `~/ddre-artifacts/gate1-2026-09-20/d03_density_ratio_diagnostic_main_2661b04.json`,
sha256 `f2468e5bb1265e107f4060d36782f6f21ba9f5f55141dde4814d5ffd11aa7f6d`.

- **Completeness:** 5,969 / 5,969 occurrences scored. Score compatibility with
  the formal cache was established on the 398-pair probe.
- **Checkpoint identity:** not established, because the Gate 1 v2 report did
  not record the Hub revision.
- **Selected fit** (the uLSIF CV choice): σ = 0.2269, λ = 1.0, 100 centres, 66
  with positive α. Raw ratios on the training data lie in [0.42, 7.50].

**Triggers:**

| Trigger | Fired | Count |
|---|---|---|
| 1. Selected fit hits the ratio clip [1e-6, 1e6] | no | 0 |
| 2. A first document that stops under the selected fit has a ratio direction (> 1 vs < 1) that flips somewhere on the σ/λ surface | **yes** | **310** |
| 3. Zero class support and a non-unanimous stop decision | no | 0 |

**Stop decisions across the surface:**
- 0 of 603 first documents have the same stop decision on all 20 (σ, λ) pairs.
- There were 85 high-to-low direction flips.

**Meaning, as preregistered:** `additional_sensitivity_required = true`. A
separate sensitivity study was needed. No cap, calibration or method change was
selected by D-03.

**D-13 (distribution shift):** measured in the same run, described and not
corrected. Wang's NBC histogram is fitted on **whole premises**; test documents
are scored as the **maximum over 400-word spans**. The two score distributions
differ:

| Entailment score | NBC pairs (n = 398) | Validation documents (n = 5,969) |
|---|---|---|
| median | 19.1 | 35.1 |
| 10–20 | 217 (55%) | 1,390 (23%) |
| 30–40 | 26 (7%) | **2,673 (45%)** |
| 70–80 | 58 (15%) | 380 (6%) |

Almost half of the test documents fall in a bin that only 26 training pairs
inform. BSE and DDRE both inherit this shift.

## 3. Sensitivity study (PR #19, run 2026-09-30, `59f4f53`)

Artifact: `d03_hyperparameter_outcome_sensitivity_main_59f4f53.json`, sha256
`fc5604854b18f3e3e712fe7ef01859b58dac569e99696041713abb11daa6dcc0`.

**Design:**
- On validation only, the DDRE outcome was replayed for each of the 20 (σ, λ)
  pairs that the uLSIF CV had scored.
- Each pair was crossed with 32 stopping bands (lower 0.05–0.20, upper
  0.60–0.95): **640 candidates**.
- A candidate is **feasible** only if it preserves BSE-official nonfactual,
  factual and balanced PR-AUC, each within 0.005.

**Result** (CV objective: lower is better):

| σ | λ | CV objective | Feasible bands |
|---|---|---|---|
| 0.2269 | 1.0 | **−2.216** (CV choice) | **0** |
| 0.1135 | 1.0 | −2.184 | 16 |
| 0.0567 | 1.0 | −1.930 | 0 |
| 0.1135 | 0.1 | 0.578 | 26 |
| 0.0567 | 0.1 | 1.146 | 32 |
| 0.1135 | 0.01 | 12.818 | 28 |
| the other 14 pairs | | | 0 |

**The CV-selected configuration had no stopping band that preserved BSE's
quality on validation.** 102 of the 640 candidates were feasible, all from
four (σ, λ) pairs.

## 4. The decision (PR #20, `83a7afa`, 2026-09-30)

**What was done:**
- The predeclared rule of `src/threshold_selection.py` was applied to all 640
  candidates.
- Among feasible candidates, the rule picks the fewest documents per sentence,
  then the highest balanced PR-AUC, and so on.
- It froze σ = 0.11346, λ = 1.0, band [0.20, 0.80].

Artifact: `ddre_validation_freeze.json`, sha256
`744abf4afc05f586ed138dbbb5a503ac8014af40ab14c2a6d13b5f512ecdf9fa`.

**Validation figures for the selected candidate:**
- 24.6% fewer documents than BSE;
- balanced PR-AUC +0.0056.
- Its σ is half the CV choice, and its CV objective (−2.184) is the second best
  of the 20.

**Why it was done:**
- The commit message gives only "the held-out evaluation needs exactly one
  configuration to run". No written decision exists.
- The evident reason, recorded here after the fact, is §3: the preregistered
  CV-only configuration could not preserve BSE quality on validation at any
  band. Running it would have been a test of a configuration already known to
  fail.

**What the protocol says about it.** `docs/confirmatory_statistical_protocol.md`
requires the CV-only selection for a confirmatory run. A post-D-03 validation
sweep is listed by name as a disqualifier (`post_d03_validation_sweep`). The
held-out run was therefore **`NOT_CONFIRMATORY` by construction**, and both its
code and its artifact record that.

**Tuning budget:** DDRE got 640 validation configurations; BSE got none. The
held-out artifact carries this disclosure verbatim.

## 5. Consequences

**Held-out run** (PR #21, `fae3eee`, 2026-09-30):
- 20.7% fewer documents and 22.0% fewer NLI calls than BSE.
- Balanced PR-AUC +0.008 [−0.021, 0.038].
- Status `NOT_CONFIRMATORY`.
- Even setting the selection aside, the lower interval bound is below the
  −0.005 margin, so preservation of quality was not shown.

**Later diagnostics on the frozen ratio:**

| Finding | Value | Source |
|---|---|---|
| Unnormalised | E_h[r] = 0.736 (a bias of −0.31 nats per document toward "hallucinated") | `~/ddre-artifacts/diagnostics/ulsif_normalisation.json`, PR #38 |
| Not monotone | log r(95) = −0.11, log r(100) = −1.09 | |
| Predicts "factual" far less often than the truth | 264 sentences predicted factual vs 417 gold; factual recall 0.48 vs BSE's 0.60 | ablation summary |

Part of the document saving plausibly comes from this bias rather than from
better evidence. This is a plausible account, not a tested cause.

**Status in plan v2:** prior exploratory work. In the paper it is reported with
the 640-configuration budget, the unnormalised ratio and the
`NOT_CONFIRMATORY` status. Plan v2 (`docs/research_plan_v2.md` §3.4) removes
the causes: hyperparameters by uLSIF CV only, always self-normalised, no
validation-tuned bands, and Wang's own Eq. 8 stopping rule.
