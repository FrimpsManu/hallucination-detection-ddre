#!/usr/bin/env python3
"""uLSIF normalisation diagnostic over the D-03 (sigma, lambda) surface.

For the production (CV-selected) fit, the frozen D configuration and every
(sigma, lambda) pair of the production CV grid, reports on the 199 + 199 NBC
training samples:

- E_hallucinated[r], which a density ratio must make 1;
- the log-ratio curve on a 0-100 grid, raw and divided by that normaliser;
- the sign of the evidence at the dominant document score (35), raw and
  normalised.

Training data only: the NBC scores come from the pinned D-03 cache, opened
read-only. No held-out or test record is read, no detector is run, nothing is
selected. The output is written once and hashed.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_ablation_heldout_replay as RM  # noqa: E402
import run_frozen_heldout as frozen  # noqa: E402
from src import ulsif_normalisation as N  # noqa: E402
from src.cache_completion import sha256_file  # noqa: E402

ARTIFACTS = Path("~/ddre-artifacts/gate1-2026-09-20").expanduser()
OUTPUT = Path("~/ddre-artifacts/diagnostics/ulsif_normalisation.json").expanduser()
FROZEN_D = (0.11346435546875, 1.0)


class DiagnosticRefused(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS))
    parser.add_argument("--wang-data-root", default=str(PROJECT_ROOT / "data" / "wang"))
    parser.add_argument("--output", default=str(OUTPUT))
    return parser.parse_args()


def label(sigma, lam):
    return f"sigma={sigma!r},lambda={lam!r}"


def main():
    from src.baseline_core import build_nbc_histograms
    from src.cached_document_scores import CachedDocumentScorer
    from src.ddre_core import ULSIFDensityRatio
    from src.hyperparameter_outcome_sensitivity import candidate_estimator, verify_production_fit
    from src.utils import SCORE_VERSION, split_text
    from src.wang_data import load_nbc_pairs

    args = parse_args()
    output = Path(args.output).expanduser()
    if output.exists():
        raise DiagnosticRefused(f"{output} already exists; the diagnostic is written once")
    name, pinned = RM.PINNED["d03_cache"]
    cache = Path(args.artifacts_dir).expanduser() / name
    before = sha256_file(cache)
    if before != pinned:
        raise DiagnosticRefused(f"D-03 cache sha256 {before}, pinned {pinned}")

    positive, negative = load_nbc_pairs(args.wang_data_root, per_class=None)
    nbc = CachedDocumentScorer(cache, frozen.OFFICIAL_MODEL, SCORE_VERSION, split_text)
    try:
        factual = nbc.score_pairs([(x["premise"], x["hypothesis"]) for x in positive])
        hallucinated = nbc.score_pairs([(x["premise"], x["hypothesis"]) for x in negative])
        pos_hist, neg_hist, _, _ = build_nbc_histograms(positive, negative, nbc)
    finally:
        nbc.close()

    production = ULSIFDensityRatio(random_state=frozen.EXPECTED_SPLIT_SEED).fit(
        factual, hallucinated)
    verify_production_fit(production)
    fx, hx = production._as_column(factual), production._as_column(hallucinated)
    pairs = sorted({(float(r["sigma"]), float(r["lambda"])) for r in production.cv_table})
    if FROZEN_D not in pairs:
        raise DiagnosticRefused(f"frozen D {FROZEN_D} is not on the production CV grid")

    rows = {}
    for sigma, lam in pairs:
        est = candidate_estimator(production, fx, hx, sigma, lam)
        rows[label(sigma, lam)] = {"sigma": sigma, "lambda": lam,
                                   **N.normalisation_row(est, factual, hallucinated)}
    production_key = label(production.sigma, production.lam)
    frozen_key = label(*FROZEN_D)

    from src.ablation_selection import HistogramRatio
    histogram = N.normalisation_row(HistogramRatio(pos_hist, neg_hist, 1.0),
                                    factual, hallucinated)

    after = sha256_file(cache)
    if after != before:
        raise DiagnosticRefused("the D-03 cache changed during the diagnostic")
    result = {
        "artifact": "ulsif-normalisation-diagnostic",
        "status": "exploratory diagnostic; training (NBC) data only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": RM.git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(RM.git("status", "--porcelain")),
        "d03_cache": {"path": str(cache), "sha256": before, "unchanged": True},
        "nbc_samples": {"factual": len(factual), "hallucinated": len(hallucinated)},
        "production": production_key,
        "frozen_D": frozen_key,
        "rows": rows,
        "bse_histogram_ratio_document_binning": histogram,
        "sign_at_mode_vs_frozen_D": N.sign_agreement(rows, frozen_key),
        "sign_at_mode_normalised_vs_frozen_D": N.sign_agreement(
            rows, frozen_key, "sign_at_mode_normalised"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")
    output.chmod(0o444)
    print(f"written {output}\n  sha256 {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DiagnosticRefused as exc:
        print(f"\nREFUSED: {exc}\nNothing was written.")
        sys.exit(1)
