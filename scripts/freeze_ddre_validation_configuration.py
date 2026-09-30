#!/usr/bin/env python3
"""Freeze ONE DDRE configuration from the existing validation sensitivity artifact.

Reads the 640 validation candidates already recorded by
``scripts/d03_hyperparameter_outcome_sensitivity.py`` and applies the
predeclared selection rule. It performs NO NLI inference, loads no model, opens
no cache, and never touches the held-out split: the only input is one JSON file
that was produced on validation data.

This DOES select -- that is its purpose, and the distinction matters. The
sensitivity study selected nothing; this step freezes a validation-tuned
configuration so the held-out evaluation has exactly one thing to run. It
chooses no cap, no calibration and no evidence transform, and changes no
production default in ``main.py``.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SOURCE_SHA256 = (
    "fc5604854b18f3e3e712fe7ef01859b58dac569e99696041713abb11daa6dcc0"
)
DEFAULT_SOURCE = (
    "~/ddre-artifacts/gate1-2026-09-20/"
    "d03_hyperparameter_outcome_sensitivity_main_59f4f53.json"
)
DEFAULT_OUTPUT = "~/ddre-artifacts/gate1-2026-09-20/ddre_validation_freeze.json"

ELIGIBILITY = ("preserves_nonfactual", "preserves_factual", "preserves_balanced")

SELECTION_RULE = (
    "Eligible iff preserves_nonfactual AND preserves_factual AND "
    "preserves_balanced are all true on the frozen validation split. Among "
    "eligible configurations, order by: (1) minimum "
    "avg_retrieved_documents_per_sentence; (2) higher balanced PR-AUC; "
    "(3) higher nonfactual PR-AUC; (4) higher factual PR-AUC; (5) lower "
    "avg_nli_span_calls_per_sentence; (6) deterministic ascending "
    "sigma, lambda, lower, upper."
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick(candidate, *names):
    for name in names:
        if candidate.get(name) is not None:
            return candidate[name]
    return None


def flatten(report):
    """Every (hyperparameter pair x threshold pair) candidate, with its identity."""
    rows = report.get("hyperparameter_rows") or []
    if not rows:
        raise SystemExit("ABORTED: the artifact carries no hyperparameter_rows.")
    flat = []
    for row in rows:
        sigma = float(row["sigma"])
        lam = float(row["lambda"])
        for candidate in row.get("threshold_candidates") or []:
            flat.append({
                "sigma": sigma,
                "lambda": lam,
                "lower_threshold": float(candidate["lower"]),
                "upper_threshold": float(candidate["upper"]),
                "is_production_hyperparameter_pair": bool(
                    row.get("is_production_pair", False)
                ),
                "ulsif_cv_objective": row.get("ulsif_cv_objective"),
                "nonfactual_auc_pr": candidate["nonfactual_auc_pr"],
                "factual_auc_pr": candidate["factual_auc_pr"],
                "balanced_pr_auc": candidate["balanced_pr_auc"],
                "accuracy": candidate["accuracy"],
                "macro_f1": candidate["macro_f1"],
                "avg_retrieved_documents_per_sentence": pick(
                    candidate, "avg_retrieved_documents_per_sentence", "avg_documents"
                ),
                "avg_retrieved_documents_per_subclaim": pick(
                    candidate, "avg_retrieved_documents_per_subclaim",
                    "avg_documents_per_subclaim",
                ),
                "avg_nli_span_calls_per_sentence": pick(
                    candidate, "avg_nli_span_calls_per_sentence", "avg_nli_span_calls"
                ),
                "nonfactual_auc_pr_delta_vs_bse": candidate[
                    "nonfactual_auc_pr_delta_vs_bse"
                ],
                "factual_auc_pr_delta_vs_bse": candidate[
                    "factual_auc_pr_delta_vs_bse"
                ],
                "balanced_pr_auc_delta_vs_bse": candidate[
                    "balanced_pr_auc_delta_vs_bse"
                ],
                "retrieved_documents_reduction_fraction": candidate.get(
                    "retrieved_documents_reduction_fraction"
                ),
                "nli_span_calls_reduction_fraction": candidate.get(
                    "nli_span_calls_reduction_fraction"
                ),
                "preserves_nonfactual": bool(candidate["preserves_nonfactual"]),
                "preserves_factual": bool(candidate["preserves_factual"]),
                "preserves_balanced": bool(candidate["preserves_balanced"]),
            })
    return flat


def eligible(candidate):
    return all(candidate[flag] for flag in ELIGIBILITY)


def order_key(candidate):
    return (
        candidate["avg_retrieved_documents_per_sentence"],
        -candidate["balanced_pr_auc"],
        -candidate["nonfactual_auc_pr"],
        -candidate["factual_auc_pr"],
        candidate["avg_nli_span_calls_per_sentence"],
        candidate["sigma"],
        candidate["lambda"],
        candidate["lower_threshold"],
        candidate["upper_threshold"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        raise SystemExit(f"ABORTED: sensitivity artifact not found: {source}")

    source_sha = sha256_file(source)
    if source_sha != args.expected_source_sha256:
        raise SystemExit(
            "ABORTED: the sensitivity artifact is not the expected one.\n"
            f"  expected {args.expected_source_sha256}\n"
            f"  observed {source_sha}\nNothing was selected."
        )
    with source.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    if report.get("held_out_scored") is not False:
        raise SystemExit(
            "ABORTED: the source artifact does not record held_out_scored=false."
        )

    candidates = flatten(report)
    winners = sorted((c for c in candidates if eligible(c)), key=order_key)

    print("=" * 112)
    print("DDRE VALIDATION FREEZE  (validation-tuned; held-out never scored)")
    print("=" * 112)
    print(f"  source            {source}")
    print(f"    sha256          {source_sha}  (verified)")
    print(f"  candidates read   {len(candidates)}")
    print(f"  ELIGIBLE          {len(winners)}"
          "   (preserves nonfactual AND factual AND balanced)")

    if not winners:
        print(
            "\nNo configuration preserved all three metrics on validation.\n"
            "Nothing is frozen: a fallback selection cannot support a "
            "confirmatory claim, and inventing one here would be exactly the "
            "post-hoc choice the protocol forbids."
        )
        return 1

    print("\n  TOP 10 ELIGIBLE")
    header = (
        f"  {'#':>2} {'sigma':>14} {'lambda':>7} {'L/U':>11} "
        f"{'docs/sent':>10} {'NLI/sent':>9} {'nonfact':>8} {'factual':>8} "
        f"{'balanced':>9} {'docs red':>9} {'NLI red':>8}"
    )
    print(header)
    print("  " + "-" * 108)
    for index, c in enumerate(winners[:10], start=1):
        def pct(v):
            return "    n/a" if v is None else f"{100.0 * v:>6.2f}%"
        print(
            f"  {index:>2} {c['sigma']:>14.10f} {c['lambda']:>7g} "
            f"{c['lower_threshold']:>5.2f}/{c['upper_threshold']:<5.2f} "
            f"{c['avg_retrieved_documents_per_sentence']:>10.4f} "
            f"{c['avg_nli_span_calls_per_sentence']:>9.3f} "
            f"{c['nonfactual_auc_pr']:>8.4f} {c['factual_auc_pr']:>8.4f} "
            f"{c['balanced_pr_auc']:>9.4f} "
            f"{pct(c['retrieved_documents_reduction_fraction']):>9} "
            f"{pct(c['nli_span_calls_reduction_fraction']):>8}"
            + ("  <- production pair" if c["is_production_hyperparameter_pair"] else "")
        )

    selected = winners[0]
    print("\n" + "=" * 112)
    print("SELECTED (frozen)")
    print("=" * 112)
    print(f"  sigma                        {selected['sigma']!r}")
    print(f"  lambda                       {selected['lambda']!r}")
    print(f"  lower threshold              {selected['lower_threshold']!r}")
    print(f"  upper threshold              {selected['upper_threshold']!r}")
    print(f"  production hyperparameter?   {selected['is_production_hyperparameter_pair']}")
    print("\n  VALIDATION METRICS")
    for label, key in (
        ("nonfactual PR-AUC", "nonfactual_auc_pr"),
        ("factual PR-AUC", "factual_auc_pr"),
        ("balanced PR-AUC", "balanced_pr_auc"),
        ("accuracy", "accuracy"),
        ("macro F1", "macro_f1"),
        ("docs / sentence", "avg_retrieved_documents_per_sentence"),
        ("docs / subclaim", "avg_retrieved_documents_per_subclaim"),
        ("NLI spans / sentence", "avg_nli_span_calls_per_sentence"),
    ):
        print(f"    {label:<26} {selected[key]}")
    print("\n  VERSUS BSE OFFICIAL ON THE SAME VALIDATION PASSAGES")
    for label, key in (
        ("nonfactual PR-AUC delta", "nonfactual_auc_pr_delta_vs_bse"),
        ("factual PR-AUC delta", "factual_auc_pr_delta_vs_bse"),
        ("balanced PR-AUC delta", "balanced_pr_auc_delta_vs_bse"),
    ):
        print(f"    {label:<26} {selected[key]:+.6f}")
    for label, key in (
        ("retrieval reduction", "retrieved_documents_reduction_fraction"),
        ("NLI-call reduction", "nli_span_calls_reduction_fraction"),
    ):
        value = selected[key]
        shown = "n/a" if value is None else f"{100.0 * value:.2f}%"
        print(f"    {label:<26} {shown}")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen = {
        "artifact": "ddre-validation-freeze",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sensitivity_artifact": str(source),
        "source_sensitivity_artifact_sha256": source_sha,
        "source_d03_base_commit": report.get("d03_base_commit"),
        "source_split": report.get("split"),
        "selection_rule": SELECTION_RULE,
        "eligibility_criteria": list(ELIGIBILITY),
        "candidates_considered": len(candidates),
        "eligible_configurations": len(winners),
        "selected_configuration": {
            "sigma": selected["sigma"],
            "lambda": selected["lambda"],
            "lower_threshold": selected["lower_threshold"],
            "upper_threshold": selected["upper_threshold"],
            "is_production_hyperparameter_pair": selected[
                "is_production_hyperparameter_pair"
            ],
        },
        "validation_metrics": selected,
        "top_10_eligible": winners[:10],
        "bse_official_validation": report.get("bse_official_validation"),
        "held_out_scored": False,
        "validation_tuned": True,
        "selected_cap": None,
        "selected_calibration": None,
        "method_change": (
            "validation configuration selection only: one (sigma, lambda, lower, "
            "upper) is frozen from candidates already measured on validation. No "
            "cap, no calibration, no evidence transform, no change to the ratio "
            "definition, the costs, P0, max_docs, the split or any main.py default."
        ),
        "production_defaults_changed": False,
        "nli_inference_performed": False,
        "next_step": (
            "Held-out evaluation of exactly this configuration. Nothing here may "
            "be re-selected after a held-out number is seen."
        ),
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, default=str)

    print(f"\n  written to  {output}")
    print(f"  sha256      {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
