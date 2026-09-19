#!/usr/bin/env python3
"""Phase 0 hardware preflight: is this host's NLI scoring path usable, and does
MPS agree with CPU closely enough to preserve the frozen computation?

Runs the EXACT production scoring path (``src.utils.EntailmentScorer``) on the
released NBC pairs, once per available backend, and compares. Nothing is
written to any cache: the scorer is bound to a throwaway database and both
``use_cache`` and ``write_cache`` are False.

The comparison that matters is NOT raw float agreement. Wang's pipeline rounds
the score to one decimal and buckets it into ten bins, and BSE's evidence is the
BUCKET. So a backend difference is scientifically irrelevant if it never changes
a bucket, and is a methodological change the moment it does. Both are reported
separately, and the bucket disagreement count is the number that decides.

No model is downloaded beyond the pinned checkpoint, no experiment artifact is
written, and no frozen decision is altered.
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = "results/diagnostics/mps_preflight.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 MPS/CPU preflight.")
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument(
        "--model-name",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    )
    parser.add_argument(
        "--pairs", type=int, default=32,
        help="How many released NBC pairs to score per backend (deterministic prefix).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Canonical Wang-fidelity path is 1. Use 8 only to inspect batching effects.",
    )
    parser.add_argument("--revision", default=None,
                        help="Pin the checkpoint revision, as the formal path does.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Report environment only; load no model.")
    return parser.parse_args()


def environment():
    import torch

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import tokenizers
        tokenizers_version = tokenizers.__version__
    except Exception:
        tokenizers_version = None

    mac = {}
    if platform.system() == "Darwin":
        for label, cmd in (
            ("macos_version", ["sw_vers", "-productVersion"]),
            ("macos_build", ["sw_vers", "-buildVersion"]),
            ("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
        ):
            try:
                mac[label] = subprocess.check_output(cmd, text=True).strip()
            except Exception:
                mac[label] = None
        try:
            mac["memory_gb"] = round(
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
                / 1024 ** 3, 1
            )
        except Exception:
            mac["memory_gb"] = None

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "apple": mac or None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers_version,
        "tokenizers": tokenizers_version,
        "mps_is_built": bool(torch.backends.mps.is_built()),
        "mps_is_available": bool(torch.backends.mps.is_available()),
        "cuda_is_available": bool(torch.cuda.is_available()),
        # What the CURRENT repository code would select. Every device site is
        # `cuda if cuda.is_available() else cpu`, with no MPS branch, so on
        # Apple silicon the GPU is ignored until that is changed.
        "device_current_repo_code_would_select": (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    }


def score_on(device_name, args, pairs):
    """Score the pairs through the production path on one backend."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.utils import EntailmentScorer

    load_kwargs = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, **load_kwargs
    )
    model = model.to(torch.device(device_name))
    model.eval()

    # Throwaway database, and caching disabled on both sides: this must not be
    # able to touch or seed any real cache.
    with tempfile.TemporaryDirectory(prefix="mps-preflight-") as scratch:
        scorer = EntailmentScorer(
            tokenizer, model, args.model_name,
            cache_path=str(Path(scratch) / "scratch.sqlite"),
            batch_size=args.batch_size,
        )
        try:
            scores = scorer.score_pairs(
                pairs, use_cache=False, write_cache=False, show_progress=False
            )
            rows_written = scorer.cache_size()
        finally:
            scorer.close()

    return {
        "device": device_name,
        "batch_size": args.batch_size,
        "dtype": str(next(model.parameters()).dtype),
        "model_max_length": int(getattr(tokenizer, "model_max_length", -1)),
        "scores": [float(s) for s in scores],
        "scratch_rows_written": int(rows_written),
    }


def compare(reference, candidate):
    """CPU vs MPS, in raw floats AND in the buckets BSE actually consumes."""
    from src.baseline_core import discretize_document_score

    a, b = reference["scores"], candidate["scores"]
    if len(a) != len(b):
        raise RuntimeError(f"score counts differ: {len(a)} vs {len(b)}")

    deltas = [abs(x - y) for x, y in zip(a, b)]
    exact = sum(1 for x, y in zip(a, b) if x == y)
    one_dp = sum(1 for x, y in zip(a, b) if round(x, 1) == round(y, 1))
    buckets_a = [discretize_document_score(x) for x in a]
    buckets_b = [discretize_document_score(y) for y in b]
    bucket_disagreements = [
        {"index": i, "cpu": x, "mps": y, "cpu_bucket": p, "mps_bucket": q}
        for i, (x, y, p, q) in enumerate(zip(a, b, buckets_a, buckets_b))
        if p != q
    ]

    return {
        "n_pairs": len(a),
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "exact_float_matches": exact,
        "one_decimal_matches": one_dp,
        "bucket_matches": len(a) - len(bucket_disagreements),
        "bucket_disagreements": bucket_disagreements,
        "buckets_identical": not bucket_disagreements,
        "verdict": (
            "MPS preserves every BSE evidence bucket on this sample"
            if not bucket_disagreements
            else f"MPS CHANGES {len(bucket_disagreements)} BSE evidence bucket(s) "
                 "on this sample -- this is a methodological difference, not a "
                 "rounding detail. Do NOT proceed on MPS without deciding."
        ),
        "bucket_note": (
            "BSE consumes the DISCRETIZED bucket, not the raw float, so raw "
            "float disagreement is only scientifically meaningful when it "
            "crosses a bucket boundary."
        ),
    }


def main():
    args = parse_args()
    report = {
        "preflight": "phase-0 hardware",
        "environment": environment(),
        "model_name": args.model_name,
        "revision": args.revision,
        "pairs_requested": args.pairs,
    }

    print("=" * 78)
    print("PHASE 0 HARDWARE PREFLIGHT")
    print("=" * 78)
    for key, value in report["environment"].items():
        print(f"  {key:38s} {value}")

    env = report["environment"]
    if args.skip_inference:
        print("\n--skip-inference: environment only, no model loaded.")
    elif not env["mps_is_available"] and env["system"] == "Darwin":
        print("\nMPS is NOT available on this host. Report this rather than "
              "working around it.")
    else:
        from src.wang_data import load_nbc_pairs

        positive, _ = load_nbc_pairs(args.data_root, per_class=None)
        pairs = [
            (item["premise"], item["hypothesis"]) for item in positive[: args.pairs]
        ]
        print(f"\nScoring {len(pairs)} released NBC pairs per backend, "
              f"batch size {args.batch_size}, caching disabled.")

        backends = ["cpu"] + (["mps"] if env["mps_is_available"] else [])
        runs = {}
        for backend in backends:
            print(f"  scoring on {backend} ...", flush=True)
            runs[backend] = score_on(backend, args, pairs)
            assert runs[backend]["scratch_rows_written"] == 0, (
                "the preflight wrote cache rows; it must write none"
            )
        report["runs"] = runs

        if "mps" in runs:
            report["cpu_vs_mps"] = compare(runs["cpu"], runs["mps"])
            print()
            for key in (
                "n_pairs", "max_abs_delta", "mean_abs_delta",
                "exact_float_matches", "one_decimal_matches", "bucket_matches",
                "buckets_identical",
            ):
                print(f"  {key:26s} {report['cpu_vs_mps'][key]}")
            print(f"\n  VERDICT: {report['cpu_vs_mps']['verdict']}")
        else:
            print("\n  Only CPU was exercised; no MPS comparison to make.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nWrote {output}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
