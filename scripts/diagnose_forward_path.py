"""Gate 1 diagnostic, Step 2: what explains the Step 1 A1 vs A3 divergence?

Step 1 established that the literal Wang scorer and this repository's scorer
tokenize identically (589/589 input_ids matched) yet disagreed on exactly one
BSE decision: positive NBC pair 169, bucket 1 under the literal path and bucket
2 under ours.

The two scorers differ in TWO independent places, not one. Besides the forward
call, they differ in how logits become a 0-100 score. Released
``utils.py:59-65`` leaves the tensor and then scales::

    probabilities = torch.softmax(output["logits"][0] / 5, -1).tolist()
    raw = float(probabilities[0]) * 100.0          # float64 multiply

while ``src/utils.py:122-124`` scales inside the tensor and then leaves it::

    probs = torch.softmax(outputs.logits / 5.0, dim=-1)
    scores = probs[:, entailment_index] * 100.0    # multiply in tensor dtype

On a half-precision tensor that difference alone reproduces the observation
exactly: for the probe pair's probability 0.199462890625, ``float(p) * 100`` is
19.9462890625 and ``float16(p * 100)`` is 19.953125 -- the two recorded values,
differing by exactly 0.0068359375. So the extraction path must be tested, and it
must be tested FIRST, because a forward-only factorial holds extraction fixed at
Wang's form for every arm and would remove the very factor under suspicion.

PRIMARY -- 2x2 over two independent factors
-------------------------------------------

    F0 = model(input_ids)                          literal Wang forward
    F1 = inference_mode: model(input_ids,          repository batch-1 forward
         attention_mask=..., token_type_ids=...)

    X0 = float(softmax(logits[0]/5,-1).tolist()[0]) * 100    Wang extraction
    X1 = float((softmax(logits/5.,-1)[:,0] * 100).tolist()[0]) repository

    D00 = F0+X0  exact A1 reconstruction    D01 = F0+X1  extraction changed only
    D10 = F1+X0  forward changed only       D11 = F1+X1  exact A3 reconstruction

Both extractions are pure functions of the logits, so one forward pass per row
serves both columns: D00/D01 share bit-identical logits, and so do D10/D11. The
extraction comparison is therefore exact by construction.

SECONDARY -- forward-path factorial
-----------------------------------

    C0  model(input_ids)                                     reference
    C1  inference_mode: model(input_ids)                     +inference_mode
    C2  model(input_ids, attention_mask=...)                 +attention_mask
    C3  inference_mode: model(input_ids, attention_mask=...) both
    C4  inference_mode: + token_type_ids                     repository bridge
    C0R C0 again, last                                       determinism control

This decomposes the forward path into its individual arguments. Every arm uses
Wang extraction, so if the primary attributes the divergence to the extraction
path, no secondary arm can reconstruct A3 and the secondary chain will say so.
The report states that expectation explicitly rather than leaving it to be
misread as a failure.

Controls that make either result attributable:

* the tokenizer and model are loaded once;
* every pair is tokenized ONCE and the identical tensors are reused everywhere;
* batch size is exactly 1 throughout, so padding is never involved;
* determinism controls re-run a cell unchanged. If a cell does not reproduce
  itself, the device is nondeterministic and attribution is withheld.

Reads no cache and writes no cache. Touches no formal artifact. Changes no
baseline behaviour, tolerance, published reference value, split, or cost
parameter.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction_path_diagnostics import (  # noqa: E402
    CELL_COMPARISONS,
    interpret_shape_and_scaling,
    summarize_shape_and_scaling,
    CELL_SPECS,
    EXTRACTION_SPECS,
    FORWARD_SPECS,
    build_primary_verdict,
    compare_cells,
    probe_cell_report,
    score_pair_both_extractions,
    secondary_expectation,
)
from src.forward_path_diagnostics import (  # noqa: E402
    ARM_SPECS,
    DETERMINISM_CONTROL,
    ENTAILMENT_INDEX,
    REFERENCE_OBSERVATION,
    SOFTMAX_TEMPERATURE,
    build_verdict,
    compare_forward_arms,
    histograms_by_polarity,
    probe_report,
    score_one_pair,
)

OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_OUTPUT = "results/diagnostics/step2_forward_path.json"

# Comparisons computed. C0-vs-arm answers the question directly; C1_vs_C3 and
# C2_vs_C3 are what the isolation rules need to tell a dominant factor from an
# interaction.
COMPARISON_PAIRS = (
    ("C0", "C1"),
    ("C0", "C2"),
    ("C0", "C3"),
    ("C1", "C3"),
    ("C2", "C3"),
    ("C1", "C2"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gate 1 Step 2: decompose the Step 1 A1/A3 divergence into a "
            "forward-path factor and an extraction/scaling factor over all 398 "
            "released NBC pairs, then decompose the forward path further."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Where to write the report. Point this at Google Drive for a formal "
            "run, e.g. /content/drive/MyDrive/ddre-gate1/diagnostics/"
            "step2_forward_path.json"
        ),
    )
    parser.add_argument("--probe-polarity", default="positive", choices=["positive", "negative"])
    parser.add_argument("--probe-index", type=int, default=169)
    parser.add_argument(
        "--skip-c4",
        action="store_true",
        help=(
            "Skip the C4 token_type_ids arm. Useful for debugging, but a run "
            "without C4 CANNOT establish the formal causal chain: the verdict "
            "becomes UNDETERMINED_BRIDGE_NOT_EVALUATED."
        ),
    )
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Also hash large checkpoint files when recording checkpoint identity.",
    )
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help=(
            "Run only the primary 2x2 decomposition and skip the C0-C4 "
            "forward-path factorial."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and count the NBC pairs, then exit without loading the model.",
    )
    return parser.parse_args()


def load_pairs(data_root):
    """All released NBC pairs, positives first, with a parallel polarity list."""
    from src.wang_data import load_nbc_pairs

    positive, negative = load_nbc_pairs(data_root, per_class=None)
    pairs = [(item["premise"], item["hypothesis"]) for item in positive]
    pairs += [(item["premise"], item["hypothesis"]) for item in negative]
    polarities = ["positive"] * len(positive) + ["negative"] * len(negative)
    return pairs, polarities, len(positive), len(negative)


def tokenize_once(tokenizer, pairs, device):
    """Tokenize every pair exactly once, using Wang's released call.

    Released ``utils.py:56``: ``tokenizer(premise, hypothesis, truncation=True,
    return_tensors="pt")`` -- no ``max_length``, no padding, batch of one. Step
    1 already proved this produces the same input_ids as the repository's call,
    so a single tokenization can serve every arm and tokenization is removed as
    a variable entirely.
    """
    encodings = []
    for premise, hypothesis in pairs:
        encoded = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        encodings.append({key: value.to(device) for key, value in encoded.items()})
    return encodings


def score_forward_arm(model, encodings, spec, torch, show_progress=True):
    """Run one arm over the pre-tokenized tensors.

    The per-pair work lives in ``src.forward_path_diagnostics.score_one_pair``
    so that the control-flow boundary -- only the model call inside
    ``torch.inference_mode()``, the softmax always outside, exactly as
    ``src/utils.py`` does it -- is unit-testable with an injected fake torch.
    """
    from tqdm import tqdm

    iterator = (
        tqdm(encodings, desc=f"arm {spec['name']}", unit="pair")
        if show_progress
        else encodings
    )

    scores = []
    start = time.perf_counter()
    for encoded in iterator:
        scores.append(score_one_pair(model, encoded, spec, torch))
    elapsed = time.perf_counter() - start
    return scores, elapsed


PURPOSE = (
    "Determine whether the Step 1 A1/A3 divergence is explained by the forward "
    "call or by the extraction/scaling path, then decompose the forward path "
    "into its individual arguments."
)


def build_protocol(pairs, n_positive, n_negative, model_block, token_type_ids_available):
    return {
        "nbc_pairs": len(pairs),
        "positive_pairs": n_positive,
        "negative_pairs": n_negative,
        "batch_size": 1,
        "entailment_index": ENTAILMENT_INDEX,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "tokenized_once": True,
        "tensors_shared_across_arms": True,
        "one_forward_per_primary_row_serves_both_extractions": True,
        "model_eval_mode": not model_block.get("training_mode", True),
        "cache_reads": 0,
        "cache_writes": 0,
        "token_type_ids_available": token_type_ids_available,
    }


def environment_summary(args, environment, model_block):
    return {
        "python_version": (environment.get("python") or {}).get("version"),
        "torch_version": (environment.get("libraries") or {}).get("torch"),
        "transformers_version": (environment.get("libraries") or {}).get("transformers"),
        "tokenizers_version": (environment.get("libraries") or {}).get("tokenizers"),
        "device": (environment.get("device") or {}).get("selected_device"),
        "gpu_name": (environment.get("device") or {}).get("gpu_name"),
        "model_name": args.model_name,
        # The decisive field for the extraction hypothesis: the half-precision
        # mechanism only operates if the probability tensor is float16.
        "model_dtype": model_block.get("dtype"),
        "model_training_mode": model_block.get("training_mode"),
        "attn_implementation": model_block.get("attn_implementation"),
        "type_vocab_size": model_block.get("type_vocab_size"),
        "resolved_hf_revision": (environment.get("checkpoint_identity") or {}).get(
            "resolved_revision"
        ),
        "git_commit": (environment.get("git") or {}).get("commit"),
    }


def run_forward_row(model, encodings, forward_spec, torch, label):
    """Score every pair once under one forward path, extracting both ways.

    One forward pass produces both the X0 and X1 scores, so the two cells in
    this row cannot differ in anything but the extraction.
    """
    from tqdm import tqdm

    iterator = tqdm(encodings, desc=label, unit="pair")
    x0, x1, probes = [], [], []
    start = time.perf_counter()
    for encoded in iterator:
        scores = score_pair_both_extractions(model, encoded, forward_spec, torch)
        x0.append(scores["X0"])
        x1.append(scores["X1"])
        probes.append(scores["shape_scaling"])
    return {
        "X0": x0,
        "X1": x1,
        "shape_scaling": probes,
        "elapsed_seconds": time.perf_counter() - start,
    }


def run_primary_decomposition(model, encodings, polarities, torch, args):
    """The 2x2, plus a determinism repeat of each forward row."""
    rows = {}
    for name in ("F0", "F1"):
        spec = FORWARD_SPECS[name]
        print(f"\nPrimary row {name} ({spec['label']}): {spec['call']}")
        rows[name] = run_forward_row(model, encodings, spec, torch, f"row {name}")

    repeats = {}
    for name in ("F0", "F1"):
        print(f"\nDeterminism control {name}R (re-running {name} unchanged)")
        repeats[name] = run_forward_row(
            model, encodings, FORWARD_SPECS[name], torch, f"row {name}R"
        )

    scores_by_cell = {
        "D00": rows["F0"]["X0"],
        "D01": rows["F0"]["X1"],
        "D10": rows["F1"]["X0"],
        "D11": rows["F1"]["X1"],
    }

    controls = {
        "F0": compare_cells(
            "D00", "D00", rows["F0"]["X0"], repeats["F0"]["X0"], polarities
        ),
        "F1": compare_cells(
            "D10", "D10", rows["F1"]["X0"], repeats["F1"]["X0"], polarities
        ),
    }

    matrix = {}
    for left, right in CELL_COMPARISONS:
        block = compare_cells(
            left, right, scores_by_cell[left], scores_by_cell[right], polarities
        )
        matrix[block["comparison"]] = block

    probe = probe_cell_report(
        args.probe_polarity, args.probe_index, scores_by_cell, polarities
    )

    # Softmax shape vs scaling order, from the logits already computed.
    flat = probe["flat_index"]
    shape_scaling = {}
    for name in ("F0", "F1"):
        probes = rows[name]["shape_scaling"]
        summary = summarize_shape_and_scaling(probes)
        probe_row = probes[flat]
        shape_scaling[name] = {
            "forward_row": name,
            "summary": summary,
            "probe_pair": probe_row,
            "interpretation": interpret_shape_and_scaling(probe_row, summary),
        }

    verdict = build_primary_verdict(
        matrix,
        probe,
        scores_by_cell,
        polarities,
        controls=controls,
        shape_scaling=shape_scaling,
    )

    return {
        "decomposition": "2x2 forward path x extraction path",
        "forward_levels": FORWARD_SPECS,
        "extraction_levels": EXTRACTION_SPECS,
        "cells": {
            cell["name"]: dict(
                cell,
                elapsed_seconds=rows[cell["forward"]]["elapsed_seconds"],
            )
            for cell in CELL_SPECS
        },
        "histograms_by_cell": {
            name: histograms_by_polarity(scores, polarities)
            for name, scores in scores_by_cell.items()
        },
        "comparisons": matrix,
        "determinism_controls": controls,
        "probe": probe,
        "shape_and_scaling": shape_scaling,
        "verdict": verdict,
    }


def print_cell_comparison(block):
    print()
    print("-" * 100)
    print(f"{block['comparison']}    isolates: {block['isolated_factor'] or 'nothing'}")
    print("-" * 100)
    print(f"  {format_deltas(block['deltas'])}")
    print(f"  difference class:            {block['difference_class']}")
    print(
        f"  one-decimal disagreements:   "
        f"{block['one_decimal_disagreements']['count']}/{block['pairs']}"
    )
    print(
        f"  NBC bucket disagreements:    "
        f"{block['nbc_bucket_disagreements']['count']}/{block['pairs']}"
    )
    for example in block["nbc_bucket_disagreements"]["examples"]:
        print(
            f"      pair {example['index']}: {example['left']!r} (bucket "
            f"{example['left_bucket']}) -> {example['right']!r} (bucket "
            f"{example['right_bucket']})"
        )
    print(f"  numerical_difference:        {block['numerical_difference']}")
    print(f"  bse_decision_impact:         {block['bse_decision_impact']}")
    print(f"  causal_candidate:            {block['causal_candidate']}")
    print(f"  positive histogram (left):   {block['histograms']['left']['positive']}")
    print(f"  positive histogram (right):  {block['histograms']['right']['positive']}")
    print(f"  negative histogram (left):   {block['histograms']['left']['negative']}")
    print(f"  negative histogram (right):  {block['histograms']['right']['negative']}")


def print_secondary_histograms(report):
    """Print the secondary arms' histograms.

    Reads ``report["secondary"]["nbc_laplace_histograms_by_arm"]``. Extracted
    into a function because this exact access regressed when the report grew a
    primary/secondary split, and a function can be unit-tested against the
    report shape the runner actually builds.
    """
    reference = REFERENCE_OBSERVATION
    histograms = report["secondary"]["nbc_laplace_histograms_by_arm"]
    print()
    print("-" * 100)
    print("LAPLACE-SMOOTHED NBC HISTOGRAMS BY ARM (secondary)")
    print("-" * 100)
    print(f"  {'Step1 A1':<8} positive: {reference['positive_histogram_literal_wang']}")
    print(f"  {'Step1 A3':<8} positive: {reference['positive_histogram_repository']}")
    for name in sorted(histograms):
        print(f"  {name:<8} positive: {histograms[name]['positive']}")
    for name in sorted(histograms):
        print(f"  {name:<8} negative: {histograms[name]['negative']}")


def print_primary(report):
    verdict = report["verdict"]
    reference = REFERENCE_OBSERVATION

    print()
    print("=" * 100)
    print("PRIMARY DECOMPOSITION: forward path x extraction path")
    print("=" * 100)
    for name, block in report["determinism_controls"].items():
        print(
            f"  determinism control {name}: bit-identical = "
            f"{block['numerical_difference'] == 'NO'}  "
            f"(max |delta| {block['deltas']['max_absolute']})"
        )

    for block in report["comparisons"].values():
        print_cell_comparison(block)

    probe = report["probe"]
    print()
    print("-" * 100)
    print(
        f"PROBE: {probe['polarity']} NBC pair {probe['index']} "
        f"(flat index {probe['flat_index']})"
    )
    print("-" * 100)
    print(f"  {'cell':<10}{'raw':>24}{'rounded':>12}{'bucket':>10}")
    print(
        f"  {'Step1 A1':<10}{reference['literal_wang']['raw']:>24.10f}"
        f"{reference['literal_wang']['rounded']:>12}"
        f"{reference['literal_wang']['nbc_bucket']:>10}   (recorded)"
    )
    print(
        f"  {'Step1 A3':<10}{reference['repository']['raw']:>24.10f}"
        f"{reference['repository']['rounded']:>12}"
        f"{reference['repository']['nbc_bucket']:>10}   (recorded)"
    )
    for name in sorted(probe["cells"]):
        row = probe["cells"][name]
        print(
            f"  {name:<10}{row['raw']:>24.10f}{row['rounded_one_decimal']:>12}"
            f"{row['nbc_bucket']:>10}"
        )

    print()
    print("-" * 100)
    print("LAPLACE-SMOOTHED NBC HISTOGRAMS BY CELL")
    print("-" * 100)
    print(f"  {'Step1 A1':<10} positive: {reference['positive_histogram_literal_wang']}")
    print(f"  {'Step1 A3':<10} positive: {reference['positive_histogram_repository']}")
    for name in sorted(report["histograms_by_cell"]):
        print(f"  {name:<10} positive: {report['histograms_by_cell'][name]['positive']}")
    for name in sorted(report["histograms_by_cell"]):
        print(f"  {name:<10} negative: {report['histograms_by_cell'][name]['negative']}")

    print()
    print("-" * 100)
    print("PRIMARY CAUSAL CHAIN")
    print("-" * 100)
    for link in verdict["causal_chain"]:
        mark = "n/a " if link["passed"] is None else ("PASS" if link["passed"] else "FAIL")
        tag = "required" if link["required"] else "informational"
        print(f"  {mark}  link {link['link']} ({tag}): {link['requirement']}")
    for label in (
        "D00_vs_step1_a1",
        "D11_vs_step1_a3",
        "D01_vs_step1_a3",
        "D10_vs_step1_a1",
    ):
        check = verdict["endpoints"][label]
        print(
            f"    {label}: raw {check['raw']:.10f} vs {check['expected_raw']:.10f}  "
            f"|delta| {check['raw_delta']:.3e} <= {check['bound']:.0e}: "
            f"{check['raw_within_bound']}; rounded {check['rounded_matches']}; "
            f"bucket {check['bucket_matches']}"
        )
    forward = verdict["forward_path"]
    print(
        f"    forward path: bucket disagreements "
        f"{forward['bucket_disagreements']}, bit-identical "
        f"{forward['bit_identical']}, max |delta| "
        f"{forward['max_absolute_delta']:.3e}"
    )
    movement = verdict["histogram_movement"]
    print(
        f"    positive histogram movement reproduced: "
        f"{movement['movement_reproduced']}"
    )
    print(f"    probe bucket flip reproduced: {verdict['probe_bucket_flip_reproduced']}")

    print()
    print("=" * 100)
    print(f"PRIMARY VERDICT: {verdict['headline']}")
    print("=" * 100)
    print(f"  numerical_difference:  {verdict['numerical_difference']}")
    print(f"  bse_decision_impact:   {verdict['bse_decision_impact']}")
    print(f"  causal_candidate:      {verdict['causal_candidate']}")
    print()
    print(
        f"  extraction path: moves raw score "
        f"{verdict['extraction_path']['moves_raw_score']}, changes NBC bucket "
        f"{verdict['extraction_path']['changes_nbc_bucket']}"
    )
    print(
        f"  forward path:    moves raw score "
        f"{verdict['forward_path']['moves_raw_score']}, changes NBC bucket "
        f"{verdict['forward_path']['changes_nbc_bucket']}"
    )

    if report.get("shape_and_scaling"):
        print()
        print("-" * 100)
        print("SUB-DIAGNOSTIC: softmax shape vs scaling order (no extra forward calls)")
        print("-" * 100)
        for name, entry in sorted(report["shape_and_scaling"].items()):
            row = entry["probe_pair"]
            summary = entry["summary"]
            print(f"  forward row {name}, probe pair:")
            print(f"    p_wang_shape        {row['p_wang_shape']!r}")
            print(f"    p_repo_shape        {row['p_repo_shape']!r}")
            print(
                f"    probability delta   {row['probability_delta']:.6e}"
                f"   bit-identical: {row['probabilities_bit_identical']}"
            )
            print(f"    scale_after         {row['scale_after']!r}")
            print(f"    scale_inside        {row['scale_inside']!r}")
            print(
                f"    scaling-only delta  {row['scaling_only_delta']:.6e}"
                f"   bit-identical: {row['scaling_order_bit_identical']}"
            )
            print(
                f"    across all pairs: softmax shape bit-identical on "
                f"{summary['softmax_shape']['bit_identical_pairs']}/"
                f"{summary['pairs']}; scaling order bit-identical on "
                f"{summary['scaling_order']['bit_identical_pairs']}/"
                f"{summary['pairs']}"
            )
            print(f"    {entry['interpretation']['message']}")
    print()
    print(f"  {verdict['reporting_note']}")


def format_deltas(deltas):
    def show(value):
        return "n/a" if value is None else f"{value:.6e}"

    return (
        f"signed_mean={show(deltas['signed_mean'])}  "
        f"abs_mean={show(deltas['absolute_mean'])}  "
        f"p50={show(deltas['p50'])}  p95={show(deltas['p95'])}  "
        f"p99={show(deltas['p99'])}  max={show(deltas['max_absolute'])}"
    )


def print_comparison(block):
    print()
    print("-" * 100)
    print(f"{block['comparison']}    isolates: {block['isolated_factor'] or 'n/a'}")
    print("-" * 100)
    print(f"  {format_deltas(block['deltas'])}")
    print(f"  difference class:            {block['difference_class']}")
    print(
        f"  one-decimal disagreements:   "
        f"{block['one_decimal_disagreements']['count']}/{block['pairs']}"
    )
    print(
        f"  NBC bucket disagreements:    "
        f"{block['nbc_bucket_disagreements']['count']}/{block['pairs']}"
    )
    for example in block["nbc_bucket_disagreements"]["examples"]:
        print(
            f"      pair {example['index']}: {example['left']!r} (bucket "
            f"{example['left_bucket']}) -> {example['right']!r} (bucket "
            f"{example['right_bucket']})"
        )
    print(f"  numerical_difference:        {block['numerical_difference']}")
    print(f"  bse_decision_impact:         {block['bse_decision_impact']}")
    print(f"  causal_candidate:            {block['causal_candidate']}")
    print(f"  positive histogram (left):   {block['histograms']['left']['positive']}")
    print(f"  positive histogram (right):  {block['histograms']['right']['positive']}")
    print(f"  negative histogram (left):   {block['histograms']['left']['negative']}")
    print(f"  negative histogram (right):  {block['histograms']['right']['negative']}")


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("GATE 1 DIAGNOSTIC -- STEP 2: SCORING-PATH DECOMPOSITION")
    print("=" * 100)

    pairs, polarities, n_positive, n_negative = load_pairs(args.data_root)
    print(f"  NBC pairs: {len(pairs)} ({n_positive} positive, {n_negative} negative)")

    specs = [spec for spec in ARM_SPECS if not (args.skip_c4 and spec["name"] == "C4")]
    primary_calls = len(pairs) * 4  # F0, F1 and one determinism repeat of each
    secondary_calls = 0 if args.skip_secondary else len(pairs) * (len(specs) + 1)
    print(
        "  PRIMARY  2x2 forward path x extraction path: cells "
        + ", ".join(cell["name"] for cell in CELL_SPECS)
        + f" -> {primary_calls} forward calls"
    )
    print(
        "           (one forward per row serves both extractions, plus a "
        "determinism repeat of each row)"
    )
    if args.skip_secondary:
        print("  SECONDARY forward-path factorial: skipped (--skip-secondary)")
    else:
        print(
            "  SECONDARY forward-path factorial: arms "
            + ", ".join(spec["name"] for spec in specs)
            + f" + {DETERMINISM_CONTROL['name']} -> {secondary_calls} forward calls"
        )
    print(f"  total forward calls: {primary_calls + secondary_calls} at batch size 1")

    if args.dry_run:
        print("\n--dry-run: pairs loaded only. No model loaded, no scoring performed.")
        print("=" * 100)
        return 0

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import collect_live_environment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    environment = collect_live_environment(
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        repo_root=PROJECT_ROOT,
        hash_all=args.hash_weights,
    )
    model_block = environment.get("model") or {}
    if model_block.get("training_mode"):
        raise RuntimeError("model.eval() did not take effect; refusing to score in train mode")

    print("Tokenizing once; the identical tensors are reused by every arm...")
    encodings = tokenize_once(tokenizer, pairs, device)
    token_type_ids_available = "token_type_ids" in encodings[0]
    if not token_type_ids_available:
        specs = [spec for spec in specs if spec["name"] != "C4"]
        print("  tokenizer emits no token_type_ids; the C4 arm is skipped")

    # ---------------- PRIMARY ----------------
    primary = run_primary_decomposition(model, encodings, polarities, torch, args)
    print_primary(primary)
    expectation = secondary_expectation(primary["verdict"])

    if args.skip_secondary:
        report = {
            "step": "2-scoring-path-decomposition",
            "purpose": PURPOSE,
            "protocol": build_protocol(pairs, n_positive, n_negative, model_block,
                                       token_type_ids_available),
            "environment": environment,
            "environment_summary": environment_summary(args, environment, model_block),
            "reference_observation": REFERENCE_OBSERVATION,
            "primary": primary,
            "secondary": None,
            "secondary_expectation": expectation,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nSecondary factorial skipped (--skip-secondary).")
        print(f"\nWritten to {output_path}")
        print("This diagnostic reports only. No issue found here has been fixed.")
        print("=" * 100)
        return 0

    print()
    print("=" * 100)
    print("SECONDARY DECOMPOSITION: forward-path factorial (C0-C4)")
    print("=" * 100)
    print(f"  {expectation}")

    scores_by_arm = {}
    timings = {}
    for spec in specs:
        print()
        print(f"Arm {spec['name']} ({spec['label']}): {spec['call']}")
        scores, elapsed = score_forward_arm(model, encodings, spec, torch)
        scores_by_arm[spec["name"]] = scores
        timings[spec["name"]] = elapsed

    print()
    print(f"Arm {DETERMINISM_CONTROL['name']} ({DETERMINISM_CONTROL['label']})")
    control_scores, control_elapsed = score_forward_arm(
        model, encodings, DETERMINISM_CONTROL, torch
    )
    scores_by_arm[DETERMINISM_CONTROL["name"]] = control_scores
    timings[DETERMINISM_CONTROL["name"]] = control_elapsed

    available = set(scores_by_arm)
    matrix = {}
    for left, right in COMPARISON_PAIRS:
        if left in available and right in available:
            block = compare_forward_arms(
                left, right, scores_by_arm[left], scores_by_arm[right], polarities
            )
            matrix[block["comparison"]] = block

    control_block = compare_forward_arms(
        "C0", "C0R", scores_by_arm["C0"], scores_by_arm["C0R"], polarities
    )

    c4_block = None
    if "C4" in available and "C3" in available:
        c4_block = compare_forward_arms(
            "C3", "C4", scores_by_arm["C3"], scores_by_arm["C4"], polarities
        )

    probe = probe_report(args.probe_polarity, args.probe_index, scores_by_arm, polarities)
    verdict = build_verdict(matrix, probe, control_block, c3_vs_c4=c4_block)

    secondary = {
        "decomposition": "forward-path factorial",
        "expectation_given_primary": expectation,
        "arms": {
            spec["name"]: dict(spec, elapsed_seconds=timings.get(spec["name"]))
            for spec in list(specs) + [DETERMINISM_CONTROL]
        },
        "nbc_laplace_histograms_by_arm": {
            name: histograms_by_polarity(scores, polarities)
            for name, scores in scores_by_arm.items()
        },
        "comparisons": matrix,
        "determinism_control": control_block,
        "c3_vs_c4": c4_block,
        "probe": probe,
        "verdict": verdict,
    }

    report = {
        "step": "2-scoring-path-decomposition",
        "purpose": PURPOSE,
        "protocol": build_protocol(
            pairs, n_positive, n_negative, model_block, token_type_ids_available
        ),
        "environment": environment,
        "environment_summary": environment_summary(args, environment, model_block),
        "reference_observation": REFERENCE_OBSERVATION,
        "primary": primary,
        "secondary": secondary,
        "secondary_expectation": expectation,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print()
    print("-" * 100)
    print("DETERMINISM CONTROL (C0 vs C0 repeated)")
    print("-" * 100)
    print(f"  {format_deltas(control_block['deltas'])}")
    print(f"  bit-identical: {control_block['numerical_difference'] == 'NO'}")
    if control_block["numerical_difference"] != "NO":
        print(
            "  *** The forward pass is NOT deterministic on this device. No factor\n"
            "      attribution below is sound. ***"
        )

    for block in matrix.values():
        print_comparison(block)

    if c4_block is not None:
        print()
        print("-" * 100)
        print(
            "C3 vs C4 -- REQUIRED LINK 4 (type_vocab_size is 0 predicts inert; "
            "this measures it)"
        )
        print("-" * 100)
        print(f"  C3 vs C4: {format_deltas(c4_block['deltas'])}")
        print(f"  numerical_difference: {c4_block['numerical_difference']}")
        print(f"  bse_decision_impact:  {c4_block['bse_decision_impact']}")

    print()
    print("-" * 100)
    print(
        f"PROBE: {probe['polarity']} NBC pair {probe['index']} "
        f"(flat index {probe['flat_index']})"
    )
    print("-" * 100)
    print(f"  {'arm':<8}{'raw':>24}{'rounded':>12}{'bucket':>10}")
    reference = REFERENCE_OBSERVATION
    print(
        f"  {'Step1 A1':<8}{reference['literal_wang']['raw']:>24.10f}"
        f"{reference['literal_wang']['rounded']:>12}"
        f"{reference['literal_wang']['nbc_bucket']:>10}   (literal Wang, recorded)"
    )
    print(
        f"  {'Step1 A3':<8}{reference['repository']['raw']:>24.10f}"
        f"{reference['repository']['rounded']:>12}"
        f"{reference['repository']['nbc_bucket']:>10}   (repository, recorded)"
    )
    for name in sorted(probe["arms"]):
        row = probe["arms"][name]
        print(
            f"  {name:<8}{row['raw']:>24.10f}{row['rounded_one_decimal']:>12}"
            f"{row['nbc_bucket']:>10}"
        )

    print_secondary_histograms(report)

    reproduction = verdict["reference_reproduction"]
    print()
    print("-" * 100)
    print("CAUSAL CHAIN")
    print("-" * 100)
    for link in reproduction["causal_chain"]:
        if link["passed"] is None:
            mark = "n/a "
        else:
            mark = "PASS" if link["passed"] else "FAIL"
        tag = "required" if link["required"] else "informational"
        print(f"  {mark}  link {link['link']} ({tag}): {link['requirement']}")
    rows = [
        (reproduction["c0_vs_step1_a1"], "C0 vs Step 1 A1", ""),
        (reproduction["bridge_vs_step1_a3"], "C4 vs Step 1 A3", ""),
        (
            reproduction["c3_vs_step1_a3_informational"],
            "C3 vs Step 1 A3",
            "  (informational: exposes a token_type_ids-driven result)",
        ),
    ]
    for check, label, suffix in rows:
        if check is None:
            continue
        print(
            f"    {label}: raw {check['raw']:.10f} vs {check['expected_raw']:.10f}  "
            f"|delta| {check['raw_delta']:.3e} <= {check['bound']:.0e}: "
            f"{check['raw_within_bound']}; rounded {check['rounded_matches']}; "
            f"bucket {check['bucket_matches']}{suffix}"
        )
    print(f"    all required links passed: {reproduction['required_links_passed']}")
    for warning in reproduction["warnings"]:
        print(f"  WARNING: {warning}")

    print()
    print("=" * 100)
    print(f"SECONDARY VERDICT: {verdict['headline']}")
    print("=" * 100)
    print(f"  numerical_difference:  {verdict['numerical_difference']}")
    print(f"  bse_decision_impact:   {verdict['bse_decision_impact']}")
    print(f"  causal_candidate:      {verdict['causal_candidate']}")
    print()
    isolation = verdict["factor_isolation"]
    if verdict["causal_attribution_withheld"]:
        print(
            "  Causal attribution is WITHHELD. The unpromoted factor-isolation\n"
            f"  reading was {isolation['headline']}, and the pairwise numbers above\n"
            "  remain valid measurements, but no factor may be named as the cause\n"
            "  until the failed guard is resolved."
        )
    else:
        print(f"  inference_mode contributes:  {isolation['inference_mode_contributes']}")
        print(f"  attention_mask contributes:  {isolation['attention_mask_contributes']}")
    print()
    print(
        f"  explains Step 1 observation: "
        f"{reproduction['explains_reference_observation']}"
    )
    print(f"  {reproduction['message']}")
    print()
    print(f"  {verdict['reporting_note']}")
    print()
    print(f"Written to {output_path}")
    print("This diagnostic reports only. No issue found here has been fixed.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
