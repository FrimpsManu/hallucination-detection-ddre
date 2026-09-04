"""Gate 1 diagnostic, Step 2: which forward-path argument explains A1 vs A3?

Step 1 established that the literal Wang scorer and this repository's scorer
tokenize identically (589/589 input_ids matched) yet disagreed on exactly one
BSE decision: positive NBC pair 169, bucket 1 under the literal path and bucket
2 under ours. ``token_type_ids`` are excluded as a cause because
``type_vocab_size`` is 0. Two candidate differences remain:
``torch.inference_mode()`` and passing ``attention_mask``.

This script runs a controlled factorial over exactly those two factors:

    C0  model(input_ids)                                     reference
    C1  inference_mode: model(input_ids)                     +inference_mode
    C2  model(input_ids, attention_mask=...)                 +attention_mask
    C3  inference_mode: model(input_ids, attention_mask=...) both
    C4  inference_mode: + token_type_ids                     confirmation only
    C0R C0 again, last                                       determinism control

Controls that make the result attributable:

* the tokenizer and model are loaded once;
* every pair is tokenized ONCE and the identical tensors are reused by every
  arm, so tokenization cannot vary;
* the score extraction is byte-identical across arms, so only the forward call
  differs;
* batch size is exactly 1 everywhere, so padding is never involved;
* C0R re-runs C0 unchanged. If C0 and C0R differ, the device is
  nondeterministic and no attribution in the report is sound.

C4 is confirmation only: ``type_vocab_size`` is 0, so no token-type embedding
exists and the argument is expected to be inert. It is included to demonstrate
that, not to test a live hypothesis.

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
            "Gate 1 Step 2: factorial isolation of torch.inference_mode() and "
            "attention_mask over all 398 released NBC pairs."
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
    print("GATE 1 DIAGNOSTIC -- STEP 2: FORWARD-PATH FACTOR ISOLATION")
    print("=" * 100)

    pairs, polarities, n_positive, n_negative = load_pairs(args.data_root)
    print(f"  NBC pairs: {len(pairs)} ({n_positive} positive, {n_negative} negative)")

    specs = [spec for spec in ARM_SPECS if not (args.skip_c4 and spec["name"] == "C4")]
    print(f"  arms: {', '.join(spec['name'] for spec in specs)} + "
          f"{DETERMINISM_CONTROL['name']} (determinism control)")
    print(f"  forward calls: {len(pairs) * (len(specs) + 1)} at batch size 1")

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
        print("  tokenizer emits no token_type_ids; the C4 confirmation arm is skipped")

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

    report = {
        "step": "2-forward-path-isolation",
        "purpose": (
            "Isolate whether torch.inference_mode() or passing attention_mask "
            "explains the single BSE decision-level disagreement Step 1 found "
            "between the literal Wang scorer and this repository's scorer."
        ),
        "protocol": {
            "nbc_pairs": len(pairs),
            "positive_pairs": n_positive,
            "negative_pairs": n_negative,
            "batch_size": 1,
            "entailment_index": ENTAILMENT_INDEX,
            "softmax_temperature": SOFTMAX_TEMPERATURE,
            "tokenized_once": True,
            "tensors_shared_across_arms": True,
            "score_extraction_identical_across_arms": True,
            "model_eval_mode": not model_block.get("training_mode", True),
            "cache_reads": 0,
            "cache_writes": 0,
            "token_type_ids_available": token_type_ids_available,
        },
        "environment": environment,
        "environment_summary": {
            "python_version": (environment.get("python") or {}).get("version"),
            "torch_version": (environment.get("libraries") or {}).get("torch"),
            "transformers_version": (environment.get("libraries") or {}).get("transformers"),
            "tokenizers_version": (environment.get("libraries") or {}).get("tokenizers"),
            "device": (environment.get("device") or {}).get("selected_device"),
            "gpu_name": (environment.get("device") or {}).get("gpu_name"),
            "model_name": args.model_name,
            "model_dtype": model_block.get("dtype"),
            "model_training_mode": model_block.get("training_mode"),
            "attn_implementation": model_block.get("attn_implementation"),
            "type_vocab_size": model_block.get("type_vocab_size"),
            "resolved_hf_revision": (environment.get("checkpoint_identity") or {}).get(
                "resolved_revision"
            ),
            "git_commit": (environment.get("git") or {}).get("commit"),
        },
        "arms": {
            spec["name"]: dict(spec, elapsed_seconds=timings.get(spec["name"]))
            for spec in list(specs) + [DETERMINISM_CONTROL]
        },
        "reference_observation": REFERENCE_OBSERVATION,
        "histograms_by_arm": {
            name: histograms_by_polarity(scores, polarities)
            for name, scores in scores_by_arm.items()
        },
        "comparisons": matrix,
        "determinism_control": control_block,
        "c4_confirmation": c4_block,
        "probe": probe,
        "verdict": verdict,
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

    print()
    print("-" * 100)
    print("LAPLACE-SMOOTHED NBC HISTOGRAMS BY ARM")
    print("-" * 100)
    print(f"  {'Step1 A1':<8} positive: {reference['positive_histogram_literal_wang']}")
    print(f"  {'Step1 A3':<8} positive: {reference['positive_histogram_repository']}")
    for name in sorted(report["histograms_by_arm"]):
        histogram = report["histograms_by_arm"][name]
        print(f"  {name:<8} positive: {histogram['positive']}")
    for name in sorted(report["histograms_by_arm"]):
        histogram = report["histograms_by_arm"][name]
        print(f"  {name:<8} negative: {histogram['negative']}")

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
    print(f"VERDICT: {verdict['headline']}")
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
