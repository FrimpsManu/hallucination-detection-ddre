"""Gate 1 diagnostic, Step 1: A/B/C comparison of three scoring arms.

One question: inside a single environment, with one loaded model, does this
repository's ``EntailmentScorer`` make the same decisions as Wang et al.'s
literal released scorer?

    A1  Literal Wang path, transcribed from released utils.py:40-68.
        batch 1; tokenizer(premise, hypothesis, truncation=True,
        return_tensors="pt"); model(inputs["input_ids"]); softmax(logits[0]/5);
        hardcoded entailment index 0; score * 100; round to one decimal.
    A2  This repository's EntailmentScorer at the production batch size of 8,
        with caching disabled in both directions.
    A3  This repository's EntailmentScorer at batch size 1, caching disabled.

    A1 vs A3 isolates the scorer's argument/inference path, because at batch 1
    padding is a no-op and only the passed arguments differ.
    A3 vs A2 isolates batching and padding numerics, because the argument path
    is identical and only the batch shape differs.

Sample: all 398 released NBC pairs (they are what the histograms are built
from, so any disagreement there propagates into every stopping decision), plus
a seeded fixed sample of 75 retrieved documents expanded into their text spans.

Caching
-------
Both repository arms run with ``use_cache=False, write_cache=False``, and their
scratch sqlite database is created inside a temporary directory that is deleted
on exit. The formal ``results/wang_nli_cache.sqlite`` is never opened. A cached
score would measure the cache rather than the model, which is the one thing
this diagnostic must not do.

Writes only under ``results/diagnostics/``. Changes no baseline behaviour, no
tolerance, no reference value, no split, and no cost parameter.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_diagnostics import (  # noqa: E402
    arm_histograms,
    compare_arms,
    content_digest,
    forward_call_accounting,
    overall_verdict,
    sample_documents,
    token_type_id_assessment,
)

OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_OUTPUT_DIR = "results/diagnostics"
DEFAULT_SEED = 20231215
DEFAULT_N_DOCUMENTS = 75
PRODUCTION_BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gate 1 Step 1: compare the literal Wang scorer against this "
            "repository's EntailmentScorer on a fixed sample, in one environment."
        )
    )
    parser.add_argument("--data-root", default="data/wang")
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument("--n-documents", type=int, default=DEFAULT_N_DOCUMENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=PRODUCTION_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Also hash large checkpoint files when recording checkpoint identity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and write the sample, report its size and the implied number "
            "of forward passes, then exit without loading the model."
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Arm A1: literal transcription of the released scorer.
# --------------------------------------------------------------------------

def wang_entailment_score(premise, hypothesis, tokenizer, model, device):
    """Released utils.py:40-68, verbatim apart from returning the token ids.

    Kept deliberately unfactored, including the hardcoded ``label_names`` order
    and the ``round(..., 1)`` at source, so the transcription can be checked
    against the released file line by line.
    """
    import torch

    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    output = model(inputs["input_ids"].to(device))

    probabilities = torch.softmax(output["logits"][0] / 5, -1).tolist()
    label_names = ["entailment", "neutral", "not_entailment"]
    prediction = {
        name: round(float(pred) * 100, 1) for pred, name in zip(probabilities, label_names)
    }
    entailment_prob = prediction["entailment"]

    # Returned alongside the score so tokenization can be compared directly;
    # the released function does not need them.
    input_ids = [int(x) for x in inputs["input_ids"][0].tolist()]
    raw_entailment = float(probabilities[0]) * 100.0
    return entailment_prob, raw_entailment, input_ids


class RecordingTokenizer:
    """Transparent proxy that records what the production scorer asks for.

    Wrapping the tokenizer rather than overriding ``EntailmentScorer._infer_batch``
    means the recorded call is the real production call, with no duplicated
    tokenization that could drift from ``src/utils.py``. ``src/utils.py`` is not
    modified by this diagnostic.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.calls = []

    def __call__(self, *args, **kwargs):
        outputs = self._tokenizer(*args, **kwargs)
        self.calls.append({"kwargs": {k: v for k, v in kwargs.items()}, "outputs": outputs})
        return outputs

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def unpadded_input_ids(outputs):
    """Recover per-row token ids from a padded batch using the attention mask."""
    ids = outputs["input_ids"]
    mask = outputs.get("attention_mask")
    recovered = []
    for row in range(ids.shape[0]):
        row_ids = [int(x) for x in ids[row].tolist()]
        if mask is None:
            recovered.append(row_ids)
            continue
        row_mask = [int(x) for x in mask[row].tolist()]
        recovered.append([token for token, keep in zip(row_ids, row_mask) if keep])
    return recovered


def run_repository_arm(tokenizer, model, model_name, pairs, batch_size, cache_dir):
    """Run this repository's EntailmentScorer with caching fully disabled."""
    from src.utils import EntailmentScorer

    recorder = RecordingTokenizer(tokenizer)
    scorer = EntailmentScorer(
        recorder,
        model,
        model_name,
        cache_path=str(Path(cache_dir) / f"scratch_bs{batch_size}.sqlite"),
        batch_size=batch_size,
    )
    try:
        start = time.perf_counter()
        scores = scorer.score_pairs(
            pairs,
            use_cache=False,
            write_cache=False,
            show_progress=True,
            description=f"arm batch_size={batch_size}",
        )
        elapsed = time.perf_counter() - start
    finally:
        scorer.close()

    token_ids = []
    emitted_keys = set()
    token_type_values = set()
    token_type_emitted = False
    for call in recorder.calls:
        outputs = call["outputs"]
        emitted_keys.update(outputs.keys())
        if "token_type_ids" in outputs:
            token_type_emitted = True
            token_type_values.update(int(x) for x in outputs["token_type_ids"].flatten().tolist())
        token_ids.extend(unpadded_input_ids(outputs))

    if len(token_ids) != len(pairs):
        raise RuntimeError(
            f"recorded {len(token_ids)} tokenizations for {len(pairs)} pairs; "
            "the tokenizer proxy did not observe every call"
        )

    return {
        "scores": scores,
        "token_ids": token_ids,
        "elapsed_seconds": elapsed,
        # One tokenizer call per batch, so this is the observed number of model
        # forward calls -- reported alongside the estimate rather than assumed.
        "forward_calls": len(recorder.calls),
        "entailment_index": scorer.entailment_index,
        "tokenizer_keys": sorted(emitted_keys),
        "token_type_ids_emitted": token_type_emitted,
        "unique_token_type_ids": sorted(token_type_values),
        "batch_size": batch_size,
    }


def run_wang_arm(tokenizer, model, pairs, device):
    from tqdm import tqdm

    rounded, raw, token_ids = [], [], []
    start = time.perf_counter()
    for premise, hypothesis in tqdm(pairs, desc="arm A1 literal Wang", unit="pair"):
        score, raw_score, ids = wang_entailment_score(
            premise, hypothesis, tokenizer, model, device
        )
        rounded.append(score)
        raw.append(raw_score)
        token_ids.append(ids)
    elapsed = time.perf_counter() - start

    # The comparators re-apply Wang's rounding to the raw score. That is only
    # legitimate if it reproduces the value the released function itself
    # returns, so it is checked rather than assumed.
    from src.scoring_diagnostics import round_one_decimal as _round

    mismatched = [i for i, (r, x) in enumerate(zip(rounded, raw)) if r != _round(x)]
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} A1 score(s) disagree with re-rounding the raw value; "
            "the transcription and the comparator are not applying the same rounding"
        )

    return {
        "scores": raw,
        "rounded_scores": rounded,
        "token_ids": token_ids,
        "elapsed_seconds": elapsed,
        "forward_calls": len(pairs),
        "batch_size": 1,
    }


# --------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------

def build_sample(data_root, n_documents, seed):
    """Assemble the frozen sample and the flat ordered pair list scored by all arms.

    Every arm scores the identical ordered list, so index i means the same
    (premise, hypothesis) pair everywhere and the comparisons are elementwise.
    Segmentation is done once, with this repository's ``split_text``; span
    deduplication is already pinned as decision-equivalent by
    ``tests/test_baseline_fidelity.py`` and is not what is under test here.
    """
    from src.utils import split_text
    from src.wang_data import load_nbc_pairs, load_sentence_records

    positive, negative = load_nbc_pairs(data_root, per_class=None)
    records = load_sentence_records(data_root, strict=True)
    documents = sample_documents(records, n_documents, seed)

    pairs = []
    units = []

    for polarity, items in (("positive", positive), ("negative", negative)):
        for index, item in enumerate(items):
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            units.append(
                {
                    "kind": "nbc",
                    "polarity": polarity,
                    "index": index,
                    "pair_index": len(pairs),
                    "digest": content_digest(premise, hypothesis),
                }
            )
            pairs.append((premise, hypothesis))

    document_units = []
    for document in documents:
        spans = split_text(document["page_content"], 400, 100)
        span_indices = []
        for span in spans:
            span_indices.append(len(pairs))
            pairs.append((span, document["subclaim_text"]))
        document_units.append(
            {
                "kind": "document",
                "passage_index": document["passage_index"],
                "sentence_index": document["sentence_index"],
                "subclaim_index": document["subclaim_index"],
                "document_index": document["document_index"],
                "url": document["url"],
                "subclaim_digest": content_digest(document["subclaim_text"]),
                "page_content_digest": content_digest(document["page_content"]),
                "page_content_words": len(str(document["page_content"]).split()),
                "span_count": len(spans),
                "pair_indices": span_indices,
            }
        )

    position_histogram = {}
    for document in document_units:
        key = str(document["document_index"])
        position_histogram[key] = position_histogram.get(key, 0) + 1

    sample = {
        "seed": seed,
        "data_root": data_root,
        "nbc": {
            "positive_pairs": len(positive),
            "negative_pairs": len(negative),
            "total_pairs": len(positive) + len(negative),
        },
        "documents": {
            "requested": n_documents,
            "selected": len(document_units),
            "total_available": sum(
                len(subclaim.documents) for r in records for subclaim in r.subclaims
            ),
            "total_spans": sum(d["span_count"] for d in document_units),
            "retrieval_position_histogram": dict(sorted(position_histogram.items(), key=lambda kv: int(kv[0]))),
        },
        "corpus": {
            "sentences": len(records),
            "passages": len({r.passage_index for r in records}),
            "subclaims": sum(len(r.subclaims) for r in records),
        },
        "total_scored_pairs": len(pairs),
        "nbc_units": units,
        "document_units": document_units,
    }
    return sample, pairs, units, document_units



# --------------------------------------------------------------------------

def format_delta_row(label, deltas):
    def show(value):
        return "n/a" if value is None else f"{value:.6e}"

    return (
        f"  {label:<22}"
        f"signed_mean={show(deltas['signed_mean'])}  "
        f"abs_mean={show(deltas['absolute_mean'])}  "
        f"p50={show(deltas['p50'])}  p95={show(deltas['p95'])}  "
        f"p99={show(deltas['p99'])}  max={show(deltas['max_absolute'])}  "
        f">1e-4: {deltas['count_above_threshold']}/{deltas['count']}"
    )


def print_comparison(block):
    print()
    print("-" * 100)
    print(f"{block['comparison']}   ->  {block['classification']['status']}")
    print("-" * 100)
    tokens = block["token_ids"]
    print(
        f"  token input_ids exact match: {tokens['exact_matches']}/{tokens['pairs']} "
        f"({100 * tokens['exact_match_rate']:.4f}%)   mismatches: {tokens['mismatch_count']}"
    )
    for mismatch in tokens["mismatches"]:
        print(
            f"    pair {mismatch['index']}: lengths {mismatch['left_length']} vs "
            f"{mismatch['right_length']}, first divergence at {mismatch['first_divergence']}"
        )
    print(format_delta_row("all pairs", block["score_deltas_all_pairs"]))
    print(format_delta_row("NBC pairs", block["nbc"]["deltas"]))
    print(format_delta_row("document spans", block["spans"]["deltas"]))
    print(
        f"  one-decimal disagreements:   {block['one_decimal_disagreements']['count']}"
        f" / {block['score_deltas_all_pairs']['count']}"
    )
    documents = block["documents"]
    print()
    print("  decision-level (what a BSE update actually consumes):")
    print(
        f"    NBC bucket disagreements:            "
        f"{block['nbc']['bucket_disagreements']['count']}/{block['nbc']['pairs']}"
    )
    print(
        f"    document-MAX bucket disagreements:   "
        f"{documents['max_score_bucket_disagreements']['count']}/{documents['documents']}"
    )
    print("  diagnostic only (not decision-level):")
    print(
        f"    span bucket disagreements:           "
        f"{block['spans']['bucket_disagreements']['count']}/{block['spans']['pairs']}"
        "   <- a non-maximal span can cross a bucket with no downstream effect"
    )
    print(
        f"    document-MAX one-decimal changes:    "
        f"{documents['max_score_one_decimal_disagreements']['count']}/{documents['documents']}"
    )
    print(
        f"    argmax span disagreements:           "
        f"{documents['argmax_span_disagreements']['count']}/{documents['documents']}"
    )
    print()
    for reason in block["classification"]["reasons"]:
        print(f"  - {reason}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample.json"
    result_path = output_dir / "step1_scorer_ab.json"

    print("=" * 100)
    print("GATE 1 DIAGNOSTIC -- STEP 1: A/B/C SCORER COMPARISON")
    print("=" * 100)
    print("Building the fixed sample...")
    sample, pairs, units, document_units = build_sample(
        args.data_root, args.n_documents, args.seed
    )
    print(f"  NBC pairs:            {sample['nbc']['total_pairs']}")
    print(
        f"  documents sampled:    {sample['documents']['selected']} of "
        f"{sample['documents']['total_available']} (seed {args.seed})"
    )
    print(f"  document spans:       {sample['documents']['total_spans']}")
    accounting = forward_call_accounting(sample["total_scored_pairs"], args.batch_size)
    sample["cost_accounting"] = accounting
    with sample_path.open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, indent=2)
    calls = accounting["forward_calls"]
    print(f"  scored pairs per arm: {accounting['pairs']}")
    print(
        f"  pair evaluations:     {accounting['pair_evaluations']} "
        f"(= 3 x {accounting['pairs']}, every arm scores every pair)"
    )
    print(
        f"  model forward calls:  ~{calls['total']}  "
        f"(A1 {calls['A1']} + A3 {calls['A3']} at batch 1, "
        f"A2 {calls['A2']} at batch {accounting['production_batch_size']})"
    )
    print(f"  sample written to     {sample_path}")

    if args.dry_run:
        print()
        print("--dry-run: sample only. No model loaded, no scoring performed.")
        print("=" * 100)
        return 0

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.diagnostic_probe import collect_live_environment
    from src.utils import SCORE_VERSION

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)

    environment = collect_live_environment(
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        repo_root=PROJECT_ROOT,
        score_version=SCORE_VERSION,
        hash_all=args.hash_weights,
    )

    with tempfile.TemporaryDirectory(prefix="gate1-diagnostic-") as cache_dir:
        print("\nArm A1: literal Wang scorer (batch 1)")
        a1 = run_wang_arm(tokenizer, model, pairs, device)
        print(f"\nArm A3: repository scorer at batch size 1")
        a3 = run_repository_arm(tokenizer, model, args.model_name, pairs, 1, cache_dir)
        print(f"\nArm A2: repository scorer at batch size {args.batch_size}")
        a2 = run_repository_arm(
            tokenizer, model, args.model_name, pairs, args.batch_size, cache_dir
        )

    a1_vs_a3 = compare_arms("A1 (literal Wang) vs A3 (repository, batch 1)", a1, a3, units, document_units)
    a3_vs_a2 = compare_arms(
        f"A3 (repository, batch 1) vs A2 (repository, batch {args.batch_size})",
        a3,
        a2,
        units,
        document_units,
    )
    a1_vs_a2 = compare_arms(
        f"A1 (literal Wang) vs A2 (repository, batch {args.batch_size})",
        a1,
        a2,
        units,
        document_units,
    )
    verdict = overall_verdict(a1_vs_a3, a3_vs_a2, a1_vs_a2)

    token_type = token_type_id_assessment(
        (environment.get("model") or {}).get("type_vocab_size"),
        a2["token_type_ids_emitted"],
        a2["unique_token_type_ids"],
    )

    histograms = {
        "A1": arm_histograms(a1["scores"], units),
        "A2": arm_histograms(a2["scores"], units),
        "A3": arm_histograms(a3["scores"], units),
    }
    histograms["identical_across_arms"] = (
        histograms["A1"] == histograms["A2"] == histograms["A3"]
    )

    result = {
        "step": "1-scorer-ab",
        "purpose": (
            "Determine whether this repository's scorer differs materially from "
            "Wang's literal released scorer under the same environment."
        ),
        "sample_path": str(sample_path),
        "sample_summary": {
            key: sample[key]
            for key in ("seed", "nbc", "documents", "corpus", "total_scored_pairs")
        },
        "cost_accounting": {
            "estimated": accounting,
            "observed_forward_calls": {
                "A1": a1["forward_calls"],
                "A3": a3["forward_calls"],
                "A2": a2["forward_calls"],
                "total": a1["forward_calls"] + a3["forward_calls"] + a2["forward_calls"],
            },
            "note": (
                "Pair evaluations are 3 x pairs because every arm scores every "
                "pair. Forward calls are fewer, because A2 batches."
            ),
        },
        "environment": environment,
        "arms": {
            "A1": {
                "description": "literal Wang get_entailment_score, released utils.py:40-68",
                "batch_size": 1,
                "caching": "not applicable",
                "elapsed_seconds": a1["elapsed_seconds"],
                "forward_calls": a1["forward_calls"],
            },
            "A2": {
                "description": "EntailmentScorer at production batch size",
                "batch_size": a2["batch_size"],
                "caching": "use_cache=False, write_cache=False, temporary database",
                "elapsed_seconds": a2["elapsed_seconds"],
                "forward_calls": a2["forward_calls"],
                "entailment_index": a2["entailment_index"],
                "tokenizer_keys": a2["tokenizer_keys"],
                "token_type_ids_emitted": a2["token_type_ids_emitted"],
                "unique_token_type_ids": a2["unique_token_type_ids"],
            },
            "A3": {
                "description": "EntailmentScorer at batch size 1",
                "batch_size": a3["batch_size"],
                "caching": "use_cache=False, write_cache=False, temporary database",
                "elapsed_seconds": a3["elapsed_seconds"],
                "forward_calls": a3["forward_calls"],
                "entailment_index": a3["entailment_index"],
                "tokenizer_keys": a3["tokenizer_keys"],
            },
        },
        "nbc_laplace_histograms": histograms,
        "token_type_id_assessment": token_type,
        "comparisons": {
            "a1_vs_a3": a1_vs_a3,
            "a3_vs_a2": a3_vs_a2,
            "a1_vs_a2": a1_vs_a2,
        },
        "verdict": verdict,
    }

    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    for block in (a1_vs_a3, a3_vs_a2, a1_vs_a2):
        print_comparison(block)

    print()
    print("-" * 100)
    print("NBC LAPLACE-SMOOTHED HISTOGRAMS PER ARM")
    print("-" * 100)
    for arm in ("A1", "A3", "A2"):
        print(f"  {arm} positive: {histograms[arm]['positive']}")
        print(f"  {arm} negative: {histograms[arm]['negative']}")
    print(f"  identical across all three arms: {histograms['identical_across_arms']}")

    print()
    print("-" * 100)
    print("token_type_ids ASSESSMENT")
    print("-" * 100)
    print(f"  type_vocab_size:        {token_type['type_vocab_size']}")
    print(f"  emitted by tokenizer:   {token_type['token_type_ids_emitted']}")
    print(f"  unique values observed: {token_type['unique_token_type_ids']}")
    print(f"  can differ from Wang:   {token_type['can_differ_from_wang']}")
    print(f"  {token_type['note']}")

    observed = result["cost_accounting"]["observed_forward_calls"]
    print()
    print("-" * 100)
    print("COST ACCOUNTING")
    print("-" * 100)
    print(
        f"  pair evaluations:     {accounting['pair_evaluations']} "
        f"(= 3 x {accounting['pairs']})"
    )
    print(
        f"  model forward calls:  {observed['total']} observed "
        f"(A1 {observed['A1']}, A3 {observed['A3']}, A2 {observed['A2']} at batch "
        f"{accounting['production_batch_size']}); estimated "
        f"{accounting['forward_calls']['total']}"
    )

    print()
    print("=" * 100)
    print(f"VERDICT: {verdict['headline']}")
    print("=" * 100)
    print(f"  A1 vs A3 (argument path):  {verdict['a1_vs_a3']}")
    print(f"  A3 vs A2 (batching):       {verdict['a3_vs_a2']}")
    print(f"  A1 vs A2 (end to end):     {verdict['a1_vs_a2']}")
    print(f"  {verdict['conclusion']}")
    print()
    print(f"Written to {result_path}")
    print("This diagnostic reports only. No issue found here has been fixed.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
