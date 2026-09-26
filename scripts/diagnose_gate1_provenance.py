"""Gate 1 diagnostic, Step 0: read the provenance the formal run already recorded.

This costs no NLI compute. It reads ``results/wang_reproduction.json`` and
prints the environment, checkpoint, and NBC-histogram facts that decide which
hypotheses about the Gate 1 discrepancy are still alive.

Several fields the diagnosis needs are not in the existing provenance block --
the tokenizers version, the model's parameter dtype, its training mode, its
attention implementation, and the tokenizer's ``is_fast`` flag. Those are
reported as NOT RECORDED rather than guessed, and a live probe (on by default,
skippable with ``--no-live``) fills them in from the current environment.

The live probe also settles the token_type_ids question, which cannot be
settled from ``type_vocab_size`` alone: it reports the keys the tokenizer
emits, whether token_type_ids appear, and their actual values.

Read-only. Writes one file, under ``results/diagnostics/`` only. Loads no
cache, touches no cache, reruns no part of Gate 1.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostic_probe import (  # noqa: E402
    collect_live_environment,
    select_device,
)
from src.scoring_diagnostics import token_type_id_assessment  # noqa: E402


DEFAULT_REPORT = "results/wang_reproduction.json"
DEFAULT_OUTPUT = "results/diagnostics/step0_provenance.json"
OFFICIAL_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

NOT_RECORDED = "NOT RECORDED"

# Fields the diagnosis wants, and where each lives in a Gate 1 report. A dotted
# path of None means the formal provenance block does not carry it at all.
REPORT_FIELDS = (
    ("python_version", "provenance.python.version"),
    ("python_platform", "provenance.python.platform"),
    ("torch_version", "provenance.libraries.torch"),
    ("transformers_version", "provenance.libraries.transformers"),
    ("tokenizers_version", None),
    ("sentencepiece_version", None),
    ("numpy_version", "provenance.libraries.numpy"),
    ("scipy_version", "provenance.libraries.scipy"),
    ("sklearn_version", "provenance.libraries.sklearn"),
    ("device", "provenance.runtime.device"),
    ("batch_size", "provenance.runtime.batch_size"),
    ("model_dtype", None),
    ("model_id2label", "provenance.nli_model.model_id2label"),
    ("model_type_vocab_size", "provenance.nli_model.model_type_vocab_size"),
    ("model_training_mode", None),
    ("model_attn_implementation", None),
    ("tokenizer_class", "provenance.nli_model.tokenizer_class"),
    ("tokenizer_model_max_length", "provenance.nli_model.tokenizer_model_max_length"),
    ("tokenizer_is_fast", None),
    ("score_version", "provenance.runtime.score_version"),
    ("git_commit", "provenance.repository.commit"),
    ("git_branch", "provenance.repository.branch"),
    ("git_dirty", "provenance.repository.dirty"),
    ("nli_model_name", "provenance.nli_model.model_name"),
    # Checkpoint identity. The formal v2 Gate report records none of these --
    # that is exactly why this diagnostic exists -- so their report path is
    # None and they appear in fields_not_recorded_in_report. Supplying them
    # live cannot establish that the historical cache was produced at this
    # revision; see checkpoint_identity_limitation below.
    ("resolved_revision", None),
    ("model_config_commit_hash", None),
    ("tokenizer_commit_hash", None),
    ("wang_source_commit", "provenance.wang_data.source_commit"),
    ("nbc_histogram_positive", "nbc_histograms_laplace_smoothed.positive"),
    ("nbc_histogram_negative", "nbc_histograms_laplace_smoothed.negative"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gate 1 Step 0: inspect the provenance of an existing reproduction "
            "report and probe the current environment for the fields it omits."
        )
    )
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--model-name", default=OFFICIAL_MODEL)
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip the live probe. Reports only what the report file contains.",
    )
    parser.add_argument(
        "--no-load-model",
        action="store_true",
        help=(
            "Probe library versions and the tokenizer but do not download or "
            "load model weights."
        ),
    )
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Also hash large checkpoint files (slow; small files are always hashed).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


def read_report(path):
    report_path = Path(path)
    if not report_path.exists():
        return None, f"{report_path} does not exist"
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def extract_from_report(payload):
    fields = {}
    missing = []
    for name, dotted in REPORT_FIELDS:
        if dotted is None:
            fields[name] = None
            missing.append(name)
            continue
        value, found = dig(payload, dotted)
        fields[name] = value
        if not found:
            missing.append(name)
    return fields, missing


def live_fields(snapshot):
    """Map the live snapshot onto the same field names as the report."""
    model = snapshot.get("model") or {}
    tokenizer = snapshot.get("tokenizer") or {}
    libraries = snapshot.get("libraries") or {}
    device = snapshot.get("device") or {}
    git = snapshot.get("git") or {}
    checkpoint = snapshot.get("checkpoint_identity") or {}
    return {
        "python_version": (snapshot.get("python") or {}).get("version"),
        "python_platform": (snapshot.get("python") or {}).get("platform"),
        "torch_version": libraries.get("torch"),
        "transformers_version": libraries.get("transformers"),
        "tokenizers_version": libraries.get("tokenizers"),
        "sentencepiece_version": libraries.get("sentencepiece"),
        "numpy_version": libraries.get("numpy"),
        "scipy_version": libraries.get("scipy"),
        "sklearn_version": libraries.get("sklearn"),
        "device": device.get("selected_device"),
        "batch_size": None,
        "model_dtype": model.get("dtype"),
        "model_id2label": model.get("id2label"),
        "model_type_vocab_size": model.get("type_vocab_size"),
        "model_training_mode": model.get("training_mode"),
        "model_attn_implementation": model.get("attn_implementation"),
        "tokenizer_class": tokenizer.get("class"),
        "tokenizer_model_max_length": tokenizer.get("model_max_length"),
        "tokenizer_is_fast": tokenizer.get("is_fast"),
        "score_version": snapshot.get("score_version"),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "nli_model_name": checkpoint.get("model_name"),
        "resolved_revision": checkpoint.get("resolved_revision"),
        "model_config_commit_hash": (
            model.get("config_commit_hash")
            or checkpoint.get("model_config_commit_hash")
        ),
        "tokenizer_commit_hash": (
            tokenizer.get("commit_hash") or checkpoint.get("tokenizer_commit_hash")
        ),
        "wang_source_commit": None,
        "nbc_histogram_positive": None,
        "nbc_histogram_negative": None,
    }


# Versions that, if they differ between the report and the live probe, mean the
# live probe is not describing the environment that produced the Gate 1 numbers.
ENVIRONMENT_KEYS = (
    "python_version",
    "torch_version",
    "transformers_version",
    "numpy_version",
    "scipy_version",
    "sklearn_version",
    "device",
)


def compare_environments(report_fields, live):
    rows = []
    for key in ENVIRONMENT_KEYS:
        reported = report_fields.get(key)
        observed = live.get(key)
        if reported is None or observed is None:
            match = None
        else:
            match = str(reported) == str(observed)
        rows.append({"field": key, "reported": reported, "live": observed, "match": match})
    definite = [row for row in rows if row["match"] is False]
    return {
        "rows": rows,
        "differences": [row["field"] for row in definite],
        "live_matches_report": (not definite) if any(r["match"] is not None for r in rows) else None,
    }


def render(value):
    if value is None:
        return NOT_RECORDED
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_scorer_environment(model_name, load_model):
    """Load tokenizer (and optionally model) purely to read their metadata.

    Returns ``(tokenizer, model, selected_device)``. The device is chosen by the
    shared CUDA -> MPS -> CPU selector and returned so the snapshot records the
    backend the weights are actually on; on Apple silicon the previous
    cuda-or-cpu expression placed the model on the CPU and then reported "cpu"
    for a machine whose accelerator is Metal.

    When no model is loaded there is no placement, so ``selected_device`` is
    ``None`` and :func:`device_state` falls back to its historical default
    rather than claiming a backend nothing ran on.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = None
    selected_device = None
    if load_model:
        import torch
        from transformers import AutoModelForSequenceClassification

        selected_device = select_device(torch)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            torch.device(selected_device)
        )
    return tokenizer, model, selected_device


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload, read_error = read_report(args.report)
    if payload is None:
        report_fields, missing = {name: None for name, _ in REPORT_FIELDS}, [
            name for name, _ in REPORT_FIELDS
        ]
    else:
        report_fields, missing = extract_from_report(payload)

    snapshot = None
    live = None
    live_error = None
    if not args.no_live:
        try:
            from src.utils import SCORE_VERSION

            tokenizer, model, selected_device = build_scorer_environment(
                args.model_name, load_model=not args.no_load_model
            )
            snapshot = collect_live_environment(
                model_name=args.model_name,
                model=model,
                tokenizer=tokenizer,
                repo_root=PROJECT_ROOT,
                score_version=SCORE_VERSION,
                hash_all=args.hash_weights,
                selected_device=selected_device,
            )
            live = live_fields(snapshot)
        except Exception as exc:  # noqa: BLE001 - a failed probe must still report
            live_error = f"{type(exc).__name__}: {exc}"

    token_type = None
    if snapshot and snapshot.get("tokenizer_emission_probe"):
        probe = snapshot["tokenizer_emission_probe"]
        repository_call = probe.get("repository_call") or {}
        token_type = token_type_id_assessment(
            (snapshot.get("model") or {}).get("type_vocab_size"),
            repository_call.get("token_type_ids_emitted", False),
            repository_call.get("unique_token_type_ids"),
        )

    comparison = compare_environments(report_fields, live) if live else None

    # The supplement can pin a revision for FUTURE runs. It cannot reach
    # backwards. The formal v2 Gate report records no resolved revision, and no
    # artifact ever will, so checkpoint identity with the historical v2 cache is
    # unestablishable -- not merely unestablished. This block states that in the
    # artifact itself so a later reader cannot mistake a live revision for
    # evidence about the cache.
    supplement_revision = (live or {}).get("resolved_revision")
    checkpoint_identity_limitation = {
        "checkpoint_identity_established": False,
        "supplement_only": True,
        "live_resolved_revision": supplement_revision,
        "revision_recorded_by_formal_gate_report": report_fields.get(
            "resolved_revision"
        ),
        "statement": (
            "The resolved Hugging Face revision recorded here describes the "
            "environment of THIS diagnostic run only. The formal v2 Gate report, "
            "the MPS preflight and the Wang probe record no resolved revision, so "
            "NO ARTIFACT ESTABLISHES that the frozen v2 NLI cache was produced at "
            "this or any other revision. Exact historical checkpoint identity "
            "cannot be established retrospectively. Pinning this revision makes "
            "future runs internally consistent and reproducible, which is "
            "strictly better than resolving against moving Hugging Face main, but "
            "it is not proof of identity and must never be reported as one."
        ),
    }

    result = {
        "step": "0-provenance",
        "purpose": (
            "Read the provenance already recorded by the formal Gate 1 run and "
            "probe the current environment for the fields it does not carry."
        ),
        "report_path": args.report,
        "report_read_error": read_error,
        "from_report": report_fields,
        "fields_not_recorded_in_report": missing,
        "live_environment": snapshot,
        "live_fields": live,
        "live_probe_error": live_error,
        "environment_comparison": comparison,
        "token_type_id_assessment": token_type,
        "checkpoint_identity_limitation": checkpoint_identity_limitation,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    print("=" * 100)
    print("GATE 1 DIAGNOSTIC -- STEP 0: PROVENANCE INSPECTION")
    print("=" * 100)
    if read_error:
        print(f"Report could not be read: {read_error}")
        print("Every 'reported' column below is therefore blank.")
    else:
        print(f"Report: {args.report}")
    if live_error:
        print(f"Live probe failed: {live_error}")
    print()
    print(f"  {'field':<32}{'reported':<40}{'live':<40}")
    print("  " + "-" * 110)
    for name, _ in REPORT_FIELDS:
        reported = render(report_fields.get(name))
        observed = render((live or {}).get(name)) if live else "-"
        print(f"  {name:<32}{reported[:38]:<40}{observed[:38]:<40}")

    if missing:
        print()
        print("Not carried by the formal provenance block (live column only):")
        for name in missing:
            print(f"  - {name}")

    if comparison and comparison["differences"]:
        print()
        print(
            "WARNING: the live environment differs from the one that produced the "
            "report in: " + ", ".join(comparison["differences"])
        )
        print(
            "  Live-only fields describe THIS environment, not the Gate 1 run. Rerun "
            "this script inside the Gate 1 environment to fill them authoritatively."
        )

    if snapshot and snapshot.get("tokenizer_emission_probe"):
        probe = snapshot["tokenizer_emission_probe"]
        print()
        print("-" * 100)
        print("TOKENIZER EMISSION PROBE (representative premise/hypothesis pair)")
        print("-" * 100)
        for label in ("wang_call", "repository_call"):
            entry = probe.get(label) or {}
            print(f"  {label}:")
            print(f"    keys emitted:            {entry.get('keys')}")
            print(f"    token_type_ids emitted:  {entry.get('token_type_ids_emitted')}")
            print(f"    unique token_type_ids:   {entry.get('unique_token_type_ids')}")
            print(f"    input_ids length:        {entry.get('input_ids_length')}")
        print(f"  input_ids identical across the two calls: {probe.get('input_ids_identical')}")

    if token_type:
        print()
        print("-" * 100)
        print("token_type_ids ASSESSMENT")
        print("-" * 100)
        print(f"  type_vocab_size:          {token_type['type_vocab_size']}")
        print(f"  token_type_ids emitted:   {token_type['token_type_ids_emitted']}")
        print(f"  unique token_type_ids:    {token_type['unique_token_type_ids']}")
        print(f"  can differ from Wang:     {token_type['can_differ_from_wang']}")
        print(f"  {token_type['note']}")

    if snapshot and snapshot.get("checkpoint_identity"):
        identity = snapshot["checkpoint_identity"]
        print()
        print("-" * 100)
        print("CHECKPOINT IDENTITY")
        print("-" * 100)
        print(f"  resolved revision:        {identity.get('resolved_revision')}")
        print(f"  config _commit_hash:      {identity.get('model_config_commit_hash')}")
        print(f"  tokenizer _commit_hash:   {identity.get('tokenizer_commit_hash')}")
        print(f"  snapshot dir:             {identity.get('snapshot_dir')}")
        for entry in identity.get("files", []):
            print(
                f"    {entry.get('name'):<40}{entry.get('size_bytes')!s:>14}  "
                f"{entry.get('sha256') or entry.get('note') or ''}"
            )
        for note in identity.get("notes", []):
            print(f"  note: {note}")

    print()
    print(f"Written to {output_path}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
