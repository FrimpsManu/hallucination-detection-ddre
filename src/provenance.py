"""Environment and data provenance capture for formal experiment runs.

Records everything needed to say exactly which code, data, model, and library
versions produced a result file. Imports only the standard library at module
scope; heavier libraries are probed lazily and their absence is recorded rather
than raised, so this module can be imported in a minimal test environment.

The NLI tokenizer/model probe exists to settle a specific baseline-fidelity
question. Wang's released ``utils.py`` calls::

    tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")

with no ``max_length``, so truncation falls back to
``tokenizer.model_max_length``. This repository's ``EntailmentScorer`` passes an
explicit ``max_length=512``. If the published tokenizer configuration specifies
``model_max_length=512`` the two are equivalent; if it carries the large
sentinel value they are not, and 400-word spans would be truncated here but not
by Wang. The probe records both numbers side by side so the question is settled
by observation rather than assumption.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


# Kept in sync by hand with the ``max_length`` argument in
# src/utils.py::EntailmentScorer._infer_batch. Recorded so that the configured
# value appears next to the probed tokenizer limit in every result file.
NLI_MAX_LENGTH_CONFIGURED = 512
NLI_MAX_LENGTH_SOURCE = "src/utils.py EntailmentScorer._infer_batch(max_length=512)"


def _run_git(args, cwd=None):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_provenance(repo_root=None):
    """Return the repository commit, branch, and dirty state."""
    cwd = str(repo_root) if repo_root else None
    status = _run_git(["status", "--porcelain"], cwd=cwd)
    return {
        "commit": _run_git(["rev-parse", "HEAD"], cwd=cwd),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd),
        "dirty": None if status is None else bool(status.strip()),
    }


def wang_data_provenance(data_root):
    """Return the recorded source commit of the downloaded Wang artifacts."""
    source_path = Path(data_root) / "SOURCE.json"
    if not source_path.exists():
        return {
            "source_path": str(source_path),
            "available": False,
            "note": "SOURCE.json not found; run scripts/prepare_wang_data.py",
        }
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        return {"source_path": str(source_path), "available": False, "note": str(exc)}
    return {
        "source_path": str(source_path),
        "available": True,
        "source_repository": metadata.get("source_repository"),
        "source_commit": metadata.get("source_commit"),
        "prepared_at_utc": metadata.get("prepared_at_utc"),
    }


def library_versions():
    """Return installed versions of the libraries that can change numerics."""
    versions = {}
    for name in ("numpy", "scipy", "sklearn", "torch", "transformers"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def nli_model_provenance(model_name, tokenizer=None, model=None):
    """Probe tokenizer/model configuration relevant to baseline fidelity."""
    probe = {
        "model_name": model_name,
        "nli_max_length_configured": NLI_MAX_LENGTH_CONFIGURED,
        "nli_max_length_configured_source": NLI_MAX_LENGTH_SOURCE,
        "tokenizer_model_max_length": None,
        "model_max_position_embeddings": None,
        "model_type_vocab_size": None,
        "model_id2label": None,
        "tokenizer_class": None,
        "truncation_matches_wang": None,
        "truncation_note": None,
    }

    if tokenizer is not None:
        probe["tokenizer_class"] = type(tokenizer).__name__
        try:
            probe["tokenizer_model_max_length"] = int(tokenizer.model_max_length)
        except Exception:
            probe["tokenizer_model_max_length"] = None

    if model is not None:
        config = getattr(model, "config", None)
        if config is not None:
            probe["model_max_position_embeddings"] = getattr(
                config, "max_position_embeddings", None
            )
            probe["model_type_vocab_size"] = getattr(config, "type_vocab_size", None)
            id2label = getattr(config, "id2label", None)
            if id2label is not None:
                probe["model_id2label"] = {str(k): str(v) for k, v in id2label.items()}

    effective = probe["tokenizer_model_max_length"]
    if effective is None:
        probe["truncation_note"] = (
            "Tokenizer limit could not be read; F1 (truncation fidelity) remains unverified."
        )
    elif effective == NLI_MAX_LENGTH_CONFIGURED:
        probe["truncation_matches_wang"] = True
        probe["truncation_note"] = (
            "tokenizer.model_max_length equals the configured max_length, so this "
            "repository's explicit truncation is equivalent to Wang's truncation=True."
        )
    else:
        probe["truncation_matches_wang"] = False
        probe["truncation_note"] = (
            f"tokenizer.model_max_length={effective} differs from the configured "
            f"max_length={NLI_MAX_LENGTH_CONFIGURED}. Wang's truncation=True would "
            "truncate at the tokenizer limit, so span inputs may differ. Investigate "
            "before accepting the reproduction."
        )
    return probe


def collect_provenance(
    *,
    model_name,
    data_root,
    device=None,
    batch_size=None,
    score_version=None,
    tokenizer=None,
    model=None,
    repo_root=None,
    extra=None,
):
    """Collect the full provenance block for a formal result file."""
    provenance = {
        "repository": git_provenance(repo_root),
        "wang_data": wang_data_provenance(data_root),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "libraries": library_versions(),
        "runtime": {
            "device": None if device is None else str(device),
            "batch_size": batch_size,
            "score_version": score_version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "nli_model": nli_model_provenance(model_name, tokenizer=tokenizer, model=model),
    }
    if extra:
        provenance.update(extra)
    return provenance
