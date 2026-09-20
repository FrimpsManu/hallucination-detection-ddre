"""Live environment and checkpoint probe for the Gate 1 scoring diagnostic.

``src/provenance.py`` records what a formal run needs to be reproducible. This
module records what a *diagnosis* needs, which is a superset: the fields the
existing provenance block does not carry (tokenizers version, parameter dtype,
training mode, attention implementation, tokenizer ``is_fast``), plus the
checkpoint identity that neither run pinned -- the resolved Hugging Face
revision SHA and file hashes.

Provenance is left untouched. This module only reads.

Everything heavy is imported lazily and every probe degrades to ``None`` with a
recorded reason rather than raising, so the module imports in a bare test
environment with no torch installed.
"""

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path


# A representative premise/hypothesis pair for the tokenizer-emission probe.
# Short and ASCII on purpose: the question is which keys the tokenizer emits
# and what values token_type_ids take for a *pair*, not how it handles hard text.
PROBE_PREMISE = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It was designed by Gustave Eiffel and completed in 1889."
)
PROBE_HYPOTHESIS = "The Eiffel Tower is located in Paris."

# Files above this size are recorded by size only unless hashing is forced.
# Reading a 1.6 GB weight file to hash it is affordable but not free, and the
# small config/tokenizer files are the ones that silently change on the Hub.
DEFAULT_HASH_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a probe must never abort a run
        return default if default is not None else {"error": f"{type(exc).__name__}: {exc}"}


def _run_git(args, cwd=None):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def git_state(repo_root=None):
    cwd = str(repo_root) if repo_root else None
    status = _run_git(["status", "--porcelain"], cwd=cwd)
    return {
        "commit": _run_git(["rev-parse", "HEAD"], cwd=cwd),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd),
        "dirty": None if status is None else bool(status.strip()),
    }


def library_versions():
    """Installed versions of every library that can move a score.

    Wider than ``src/provenance.py``'s list: ``tokenizers`` and
    ``sentencepiece`` are the two that decide how a DeBERTa-v3 premise/hypothesis
    pair becomes token ids, and neither is currently recorded anywhere.
    """
    names = (
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "huggingface_hub",
    )
    versions = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def python_state():
    return {
        "version": sys.version.split()[0],
        "full_version": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def mps_available(torch_module):
    """Is Apple Metal usable? Defensive because old torch has no ``backends.mps``."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:  # noqa: BLE001 - an unusable probe is an unusable backend
        return False


def select_device(torch_module=None):
    """The accelerator to run on, in a fixed priority order.

        CUDA  ->  MPS  ->  CPU

    CUDA keeps absolute priority so every existing CUDA run selects exactly
    what it selected before; Apple Metal is consulted only where CUDA is
    absent, which is precisely the machines that were falling back to CPU.

    This decides only WHERE the arithmetic runs. It does not touch the scoring
    mathematics, the score version, host-side probability scaling, truncation,
    or any frozen protocol value -- the backend is not part of the estimand.
    The caller is expected to record the returned string as the run's device,
    so provenance states the backend that was actually used.

    ``torch_module`` is injectable so the three branches can be tested without
    a GPU, and without torch at all.
    """
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415 - lazy, keeps CI torch-free

    cuda = getattr(torch_module, "cuda", None)
    cuda_available = getattr(cuda, "is_available", None)
    if cuda_available is not None and bool(cuda_available()):
        return "cuda"
    if mps_available(torch_module):
        return "mps"
    return "cpu"


def device_state():
    """Device and accelerator identity, including the exact GPU."""
    try:
        import torch
    except Exception:
        return {"available": False, "note": "torch is not importable"}

    state = {
        "available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": None,
        "gpu_capability": None,
    }
    if state["cuda_available"]:
        state["gpu_name"] = _safe(lambda: torch.cuda.get_device_name(0))
        capability = _safe(lambda: torch.cuda.get_device_capability(0))
        if isinstance(capability, tuple):
            state["gpu_capability"] = f"sm_{capability[0]}{capability[1]}"
        # TF32 only exists from sm_80 (Ampere) onward; on older cards fp32
        # matmuls are plain fp32 whatever the flags say, which matters when
        # comparing against a run on unknown hardware.
        state["matmul_tf32_enabled"] = _safe(
            lambda: bool(torch.backends.cuda.matmul.allow_tf32), default=False
        )
        state["cudnn_tf32_enabled"] = _safe(
            lambda: bool(torch.backends.cudnn.allow_tf32), default=False
        )
    return state


def model_state(model):
    """Configuration and runtime facts about the loaded model."""
    config = getattr(model, "config", None)
    parameter = None
    try:
        parameter = next(model.parameters())
    except Exception:
        parameter = None

    return {
        "class": type(model).__name__,
        "dtype": None if parameter is None else str(parameter.dtype),
        "device": None if parameter is None else str(parameter.device),
        # from_pretrained sets eval() before returning, so Wang's released
        # main.py -- which never calls .eval() -- also runs with dropout off.
        # Recorded so that stays a measured fact rather than an assumption.
        "training_mode": bool(getattr(model, "training", False)),
        "id2label": (
            None
            if config is None or getattr(config, "id2label", None) is None
            else {str(k): str(v) for k, v in config.id2label.items()}
        ),
        "type_vocab_size": None if config is None else getattr(config, "type_vocab_size", None),
        "max_position_embeddings": (
            None if config is None else getattr(config, "max_position_embeddings", None)
        ),
        "model_type": None if config is None else getattr(config, "model_type", None),
        "attn_implementation": (
            None
            if config is None
            else getattr(config, "_attn_implementation", None)
            or getattr(model, "_attn_implementation", None)
        ),
        "config_commit_hash": None if config is None else getattr(config, "_commit_hash", None),
        "config_dtype_field": (
            None
            if config is None
            else str(getattr(config, "dtype", None) or getattr(config, "torch_dtype", None))
        ),
    }


def tokenizer_state(tokenizer):
    return {
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "model_max_length": _safe(lambda: int(tokenizer.model_max_length), default=None),
        "vocab_size": _safe(lambda: int(tokenizer.vocab_size), default=None),
        "commit_hash": getattr(tokenizer, "_commit_hash", None),
        "name_or_path": getattr(tokenizer, "name_or_path", None),
    }


def tokenizer_emission_probe(tokenizer, premise=PROBE_PREMISE, hypothesis=PROBE_HYPOTHESIS):
    """Record exactly what the tokenizer emits for a representative pair.

    This is the observation that decides whether ``model(**inputs)`` can differ
    from Wang's ``model(inputs["input_ids"])``. It reports the emitted keys, the
    presence of token_type_ids, and their actual distinct values -- all three,
    because none of them is conclusive alone.
    """
    probe = {
        "premise": premise,
        "hypothesis": hypothesis,
        "wang_call": {},
        "repository_call": {},
        "error": None,
    }
    try:
        # Wang, released utils.py:56 -- no max_length, no padding.
        wang_inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        # This repository, src/utils.py:110-117.
        repo_inputs = tokenizer(
            [premise],
            [hypothesis],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
    except Exception as exc:  # noqa: BLE001
        probe["error"] = f"{type(exc).__name__}: {exc}"
        return probe

    for label, inputs in (("wang_call", wang_inputs), ("repository_call", repo_inputs)):
        keys = sorted(inputs.keys())
        entry = {
            "keys": keys,
            "token_type_ids_emitted": "token_type_ids" in keys,
            "unique_token_type_ids": None,
            "input_ids_length": int(inputs["input_ids"].shape[-1]),
            "input_ids_head": [int(x) for x in inputs["input_ids"][0][:24].tolist()],
        }
        if entry["token_type_ids_emitted"]:
            entry["unique_token_type_ids"] = sorted(
                {int(x) for x in inputs["token_type_ids"][0].tolist()}
            )
        probe[label] = entry

    probe["input_ids_identical"] = (
        wang_inputs["input_ids"][0].tolist() == repo_inputs["input_ids"][0].tolist()
    )
    return probe


# --------------------------------------------------------------------------
# Checkpoint identity
# --------------------------------------------------------------------------

def _sha256_file(path, size_limit, force):
    size = path.stat().st_size
    entry = {"name": path.name, "size_bytes": size, "sha256": None}
    if force or size <= size_limit:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entry["sha256"] = digest.hexdigest()
    else:
        entry["note"] = (
            f"larger than {size_limit} bytes; rerun with --hash-weights to hash it"
        )
    return entry


def checkpoint_identity(
    model_name,
    model=None,
    tokenizer=None,
    hash_size_limit=DEFAULT_HASH_SIZE_LIMIT_BYTES,
    hash_all=False,
):
    """Resolve the Hub revision and hash the cached files, best effort.

    Neither the published Wang run nor our Gate 1 run pinned a revision, so the
    checkpoint is currently an uncontrolled variable. Recording the resolved SHA
    and file digests is what makes "checkpoint drift" a testable hypothesis
    instead of an unfalsifiable one.
    """
    identity = {
        "model_name": model_name,
        "model_config_commit_hash": None,
        "tokenizer_commit_hash": None,
        "hub_cache_dir": None,
        "resolved_revision": None,
        "snapshot_dir": None,
        "files": [],
        "notes": [],
    }

    if model is not None:
        identity["model_config_commit_hash"] = getattr(
            getattr(model, "config", None), "_commit_hash", None
        )
    if tokenizer is not None:
        identity["tokenizer_commit_hash"] = getattr(tokenizer, "_commit_hash", None)

    try:
        from huggingface_hub import constants as hub_constants

        cache_dir = Path(hub_constants.HF_HUB_CACHE)
    except Exception:
        cache_dir = Path(
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or (Path.home() / ".cache" / "huggingface" / "hub")
        )
    identity["hub_cache_dir"] = str(cache_dir)

    repo_dir = cache_dir / ("models--" + model_name.replace("/", "--"))
    if not repo_dir.exists():
        identity["notes"].append(
            f"No Hub cache directory at {repo_dir}; revision cannot be resolved from disk."
        )
        return identity

    main_ref = repo_dir / "refs" / "main"
    if main_ref.exists():
        identity["resolved_revision"] = _safe(
            lambda: main_ref.read_text(encoding="utf-8").strip(), default=None
        )

    snapshots = repo_dir / "snapshots"
    snapshot_dir = None
    if snapshots.exists():
        candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if identity["resolved_revision"]:
            for candidate in candidates:
                if candidate.name == identity["resolved_revision"]:
                    snapshot_dir = candidate
                    break
        if snapshot_dir is None and candidates:
            snapshot_dir = candidates[-1]
            if len(candidates) > 1:
                identity["notes"].append(
                    f"{len(candidates)} snapshots cached; hashing {snapshot_dir.name}. "
                    "More than one revision on disk is itself worth noting."
                )
        if snapshot_dir is not None and not identity["resolved_revision"]:
            identity["resolved_revision"] = snapshot_dir.name

    if snapshot_dir is None:
        identity["notes"].append("No snapshot directory found; no files hashed.")
        return identity

    identity["snapshot_dir"] = str(snapshot_dir)
    for path in sorted(snapshot_dir.rglob("*")):
        if path.is_file() or path.is_symlink():
            resolved = _safe(lambda p=path: p.resolve(strict=True), default=None)
            if resolved is None or not Path(resolved).is_file():
                continue
            identity["files"].append(
                _safe(
                    lambda p=Path(resolved): _sha256_file(p, hash_size_limit, hash_all),
                    default={"name": path.name, "sha256": None, "error": "unreadable"},
                )
            )
    return identity


def collect_live_environment(
    *,
    model_name,
    model=None,
    tokenizer=None,
    repo_root=None,
    score_version=None,
    hash_all=False,
):
    """Full live snapshot, with every model/tokenizer section optional."""
    snapshot = {
        "python": python_state(),
        "libraries": library_versions(),
        "device": device_state(),
        "git": git_state(repo_root),
        "score_version": score_version,
        "model": None,
        "tokenizer": None,
        "tokenizer_emission_probe": None,
        "checkpoint_identity": None,
    }
    if model is not None:
        snapshot["model"] = model_state(model)
    if tokenizer is not None:
        snapshot["tokenizer"] = tokenizer_state(tokenizer)
        snapshot["tokenizer_emission_probe"] = tokenizer_emission_probe(tokenizer)
    if model is not None or tokenizer is not None:
        snapshot["checkpoint_identity"] = checkpoint_identity(
            model_name, model=model, tokenizer=tokenizer, hash_all=hash_all
        )
    return snapshot
