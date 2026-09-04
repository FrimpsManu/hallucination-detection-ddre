"""Pre-write provenance guard for NLI cache completion.

Adding scores to an existing cache is only sound if they are produced under the
same scorer, checkpoint and runtime semantics as the scores already in it. A
cache holding rows from two different checkpoints is worse than an incomplete
one: the incompleteness is visible and reported, the mixture is not.

``from_pretrained(model_name)`` resolves against Hugging Face main, which moves.
Recording the environment *after* loading proves only what was loaded, not that
it matched the formal run. So this module compares the current environment
against the **previously recorded formal provenance** and fails closed.

Every check runs before the derived cache is opened for writes and before any
inference. On a mismatch the caller aborts having performed zero inference and
written zero rows.

Unverifiable is treated as failed. A reference that does not record a field
cannot establish that the field matches, and silently accepting the current
value is exactly the failure mode the guard exists to prevent.

Standard library only, so the whole guard is testable without torch or a model.
"""

# The corrected Wang-fidelity scorer. Cache rows written under any other score
# version encode a different scaling convention.
REQUIRED_SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2"

# The pinned released-artifact commit the whole experiment is tied to.
REQUIRED_WANG_SOURCE_COMMIT = "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea"

# Wang's released inference scores one pair at a time.
REQUIRED_BATCH_SIZE = 1

OFFICIAL_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

# Libraries whose version can move an NLI score: the tensor library, the
# modelling code, and the tokenizer. numpy/scipy/sklearn are recorded for the
# record but cannot change a forward pass, so they are reported and not gated.
SCORE_AFFECTING_LIBRARIES = ("torch", "transformers", "tokenizers")
ADVISORY_LIBRARIES = ("numpy", "scipy", "sklearn", "sentencepiece")


def _dig(payload, dotted):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _first(payload, *paths):
    """First non-None value among several candidate dotted paths."""
    for path in paths:
        value = _dig(payload, path)
        if value is not None:
            return value
    return None


def extract_reference(payload):
    """Normalize a recorded provenance artifact into the fields the guard needs.

    Accepts the shapes this repository actually writes: the Step 2 scoring-path
    report, a raw ``collect_live_environment`` snapshot, the Gate 1 reproduction
    report's ``provenance`` block, and a previous cache-completion report.
    Missing fields come back as ``None`` and are treated as failures downstream.
    """
    reference = {
        "model_name": _first(
            payload,
            "environment_summary.model_name",
            "checkpoint_identity.model_name",
            "environment.checkpoint_identity.model_name",
            "provenance.nli_model.model_name",
            "nli_model.model_name",
        ),
        "resolved_revision": _first(
            payload,
            "environment_summary.resolved_hf_revision",
            "checkpoint_identity.resolved_revision",
            "environment.checkpoint_identity.resolved_revision",
            "primary.verdict.environment.checkpoint_identity.resolved_revision",
        ),
        "model_config_commit_hash": _first(
            payload,
            "checkpoint_identity.model_config_commit_hash",
            "environment.checkpoint_identity.model_config_commit_hash",
            "model.config_commit_hash",
            "environment.model.config_commit_hash",
        ),
        "tokenizer_commit_hash": _first(
            payload,
            "checkpoint_identity.tokenizer_commit_hash",
            "environment.checkpoint_identity.tokenizer_commit_hash",
            "tokenizer.commit_hash",
            "environment.tokenizer.commit_hash",
        ),
        "score_version": _first(
            payload,
            "score_version",
            "environment.score_version",
            "provenance.runtime.score_version",
            "environment_summary.score_version",
        ),
        "wang_source_commit": _first(
            payload,
            "wang_source_commit",
            "provenance.wang_data.source_commit",
            "environment.wang_data.source_commit",
        ),
        "batch_size": _first(
            payload,
            "protocol.batch_size",
            "provenance.runtime.batch_size",
            "environment_summary.batch_size",
        ),
        "model_dtype": _first(
            payload,
            "environment_summary.model_dtype",
            "model.dtype",
            "environment.model.dtype",
        ),
        "tokenizer_model_max_length": _first(
            payload,
            "tokenizer.model_max_length",
            "environment.tokenizer.model_max_length",
            "provenance.nli_model.tokenizer_model_max_length",
        ),
        "device": _first(
            payload,
            "environment_summary.device",
            "device.selected_device",
            "environment.device.selected_device",
            "provenance.runtime.device",
        ),
        "gpu_name": _first(
            payload,
            "environment_summary.gpu_name",
            "device.gpu_name",
            "environment.device.gpu_name",
        ),
        "libraries": _first(
            payload, "libraries", "environment.libraries", "provenance.libraries"
        )
        or {},
    }
    # The Step 2 summary flattens library versions; fall back to those.
    for name in SCORE_AFFECTING_LIBRARIES + ADVISORY_LIBRARIES:
        if reference["libraries"].get(name) is None:
            flat = _first(payload, f"environment_summary.{name}_version")
            if flat is not None:
                reference["libraries"][name] = flat
    return reference


def _check(name, expected, observed, ok, message):
    return {
        "name": name,
        "status": STATUS_PASS if ok else STATUS_FAIL,
        "expected": expected,
        "observed": observed,
        "message": message,
    }


def _match(name, expected, observed, what, *, advisory=False):
    """Equality check where a missing value on either side is a failure.

    ``advisory`` means the check is reported but never gates. It is used for
    quantities that cannot change an NLI forward pass, and for the checkpoint
    revision under the loudly-named debug override. An advisory check records
    the difference in its message so the run is still auditable.
    """
    if expected is None or observed is None:
        verified = False
        detail = (
            f"{what} could not be verified (reference={expected!r}, "
            f"observed={observed!r}). Unverifiable is treated as failed: a "
            "reference that does not record the field cannot establish that "
            "it matches."
        )
    else:
        verified = str(expected) == str(observed)
        detail = (
            f"{what} matches the formal run."
            if verified
            else f"{what} differs from the formal run."
        )

    if advisory and not verified:
        detail += " Reported only; this check does not gate the run."
    return _check(name, expected, observed, verified or advisory, detail)


def check_static_preconditions(
    reference,
    *,
    score_version,
    batch_size,
    wang_source_commit,
    model_name,
    required_model=OFFICIAL_NLI_MODEL,
    allow_local_checkpoint=False,
):
    """Checks that need no model load. Run these first and cheapest.

    ``wang_source_commit`` is the commit recorded in the local ``data/wang``
    SOURCE.json, so the completion scores the same released artifacts.

    ``allow_local_checkpoint`` downgrades the checkpoint-identity checks to
    advisory so the tool can be exercised against a local debug checkpoint,
    which by definition has neither the official name nor a Hub revision. It is
    never used by the documented formal command, and it does not relax anything
    else: score version, batch size, Wang commit, truncation, dtype, device and
    the score-affecting library versions all still gate.
    """
    checks = [
        _check(
            "score_version",
            REQUIRED_SCORE_VERSION,
            score_version,
            score_version == REQUIRED_SCORE_VERSION,
            "SCORE_VERSION is the corrected Wang-fidelity v2 convention."
            if score_version == REQUIRED_SCORE_VERSION
            else (
                "SCORE_VERSION is not the corrected v2 convention. Rows written "
                "now would encode a different scaling order than the cache holds."
            ),
        ),
        _check(
            "batch_size",
            REQUIRED_BATCH_SIZE,
            batch_size,
            batch_size == REQUIRED_BATCH_SIZE,
            "Batch size 1 matches the released Wang inference semantics."
            if batch_size == REQUIRED_BATCH_SIZE
            else "Batch size must be 1 to match the formal batch-1 cache.",
        ),
        _check(
            "wang_source_commit",
            REQUIRED_WANG_SOURCE_COMMIT,
            wang_source_commit,
            wang_source_commit == REQUIRED_WANG_SOURCE_COMMIT,
            "Wang released artifacts are the pinned commit."
            if wang_source_commit == REQUIRED_WANG_SOURCE_COMMIT
            else "Wang source data is not the pinned released commit.",
        ),
        _check(
            "official_model_name",
            required_model,
            model_name,
            model_name == required_model or allow_local_checkpoint,
            "NLI model is the official released model."
            if model_name == required_model
            else (
                "NLI model is not the official released model."
                + (
                    " Reported only; --unsafe-allow-local-checkpoint is set."
                    if allow_local_checkpoint
                    else ""
                )
            ),
        ),
        _match(
            "reference_model_name",
            reference.get("model_name"),
            model_name,
            "Model name recorded in the reference provenance",
            advisory=allow_local_checkpoint,
        ),
        _match(
            "reference_score_version",
            reference.get("score_version"),
            score_version,
            "SCORE_VERSION recorded in the reference provenance",
        ),
    ]
    return checks


def check_runtime_preconditions(reference, observed, *, allow_local_checkpoint=False):
    """Checks that need the loaded, revision-pinned model and tokenizer.

    ``observed`` is a ``collect_live_environment`` snapshot taken after loading
    with ``revision=`` pinned to the reference. These still run before the
    derived cache is opened for writes and before any inference.

    ``allow_local_checkpoint`` downgrades only the checkpoint-identity checks.
    """
    checks = []

    revision_advisory = allow_local_checkpoint
    checks.append(
        _match(
            "checkpoint_revision",
            reference.get("resolved_revision"),
            _first(observed, "checkpoint_identity.resolved_revision"),
            "Resolved Hugging Face checkpoint revision",
            advisory=revision_advisory,
        )
    )
    checks.append(
        _match(
            "model_config_commit_hash",
            reference.get("model_config_commit_hash"),
            _first(observed, "model.config_commit_hash",
                   "checkpoint_identity.model_config_commit_hash"),
            "Model config commit hash",
            advisory=revision_advisory,
        )
    )
    checks.append(
        _match(
            "tokenizer_commit_hash",
            reference.get("tokenizer_commit_hash"),
            _first(observed, "tokenizer.commit_hash",
                   "checkpoint_identity.tokenizer_commit_hash"),
            "Tokenizer commit hash",
            advisory=revision_advisory,
        )
    )

    # Truncation equivalence: the tokenizer limit must equal the configured
    # max_length, so this repository's explicit truncation is equivalent to
    # Wang's truncation=True.
    tokenizer_limit = _first(observed, "tokenizer.model_max_length")
    configured = 512
    truncation_ok = tokenizer_limit == configured
    checks.append(
        _check(
            "truncation_equivalence",
            configured,
            tokenizer_limit,
            truncation_ok,
            "Tokenizer limit equals the configured max_length, so truncation is "
            "equivalent to Wang's truncation=True."
            if truncation_ok
            else (
                "Tokenizer limit differs from the configured max_length. Spans "
                "would be truncated differently from the cached scores."
            ),
        )
    )

    checks.append(
        _match(
            "model_dtype",
            reference.get("model_dtype"),
            _first(observed, "model.dtype"),
            "Model parameter dtype",
        )
    )
    checks.append(
        _match(
            "device",
            reference.get("device"),
            _first(observed, "device.selected_device"),
            "Device",
        )
    )
    # The GPU identity only exists to compare when both runs were on a GPU.
    # cpu-vs-cuda is already gated by the device check above, and a CPU run has
    # no GPU identity to verify, so requiring one there would be unsatisfiable
    # rather than strict.
    reference_device = reference.get("device")
    observed_device = _first(observed, "device.selected_device")
    gpu_applicable = "cuda" in {str(reference_device), str(observed_device)}
    if gpu_applicable:
        checks.append(
            _match(
                "gpu_name",
                reference.get("gpu_name"),
                _first(observed, "device.gpu_name"),
                "GPU",
            )
        )
    else:
        checks.append(
            _check(
                "gpu_name",
                None,
                None,
                True,
                "Not applicable: neither run used a GPU, and cpu-vs-cuda is "
                "already gated by the device check.",
            )
        )

    observed_libraries = _first(observed, "libraries") or {}
    reference_libraries = reference.get("libraries") or {}
    for name in SCORE_AFFECTING_LIBRARIES:
        checks.append(
            _match(
                f"library_{name}",
                reference_libraries.get(name),
                observed_libraries.get(name),
                f"{name} version",
            )
        )
    for name in ADVISORY_LIBRARIES:
        checks.append(
            _match(
                f"library_{name}_advisory",
                reference_libraries.get(name),
                observed_libraries.get(name),
                f"{name} version (advisory: cannot change an NLI score)",
                advisory=True,
            )
        )

    training_mode = _first(observed, "model.training_mode")
    checks.append(
        _check(
            "model_eval_mode",
            False,
            training_mode,
            training_mode is False,
            "Model is in eval mode."
            if training_mode is False
            else "Model is not in eval mode; refusing to score.",
        )
    )
    return checks


def guard_report(checks, *, reference_path=None):
    """Aggregate checks into a pass/fail decision.

    ``passed`` is False if any check failed. The caller must perform zero
    inference and write zero cache rows when it is False.
    """
    failures = [check for check in checks if check["status"] != STATUS_PASS]
    return {
        "reference_path": reference_path,
        "checks": checks,
        "failed_checks": [check["name"] for check in failures],
        "failure_messages": [
            f"{check['name']}: {check['message']} "
            f"(expected {check['expected']!r}, observed {check['observed']!r})"
            for check in failures
        ],
        "passed": not failures,
        "policy": (
            "All provenance checks run BEFORE the derived cache is opened for "
            "writes and BEFORE any inference. On any failure the run aborts "
            "having performed zero inference and written zero cache rows. "
            "Unverifiable fields are treated as failures."
        ),
    }


def format_guard(report):
    lines = ["-" * 100, "PRE-WRITE PROVENANCE GUARD", "-" * 100]
    for check in report["checks"]:
        lines.append(f"  {check['name']:<32}{check['status']}")
        if check["status"] != STATUS_PASS:
            lines.append(f"      expected: {check['expected']!r}")
            lines.append(f"      observed: {check['observed']!r}")
            lines.append(f"      {check['message']}")
    lines.append(f"  {'overall':<32}{'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        lines.append(
            "  ABORTING: zero inference performed, zero cache rows written, "
            "derived cache not created."
        )
    lines.append("-" * 100)
    return "\n".join(lines)
