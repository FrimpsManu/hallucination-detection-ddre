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

The reference is a **bundle of two artifacts**, because no single artifact
records everything the guard needs:

* the formal v2 batch-1 Gate report is authoritative for the corrected score
  version, the Wang source commit, batch size, model name, the truncation
  precondition, and the runtime/library fields it actually records;
* a checkpoint-provenance artifact (the Step 2 scoring-path diagnostic)
  supplements the fields the Gate report does not record at all: the resolved
  Hugging Face revision, the model/tokenizer commit hashes, the model dtype,
  the GPU identity, and the ``tokenizers``/``sentencepiece`` versions.

The formal report wins wherever it records a field. Where both record a field
they must agree, and a disagreement aborts the run — two artifacts describing
different environments cannot be spliced into one reference. The one exception
is ``score_version``: the Step 2 diagnostic predates the PR #4 scorer
correction, so a divergence there is structural rather than a sign of two
different machines. It does not gate, but it is recorded, and it is one of the
reasons exact checkpoint identity with the v2 cache cannot be established.

That limitation is stated rather than papered over. The formal v2 Gate report
does not record a resolved revision, so **no artifact establishes which Hub
revision produced the v2 cache rows.** The guard pins the revision the
supplement records and verifies the loaded checkpoint against it, which makes
this run internally consistent and reproducible; it does not prove the v2 cache
was produced at that revision, and the report says so.

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
    report's ``provenance`` block, a previous cache-completion report, and the
    checkpoint-provenance supplement from ``scripts/diagnose_gate1_provenance.py``,
    which nests the same values under ``live_environment`` (the raw snapshot) and
    ``live_fields`` (its flattened view).

    The ``live_*`` paths are appended to each candidate list rather than inserted,
    so precedence for every shape already accepted is unchanged. Missing fields
    come back as ``None`` and are treated as failures downstream: nothing here is
    made optional.
    """
    reference = {
        "model_name": _first(
            payload,
            "environment_summary.model_name",
            "checkpoint_identity.model_name",
            "environment.checkpoint_identity.model_name",
            "provenance.nli_model.model_name",
            "nli_model.model_name",
            "live_environment.checkpoint_identity.model_name",
            "live_fields.nli_model_name",
        ),
        "resolved_revision": _first(
            payload,
            "environment_summary.resolved_hf_revision",
            "checkpoint_identity.resolved_revision",
            "environment.checkpoint_identity.resolved_revision",
            "primary.verdict.environment.checkpoint_identity.resolved_revision",
            "live_environment.checkpoint_identity.resolved_revision",
            "live_fields.resolved_revision",
        ),
        "model_config_commit_hash": _first(
            payload,
            "checkpoint_identity.model_config_commit_hash",
            "environment.checkpoint_identity.model_config_commit_hash",
            "model.config_commit_hash",
            "environment.model.config_commit_hash",
            "live_environment.checkpoint_identity.model_config_commit_hash",
            "live_environment.model.config_commit_hash",
            "live_fields.model_config_commit_hash",
        ),
        "tokenizer_commit_hash": _first(
            payload,
            "checkpoint_identity.tokenizer_commit_hash",
            "environment.checkpoint_identity.tokenizer_commit_hash",
            "tokenizer.commit_hash",
            "environment.tokenizer.commit_hash",
            "live_environment.checkpoint_identity.tokenizer_commit_hash",
            "live_environment.tokenizer.commit_hash",
            "live_fields.tokenizer_commit_hash",
        ),
        "score_version": _first(
            payload,
            "score_version",
            "environment.score_version",
            "provenance.runtime.score_version",
            "environment_summary.score_version",
            "live_environment.score_version",
            "live_fields.score_version",
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
            "live_environment.model.dtype",
            "live_fields.model_dtype",
        ),
        "tokenizer_model_max_length": _first(
            payload,
            "tokenizer.model_max_length",
            "environment.tokenizer.model_max_length",
            "provenance.nli_model.tokenizer_model_max_length",
            "live_environment.tokenizer.model_max_length",
            "live_fields.tokenizer_model_max_length",
        ),
        "device": _first(
            payload,
            "environment_summary.device",
            "device.selected_device",
            "environment.device.selected_device",
            "provenance.runtime.device",
            "live_environment.device.selected_device",
            "live_fields.device",
        ),
        "gpu_name": _first(
            payload,
            "environment_summary.gpu_name",
            "device.gpu_name",
            "environment.device.gpu_name",
            "live_environment.device.gpu_name",
        ),
        "libraries": dict(
            _first(
                payload,
                "libraries",
                "environment.libraries",
                "provenance.libraries",
                "live_environment.libraries",
            )
            or {}
        ),
    }
    # The Step 2 summary and the checkpoint-provenance supplement's live_fields
    # both flatten library versions as "<name>_version"; fall back to those.
    for name in SCORE_AFFECTING_LIBRARIES + ADVISORY_LIBRARIES:
        if reference["libraries"].get(name) is None:
            flat = _first(
                payload,
                f"environment_summary.{name}_version",
                f"live_fields.{name}_version",
            )
            if flat is not None:
                reference["libraries"][name] = flat
    return reference


# --------------------------------------------------------------------------
# Reference bundle: the formal v2 Gate report plus a checkpoint supplement.
# --------------------------------------------------------------------------

FORMAL_SOURCE = "formal_v2_gate_report"
CHECKPOINT_SOURCE = "checkpoint_provenance"

# Fields the formal v2 batch-1 Gate report records and is authoritative for.
FORMAL_AUTHORITATIVE_FIELDS = (
    "model_name",
    "score_version",
    "wang_source_commit",
    "batch_size",
    "device",
    "tokenizer_model_max_length",
)

# Fields the formal Gate report does not record at all. The checkpoint
# provenance artifact supplements exactly these.
CHECKPOINT_SUPPLEMENTED_FIELDS = (
    "resolved_revision",
    "model_config_commit_hash",
    "tokenizer_commit_hash",
    "model_dtype",
    "gpu_name",
)

BUNDLE_FIELDS = FORMAL_AUTHORITATIVE_FIELDS + CHECKPOINT_SUPPLEMENTED_FIELDS

# A cross-artifact disagreement here is structural rather than evidence of two
# different environments: the checkpoint provenance artifact was produced by an
# earlier diagnostic that predates the PR #4 scorer correction, so it records
# the pre-correction score version by construction. It is recorded, and it
# counts against exact checkpoint identity, but it does not gate.
NON_GATING_CROSS_ARTIFACT_FIELDS = ("score_version",)

AGREE_BOTH = "both_agree"
AGREE_CONFLICT = "conflict"
AGREE_FORMAL_ONLY = "formal_only"
AGREE_CHECKPOINT_ONLY = "checkpoint_only"
AGREE_UNRECORDED = "unrecorded"


def _merge_field(field, formal_value, checkpoint_value):
    """Resolve one field across the two artifacts, recording where it came from.

    The formal report wins wherever it records the field. The checkpoint
    artifact supplements only what the formal report left out. Where both
    record it they must agree.
    """
    entry = {
        "field": field,
        "formal": formal_value,
        "checkpoint": checkpoint_value,
        "gating": False,
    }
    if formal_value is not None and checkpoint_value is not None:
        agree = str(formal_value) == str(checkpoint_value)
        entry["agreement"] = AGREE_BOTH if agree else AGREE_CONFLICT
        entry["source"] = FORMAL_SOURCE
        entry["value"] = formal_value
        entry["gating"] = (
            not agree and field not in NON_GATING_CROSS_ARTIFACT_FIELDS
        )
    elif formal_value is not None:
        entry["agreement"] = AGREE_FORMAL_ONLY
        entry["source"] = FORMAL_SOURCE
        entry["value"] = formal_value
    elif checkpoint_value is not None:
        entry["agreement"] = AGREE_CHECKPOINT_ONLY
        entry["source"] = CHECKPOINT_SOURCE
        entry["value"] = checkpoint_value
    else:
        entry["agreement"] = AGREE_UNRECORDED
        entry["source"] = None
        entry["value"] = None
    return entry


def merge_reference(
    formal, checkpoint=None, *, formal_path=None, checkpoint_path=None
):
    """Combine the authoritative formal reference with a checkpoint supplement.

    ``formal`` and ``checkpoint`` are ``extract_reference`` outputs. Returns a
    bundle carrying the merged reference, the per-field provenance of every
    guarded value, the gating conflicts, the recorded non-gating divergences,
    and the limitations that follow from what the artifacts do not establish.
    """
    checkpoint = checkpoint or {}
    entries = [
        _merge_field(field, formal.get(field), checkpoint.get(field))
        for field in BUNDLE_FIELDS
    ]

    formal_libraries = formal.get("libraries") or {}
    checkpoint_libraries = checkpoint.get("libraries") or {}
    libraries = {}
    for name in SCORE_AFFECTING_LIBRARIES + ADVISORY_LIBRARIES:
        entry = _merge_field(
            f"library:{name}", formal_libraries.get(name), checkpoint_libraries.get(name)
        )
        # An advisory library cannot change a forward pass, so a cross-artifact
        # difference in one is not evidence that the artifacts describe
        # different environments.
        if name in ADVISORY_LIBRARIES:
            entry["gating"] = False
        entries.append(entry)
        if entry["value"] is not None:
            libraries[name] = entry["value"]

    reference = {entry["field"]: entry["value"] for entry in entries
                 if not entry["field"].startswith("library:")}
    reference["libraries"] = libraries

    conflicts = [e for e in entries if e["agreement"] == AGREE_CONFLICT and e["gating"]]
    divergences = [
        e for e in entries if e["agreement"] == AGREE_CONFLICT and not e["gating"]
    ]
    by_field = {entry["field"]: entry for entry in entries}

    revision_entry = by_field["resolved_revision"]
    identity_established = revision_entry["source"] == FORMAL_SOURCE

    limitations = []
    if revision_entry["agreement"] == AGREE_UNRECORDED:
        limitations.append(
            "Neither artifact records a resolved Hugging Face checkpoint "
            "revision, so the checkpoint cannot be pinned. The static "
            "resolved_revision_present check gates on this."
        )
    elif not identity_established:
        limitations.append(
            "The formal v2 Gate report does not record a resolved Hugging Face "
            f"revision. The pinned revision {revision_entry['value']!r} comes "
            f"from the checkpoint provenance artifact ({checkpoint_path}), a "
            "separate diagnostic run. NO ARTIFACT ESTABLISHES that the v2 cache "
            "rows were produced at that revision; exact checkpoint identity with "
            "the formal v2 run cannot be established retrospectively. Pinning it "
            "makes this completion internally consistent and reproducible, which "
            "is strictly better than resolving against moving Hugging Face main, "
            "but it is not proof of identity."
        )
    for entry in divergences:
        limitations.append(
            f"{entry['field']} differs between the artifacts "
            f"(formal={entry['formal']!r}, checkpoint={entry['checkpoint']!r}). "
            "The checkpoint provenance artifact predates the PR #4 scorer "
            "correction, so it describes a run under a different scoring "
            "convention. The formal value is used; the divergence is further "
            "reason the supplement cannot establish v2 checkpoint identity."
        )
    for entry in entries:
        if (
            entry["field"] in FORMAL_AUTHORITATIVE_FIELDS
            and entry["agreement"] == AGREE_CHECKPOINT_ONLY
        ):
            limitations.append(
                f"{entry['field']} was not recorded by the formal v2 Gate report "
                "and was supplied by the checkpoint provenance artifact."
            )

    return {
        "formal_path": formal_path,
        "checkpoint_path": checkpoint_path,
        "reference": reference,
        "field_sources": entries,
        "conflicts": conflicts,
        "recorded_divergences": divergences,
        "limitations": limitations,
        "checkpoint_identity_established": identity_established,
        "policy": (
            "The formal v2 Gate report is authoritative wherever it records a "
            "field. The checkpoint provenance artifact supplements only fields "
            "the Gate report does not record. A field recorded by both must "
            "agree; a gating disagreement aborts the run."
        ),
    }


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


def check_reference_bundle(bundle):
    """Gating checks over the two-artifact reference bundle.

    One check per field both artifacts record, so every cross-artifact
    comparison is visible in the report rather than only the failures, plus a
    summary check that fails if any gating conflict exists.
    """
    checks = []
    for entry in bundle["field_sources"]:
        if entry["agreement"] not in (AGREE_BOTH, AGREE_CONFLICT):
            continue
        agree = entry["agreement"] == AGREE_BOTH
        if agree:
            message = (
                f"{entry['field']} is recorded identically by the formal v2 Gate "
                "report and the checkpoint provenance artifact."
            )
        elif entry["gating"]:
            message = (
                f"{entry['field']} disagrees between the two reference artifacts. "
                "They describe different environments and cannot be spliced into "
                "one reference."
            )
        else:
            message = (
                f"{entry['field']} disagrees between the two reference artifacts. "
                "Recorded and reported; the formal value is authoritative and "
                "this difference does not gate. See the bundle limitations."
            )
        checks.append(
            _check(
                f"bundle_agreement_{entry['field'].replace(':', '_')}",
                entry["formal"],
                entry["checkpoint"],
                agree or not entry["gating"],
                message,
            )
        )

    conflicts = [entry["field"] for entry in bundle["conflicts"]]
    checks.append(
        _check(
            "reference_bundle_consistent",
            [],
            conflicts,
            not conflicts,
            "The formal and checkpoint provenance artifacts agree on every field "
            "they both record."
            if not conflicts
            else (
                "The formal and checkpoint provenance artifacts disagree on "
                f"{', '.join(conflicts)}. Refusing to merge two artifacts that "
                "describe different environments."
            ),
        )
    )
    return checks


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
        # STATIC, and deliberately so: this must abort before from_pretrained.
        # Without a recorded revision the only thing left to load is moving
        # Hugging Face main, which is exactly the uncontrolled variable the
        # guard exists to eliminate. There is no fallback.
        _check(
            "resolved_revision_present",
            "a resolved Hugging Face revision",
            reference.get("resolved_revision"),
            bool(reference.get("resolved_revision")) or allow_local_checkpoint,
            "A resolved checkpoint revision is recorded, so the model and "
            "tokenizer can be pinned to it."
            if reference.get("resolved_revision")
            else (
                "No resolved checkpoint revision is recorded by either "
                "reference artifact. Loading would fall back to moving Hugging "
                "Face main, which cannot be shown to match the cached scores. "
                "Aborting before from_pretrained."
                + (
                    " Reported only; --unsafe-allow-local-checkpoint is set."
                    if allow_local_checkpoint
                    else ""
                )
            ),
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

    # The Gate report is authoritative for the truncation precondition, so the
    # live tokenizer limit is checked against the recorded one as well as
    # against the configured max_length.
    checks.append(
        _match(
            "reference_tokenizer_model_max_length",
            reference.get("tokenizer_model_max_length"),
            tokenizer_limit,
            "Tokenizer model_max_length recorded in the reference provenance",
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
    width = max([32] + [len(check["name"]) + 2 for check in report["checks"]])
    for check in report["checks"]:
        lines.append(f"  {check['name']:<{width}}{check['status']}")
        if check["status"] != STATUS_PASS:
            lines.append(f"      expected: {check['expected']!r}")
            lines.append(f"      observed: {check['observed']!r}")
            lines.append(f"      {check['message']}")
    lines.append(f"  {'overall':<{width}}{'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        lines.append(
            "  ABORTING: zero inference performed, zero cache rows written, "
            "derived cache not created."
        )
    lines.append("-" * 100)
    return "\n".join(lines)


def format_bundle(bundle):
    """Human-readable account of where every guarded value came from."""
    lines = ["-" * 100, "REFERENCE PROVENANCE BUNDLE", "-" * 100]
    lines.append(f"  formal (authoritative): {bundle['formal_path']}")
    lines.append(f"  checkpoint (supplement): {bundle['checkpoint_path']}")
    lines.append("")
    lines.append(f"  {'field':<34}{'source':<24}{'agreement'}")
    for entry in bundle["field_sources"]:
        source = entry["source"] or "-"
        lines.append(f"  {entry['field']:<34}{source:<24}{entry['agreement']}")
    lines.append("")
    lines.append(
        f"  exact v2 checkpoint identity established: "
        f"{bundle['checkpoint_identity_established']}"
    )
    if bundle["limitations"]:
        lines.append("  LIMITATIONS:")
        for limitation in bundle["limitations"]:
            lines.append(f"    - {limitation}")
    lines.append("-" * 100)
    return "\n".join(lines)
