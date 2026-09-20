"""Run Wang et al.'s ORIGINAL released implementation, untouched, and parse it.

The question this exists to answer: when Wang's own released code runs on Wang's
own released artifacts, does it produce **our** numbers or the **paper's** Table 1
numbers? Our corrected-v2 Gate 1 passes at CM=28/CFA=96 and fails at
CM=14/CFA=24, and that failure is either ours or theirs. Nothing short of
executing their code settles it.

This module is deliberately **independent of our implementation**. It imports no
``BSEDetector``, no ``EntailmentScorer``, no ``reproduction_gate``, no histogram
code and no cache. It knows how to verify a checkout, fingerprint the released
data, build a command line, and read the numbers Wang's ``main.py`` prints. The
science is entirely theirs.

**Wang's source and data are never modified.** The harness verifies the pinned
commit, records whether the checkout is dirty, and refuses to run a checkout it
cannot vouch for.

Standard library only, so the parser and provenance logic are testable without
torch, without a network and without a Wang checkout present.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

WANG_REPO_URL = "https://github.com/xhwang22/HallucinationDetection"
WANG_PINNED_COMMIT = "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea"

# The two cost settings to run. CM=28/CFA=96 is run.sh's intent and main.py's
# default; CM=14/CFA=24 is the paper's other published column and the one our
# Gate 1 fails.
COST_CONFIGURATIONS = (
    ("CM_28_CFA_96", 28, 96),
    ("CM_14_CFA_24", 14, 24),
)

# Wang's Table 1, transcribed from the paper. Declared here rather than imported
# from src/reproduction_gate.py so this harness stays independent of our
# implementation; a test cross-checks the two so they cannot silently diverge.
PUBLISHED_TABLE1 = {
    "CM_14_CFA_24": {
        "accuracy": 0.8024,
        "nonfactual_auc_pr": 0.8242,
        "factual_auc_pr": 0.5701,
        "pearson": 0.7137,
        "spearman": 0.6455,
        "evidence_num_per_sentence": 3.05,
        "evidence_num_per_subclaim": None,  # not published
    },
    "CM_28_CFA_96": {
        "accuracy": 0.8239,
        "nonfactual_auc_pr": 0.8645,
        "factual_auc_pr": 0.6196,
        "pearson": 0.8118,
        "spearman": 0.7420,
        "evidence_num_per_sentence": 6.22,
        "evidence_num_per_subclaim": None,
    },
}

# The released data files whose integrity is fingerprinted before a run.
RELEASED_DATA_FILES = (
    "dataset/NBC/NBC_positive.json",
    "dataset/NBC/NBC_negative.json",
    "dataset/selfcheckgpt/dataset.json",
)

# Wang's printed metric lines, mapped to our comparison names.
#
# The two evidence counts are easy to transpose, so the mapping is stated
# explicitly against what main.py actually accumulates:
#
#   "avg_sentence_search_time"   averages hypothesis_search_time_list, whose
#                                entries are PER-SENTENCE document totals
#                                -> evidence_num_per_sentence  (the Table 1 metric)
#
#   "hypothesis_avg_search_time" averages sentence_search_time_list, whose
#                                entries are PER-SUBCLAIM document counts
#                                -> evidence_num_per_subclaim
#
# The list names in Wang's source are the reverse of their print labels. The
# labels are what this mapping follows, because the labels describe the value.
# A float as Python's print() renders it, including a signed exponent and the
# nan/inf that scipy.stats can legitimately return (pearsonr over a constant
# vector is nan). Those must be CAPTURED, not treated as unparseable: "the
# correlation was nan" is a finding, and silently failing to parse it would
# look like a crashed run.
_NUMBER = r"(-?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?))"

METRIC_PATTERNS = (
    ("accuracy", r"^acc:\s+" + _NUMBER + r"\s*$"),
    ("nonfactual_auc_pr", r"^Non_fact_auc_precision_recall:\s+" + _NUMBER + r"\s*$"),
    ("factual_auc_pr", r"^fact_auc_precision_recall:\s+" + _NUMBER + r"\s*$"),
    ("evidence_num_per_sentence", r"^avg_sentence_search_time:\s+" + _NUMBER + r"\s*$"),
    ("pearson", r"^pearson:\s+" + _NUMBER + r"\s*$"),
    ("spearman", r"^Spearman:\s+" + _NUMBER + r"\s*$"),
    ("evidence_num_per_subclaim",
     r"^hypothesis_avg_search_time:\s+" + _NUMBER + r"\s*$"),
)

# main.py prints the Laplace-smoothed histograms as two Python lists before
# evaluating. Capturing them is free and makes the NBC state of the run visible.
HISTOGRAM_PATTERN = re.compile(r"^(\[[0-9,\s]+\])\s+(\[[0-9,\s]+\])\s*$")

INTERPRETATIONS = (
    "WANG_RELEASE_MATCHES_OUR_IMPLEMENTATION",
    "WANG_RELEASE_MATCHES_PUBLISHED_TABLE",
    "ALL_THREE_DIFFER",
)


class WangCheckoutInvalid(RuntimeError):
    """The Wang checkout is not the pinned, clean commit. Refuse to run."""


class WangOutputUnparsed(RuntimeError):
    """Wang's stdout did not contain every expected metric. Never guessed."""


def _git(checkout, *args):
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise WangCheckoutInvalid(
            f"git {' '.join(args)} failed in {checkout}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_checkout(checkout, expected_commit=WANG_PINNED_COMMIT):
    """The checkout must BE the pinned commit, and must be clean.

    A dirty checkout is refused rather than reported and run: the whole point
    is to execute Wang's code as released, and a modified working tree means
    the artifact would describe something else. ``git status --porcelain`` is
    the evidence, and it is recorded either way.
    """
    checkout = Path(checkout)
    if not (checkout / ".git").exists():
        raise WangCheckoutInvalid(f"not a git checkout: {checkout}")

    head = _git(checkout, "rev-parse", "HEAD")
    porcelain = _git(checkout, "status", "--porcelain")
    dirty_entries = [line for line in porcelain.split("\n") if line.strip()]

    state = {
        "checkout_path": str(checkout),
        "head_commit": head,
        "expected_commit": expected_commit,
        "commit_matches": head == expected_commit,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
    }
    if not state["commit_matches"]:
        raise WangCheckoutInvalid(
            f"Wang checkout is at {head}, not the pinned {expected_commit}. "
            "Refusing to run: a different commit is a different experiment."
        )
    if state["dirty"]:
        raise WangCheckoutInvalid(
            f"Wang checkout has {len(dirty_entries)} modified or untracked "
            f"path(s): {dirty_entries[:5]}. Refusing to run: this harness "
            "executes Wang's code AS RELEASED, and a dirty tree is not that."
        )
    return state


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def released_data_fingerprint(checkout, files=RELEASED_DATA_FILES):
    """Digest and size every released data file the run depends on."""
    checkout = Path(checkout)
    fingerprint = {}
    for relative in files:
        path = checkout / relative
        if not path.exists():
            raise WangCheckoutInvalid(f"released data file missing: {relative}")
        fingerprint[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return fingerprint


def nbc_counts(checkout):
    """The NBC example counts actually present. OBSERVED, never forced to 200.

    Wang's ``NBC_feature.get_NBC_features`` iterates ``for data in pos_data``
    and ``for data in neg_data``; it never indexes a fixed length and never
    asserts 200. Whatever the released files contain is what the histograms are
    built from, so the count is recorded as a measurement.
    """
    checkout = Path(checkout)
    counts = {}
    for label, relative in (
        ("positive", "dataset/NBC/NBC_positive.json"),
        ("negative", "dataset/NBC/NBC_negative.json"),
    ):
        raw = (checkout / relative).read_bytes()
        counts[label] = len(json.loads(raw.decode("utf-8", errors="ignore")))
    counts["note"] = (
        "Observed counts from the released files. Wang's NBC_feature.py "
        "iterates over every example present and does not require 200. No "
        "hypothetical 200th example is invented, reconstructed or inserted."
    )
    return counts


def build_command(c_miss, c_false_alarm, python_executable="python"):
    """The exact argv used to invoke Wang's original main.py.

    ``main.py`` is invoked directly rather than through ``bash run.sh``.

    At the pinned commit ``run.sh`` contains ``C_M = 28`` with spaces around the
    ``=``, which is not a bash assignment: bash tries to execute a command named
    ``C_M``, fails with 127, and leaves ``$C_M`` empty. The final line therefore
    expands to ``python -m main --C_M --C_FA``, which argparse rejects with
    "argument --C_M: expected one argument". So ``bash run.sh`` does not run at
    all, and cannot be the vehicle for the reproduction.

    Passing the values explicitly supplies exactly what run.sh intended and what
    main.py already defaults to for the 28/96 column. Wang's source is not
    edited to work around this.
    """
    return [
        python_executable, "-m", "main",
        "--C_M", str(int(c_miss)),
        "--C_FA", str(int(c_false_alarm)),
    ]


def parse_metrics(stdout):
    """Extract Wang's printed metrics. Every one required; none inferred.

    Wang's stdout interleaves progress noise -- a tqdm bar on stderr, and a
    ``wrong_num total_num`` pair printed on every misclassification -- so the
    patterns are anchored to whole lines and matched against the last
    occurrence, and a missing metric raises rather than defaulting.
    """
    lines = stdout.split("\n")
    metrics = {}
    for name, pattern in METRIC_PATTERNS:
        compiled = re.compile(pattern)
        for line in reversed(lines):
            match = compiled.match(line.strip())
            if match:
                metrics[name] = float(match.group(1))
                break

    missing = [name for name, _ in METRIC_PATTERNS if name not in metrics]
    if missing:
        raise WangOutputUnparsed(
            f"Wang's output did not contain: {missing}. The run is not "
            "interpretable and no value is guessed or defaulted. Check the "
            "captured log for a crash or an early exit."
        )
    return metrics


def parse_histograms(stdout):
    """The Laplace-smoothed (neg, pos) histograms main.py prints, if present."""
    for line in stdout.split("\n"):
        match = HISTOGRAM_PATTERN.match(line.strip())
        if match:
            return {
                "neg_features_smoothed": json.loads(match.group(1)),
                "pos_features_smoothed": json.loads(match.group(2)),
            }
    return None


def environment_provenance():
    """Versions and platform of the interpreter that will run Wang's code."""
    import platform
    import sys

    versions = {}
    for module in ("numpy", "scipy", "sklearn", "torch", "transformers", "tqdm"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - absence is itself the record
            versions[module] = None

    device = {"cuda_available": None, "selected_device": None, "gpu_name": None}
    try:
        import torch

        device["cuda_available"] = bool(torch.cuda.is_available())
        # Wang's main.py: torch.device("cuda") if cuda.is_available() else cpu.
        # No MPS branch, and none is added -- this mirrors their selection.
        device["selected_device"] = "cuda" if device["cuda_available"] else "cpu"
        if device["cuda_available"]:
            device["gpu_name"] = torch.cuda.get_device_name(0)
        device["mps_available"] = bool(
            getattr(getattr(torch.backends, "mps", None), "is_available", bool)()
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": versions,
        "device": device,
        "model_name": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "hardware_note": (
            "Wang's code selects CUDA if available, else CPU. That selection is "
            "preserved exactly; MPS is NOT substituted into their source. A run "
            "on a machine without CUDA is a CPU diagnostic, not a "
            "hardware-equivalent reproduction."
        ),
    }


def run_label(config_name, device):
    """How a run must be described, given the hardware it actually used."""
    if device == "cuda":
        return "released-code reproduction (CUDA, Wang's original device path)"
    return (
        f"CPU diagnostic on non-original hardware ({device}); NOT a "
        "hardware-equivalent reproduction of Wang et al."
    )


def comparison(wang_results, our_results=None, published=None):
    """Published Table 1 vs Wang's released code vs our bse_official.

    Measurements only. No pass/fail tolerance is applied here -- the point of
    this experiment is to record what the three sources say before deciding what
    the difference means.
    """
    published = PUBLISHED_TABLE1 if published is None else published
    metrics = [name for name, _ in METRIC_PATTERNS]

    rows = {}
    for config_name, _, _ in COST_CONFIGURATIONS:
        rows[config_name] = {}
        for metric in metrics:
            wang_value = (wang_results.get(config_name) or {}).get(metric)
            ours = (our_results or {}).get(config_name, {}).get(metric)
            published_value = (published.get(config_name) or {}).get(metric)
            rows[config_name][metric] = {
                "published_table1": published_value,
                "wang_released_code": wang_value,
                "our_bse_official": ours,
                "wang_minus_published": (
                    None if wang_value is None or published_value is None
                    else wang_value - published_value
                ),
                "wang_minus_ours": (
                    None if wang_value is None or ours is None
                    else wang_value - ours
                ),
            }
    return {
        "metrics": metrics,
        "configurations": rows,
        "tolerances_applied": False,
        "note": (
            "Measurements recorded side by side. No pass/fail tolerance is "
            "applied at this stage, and no interpretation category is asserted "
            "until the measurements support one."
        ),
        "interpretation_categories": list(INTERPRETATIONS),
        "interpretation": None,
    }
