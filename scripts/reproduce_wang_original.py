#!/usr/bin/env python3
"""Execute Wang et al.'s ORIGINAL released implementation and record what it says.

Our corrected-v2 Gate 1 passes at CM=28/CFA=96 and fails at CM=14/CFA=24 on
Pearson, Spearman and average retrieved documents per sentence. That failure is
either our implementation's or the release's. This runs their code, unmodified,
on their data, and puts the three sources side by side: published Table 1,
Wang's released code, and our bse_official.

**Wang's source and dataset are never written to.** The checkout is verified to
be the pinned commit and clean, and refused otherwise. Every artifact this
produces lands under our ``results/original_wang/``.

**Our formal Gate 1 cache is never touched.** This harness imports none of our
implementation -- no BSEDetector, no EntailmentScorer, no reproduction_gate, no
histogram code, no NLI cache -- and Wang's code has no cache of its own.

``--dry-run`` performs every cheap validation (checkout, commit, cleanliness,
data fingerprints, NBC counts, command construction, environment) and writes
nothing but the provenance record, so the setup is reviewable before spending
hours of inference.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wang_original import (  # noqa: E402
    COST_CONFIGURATIONS,
    WANG_PINNED_COMMIT,
    WANG_REPO_URL,
    WangCheckoutInvalid,
    WangOutputUnparsed,
    build_command,
    comparison,
    environment_provenance,
    nbc_counts,
    parse_histograms,
    parse_metrics,
    released_data_fingerprint,
    run_label,
    verify_checkout,
)

DEFAULT_CHECKOUT = "external/HallucinationDetection"
DEFAULT_OUTPUT_DIR = "results/original_wang"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Wang et al.'s released implementation at the pinned commit "
            "and compare against Table 1 and our bse_official."
        )
    )
    parser.add_argument(
        "--wang-checkout", default=DEFAULT_CHECKOUT,
        help="Separate checkout of Wang's repository. Never modified.",
    )
    parser.add_argument(
        "--clone", action="store_true",
        help="Clone the repository to --wang-checkout if it is not there yet.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--our-gate1-report", default=None, metavar="PATH",
        help=(
            "Our formal Gate 1 JSON, read READ-ONLY to fill the comparison's "
            "third column. Optional; omitted leaves that column null."
        ),
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter used to run Wang's main.py. Point this at the "
             "separate Wang environment, not our project venv.",
    )
    parser.add_argument(
        "--config", action="append", default=None, metavar="NAME",
        help="Limit to named configurations (default: both).",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=None,
        help="Abort a configuration that exceeds this wall-clock budget.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate everything cheap and exit. No model, no inference.",
    )
    return parser.parse_args()


def clone_pinned(checkout):
    """Clone Wang's repository and detach at the pinned commit."""
    checkout = Path(checkout)
    if checkout.exists():
        return False
    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {WANG_REPO_URL} -> {checkout}")
    subprocess.run(
        ["git", "clone", WANG_REPO_URL, str(checkout)], check=True
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", WANG_PINNED_COMMIT],
        check=True,
    )
    return True


def selected_configurations(args):
    if not args.config:
        return list(COST_CONFIGURATIONS)
    wanted = set(args.config)
    chosen = [c for c in COST_CONFIGURATIONS if c[0] in wanted]
    unknown = wanted - {c[0] for c in COST_CONFIGURATIONS}
    if unknown:
        raise SystemExit(f"unknown configuration(s): {sorted(unknown)}")
    return chosen


def our_gate1_metrics(path):
    """Read our formal Gate 1 report READ-ONLY for the comparison column.

    Only the reproduced numbers are lifted out. None of our code is imported
    and the file is never written.
    """
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reproduced = payload.get("reproduced") or {}
    out = {}
    for config_name, block in reproduced.items():
        if isinstance(block, dict):
            out[config_name] = {
                key: block.get(key)
                for key in (
                    "accuracy", "nonfactual_auc_pr", "factual_auc_pr",
                    "pearson", "spearman", "evidence_num_per_sentence",
                    "evidence_num_per_subclaim",
                )
            }
    return out or None


def run_configuration(config_name, c_miss, c_false_alarm, args, output_dir, device):
    """One invocation of Wang's main.py, with stdout and stderr kept apart."""
    command = build_command(c_miss, c_false_alarm, python_executable=args.python)
    print(f"\n[{config_name}] {' '.join(command)}")
    print(f"[{config_name}] cwd = {args.wang_checkout}")

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command, cwd=args.wang_checkout, capture_output=True, text=True,
            timeout=args.timeout_seconds,
        )
        stdout, stderr, returncode = (
            completed.stdout, completed.stderr, completed.returncode
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {args.timeout_seconds}s"
        returncode = None
    elapsed = time.perf_counter() - clock
    finished = datetime.now(timezone.utc)

    (output_dir / f"{config_name}.log").write_text(stdout, encoding="utf-8")
    (output_dir / f"{config_name}.stderr.log").write_text(stderr, encoding="utf-8")

    record = {
        "configuration": config_name,
        "c_miss": c_miss,
        "c_false_alarm": c_false_alarm,
        "command": command,
        "cwd": str(args.wang_checkout),
        "returncode": returncode,
        "timed_out": timed_out,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_clock_seconds": elapsed,
        "device": device,
        "run_label": run_label(config_name, device),
        "stdout_log": str(output_dir / f"{config_name}.log"),
        "stderr_log": str(output_dir / f"{config_name}.stderr.log"),
        "smoothed_histograms": parse_histograms(stdout),
    }
    try:
        record["metrics"] = parse_metrics(stdout)
        record["parsed"] = True
    except WangOutputUnparsed as exc:
        record["metrics"] = None
        record["parsed"] = False
        record["parse_error"] = str(exc)

    (output_dir / f"{config_name}_metrics.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"[{config_name}] rc={returncode} parsed={record['parsed']} "
          f"{elapsed:.1f}s")
    if record["parsed"]:
        for key, value in record["metrics"].items():
            print(f"    {key:30s} {value}")
    return record


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = selected_configurations(args)

    print("=" * 84)
    print("WANG ET AL. ORIGINAL RELEASED IMPLEMENTATION")
    print("=" * 84)
    print(f"  repository:      {WANG_REPO_URL}")
    print(f"  pinned commit:   {WANG_PINNED_COMMIT}")
    print(f"  checkout:        {args.wang_checkout}   (never modified)")
    print(f"  output:          {output_dir}")
    print(f"  interpreter:     {args.python}")

    if args.clone:
        clone_pinned(args.wang_checkout)

    # Cheap validation, all of it, before anything expensive.
    try:
        checkout_state = verify_checkout(args.wang_checkout)
    except WangCheckoutInvalid as exc:
        print(f"\nABORTED: {exc}")
        return 1
    fingerprint = released_data_fingerprint(args.wang_checkout)
    counts = nbc_counts(args.wang_checkout)
    environment = environment_provenance()
    device = environment["device"].get("selected_device") or "cpu"

    print(f"\n  HEAD:            {checkout_state['head_commit']}")
    print(f"  commit matches:  {checkout_state['commit_matches']}")
    print(f"  clean:           {not checkout_state['dirty']}")
    print(f"  NBC positive:    {counts['positive']}   (observed, not forced to 200)")
    print(f"  NBC negative:    {counts['negative']}   (observed, not forced to 200)")
    print(f"  device:          {device}")
    print(f"  run label:       {run_label('*', device)}")
    print("\n  released data fingerprints:")
    for relative, info in fingerprint.items():
        print(f"    {relative:40s} {info['sha256'][:16]}...  {info['bytes']:>9,} B")
    print("\n  commands that would run:")
    for name, c_miss, c_false_alarm in configurations:
        print(f"    {name:16s} {' '.join(build_command(c_miss, c_false_alarm, args.python))}")

    provenance = {
        "experiment": "wang-original-released-code",
        "question": (
            "Does Wang's own released code, run on Wang's own released "
            "artifacts, produce our result or the published Table 1 result?"
        ),
        "wang_repository": WANG_REPO_URL,
        "wang_pinned_commit": WANG_PINNED_COMMIT,
        "wang_checkout": checkout_state,
        "wang_source_modified": False,
        "wang_dataset_modified": False,
        "our_gate1_cache_touched": False,
        "our_implementation_imported": False,
        "released_data_fingerprint": fingerprint,
        "nbc_counts": counts,
        "environment": environment,
        "configurations": [
            {
                "name": name, "c_miss": c_miss, "c_false_alarm": c_false_alarm,
                "command": build_command(c_miss, c_false_alarm, args.python),
            }
            for name, c_miss, c_false_alarm in configurations
        ],
        "run_sh_note": (
            "bash run.sh is NOT used. At the pinned commit it contains "
            "'C_M = 28' with spaces around '=', which bash does not treat as "
            "an assignment: it tries to execute a command named C_M, exits "
            "127, and leaves $C_M empty. The final line expands to "
            "'python -m main --C_M --C_FA', which argparse rejects with "
            "'argument --C_M: expected one argument'. main.py is therefore "
            "invoked directly with the values run.sh intended. Wang's source "
            "is not edited."
        ),
        "dry_run": bool(args.dry_run),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if args.dry_run:
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        print(f"\n--dry-run: validated only. No model loaded, no inference run.")
        print(f"Wrote {output_dir / 'provenance.json'}")
        print("=" * 84)
        return 0

    results, wang_metrics = [], {}
    for name, c_miss, c_false_alarm in configurations:
        record = run_configuration(
            name, c_miss, c_false_alarm, args, output_dir, device
        )
        results.append(record)
        if record["parsed"]:
            wang_metrics[name] = record["metrics"]

    provenance["runs"] = [
        {k: v for k, v in r.items() if k != "metrics"} for r in results
    ]
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    side_by_side = comparison(
        wang_metrics, our_results=our_gate1_metrics(args.our_gate1_report)
    )
    side_by_side["wang_checkout"] = checkout_state
    side_by_side["nbc_counts"] = counts
    side_by_side["device"] = device
    side_by_side["run_label"] = run_label("*", device)
    (output_dir / "comparison.json").write_text(
        json.dumps(side_by_side, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {output_dir / 'provenance.json'}")
    print(f"Wrote {output_dir / 'comparison.json'}")
    print("\nNo interpretation category is asserted; the measurements are "
          "recorded for review.")
    print("=" * 84)
    return 0 if all(r["parsed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
