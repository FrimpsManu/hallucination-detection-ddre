"""Tests for the Wang-original reproduction harness and its output parser.

The harness executes Wang et al.'s released implementation unmodified and
records what it prints. These tests cover OUR harness only -- the checkout
guard, the command construction, the parser and the provenance record. Wang's
algorithm is not under test and is never imported.

Synthetic fixtures throughout: no network, no Wang checkout, no model, no
inference.
"""

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.wang_original import (
    COST_CONFIGURATIONS,
    OUR_GATE1_ROW_METRICS,
    OurGate1Unreadable,
    REQUIRED_WANG_RUNTIME,
    WangEnvironmentUnusable,
    WangInterpreterUnresolvable,
    METRIC_PATTERNS,
    PUBLISHED_TABLE1,
    RELEASED_DATA_FILES,
    WANG_PINNED_COMMIT,
    WANG_REPO_URL,
    WangCheckoutInvalid,
    WangOutputUnparsed,
    build_command,
    comparison,
    environment_provenance,
    failure_reasons,
    nbc_counts,
    our_gate1_metrics,
    parse_histograms,
    parse_metrics,
    released_data_fingerprint,
    resolve_interpreter,
    run_label,
    run_succeeded,
    verify_checkout,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def executable_code(source):
    """Source with docstrings and comments removed, via the AST.

    Lets a test assert what the code DOES without forbidding the prose from
    describing it -- these modules deliberately name the things they exclude.
    Comments never reach the AST; docstrings are dropped by node identity
    rather than by guessing from token position, so ordinary string literals
    (dict keys, argv fragments) survive intact.
    """
    import ast as _ast

    tree = _ast.parse(source)
    for node in _ast.walk(tree):
        if not isinstance(
            node, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], _ast.Expr)
            and isinstance(body[0].value, _ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [_ast.Pass()]
    return _ast.unparse(_ast.fix_missing_locations(tree))


# A faithful sample of what Wang's main.py prints: loading chatter, the
# smoothed histograms, misclassification noise, then the seven metrics.
WANG_STDOUT = """loading tokenizer...
loading model...
[1, 142, 39, 16, 5, 2, 4, 5, 2, 2] [1, 79, 20, 14, 7, 6, 29, 58, 2, 2]
evaluting...
1 0
2 7
17 122
acc:  0.8123
Non_fact_auc_precision_recall:  0.85123
fact_auc_precision_recall:  0.60321
avg_sentence_search_time: 5.8842
pearson:  0.7991
Spearman:  0.7311
hypothesis_avg_search_time: 3.7551
"""


def stub_env_interpreter(tmp, name, cuda=False, mps=False):
    """A launcher that runs the REAL probe against a STUBBED torch.

    Not a fabricated payload: this is the actual interpreter, with a stub
    package directory ahead of it on PYTHONPATH, executing the real
    ``_ENVIRONMENT_PROBE`` source. That makes the device rule itself testable
    -- including combinations no real machine here offers, such as MPS present
    while CUDA is absent -- instead of merely asserting whatever the host
    happens to report.
    """
    import stat

    stubs = tmp / f"{name}-stubs"
    stubs.mkdir(parents=True, exist_ok=True)
    (stubs / "torch.py").write_text(
        "__version__ = '0.0.0-stub'\n"
        "class _Cuda:\n"
        f"    @staticmethod\n    def is_available(): return {bool(cuda)!r}\n"
        "    @staticmethod\n    def get_device_name(index): return 'StubGPU'\n"
        "class _Mps:\n"
        f"    @staticmethod\n    def is_available(): return {bool(mps)!r}\n"
        "class _Backends:\n    mps = _Mps()\n"
        "class _Version:\n    cuda = '11.8'\n"
        "cuda = _Cuda()\nbackends = _Backends()\nversion = _Version()\n",
        encoding="utf-8",
    )
    for module in REQUIRED_WANG_RUNTIME:
        if module == "torch":
            continue
        (stubs / f"{module}.py").write_text(
            f"__version__ = '0.0.0-stub-{module}'\n", encoding="utf-8"
        )

    launcher = tmp / f"{name}-python"
    launcher.write_text(
        f'#!/bin/sh\nPYTHONPATH={stubs} exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return str(launcher)


def make_repo(tmp, commit_files=True, dirty=False):
    """A throwaway git repo standing in for a Wang checkout."""
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.name", "t"], check=True)
    if commit_files:
        for relative in RELEASED_DATA_FILES:
            path = tmp / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                [{"premise": "p", "hypothesis": "h"}] * 199
                if "NBC" in relative else [{"gpt3_sentences": []}]
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(tmp), "commit", "-q", "-m", "released"], check=True
        )
    if dirty:
        (tmp / "dirty.txt").write_text("modified", encoding="utf-8")
    return subprocess.run(
        ["git", "-C", str(tmp), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


class TestCheckoutGuard(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="wang-checkout-")
        self.repo = Path(self._tmp.name) / "wang"
        self.head = make_repo(self.repo)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_pinned_commit_is_required(self):
        # A different commit is a different experiment.
        with self.assertRaises(WangCheckoutInvalid) as caught:
            verify_checkout(self.repo, expected_commit=WANG_PINNED_COMMIT)
        self.assertIn("not the pinned", str(caught.exception))
        self.assertIn(WANG_PINNED_COMMIT, str(caught.exception))

    def test_the_matching_commit_is_accepted_and_recorded(self):
        state = verify_checkout(self.repo, expected_commit=self.head)
        self.assertTrue(state["commit_matches"])
        self.assertFalse(state["dirty"])
        self.assertEqual(state["head_commit"], self.head)
        self.assertEqual(state["expected_commit"], self.head)

    def test_a_dirty_checkout_is_detected_and_refused(self):
        (self.repo / "NBC_feature.py").write_text("tampered", encoding="utf-8")
        with self.assertRaises(WangCheckoutInvalid) as caught:
            verify_checkout(self.repo, expected_commit=self.head)
        message = str(caught.exception)
        self.assertIn("modified or untracked", message)
        self.assertIn("AS RELEASED", message)

    def test_an_untracked_file_also_counts_as_dirty(self):
        (self.repo / "scratch.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(WangCheckoutInvalid):
            verify_checkout(self.repo, expected_commit=self.head)

    def test_a_non_git_directory_is_refused(self):
        with self.assertRaises(WangCheckoutInvalid) as caught:
            verify_checkout(self.repo.parent, expected_commit=self.head)
        self.assertIn("not a git checkout", str(caught.exception))

    def test_the_pinned_commit_constant_is_the_expected_one(self):
        self.assertEqual(
            WANG_PINNED_COMMIT, "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea"
        )
        self.assertIn("xhwang22/HallucinationDetection", WANG_REPO_URL)

    def test_released_data_files_are_fingerprinted(self):
        fingerprint = released_data_fingerprint(self.repo)
        self.assertEqual(sorted(fingerprint), sorted(RELEASED_DATA_FILES))
        for info in fingerprint.values():
            self.assertEqual(len(info["sha256"]), 64)
            self.assertGreater(info["bytes"], 0)

    def test_a_missing_released_data_file_is_refused(self):
        (self.repo / "dataset/NBC/NBC_positive.json").unlink()
        with self.assertRaises(WangCheckoutInvalid) as caught:
            released_data_fingerprint(self.repo)
        self.assertIn("missing", str(caught.exception))


class TestNbcCountsAreObservedNotForced(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="wang-nbc-")
        self.repo = Path(self._tmp.name) / "wang"
        make_repo(self.repo)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_counts_are_recorded_as_measured(self):
        counts = nbc_counts(self.repo)
        self.assertEqual(counts["positive"], 199)
        self.assertEqual(counts["negative"], 199)

    def test_the_counts_are_never_forced_to_200(self):
        # The released files hold 199 each and that is the source of truth. No
        # hypothetical 200th example is invented, reconstructed or inserted.
        counts = nbc_counts(self.repo)
        self.assertNotEqual(counts["positive"], 200)
        self.assertNotEqual(counts["negative"], 200)
        self.assertIn("does not require 200", counts["note"])
        self.assertIn("hypothetical 200th example", counts["note"])

    def test_a_different_count_is_reported_rather_than_corrected(self):
        path = self.repo / "dataset/NBC/NBC_positive.json"
        path.write_text(
            json.dumps([{"premise": "p", "hypothesis": "h"}] * 7), encoding="utf-8"
        )
        self.assertEqual(nbc_counts(self.repo)["positive"], 7)

    def test_the_harness_contains_no_padding_or_reconstruction_logic(self):
        """Structural, not textual: nbc_counts may only COUNT.

        Asserted on the AST so the prose is free to say "never forced to 200"
        without the guard mistaking documentation for behaviour. The function
        must contain no literal 200, and must not call append/insert/extend --
        the operations by which a 200th example could be manufactured.
        """
        tree = ast.parse((PROJECT_ROOT / "src" / "wang_original.py").read_text("utf-8"))
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "nbc_counts"
        )
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and node.value == 200:
                self.fail("nbc_counts contains the literal 200")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(
                    node.func.attr, {"append", "insert", "extend"},
                    f"nbc_counts calls {node.func.attr}, which could pad the data",
                )


class TestCommandConstruction(unittest.TestCase):
    def test_both_cost_configurations_are_defined(self):
        self.assertEqual(
            COST_CONFIGURATIONS,
            (("CM_28_CFA_96", 28, 96), ("CM_14_CFA_24", 14, 24)),
        )

    def test_the_command_is_exact(self):
        self.assertEqual(
            build_command(28, 96, python_executable="/usr/bin/python3"),
            ["/usr/bin/python3", "-m", "main", "--C_M", "28", "--C_FA", "96"],
        )
        self.assertEqual(
            build_command(14, 24, python_executable="py"),
            ["py", "-m", "main", "--C_M", "14", "--C_FA", "24"],
        )

    def test_run_sh_is_not_used_and_the_reason_is_recorded(self):
        # run.sh at the pinned commit contains "C_M = 28", which bash does not
        # treat as an assignment, so it expands to "python -m main --C_M
        # --C_FA" and argparse rejects it. main.py is invoked directly with the
        # values run.sh intended; Wang's source is not edited.
        self.assertIn("run.sh", build_command.__doc__)
        self.assertIn("expected one argument", build_command.__doc__)
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn("run_sh_note", script)
        # The harness must INVOKE main.py, never shell out to run.sh. The note
        # explaining why run.sh is unusable is provenance data, so the check is
        # on what gets executed, not on whether the string appears.
        code = executable_code(script)
        self.assertNotIn("'bash'", code)
        self.assertNotIn("run.sh'", code)
        self.assertIn("'-m', 'main'", executable_code(
            (PROJECT_ROOT / "src" / "wang_original.py").read_text("utf-8")
        ))

    def test_the_values_passed_match_the_two_published_columns(self):
        for name, c_miss, c_false_alarm in COST_CONFIGURATIONS:
            with self.subTest(configuration=name):
                self.assertEqual(name, f"CM_{c_miss}_CFA_{c_false_alarm}")
                self.assertIn(name, PUBLISHED_TABLE1)


class TestOutputParser(unittest.TestCase):
    def test_every_printed_metric_is_extracted(self):
        metrics = parse_metrics(WANG_STDOUT)
        self.assertEqual(metrics["accuracy"], 0.8123)
        self.assertEqual(metrics["nonfactual_auc_pr"], 0.85123)
        self.assertEqual(metrics["factual_auc_pr"], 0.60321)
        self.assertEqual(metrics["pearson"], 0.7991)
        self.assertEqual(metrics["spearman"], 0.7311)
        self.assertEqual(metrics["evidence_num_per_sentence"], 5.8842)
        self.assertEqual(metrics["evidence_num_per_subclaim"], 3.7551)

    def test_the_two_evidence_counts_are_not_transposed(self):
        # main.py's list names are the reverse of its print labels:
        # "avg_sentence_search_time" averages hypothesis_search_time_list
        # (per-SENTENCE totals) and is the Table 1 metric, while
        # "hypothesis_avg_search_time" averages sentence_search_time_list
        # (per-SUBCLAIM counts). Getting these the wrong way round would
        # silently compare the wrong quantity against Table 1.
        metrics = parse_metrics(WANG_STDOUT)
        self.assertEqual(metrics["evidence_num_per_sentence"], 5.8842)
        self.assertEqual(metrics["evidence_num_per_subclaim"], 3.7551)
        self.assertGreater(
            metrics["evidence_num_per_sentence"],
            metrics["evidence_num_per_subclaim"],
        )

    def test_misclassification_noise_does_not_confuse_the_parser(self):
        # main.py prints "wrong_num total_num" pairs throughout evaluation.
        self.assertEqual(parse_metrics(WANG_STDOUT)["accuracy"], 0.8123)

    def test_a_missing_metric_raises_rather_than_defaulting(self):
        truncated = WANG_STDOUT.replace("pearson:  0.7991\n", "")
        with self.assertRaises(WangOutputUnparsed) as caught:
            parse_metrics(truncated)
        message = str(caught.exception)
        self.assertIn("pearson", message)
        self.assertIn("no value is guessed", message)

    def test_empty_output_raises(self):
        with self.assertRaises(WangOutputUnparsed):
            parse_metrics("")

    def test_negative_and_exponent_values_parse(self):
        stdout = WANG_STDOUT.replace("pearson:  0.7991", "pearson:  -1.2e-03")
        self.assertAlmostEqual(parse_metrics(stdout)["pearson"], -0.0012)

    def test_a_nan_correlation_is_captured_not_treated_as_a_parse_failure(self):
        # scipy.stats.pearsonr returns nan over a constant vector. "The
        # correlation was nan" is a finding about the run; failing to parse it
        # would make a completed run look like a crashed one.
        import math

        stdout = WANG_STDOUT.replace("pearson:  0.7991", "pearson:  nan")
        self.assertTrue(math.isnan(parse_metrics(stdout)["pearson"]))

    def test_the_last_occurrence_wins(self):
        doubled = WANG_STDOUT + "acc:  0.9999\n"
        self.assertEqual(parse_metrics(doubled)["accuracy"], 0.9999)

    def test_the_smoothed_histograms_are_captured(self):
        histograms = parse_histograms(WANG_STDOUT)
        self.assertEqual(len(histograms["neg_features_smoothed"]), 10)
        self.assertEqual(len(histograms["pos_features_smoothed"]), 10)
        self.assertEqual(histograms["neg_features_smoothed"][1], 142)

    def test_absent_histograms_return_none_rather_than_raising(self):
        self.assertIsNone(parse_histograms("acc:  0.5\n"))

    def test_all_seven_metric_names_are_covered(self):
        self.assertEqual(
            [name for name, _ in METRIC_PATTERNS],
            [
                "accuracy", "nonfactual_auc_pr", "factual_auc_pr",
                "evidence_num_per_sentence", "pearson", "spearman",
                "evidence_num_per_subclaim",
            ],
        )


class TestComparisonRecord(unittest.TestCase):
    def test_three_sources_sit_side_by_side(self):
        wang = {"CM_14_CFA_24": {"pearson": 0.70, "accuracy": 0.80}}
        ours = {"CM_14_CFA_24": {"pearson": 0.66, "accuracy": 0.79}}
        cell = comparison(wang, ours)["configurations"]["CM_14_CFA_24"]["pearson"]
        self.assertEqual(cell["published_table1"], 0.7137)
        self.assertEqual(cell["wang_released_code"], 0.70)
        self.assertEqual(cell["our_bse_official"], 0.66)
        self.assertAlmostEqual(cell["wang_minus_published"], 0.70 - 0.7137)
        self.assertAlmostEqual(cell["wang_minus_ours"], 0.70 - 0.66)

    def test_missing_values_are_null_not_zero(self):
        cell = comparison({}, None)["configurations"]["CM_28_CFA_96"]["pearson"]
        self.assertIsNone(cell["wang_released_code"])
        self.assertIsNone(cell["our_bse_official"])
        self.assertIsNone(cell["wang_minus_published"])
        self.assertIsNone(cell["wang_minus_ours"])
        self.assertEqual(cell["published_table1"], 0.8118)

    def test_no_tolerance_is_applied_and_no_category_is_asserted(self):
        record = comparison({}, None)
        self.assertFalse(record["tolerances_applied"])
        self.assertIsNone(record["interpretation"])
        self.assertEqual(
            record["interpretation_categories"],
            [
                "WANG_RELEASE_MATCHES_OUR_IMPLEMENTATION",
                "WANG_RELEASE_MATCHES_PUBLISHED_TABLE",
                "ALL_THREE_DIFFER",
            ],
        )

    def test_the_published_values_match_our_gate1_constants(self):
        # The harness declares Table 1 independently so it does not import our
        # implementation. This cross-check is a TEST, not a harness dependency,
        # and exists so the two transcriptions cannot silently diverge.
        from src.reproduction_gate import PUBLISHED_TABLE1 as GATE_TABLE1

        for config_name, published in PUBLISHED_TABLE1.items():
            for metric, value in published.items():
                if value is None:
                    continue
                with self.subTest(configuration=config_name, metric=metric):
                    self.assertEqual(value, GATE_TABLE1[config_name][metric])


class TestHardwarePolicy(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="wang-hardware-")
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_non_cuda_run_is_labelled_a_cpu_diagnostic(self):
        label = run_label("CM_28_CFA_96", "cpu")
        self.assertIn("CPU diagnostic", label)
        self.assertIn("NOT a hardware-equivalent reproduction", label)

    def test_a_cuda_run_is_labelled_a_reproduction(self):
        self.assertIn("released-code reproduction", run_label("x", "cuda"))
        self.assertIn("CUDA", run_label("x", "cuda"))

    def test_mps_is_never_substituted_into_wangs_device_selection(self):
        # Wang's main.py and utils.py both select cuda-or-cpu. Preserving that
        # is the point of a released-code reproduction.
        harness = (PROJECT_ROOT / "src" / "wang_original.py").read_text("utf-8")
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn("MPS is NOT substituted", harness)
        for source in (harness, script):
            self.assertNotIn('torch.device("mps")', source)
            self.assertNotIn("select_device", source)

    def test_the_environment_record_reports_the_wang_device_rule(self):
        # Probed against a CONTROLLED interpreter, never sys.executable. The
        # record must describe the interpreter that will RUN Wang, so the
        # harness's own environment is irrelevant to it -- and asserting
        # against the host would make this test pass or fail on what happens
        # to be installed rather than on the rule under test.
        interpreter = stub_env_interpreter(self.tmp, "rule", cuda=True)
        environment = environment_provenance(interpreter)

        self.assertEqual(environment["device"]["selected_device"], "cuda")
        self.assertTrue(environment["device"]["cuda_available"])
        self.assertEqual(environment["device"]["gpu_name"], "StubGPU")
        for library in REQUIRED_WANG_RUNTIME:
            self.assertIn(library, environment["libraries"])
            self.assertIsNotNone(environment["libraries"][library])
        self.assertIn("MPS is NOT substituted", environment["hardware_note"])
        self.assertIn("diagnostic only", environment["hardware_note"])
        self.assertEqual(
            environment["model_name"],
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        )

    def test_mps_availability_never_changes_the_selected_device(self):
        """Every cuda/mps combination, including ones no host here offers.

        The decisive case is MPS present with CUDA absent -- an Apple machine.
        Wang's rule must still choose cpu there, because substituting MPS into
        their device selection would change the computation being reproduced.
        """
        cases = (
            (False, True, "cpu"),   # Apple silicon: the case that matters
            (False, False, "cpu"),
            (True, True, "cuda"),
            (True, False, "cuda"),
        )
        for cuda, mps, expected in cases:
            with self.subTest(cuda=cuda, mps=mps):
                interpreter = stub_env_interpreter(
                    self.tmp, f"dev-{int(cuda)}{int(mps)}", cuda=cuda, mps=mps
                )
                device = environment_provenance(interpreter)["device"]
                self.assertEqual(device["selected_device"], expected)
                self.assertEqual(device["cuda_available"], cuda)
                # Recorded, and recorded accurately -- but never consulted.
                self.assertEqual(device["mps_available"], mps)


class TestEnvironmentProbeUsesTheTargetInterpreter(unittest.TestCase):
    """Provenance must describe the interpreter that RUNS Wang, not the harness.

    The harness may live in our DDRE virtualenv while Wang runs under a
    separate one supplied via --python. Importing numpy/torch/transformers in
    the harness process would record OUR versions against THEIR run.
    """

    def fake_interpreter(self, tmp, name, libraries, cuda=False, rc=0):
        """A stand-in 'python' that answers the probe with chosen versions.

        The payload is serialised here and emitted verbatim, so the stub cannot
        pick up anything from the harness process.
        """
        import stat

        payload = json.dumps({
            "python": f"9.9.9-{name}",
            "python_full_version": f"9.9.9-{name} (fake)",
            "python_executable": str(tmp / name),
            "platform": f"FakeOS-{name}",
            "machine": "fake64",
            "libraries": dict(libraries),
            "device": {
                "cuda_available": bool(cuda),
                "selected_device": "cuda" if cuda else "cpu",
                "gpu_name": "FakeGPU" if cuda else None,
                "cuda_version": "11.7" if cuda else None,
                "mps_available": False,
            },
        })
        script = tmp / f"{name}.py"
        script.write_text(
            "import sys\n"
            f"sys.stdout.write({('<<<WANG_ENV_JSON>>>' + payload)!r})\n"
            f"sys.exit({int(rc)})\n",
            encoding="utf-8",
        )
        launcher = tmp / name
        launcher.write_text(
            f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n', encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        return str(launcher)

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="wang-interp-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_two_interpreters_yield_two_different_environments(self):
        # The decisive test: the same harness process probes two different
        # "interpreters" and gets each one's OWN versions back. If the harness
        # were importing libraries itself, both would report identically.
        alpha = self.fake_interpreter(
            self.tmp, "alpha",
            {"numpy": "1.21.5", "scipy": "1.7.3", "sklearn": "1.0.2",
             "torch": "2.0.1+cu117", "transformers": "4.29.2", "tqdm": "4.64.0"},
            cuda=True,
        )
        beta = self.fake_interpreter(
            self.tmp, "beta",
            {"numpy": "2.4.6", "scipy": "1.17.1", "sklearn": "1.9.0",
             "torch": "9.9.9", "transformers": "5.17.0", "tqdm": "4.70.0"},
            cuda=False,
        )
        a = environment_provenance(alpha)
        b = environment_provenance(beta)

        self.assertEqual(a["libraries"]["torch"], "2.0.1+cu117")
        self.assertEqual(b["libraries"]["torch"], "9.9.9")
        self.assertNotEqual(a["libraries"], b["libraries"])
        self.assertEqual(a["device"]["selected_device"], "cuda")
        self.assertEqual(b["device"]["selected_device"], "cpu")
        self.assertEqual(a["probed_interpreter"], alpha)
        self.assertEqual(b["probed_interpreter"], beta)

    def test_the_harness_process_versions_cannot_leak_in(self):
        # Probe an interpreter whose reported versions are deliberately unlike
        # anything installed here. If any harness-process version appeared, it
        # would show up as a mismatch against these fabricated ones.
        import importlib

        fabricated = {
            "numpy": "0.0.1-probe", "scipy": "0.0.2-probe",
            "sklearn": "0.0.3-probe", "torch": "0.0.4-probe",
            "transformers": "0.0.5-probe", "tqdm": "0.0.6-probe",
        }
        interpreter = self.fake_interpreter(self.tmp, "isolated", fabricated)
        environment = environment_provenance(interpreter)

        self.assertEqual(environment["libraries"], fabricated)
        for module in REQUIRED_WANG_RUNTIME:
            try:
                ours = importlib.import_module(module).__version__
            except Exception:  # noqa: BLE001
                continue
            with self.subTest(module=module):
                self.assertNotEqual(
                    environment["libraries"][module], ours,
                    f"{module} version leaked from the harness process",
                )
        self.assertTrue(environment["probed_in_subprocess"])

    def test_a_missing_runtime_dependency_fails_closed(self):
        crippled = {
            "numpy": "1.21.5", "scipy": "1.7.3", "sklearn": "1.0.2",
            "torch": None, "transformers": "4.29.2", "tqdm": "4.64.0",
        }
        interpreter = self.fake_interpreter(self.tmp, "notorch", crippled)
        with self.assertRaises(WangEnvironmentUnusable) as caught:
            environment_provenance(interpreter)
        message = str(caught.exception)
        self.assertIn("torch", message)
        self.assertIn("cannot happen", message)

    def test_a_nonexistent_interpreter_fails_closed(self):
        with self.assertRaises(WangEnvironmentUnusable) as caught:
            environment_provenance(str(self.tmp / "does-not-exist"))
        self.assertIn("cannot execute", str(caught.exception))

    def test_a_probe_that_exits_nonzero_fails_closed(self):
        interpreter = self.fake_interpreter(
            self.tmp, "broken",
            {name: "1.0" for name in REQUIRED_WANG_RUNTIME}, rc=3,
        )
        with self.assertRaises(WangEnvironmentUnusable) as caught:
            environment_provenance(interpreter)
        self.assertIn("rc=3", str(caught.exception))

    def test_a_probe_returning_garbage_fails_closed(self):
        import stat

        launcher = self.tmp / "garbage"
        launcher.write_text(
            "#!/bin/sh\nprintf '<<<WANG_ENV_JSON>>>not json'\n", encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        with self.assertRaises(WangEnvironmentUnusable) as caught:
            environment_provenance(str(launcher))
        self.assertIn("unparseable JSON", str(caught.exception))

    def test_the_harness_never_imports_the_runtime_libraries_itself(self):
        # Structural backstop for the leak. Checked as real import STATEMENTS,
        # because the probe's source text legitimately contains "import torch"
        # -- that text is executed by the target interpreter, not by us.
        tree = ast.parse(
            (PROJECT_ROOT / "src" / "wang_original.py").read_text("utf-8")
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for module in REQUIRED_WANG_RUNTIME:
            with self.subTest(module=module):
                self.assertNotIn(module, imported)

    def test_the_runner_probes_with_the_supplied_interpreter(self):
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn("environment_provenance(args.python)", script)


class TestSuccessRequiresACleanProcessExit(unittest.TestCase):
    """Parsing metric-looking lines is not the same as a successful run."""

    def record(self, **overrides):
        base = {"timed_out": False, "returncode": 0, "parsed": True}
        base.update(overrides)
        return base

    def test_clean_exit_with_complete_metrics_succeeds(self):
        self.assertTrue(run_succeeded(self.record()))
        self.assertEqual(failure_reasons(self.record()), [])

    def test_nonzero_exit_fails_even_with_complete_metrics(self):
        # main.py prints its metric block before it can still fail afterwards.
        failing = self.record(returncode=1)
        self.assertFalse(run_succeeded(failing))
        self.assertIn("returncode 1", " ".join(failure_reasons(failing)))

    def test_a_timeout_fails_even_with_complete_metrics(self):
        timed_out = self.record(timed_out=True, returncode=None)
        self.assertFalse(run_succeeded(timed_out))
        reasons = " ".join(failure_reasons(timed_out))
        self.assertIn("wall-clock budget", reasons)

    def test_clean_exit_with_a_missing_metric_fails(self):
        unparsed = self.record(parsed=False)
        self.assertFalse(run_succeeded(unparsed))
        self.assertIn("not every expected metric", " ".join(failure_reasons(unparsed)))

    def test_every_failing_combination(self):
        for timed_out, returncode, parsed, expected in (
            (False, 0, True, True),
            (False, 1, True, False),
            (False, 137, True, False),
            (True, None, True, False),
            (True, 0, True, False),
            (False, 0, False, False),
            (False, 1, False, False),
            (True, None, False, False),
        ):
            with self.subTest(timed_out=timed_out, rc=returncode, parsed=parsed):
                self.assertIs(
                    run_succeeded(self.record(
                        timed_out=timed_out, returncode=returncode, parsed=parsed
                    )),
                    expected,
                )

    def test_the_runner_gates_its_exit_code_on_succeeded(self):
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn("record[\"succeeded\"] = run_succeeded(record)", script)
        self.assertIn('if not r["succeeded"]', script)
        self.assertIn("return 0 if not failed else 1", script)
        # The old parse-only gate must be gone.
        self.assertNotIn('all(r["parsed"] for r in results)', script)

    def test_only_a_succeeded_run_contributes_metrics(self):
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn('if record["succeeded"]:', script)
        self.assertNotIn('if record["parsed"]:\n            wang_metrics', script)


REAL_GATE1_REPORT = {
    "mode": "gate",
    "purpose": "Gate 1: does this repository reproduce Wang et al. ...",
    "provenance": {"runtime": {"device": "mps", "batch_size": 1}},
    "dataset": {"sentences": 1908, "passages": 238, "subclaims": 2990},
    "nbc": {"positive_used": 199, "negative_used": 199},
    "nbc_histograms_laplace_smoothed": {"positive": [1] * 10, "negative": [1] * 10},
    "gate": {
        "formal_gate1_run": True,
        "overall_verdict": "FAIL",
        "configurations": {
            "CM_28_CFA_96": {
                "configuration": "CM_28_CFA_96",
                "c_miss": 28, "c_false_alarm": 96,
                "metrics": [
                    {"metric": "nonfactual_auc_pr", "published": 0.8645,
                     "reproduced": 0.8601, "status": "PASS"},
                    {"metric": "factual_auc_pr", "published": 0.6196,
                     "reproduced": 0.6104, "status": "WARN"},
                    {"metric": "accuracy", "published": 0.8239,
                     "reproduced": 0.8201, "status": "PASS"},
                    {"metric": "pearson", "published": 0.8118,
                     "reproduced": 0.8050, "status": "PASS"},
                    {"metric": "spearman", "published": 0.7420,
                     "reproduced": 0.7366, "status": "PASS"},
                    {"metric": "evidence_num_per_sentence", "published": 6.22,
                     "reproduced": 6.1903, "status": "PASS"},
                ],
                "diagnostics": {
                    "avg_retrieved_documents_per_subclaim": 3.9481,
                    "total_retrieved_documents": 11811,
                    "total_nli_span_calls": 34120,
                },
                "zero_retrieval": False,
                "verdict": "WARN",
            },
            "CM_14_CFA_24": {
                "configuration": "CM_14_CFA_24",
                "c_miss": 14, "c_false_alarm": 24,
                "metrics": [
                    {"metric": "nonfactual_auc_pr", "published": 0.8242,
                     "reproduced": 0.8119, "status": "WARN"},
                    {"metric": "factual_auc_pr", "published": 0.5701,
                     "reproduced": 0.5588, "status": "WARN"},
                    {"metric": "accuracy", "published": 0.8024,
                     "reproduced": 0.7998, "status": "PASS"},
                    {"metric": "pearson", "published": 0.7137,
                     "reproduced": 0.6402, "status": "FAIL"},
                    {"metric": "spearman", "published": 0.6455,
                     "reproduced": 0.5711, "status": "FAIL"},
                    {"metric": "evidence_num_per_sentence", "published": 3.05,
                     "reproduced": 2.4107, "status": "FAIL"},
                ],
                "diagnostics": {
                    "avg_retrieved_documents_per_subclaim": 1.5379,
                    "total_retrieved_documents": 4600,
                    "total_nli_span_calls": 13290,
                },
                "zero_retrieval": False,
                "verdict": "FAIL",
            },
        },
    },
    "full_metrics": {"CM_28_CFA_96": {}, "CM_14_CFA_24": {}},
}


class TestOurGate1IsReadFromTheRealSchema(unittest.TestCase):
    """The artifact has no top-level 'reproduced' key. Read what it really has."""

    def test_every_metric_populates_for_both_configurations(self):
        ours = our_gate1_metrics(REAL_GATE1_REPORT)
        self.assertEqual(sorted(ours), ["CM_14_CFA_24", "CM_28_CFA_96"])
        for config_name, values in ours.items():
            with self.subTest(configuration=config_name):
                for metric in (
                    "accuracy", "nonfactual_auc_pr", "factual_auc_pr",
                    "pearson", "spearman", "evidence_num_per_sentence",
                    "evidence_num_per_subclaim",
                ):
                    self.assertIsNotNone(values.get(metric), metric)

    def test_the_known_numbers_are_read_exactly(self):
        ours = our_gate1_metrics(REAL_GATE1_REPORT)
        failing = ours["CM_14_CFA_24"]
        self.assertEqual(failing["pearson"], 0.6402)
        self.assertEqual(failing["spearman"], 0.5711)
        self.assertEqual(failing["evidence_num_per_sentence"], 2.4107)
        self.assertEqual(failing["evidence_num_per_subclaim"], 1.5379)
        passing = ours["CM_28_CFA_96"]
        self.assertEqual(passing["accuracy"], 0.8201)
        self.assertEqual(passing["nonfactual_auc_pr"], 0.8601)
        self.assertEqual(passing["evidence_num_per_subclaim"], 3.9481)

    def test_the_comparison_column_is_fully_populated(self):
        record = comparison(
            {"CM_14_CFA_24": {"pearson": 0.71}},
            our_results=our_gate1_metrics(REAL_GATE1_REPORT),
        )
        for config_name in ("CM_14_CFA_24", "CM_28_CFA_96"):
            for metric in record["metrics"]:
                cell = record["configurations"][config_name][metric]
                with self.subTest(configuration=config_name, metric=metric):
                    self.assertIsNotNone(
                        cell["our_bse_official"],
                        f"our_bse_official is blank for {config_name}.{metric}",
                    )
        cell = record["configurations"]["CM_14_CFA_24"]["pearson"]
        self.assertAlmostEqual(cell["wang_minus_ours"], 0.71 - 0.6402)

    def test_the_old_nonexistent_reproduced_key_is_not_used(self):
        harness = executable_code(
            (PROJECT_ROOT / "src" / "wang_original.py").read_text("utf-8")
        )
        script = executable_code(
            (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text("utf-8")
        )
        # The wrong access was payload.get("reproduced") at top level. The
        # REAL schema does use a per-row "reproduced" key, so the check names
        # the bad pattern rather than the word.
        for source in (harness, script):
            self.assertNotIn("payload.get('reproduced')", source)
            self.assertNotIn("get('reproduced') or {}", source)
        # And the real path IS used.
        self.assertIn("'gate'", harness)
        self.assertIn("'configurations'", harness)
        self.assertIn("'diagnostics'", harness)
        self.assertIn("rows[metric].get('reproduced')", harness)

    def test_a_report_without_the_gate_block_fails_closed(self):
        with self.assertRaises(OurGate1Unreadable) as caught:
            our_gate1_metrics({"reproduced": {"CM_14_CFA_24": {"pearson": 0.6}}})
        self.assertIn("gate.configurations", str(caught.exception))

    def test_a_missing_configuration_fails_closed(self):
        import copy

        payload = copy.deepcopy(REAL_GATE1_REPORT)
        del payload["gate"]["configurations"]["CM_14_CFA_24"]
        with self.assertRaises(OurGate1Unreadable) as caught:
            our_gate1_metrics(payload)
        self.assertIn("CM_14_CFA_24", str(caught.exception))

    def test_a_missing_metric_row_fails_closed_rather_than_returning_none(self):
        import copy

        payload = copy.deepcopy(REAL_GATE1_REPORT)
        rows = payload["gate"]["configurations"]["CM_28_CFA_96"]["metrics"]
        payload["gate"]["configurations"]["CM_28_CFA_96"]["metrics"] = [
            row for row in rows if row["metric"] != "spearman"
        ]
        with self.assertRaises(OurGate1Unreadable) as caught:
            our_gate1_metrics(payload)
        self.assertIn("spearman", str(caught.exception))

    def test_a_null_reproduced_value_fails_closed(self):
        import copy

        payload = copy.deepcopy(REAL_GATE1_REPORT)
        for row in payload["gate"]["configurations"]["CM_28_CFA_96"]["metrics"]:
            if row["metric"] == "pearson":
                row["reproduced"] = None
        with self.assertRaises(OurGate1Unreadable) as caught:
            our_gate1_metrics(payload)
        self.assertIn("null", str(caught.exception))

    def test_a_missing_subclaim_diagnostic_fails_closed(self):
        import copy

        payload = copy.deepcopy(REAL_GATE1_REPORT)
        del payload["gate"]["configurations"]["CM_14_CFA_24"]["diagnostics"][
            "avg_retrieved_documents_per_subclaim"
        ]
        with self.assertRaises(OurGate1Unreadable) as caught:
            our_gate1_metrics(payload)
        self.assertIn("avg_retrieved_documents_per_subclaim", str(caught.exception))

    def test_the_row_metric_names_match_our_gate_module(self):
        from src.reproduction_gate import COMPARED_METRICS

        self.assertEqual(set(OUR_GATE1_ROW_METRICS), set(COMPARED_METRICS))

    def test_the_gate1_report_is_only_ever_read(self):
        script = executable_code(
            (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text("utf-8")
        )
        self.assertIn("read_text", script)
        self.assertNotIn("our_gate1_report).write_text", script)


class TestIsolationFromOurImplementation(unittest.TestCase):
    """This experiment must be an independent execution of Wang's code."""

    FORBIDDEN = (
        "BSEDetector", "EntailmentScorer", "reproduction_gate",
        "build_nbc_histograms", "baseline_core", "ddre_core",
        "cache_completion", "score_compatibility", "evaluate_detector",
    )

    def sources(self):
        return {
            "src/wang_original.py": (
                PROJECT_ROOT / "src" / "wang_original.py"
            ).read_text("utf-8"),
            "scripts/reproduce_wang_original.py": (
                PROJECT_ROOT / "scripts" / "reproduce_wang_original.py"
            ).read_text("utf-8"),
        }

    def test_none_of_our_implementation_is_imported(self):
        for path, source in self.sources().items():
            imported = set()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                    imported.add(node.module)
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
            with self.subTest(path=path):
                self.assertEqual(
                    {name for name in imported if name.startswith("src.")}
                    - {"src.wang_original"},
                    set(),
                    f"{path} imports our implementation: {sorted(imported)}",
                )

    def test_no_forbidden_symbol_appears(self):
        # Scans executable code only. The module docstrings legitimately NAME
        # what is excluded ("imports no BSEDetector, no EntailmentScorer..."),
        # and a scan that cannot tell documentation from a call would forbid
        # saying so.
        for path, source in self.sources().items():
            code = executable_code(source)
            for symbol in self.FORBIDDEN:
                with self.subTest(path=path, symbol=symbol):
                    self.assertNotIn(symbol, code)

    def test_our_nli_cache_is_never_referenced(self):
        for path, source in self.sources().items():
            with self.subTest(path=path):
                self.assertNotIn(".sqlite", source)
                self.assertNotIn("nli_cache", source)
                self.assertNotIn("cache_path", source)

    def test_wang_source_and_dataset_are_never_written(self):
        script = self.sources()["scripts/reproduce_wang_original.py"]
        # Every write target is under our own results directory.
        self.assertIn("output_dir / ", script)
        self.assertNotIn("wang_checkout / ", script)
        for writer in ("shutil.copy", "shutil.move", "os.remove", "unlink("):
            self.assertNotIn(writer, script)

    def test_the_provenance_asserts_the_isolation_explicitly(self):
        script = self.sources()["scripts/reproduce_wang_original.py"]
        for field in (
            '"wang_source_modified": False',
            '"wang_dataset_modified": False',
            '"our_gate1_cache_touched": False',
            '"our_implementation_imported": False',
        ):
            self.assertIn(field, script)


class TestProvenanceFields(unittest.TestCase):
    def test_the_required_provenance_fields_are_recorded(self):
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        for field in (
            "wang_pinned_commit", "wang_checkout", "released_data_fingerprint",
            "nbc_counts", "environment", "configurations", "started_utc",
            "finished_utc", "wall_clock_seconds", "command", "returncode",
            "device", "run_label", "generated_utc", "run_sh_note",
        ):
            with self.subTest(field=field):
                self.assertIn(field, script)

    def test_stdout_and_stderr_are_captured_separately(self):
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn('f"{config_name}.log"', script)
        self.assertIn('f"{config_name}.stderr.log"', script)
        self.assertIn('f"{config_name}_metrics.json"', script)
        self.assertIn("comparison.json", script)
        self.assertIn("provenance.json", script)

    def test_the_dry_run_loads_no_model_and_runs_no_inference(self):
        source = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        body = ast.get_source_segment(
            source,
            next(
                node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main"
            ),
        )
        dry = body[body.index("if args.dry_run:"):]
        self.assertIn("No model loaded, no inference run", dry)


class TestTheInterpreterIsResolvedBeforeTheCwdChange(unittest.TestCase):
    """--python must survive the move to Wang's working directory.

    Wang's main.py is launched with ``cwd`` set to Wang's own checkout, because
    their code opens ``dataset/...`` relative to the repository root. The
    documented invocation passes ``--python .venv-wang/bin/python``, which is
    written relative to OUR project root. Left literal, that string would be
    looked up inside ``external/HallucinationDetection`` at run time -- after an
    environment probe (which does not change directory) had already certified
    the interpreter as usable. The resolution therefore has to happen once, up
    front, and the same absolute string has to reach both.
    """

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="wang-resolve-")
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def executable(self, relative):
        """An executable stub at ``relative`` under the temp root."""
        import stat

        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def test_the_documented_relative_path_becomes_absolute(self):
        import os

        created = self.executable(".venv-wang/bin/python")
        wang_checkout = self.tmp / "external" / "HallucinationDetection"
        wang_checkout.mkdir(parents=True)

        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            resolved = resolve_interpreter(".venv-wang/bin/python")
        finally:
            os.chdir(cwd)

        self.assertTrue(os.path.isabs(resolved), resolved)
        self.assertEqual(Path(resolved), created)
        # The point of the fix: under Wang's cwd the original string names
        # nothing, so an unresolved argv would have died at launch.
        self.assertFalse((wang_checkout / ".venv-wang/bin/python").exists())

    def test_an_absolute_path_is_unchanged(self):
        created = self.executable("envs/wang/bin/python")
        self.assertEqual(resolve_interpreter(str(created)), str(created))

    def test_a_bare_command_name_goes_through_shutil_which(self):
        from unittest import mock

        created = self.executable("bin/python3")
        with mock.patch("src.wang_original.shutil.which") as which:
            which.return_value = str(created)
            resolved = resolve_interpreter("python3")
        which.assert_called_once_with("python3")
        self.assertEqual(resolved, str(created))

    def test_a_bare_command_name_missing_from_path_fails_closed(self):
        with self.assertRaises(WangInterpreterUnresolvable) as caught:
            resolve_interpreter("python-that-is-not-installed-anywhere")
        self.assertIn("not found on PATH", str(caught.exception))

    def test_a_which_result_is_made_absolute(self):
        # Defence in depth: a relative PATH entry would otherwise leak through.
        created = self.executable("bin/python3")
        resolved = resolve_interpreter(
            "python3", which=lambda name: str(created)
        )
        self.assertEqual(resolved, str(created))

    def test_a_symlinked_venv_python_is_not_dereferenced(self):
        """A venv's bin/python is a symlink; following it swaps the env.

        Python picks its site-packages from the executable path it was started
        with. Dereferencing ``.venv-wang/bin/python`` to the base interpreter
        would run Wang against a different set of libraries than the probe
        measured -- exactly the divergence this whole fix exists to prevent.
        """
        target = self.executable("system/python3.11")
        link = self.tmp / ".venv-wang" / "bin" / "python"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

        resolved = resolve_interpreter(str(link))
        self.assertEqual(Path(resolved), link)
        self.assertNotEqual(Path(resolved), target)

    def test_a_nonexistent_interpreter_fails_closed(self):
        with self.assertRaises(WangInterpreterUnresolvable) as caught:
            resolve_interpreter(str(self.tmp / "no-such-python"))
        self.assertIn("not an existing file", str(caught.exception))

    def test_a_non_executable_file_fails_closed(self):
        path = self.tmp / "not-executable"
        path.write_text("", encoding="utf-8")
        with self.assertRaises(WangInterpreterUnresolvable) as caught:
            resolve_interpreter(str(path))
        self.assertIn("not executable", str(caught.exception))

    def test_the_command_keeps_wangs_argv_with_an_absolute_interpreter(self):
        import os

        created = self.executable(".venv-wang/bin/python")
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            resolved = resolve_interpreter(".venv-wang/bin/python")
        finally:
            os.chdir(cwd)

        for name, c_miss, c_false_alarm in COST_CONFIGURATIONS:
            with self.subTest(configuration=name):
                command = build_command(c_miss, c_false_alarm, resolved)
                self.assertEqual(
                    command,
                    [str(created), "-m", "main",
                     "--C_M", str(c_miss), "--C_FA", str(c_false_alarm)],
                )
                self.assertTrue(os.path.isabs(command[0]))

    def test_the_probe_and_the_command_get_the_identical_string(self):
        import os
        import stat

        # A working fake interpreter reachable by a relative path, so the
        # resolution is what makes both usable.
        payload = json.dumps({
            "python": "9.9.9-shared",
            "platform": "FakeOS", "machine": "fake64",
            "libraries": {name: "1.0" for name in REQUIRED_WANG_RUNTIME},
            "device": {"cuda_available": False, "selected_device": "cpu"},
        })
        script = self.tmp / "shared.py"
        script.write_text(
            "import sys\n"
            f"sys.stdout.write({('<<<WANG_ENV_JSON>>>' + payload)!r})\n",
            encoding="utf-8",
        )
        launcher = self.tmp / ".venv-wang" / "bin" / "python"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text(
            f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n', encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)

        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            resolved = resolve_interpreter(".venv-wang/bin/python")
        finally:
            os.chdir(cwd)

        environment = environment_provenance(resolved)
        command = build_command(28, 96, resolved)
        self.assertEqual(environment["probed_interpreter"], resolved)
        self.assertEqual(command[0], resolved)
        self.assertEqual(environment["probed_interpreter"], command[0])

    def test_the_runner_resolves_before_it_probes_or_builds_a_command(self):
        """Ordering is the whole guarantee, so it is asserted structurally."""
        tree = ast.parse(
            (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
                "utf-8"
            )
        )
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = {}
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.setdefault(node.func.id, []).append(node.lineno)

        self.assertIn("resolve_interpreter", calls)
        self.assertIn("environment_provenance", calls)
        self.assertIn("build_command", calls)
        resolved_at = min(calls["resolve_interpreter"])
        self.assertLess(resolved_at, min(calls["environment_provenance"]))
        self.assertLess(resolved_at, min(calls["build_command"]))

        # One carrier, so the two cannot drift apart.
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn("args.python = resolve_interpreter(args.python)", script)
        self.assertIn("environment_provenance(args.python)", script)

    def test_the_runner_aborts_on_an_unresolvable_interpreter(self):
        """End to end: no checkout work, no probe, no model -- just a refusal."""
        completed = subprocess.run(
            [sys.executable, "scripts/reproduce_wang_original.py",
             "--python", str(self.tmp / "absent-python"), "--dry-run"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn("ABORTED", completed.stdout)
        self.assertIn("not an existing file", completed.stdout)
        # It refused before doing anything else at all.
        self.assertNotIn("released data fingerprints", completed.stdout)

    def test_cwd_for_wangs_process_is_still_wangs_checkout(self):
        script = (PROJECT_ROOT / "scripts" / "reproduce_wang_original.py").read_text(
            "utf-8"
        )
        self.assertIn("cwd=args.wang_checkout", script)

if __name__ == "__main__":
    unittest.main()
