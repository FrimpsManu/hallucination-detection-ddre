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
import unittest
from pathlib import Path

from src.wang_original import (
    COST_CONFIGURATIONS,
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
    nbc_counts,
    parse_histograms,
    parse_metrics,
    released_data_fingerprint,
    run_label,
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
        source = (PROJECT_ROOT / "src" / "wang_original.py").read_text("utf-8")
        for forbidden in ("* 200", "range(200)", "append(", "pad", "interpolat"):
            self.assertNotIn(forbidden, source.lower().replace("appended", ""))


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
        environment = environment_provenance()
        self.assertIn("selected_device", environment["device"])
        self.assertIn("libraries", environment)
        for library in ("numpy", "scipy", "sklearn", "torch", "transformers"):
            self.assertIn(library, environment["libraries"])
        self.assertIn("MPS is NOT substituted", environment["hardware_note"])
        self.assertEqual(
            environment["model_name"],
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        )


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


if __name__ == "__main__":
    unittest.main()
