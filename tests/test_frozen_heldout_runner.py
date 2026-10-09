"""THE held-out experiment runner: BSE official versus the frozen DDRE config.

These tests exist to make one guarantee testable: the runner can evaluate the
frozen configuration on held-out data and nothing else. Every precondition
fails closed, the four frozen numbers are literals with no command-line route
to a different value, no tuner or threshold grid is reachable, ``--dry-run``
scores nothing, and the artifact carries the tuning-budget disclosure rather
than a laundered SUPPORTED label.

No model is loaded and no NLI inference runs here. ``frozen_split`` reads the
released Wang records to reconstruct the split, which is arithmetic over
passage indices; it scores nothing.
"""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.cache_completion import sha256_file
from src.ddre_core import ULSIFDensityRatio
from src.hyperparameter_outcome_sensitivity import (
    EXPECTED_HELD_OUT_PASSAGES,
    EXPECTED_SPLIT_SEED,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VALIDATION_PASSAGES,
    candidate_estimator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_frozen_heldout.py"
DATA_ROOT = PROJECT_ROOT / "data" / "wang"

# The frozen configuration, transcribed from the freeze decision and repeated
# here so a silent edit to the runner's literals breaks a test.
FROZEN_SIGMA = 0.11346435546875
FROZEN_LAMBDA = 1.0
FROZEN_LOWER = 0.20
FROZEN_UPPER = 0.80

FREEZE_SHA256 = "744abf4afc05f586ed138dbbb5a503ac8014af40ab14c2a6d13b5f512ecdf9fa"
SENSITIVITY_SHA256 = (
    "fc5604854b18f3e3e712fe7ef01859b58dac569e99696041713abb11daa6dcc0"
)
D03_CACHE_SHA256 = (
    "66b4715fa555c503f16ac40f28fd3cab26aa3b67cb498bb79a136300b1553776"
)

# The split reconstructed from the released data at fraction 0.20, seed 42.
HELD_OUT_SENTENCES = 1525
HELD_OUT_SUBCLAIMS = 2387
VALIDATION_IDS_SHA256 = (
    "618f248e3ddad32881a787d445badc6ddf010050c4a645f7e185cb017ce7bf67"
)
HELD_OUT_IDS_SHA256 = (
    "eb25bd2bbc0beaabf3704288ad644f01d4bb315f0f897d5432771457974a1732"
)


def runner():
    spec = importlib.util.spec_from_file_location("run_frozen_heldout", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = runner()
SOURCE = SCRIPT.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def freeze_payload(**overrides):
    """A freeze artifact shaped like the real one."""
    payload = {
        "artifact": "ddre-validation-freeze",
        "held_out_scored": False,
        "validation_tuned": True,
        "source_sensitivity_artifact_sha256": SENSITIVITY_SHA256,
        "selected_configuration": {
            "sigma": FROZEN_SIGMA,
            "lambda": FROZEN_LAMBDA,
            "lower_threshold": FROZEN_LOWER,
            "upper_threshold": FROZEN_UPPER,
        },
    }
    selected = overrides.pop("selected_configuration", None)
    if selected is not None:
        payload["selected_configuration"].update(selected)
    payload.update(overrides)
    return payload


class FreezeFixture:
    """A freeze file on disk, with the module's expected digest repointed.

    Repointing ``FREEZE_SHA256`` is what lets the FIELD checks be tested at
    all: the real artifact's bytes cannot be synthesized. The digest gate
    itself is tested separately, against the unmodified literal.
    """

    def __init__(self, payload):
        self._tmp = tempfile.TemporaryDirectory(prefix="frozen-heldout-")
        self.path = Path(self._tmp.name) / "ddre_validation_freeze.json"
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._saved = MODULE.FREEZE_SHA256
        MODULE.FREEZE_SHA256 = sha256_file(self.path)

    def close(self):
        MODULE.FREEZE_SHA256 = self._saved
        self._tmp.cleanup()


# ---------------------------------------------------------------- 1. freeze

class TestFrozenConstantsAreTheFrozenOnes(unittest.TestCase):
    def test_the_four_numbers_are_the_frozen_configuration(self):
        self.assertEqual(MODULE.FROZEN_SIGMA, FROZEN_SIGMA)
        self.assertEqual(MODULE.FROZEN_LAMBDA, FROZEN_LAMBDA)
        self.assertEqual(MODULE.FROZEN_LOWER, FROZEN_LOWER)
        self.assertEqual(MODULE.FROZEN_UPPER, FROZEN_UPPER)

    def test_sigma_is_pinned_bit_exactly_not_rounded(self):
        """0.11346435546875 is exact in binary; a rounded literal is a
        different estimator."""
        self.assertEqual(repr(MODULE.FROZEN_SIGMA), "0.11346435546875")

    def test_the_expected_digests_are_the_frozen_ones(self):
        self.assertEqual(MODULE.FREEZE_SHA256, FREEZE_SHA256)
        self.assertEqual(MODULE.SENSITIVITY_SHA256, SENSITIVITY_SHA256)
        self.assertEqual(MODULE.D03_CACHE_SHA256, D03_CACHE_SHA256)

    def test_the_tuning_budget_is_recorded_as_640_of_which_102_eligible(self):
        self.assertEqual(MODULE.VALIDATION_CONFIGURATIONS_CONSIDERED, 640)
        self.assertEqual(MODULE.ELIGIBLE_VALIDATION_CONFIGURATIONS, 102)


class TestFreezeVerificationFailsClosed(unittest.TestCase):
    def test_a_wrong_freeze_digest_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freeze.json"
            path.write_text(json.dumps(freeze_payload()), encoding="utf-8")
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.verify_freeze(path)
        message = str(caught.exception)
        self.assertIn("not the expected one", message)
        self.assertIn(FREEZE_SHA256, message)

    def test_a_missing_freeze_artifact_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.verify_freeze(Path(tmp) / "absent.json")
        self.assertIn("not found", str(caught.exception))

    def _reject(self, **overrides):
        fixture = FreezeFixture(freeze_payload(**overrides))
        try:
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.verify_freeze(fixture.path)
            return str(caught.exception)
        finally:
            fixture.close()

    def test_a_matching_freeze_artifact_verifies(self):
        fixture = FreezeFixture(freeze_payload())
        try:
            record = MODULE.verify_freeze(fixture.path)
        finally:
            fixture.close()
        self.assertTrue(record["verified"])
        self.assertIs(record["held_out_scored_in_freeze"], False)
        self.assertIs(record["validation_tuned"], True)
        self.assertEqual(
            record["source_sensitivity_artifact_sha256"], SENSITIVITY_SHA256
        )

    def test_a_freeze_claiming_held_out_was_scored_aborts(self):
        self.assertIn("held_out_scored", self._reject(held_out_scored=True))

    def test_a_freeze_missing_held_out_scored_aborts(self):
        payload = freeze_payload()
        del payload["held_out_scored"]
        fixture = FreezeFixture(payload)
        try:
            with self.assertRaises(MODULE.Aborted):
                MODULE.verify_freeze(fixture.path)
        finally:
            fixture.close()

    def test_a_freeze_not_marked_validation_tuned_aborts(self):
        self.assertIn("validation_tuned", self._reject(validation_tuned=False))

    def test_a_wrong_sensitivity_source_digest_aborts(self):
        message = self._reject(source_sensitivity_artifact_sha256="0" * 64)
        self.assertIn("sensitivity", message)
        self.assertIn(SENSITIVITY_SHA256, message)

    def test_a_different_sigma_in_the_freeze_aborts(self):
        message = self._reject(selected_configuration={"sigma": 0.2269287109375})
        self.assertIn("sigma", message)

    def test_a_rounded_sigma_in_the_freeze_aborts(self):
        """Near is not equal: the frozen estimator is one exact sigma."""
        self.assertIn("sigma", self._reject(
            selected_configuration={"sigma": 0.1134643554687}
        ))

    def test_a_different_lambda_in_the_freeze_aborts(self):
        self.assertIn("lambda", self._reject(
            selected_configuration={"lambda": 0.1}
        ))

    def test_a_different_lower_threshold_in_the_freeze_aborts(self):
        self.assertIn("lower_threshold", self._reject(
            selected_configuration={"lower_threshold": 0.25}
        ))

    def test_a_different_upper_threshold_in_the_freeze_aborts(self):
        self.assertIn("upper_threshold", self._reject(
            selected_configuration={"upper_threshold": 0.75}
        ))

    def test_every_mismatch_is_reported_at_once(self):
        message = self._reject(
            held_out_scored=True,
            selected_configuration={"sigma": 1.0, "lambda": 2.0},
        )
        for fragment in ("held_out_scored", "sigma", "lambda"):
            self.assertIn(fragment, message)


class TestD03CacheVerificationFailsClosed(unittest.TestCase):
    def test_a_wrong_cache_digest_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            path.write_bytes(b"not the D-03 cache")
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.verify_d03_cache(path)
        message = str(caught.exception)
        self.assertIn(D03_CACHE_SHA256, message)

    def test_a_missing_cache_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.verify_d03_cache(Path(tmp) / "absent.sqlite")
        self.assertIn("not found", str(caught.exception))


# ------------------------------------------------------------------ 2. split

class TestFrozenSplitIdentity(unittest.TestCase):
    """The split is verified by passage IDENTITY, not by count alone."""

    @classmethod
    def setUpClass(cls):
        if not DATA_ROOT.exists():
            raise unittest.SkipTest(f"released Wang data absent at {DATA_ROOT}")
        cls.held_out, cls.split = MODULE.frozen_split(str(DATA_ROOT))

    def test_the_frozen_split_reconstructs(self):
        self.assertEqual(
            self.split["validation_fraction"], EXPECTED_VALIDATION_FRACTION
        )
        self.assertEqual(self.split["split_seed"], EXPECTED_SPLIT_SEED)
        self.assertEqual(
            self.split["validation_passages"], EXPECTED_VALIDATION_PASSAGES
        )
        self.assertEqual(
            self.split["held_out_passages"], EXPECTED_HELD_OUT_PASSAGES
        )
        self.assertTrue(self.split["identity_matches"])

    def test_the_passage_id_digests_are_the_frozen_ones(self):
        self.assertEqual(
            self.split["validation_passage_ids_sha256"], VALIDATION_IDS_SHA256
        )
        self.assertEqual(
            self.split["held_out_passage_ids_sha256"], HELD_OUT_IDS_SHA256
        )

    def test_the_held_out_half_is_the_expected_size(self):
        self.assertEqual(len(self.held_out), HELD_OUT_SENTENCES)
        self.assertEqual(self.split["held_out_sentences"], HELD_OUT_SENTENCES)
        self.assertEqual(self.split["held_out_subclaims"], HELD_OUT_SUBCLAIMS)

    def test_the_two_halves_are_disjoint(self):
        overlap = set(self.split["validation_passage_ids"]) & set(
            self.split["held_out_passage_ids"]
        )
        self.assertEqual(overlap, set())

    def test_the_returned_records_are_exactly_the_held_out_passages(self):
        self.assertEqual(
            sorted({int(r.passage_index) for r in self.held_out}),
            sorted(self.split["held_out_passage_ids"]),
        )
        validation = set(self.split["validation_passage_ids"])
        self.assertEqual(
            [r for r in self.held_out if int(r.passage_index) in validation], []
        )


class TestASplitThatIsNotTheFrozenOneAborts(unittest.TestCase):
    """Any drift in the split is an abort, not a silently different sample."""

    def _split_returning(self, validation_ids, held_out_ids, observed_ids=None):
        observed = held_out_ids if observed_ids is None else observed_ids

        class Record:
            def __init__(self, passage_index):
                self.passage_index = passage_index
                self.subclaims = []

        def fake_group_split(records, validation_fraction, random_state):
            return (
                [Record(i) for i in validation_ids],
                [Record(i) for i in observed],
                {
                    "validation_passage_ids": list(validation_ids),
                    "test_passage_ids": list(held_out_ids),
                },
            )

        import src.wang_data as wang_data

        return mock.patch.multiple(
            wang_data,
            group_split_records=fake_group_split,
            load_sentence_records=lambda *a, **k: [],
        )

    def test_a_held_out_half_that_is_not_the_frozen_one_aborts(self):
        with self._split_returning(
            range(48), range(48, 238), observed_ids=range(48, 237)
        ):
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.frozen_split("data/wang")
        self.assertIn("not the frozen ones", str(caught.exception))

    def test_a_wrong_validation_count_aborts(self):
        with self._split_returning(range(47), range(47, 238)):
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.frozen_split("data/wang")
        self.assertIn("validation passages", str(caught.exception))

    def test_a_wrong_held_out_count_aborts(self):
        with self._split_returning(range(48), range(48, 237)):
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.frozen_split("data/wang")
        self.assertIn("held-out passages", str(caught.exception))

    def test_overlapping_halves_abort(self):
        with self._split_returning(range(48), range(47, 237)):
            with self.assertRaises(MODULE.Aborted) as caught:
                MODULE.frozen_split("data/wang")
        self.assertIn("overlap", str(caught.exception))


# ------------------------------------------------------------- 3. checkpoint

class TestCheckpointProvenanceFailsClosed(unittest.TestCase):
    def _report(self, payload):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "d03.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def tearDown(self):
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_a_report_without_a_resolved_revision_aborts(self):
        path = self._report({"provenance": {}, "checkpoint_identity_established": False})
        with self.assertRaises(MODULE.Aborted) as caught:
            MODULE.d03_checkpoint(path)
        message = str(caught.exception)
        self.assertIn("no resolved Hugging Face revision", message)
        self.assertIn("unpinned newer checkpoint", message)

    def test_a_missing_report_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MODULE.Aborted):
                MODULE.d03_checkpoint(Path(tmp) / "absent.json")

    def test_the_resolved_revision_is_read_and_the_limitation_preserved(self):
        revision = "a" * 40
        path = self._report({
            "provenance": {
                "environment": {"checkpoint_identity": {"resolved_revision": revision}}
            },
            "checkpoint_identity_established": False,
            "checkpoint_identity_note": "historical cache revision unknown",
        })
        record = MODULE.d03_checkpoint(path)
        self.assertEqual(record["resolved_revision"], revision)
        # The historical limitation is carried through, never upgraded.
        self.assertIs(record["checkpoint_identity_established"], False)
        self.assertEqual(
            record["checkpoint_identity_limitation"],
            "historical cache revision unknown",
        )
        self.assertIs(record["limitation_preserved"], True)


# --------------------------------------------------------------- 4. estimator

class TestTheEstimatorIsTheFrozenOneOnTheProductionCentres(unittest.TestCase):
    def setUp(self):
        self.factual = [0.9, 0.8, 0.95, 0.7, 0.85, 0.75, 0.88, 0.92]
        self.hallucinated = [0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.12, 0.08]
        self.production = ULSIFDensityRatio(random_state=EXPECTED_SPLIT_SEED).fit(
            self.factual, self.hallucinated
        )

    def _frozen(self):
        return candidate_estimator(
            self.production,
            self.production._as_column(self.factual),
            self.production._as_column(self.hallucinated),
            FROZEN_SIGMA,
            FROZEN_LAMBDA,
        )

    def test_the_centre_set_is_the_production_one_not_resampled(self):
        candidate = self._frozen()
        self.assertIs(candidate.centers, self.production.centers)

    def test_the_frozen_hyperparameters_are_installed_exactly(self):
        candidate = self._frozen()
        self.assertEqual(candidate.sigma, FROZEN_SIGMA)
        self.assertEqual(candidate.lam, FROZEN_LAMBDA)

    def test_only_alpha_is_re_solved(self):
        candidate = self._frozen()
        self.assertEqual(candidate.alpha.shape, self.production.alpha.shape)
        self.assertNotEqual(candidate.sigma, self.production.sigma)

    def test_the_production_fit_is_left_untouched(self):
        before_sigma = self.production.sigma
        before_alpha = self.production.alpha.copy()
        self._frozen()
        self.assertEqual(self.production.sigma, before_sigma)
        self.assertTrue((self.production.alpha == before_alpha).all())


# ----------------------------------------------------- 5. nothing is selected

class TestNoSelectionIsReachableFromTheRunner(unittest.TestCase):
    """The runner is an evaluator. No tuner, grid or search is reachable."""

    def test_no_tuning_or_selection_function_is_imported(self):
        imported = set()
        for node in ast.walk(TREE):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        banned = {
            "select_threshold_configuration",
            "cost_consistent_thresholds",
            "production_hyperparameter_pairs",
            "robustness_summary",
            "CANDIDATE_LOWER_GRID",
            "CANDIDATE_UPPER_GRID",
            "select_configuration",
            "GridSearchCV",
        }
        self.assertEqual(imported & banned, set())

    def test_no_threshold_or_hyperparameter_grid_is_iterated(self):
        for name in (
            "CANDIDATE_LOWER_GRID",
            "CANDIDATE_UPPER_GRID",
            "cost_consistent_thresholds(",
            "production_hyperparameter_pairs(",
            "select_threshold_configuration(",
            "itertools.product",
        ):
            self.assertNotIn(name, SOURCE, f"{name} must be unreachable here")

    def test_each_detector_is_constructed_exactly_once(self):
        self.assertEqual(SOURCE.count("BSEDetector("), 1)
        self.assertEqual(SOURCE.count("DDREDetector("), 1)

    def test_each_method_is_evaluated_exactly_once(self):
        self.assertEqual(SOURCE.count("evaluate_detector_with_traces("), 2)

    def test_the_frozen_numbers_are_the_only_ddre_configuration_used(self):
        """DDREDetector is given the frozen constants, not variables."""
        self.assertIn("lower_threshold=FROZEN_LOWER", SOURCE)
        self.assertIn("upper_threshold=FROZEN_UPPER", SOURCE)
        self.assertIn("FROZEN_SIGMA,\n        FROZEN_LAMBDA,", SOURCE)

    def test_the_runner_records_that_it_tuned_nothing(self):
        self.assertIn('"tuning_performed_in_this_run": False', SOURCE)
        self.assertIn('"held_out_configuration_changed": False', SOURCE)
        self.assertIn('"held_out_evaluated_once": True', SOURCE)


class TestTheFrozenConfigurationHasNoCommandLineRoute(unittest.TestCase):
    """Requirement: no CLI argument may change sigma, lambda, lower or upper."""

    def setUp(self):
        self.destinations = set()
        self.flags = set()
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "add_argument"
            ):
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    self.flags.add(argument.value)
                    self.destinations.add(
                        argument.value.lstrip("-").replace("-", "_")
                    )
            for keyword in node.keywords:
                if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                    self.destinations.add(keyword.value.value)

    def test_the_parser_was_actually_found(self):
        """Guard the guard: an empty flag set would pass everything below."""
        self.assertIn("--data-root", self.flags)
        self.assertIn("--dry-run", self.flags)

    def test_no_argument_can_set_sigma_lambda_or_either_threshold(self):
        for banned in ("sigma", "lambda", "lower", "upper", "lower_threshold",
                       "upper_threshold", "threshold", "thresholds"):
            self.assertNotIn(banned, self.destinations)

    def test_no_flag_name_mentions_a_frozen_quantity(self):
        for flag in self.flags:
            lowered = flag.lower()
            for banned in ("sigma", "lambda", "lower", "upper", "threshold"):
                self.assertNotIn(banned, lowered, f"{flag} exposes {banned}")

    def test_the_frozen_constants_are_module_literals(self):
        """Assigned once, at module level, from a numeric constant."""
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("FROZEN_"):
                    assignments[target.id] = node.value
        self.assertEqual(
            sorted(assignments),
            ["FROZEN_LAMBDA", "FROZEN_LOWER", "FROZEN_SIGMA", "FROZEN_UPPER"],
        )
        for name, value in assignments.items():
            self.assertIsInstance(value, ast.Constant, name)
            self.assertIsInstance(value.value, float, name)
        # And never rebound: exactly one Store each, the defining assignment.
        stores = [
            node.id for node in ast.walk(TREE)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in assignments
        ]
        for name in assignments:
            self.assertEqual(
                stores.count(name), 1, f"{name} is assigned more than once"
            )


# ------------------------------------------------------- 6. the held-out gate

class TestDryRunScoresNothing(unittest.TestCase):
    def test_dry_run_returns_before_any_inference(self):
        """The dry run exits before the cache, the model and the detectors."""
        lines = SOURCE.splitlines()
        dry_run_exit = next(
            index for index, line in enumerate(lines)
            if line.strip() == "if args.dry_run:"
        )
        for later in (
            "import torch",
            "from transformers import",
            "prepare_derived_cache(",
            "EntailmentScorer(",
            "evaluate_detector_with_traces(",
            "BSEDetector(",
            "DDREDetector(",
        ):
            index = next(
                i for i, line in enumerate(lines) if later in line and i > 40
            )
            self.assertGreater(
                index, dry_run_exit, f"{later} is reachable before --dry-run exits"
            )

    def test_the_dry_run_says_it_scored_nothing(self):
        self.assertIn("NO held-out evidence was scored", SOURCE)
        self.assertIn("no model was loaded", SOURCE)

    def test_no_model_is_loaded_at_import_time(self):
        """Importing the runner must not touch torch or transformers."""
        top_level = set()
        for node in TREE.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        self.assertNotIn("torch", top_level)
        self.assertNotIn("transformers", top_level)


class TestHeldOutIsScoredOnlyByTheTwoFrozenMethods(unittest.TestCase):
    def test_the_held_out_records_reach_only_the_two_evaluations(self):
        calls = [
            node for node in ast.walk(TREE)
            if isinstance(node, ast.Call)
            and any(
                isinstance(argument, ast.Name) and argument.id == "held_out"
                for argument in node.args
            )
        ]
        names = sorted(
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr
            for node in calls
        )
        # len() counts sentences for the report; the two evaluations and the
        # two prediction-row builds are the only things that READ the records.
        self.assertEqual(
            names,
            [
                "evaluate_detector_with_traces",
                "evaluate_detector_with_traces",
                "len",
                "prediction_rows",
                "prediction_rows",
            ],
        )

    def test_the_nbc_cache_reader_is_the_read_only_scorer(self):
        """NBC histograms replay the D-03 cache; they never call a model."""
        self.assertIn("CachedDocumentScorer(", SOURCE)
        self.assertIn("MissingPairScore", SOURCE)
        self.assertIn("the D-03 cache is missing an NBC score", SOURCE)

    def test_the_held_out_cache_is_a_separate_artifact_from_the_d03_cache(self):
        self.assertIn("prepare_derived_cache(", SOURCE)
        self.assertIn("--held-out-cache", SOURCE)
        self.assertIn('copy_faithful") is not True', SOURCE)


# ---------------------------------------------------------- 7. honest output

class TestTheArtifactRecordsTheSelectionHistory(unittest.TestCase):
    def test_the_output_records_validation_tuning_and_its_budget(self):
        self.assertIn('"validation_tuned": True', SOURCE)
        self.assertIn(
            '"validation_configurations_considered": '
            "VALIDATION_CONFIGURATIONS_CONSIDERED",
            SOURCE,
        )
        self.assertIn(
            '"eligible_validation_configurations": '
            "ELIGIBLE_VALIDATION_CONFIGURATIONS",
            SOURCE,
        )

    def test_the_tuning_budget_disclosure_is_verbatim_and_in_the_artifact(self):
        self.assertEqual(
            MODULE.TUNING_BUDGET_DISCLOSURE,
            "DDRE received validation-based model/threshold selection across "
            "640 configurations whereas BSE remains the fixed published "
            "baseline. This tuning-budget asymmetry must be disclosed in the "
            "paper.",
        )
        self.assertIn('"tuning_budget_disclosure": TUNING_BUDGET_DISCLOSURE', SOURCE)

    def test_the_selection_history_is_not_claimed_to_be_pre_registered(self):
        note = MODULE.SELECTION_HISTORY_NOTE
        self.assertIn("NOT the original pre-registered", note)
        self.assertIn("640", note)
        self.assertIn("NOT_CONFIRMATORY", note)
        self.assertIn('"selection_history_note": SELECTION_HISTORY_NOTE', SOURCE)

    def test_the_claim_is_assessed_as_non_confirmatory(self):
        """A SUPPORTED label must not be obtained by rewriting the history."""
        self.assertIn("validation_selection=VALIDATION_SELECTION_POST_D03_SWEEP", SOURCE)
        self.assertNotIn("VALIDATION_SELECTION_PREREGISTERED_CV", SOURCE)
        self.assertNotIn("validation_selection_confirmatory=True", SOURCE)

    def test_the_selection_history_is_not_reported_as_the_fallback(self):
        """The frozen config came from the post-D-03 sweep, not the fallback.

        The original run passed a bare False, which assess_claim rendered as
        "used the fallback objective" -- false for a selection with 102
        eligible configurations.
        """
        self.assertNotIn("VALIDATION_SELECTION_FALLBACK", SOURCE)
        self.assertNotIn("validation_selection_confirmatory=False", SOURCE)

    def test_the_verified_split_is_passed_to_the_claim_assessment(self):
        """frozen_split verifies the split; assess_claim must be told so."""
        call = SOURCE[SOURCE.index("claim = assess_claim("):]
        call = call[: call.index("\n    )\n")]
        self.assertIn("split_metadata=claim_split_metadata(split)", call)
        self.assertIn('expected_validation_passage_ids=split["validation_passage_ids"]', call)
        self.assertIn('expected_test_passage_ids=split["held_out_passage_ids"]', call)

    def test_claim_split_metadata_compares_the_evaluated_passages(self):
        from src.paired_bootstrap import (
            VALIDATION_SELECTION_POST_D03_SWEEP,
            assess_claim,
        )

        split = {
            "validation_fraction": EXPECTED_VALIDATION_FRACTION,
            "split_seed": EXPECTED_SPLIT_SEED,
            "validation_passage_ids": [1, 2],
            "held_out_passage_ids": [3, 4, 5],
            "observed_held_out_passage_ids": [3, 4, 5],
        }
        meta = MODULE.claim_split_metadata(split)
        self.assertEqual(meta["test_passage_ids"], [3, 4, 5])
        self.assertEqual(meta["random_state"], EXPECTED_SPLIT_SEED)

        def assess(observed):
            return assess_claim(
                None,
                validation_selection=VALIDATION_SELECTION_POST_D03_SWEEP,
                quality_tolerance=0.005,
                bootstrap_error="not needed here",
                split_metadata=MODULE.claim_split_metadata(
                    dict(split, observed_held_out_passage_ids=observed)
                ),
                expected_validation_passage_ids=split["validation_passage_ids"],
                expected_test_passage_ids=split["held_out_passage_ids"],
            )

        verified = assess([3, 4, 5])
        self.assertTrue(verified["split_matches_frozen"])
        self.assertEqual(verified["split_mismatches"], [])
        reasons = " ".join(verified["confirmatory_disqualifiers"])
        self.assertIn("after the D-03 diagnostic", reasons)
        self.assertNotIn("fallback", reasons)
        self.assertNotIn("split was not verified", reasons)
        self.assertEqual(verified["claim_status"], "NOT_CONFIRMATORY")

        drifted = assess([3, 4, 6])
        self.assertFalse(drifted["split_matches_frozen"])

    def test_the_two_primary_methods_are_bse_official_and_frozen_ddre(self):
        self.assertIn(
            '"methods_evaluated": ["bse_official", "frozen_ddre"]', SOURCE
        )
        self.assertIn('mode="official"', SOURCE)

    def test_the_published_cost_configuration_is_unchanged(self):
        self.assertEqual(MODULE.C_MISS, 28.0)
        self.assertEqual(MODULE.C_FALSE_ALARM, 96.0)
        self.assertEqual(MODULE.C_RETRIEVE, 1.0)
        self.assertEqual(MODULE.P0, 0.5)
        self.assertEqual(MODULE.MAX_DOCS, 10)

    def test_no_fairness_correction_is_applied_to_the_endpoint(self):
        self.assertIn('"no_fairness_correction_applied": True', SOURCE)
        self.assertIn('"retrieval_protocol_asymmetry": asymmetry', SOURCE)

    def test_the_bootstrap_is_the_frozen_confirmatory_protocol(self):
        """No resample count, seed or CI level is chosen here."""
        self.assertIn("paired_passage_bootstrap(ddre_obs, bse_obs)", SOURCE)
        self.assertIn('"confirmatory_protocol": confirmatory_protocol()', SOURCE)
        for banned in ("n_resamples=", "ci_level=", "seed="):
            self.assertNotIn(banned, SOURCE, f"{banned} must not be set here")


class TestTheReportReadsTheRealAsymmetryKeys(unittest.TestCase):
    """The D-02 disclosure must print, not silently vanish on a key rename."""

    def test_the_report_indexes_the_keys_src_evaluation_actually_returns(self):
        from src.evaluation import retrieval_protocol_asymmetry  # noqa: F401

        for key in (
            "asymmetry['bse_official']['zero_retrieval_nonempty_subclaims']",
            "asymmetry['ddre_ulsif']['zero_retrieval_nonempty_subclaims']",
            "asymmetry['ddre_first_retrieval_floor_documents']",
            "asymmetry['ddre_documents_above_first_retrieval_floor']",
        ):
            self.assertIn(key, SOURCE)

    def test_the_report_does_not_probe_for_optional_keys(self):
        """`if key in asymmetry` would hide a rename instead of failing."""
        self.assertNotIn("if key in asymmetry", SOURCE)

    def test_the_printed_keys_are_the_ones_the_producer_emits(self):
        """Cross-checked against src.evaluation's own return dict.

        A live call needs trace-bearing observations, which needs NLI
        inference; the producer's AST gives the same guarantee offline.
        """
        producer = ast.parse(
            (PROJECT_ROOT / "src" / "evaluation.py").read_text(encoding="utf-8")
        )
        function = next(
            node for node in ast.walk(producer)
            if isinstance(node, ast.FunctionDef)
            and node.name == "retrieval_protocol_asymmetry"
        )
        emitted = {
            key.value for node in ast.walk(function)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        for key in (
            "bse_official",
            "ddre_ulsif",
            "zero_retrieval_nonempty_subclaims",
            "ddre_first_retrieval_floor_documents",
            "ddre_documents_above_first_retrieval_floor",
            "confirmatory_endpoint_unchanged",
        ):
            self.assertIn(key, emitted)
        # And the names the earlier draft guessed are NOT what it emits.
        for wrong in (
            "bse_zero_retrieval_non_empty_subclaims",
            "ddre_first_retrieval_floor",
            "ddre_documents_above_floor",
        ):
            self.assertNotIn(wrong, emitted)


if __name__ == "__main__":
    unittest.main()
