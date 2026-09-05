"""Tests for audit findings D-11 and D-12.

**D-11:** the fallback objective's retrieval penalty must be *dimensionless*.
It divided a per-sentence document count by a per-subclaim budget.

Every threshold candidate in one tuning run is evaluated on the SAME
``validation_records``, so the sentence and subclaim counts are fixed and

    avg_docs_per_sentence = avg_docs_per_subclaim * subclaims_per_sentence

with ``subclaims_per_sentence`` a constant (~1.57 on this dataset). The old
quantity was therefore the correct cost multiplied by that constant, i.e. the
fallback used the wrong **exchange rate** between balanced PR-AUC and retrieval
cost -- and because balanced PR-AUC is not scaled with it, that can select a
different fallback configuration, which ``TestRealisticFallbackRegression``
exhibits. The corrected cost is
``avg_retrieved_documents_per_subclaim / max_documents_per_subclaim``.

**D-12:** publishing result artifacts must be opt-in, and an automatic push
must never touch a protected branch.

CPU-only, synthetic. No model, no GPU, no tuning on real data, no held-out
inference. ``main.py`` imports torch, so its behaviour is asserted on its AST
and on the functions that can be imported without it.
"""

import argparse
import ast
import contextlib
import io
import re
import unittest
from pathlib import Path

from src.threshold_selection import (
    FALLBACK_COST_NORMALIZATION,
    FALLBACK_COST_UNITS,
    FALLBACK_SELECTION_RULE,
    SAFEGUARD_NOTE,
    candidate_record,
    select_threshold_configuration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collapsed(source):
    """Source with Python's implicit string concatenation joined up.

    Lets a test assert on the *message a user sees* rather than on where the
    line happened to wrap.
    """
    return re.sub(r'"\s*\n\s*"', "", source)


BASELINE = {
    "nonfactual": {"auc_pr": 0.80},
    "factual": {"auc_pr": 0.60},
    "balanced_pr_auc": 0.70,
}


def metrics_for(
    *,
    documents_per_sentence,
    documents_per_subclaim,
    nonfactual=0.80,
    factual=0.60,
):
    """A metrics dict shaped like ``summarize_method``'s output.

    The two document counts are supplied independently, because their being
    *different* quantities is the whole of D-11.
    """
    return {
        "nonfactual": {"auc_pr": nonfactual},
        "factual": {"auc_pr": factual},
        "balanced_pr_auc": 0.5 * (nonfactual + factual),
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "efficiency": {
            "avg_retrieved_documents_per_sentence": documents_per_sentence,
            "avg_retrieved_documents_per_subclaim": documents_per_subclaim,
            "avg_nli_span_calls_per_sentence": documents_per_sentence * 3.0,
        },
    }


def record_for(
    *,
    documents_per_sentence,
    documents_per_subclaim,
    max_docs=10,
    retrieval_penalty=0.05,
    nonfactual=0.80,
    factual=0.60,
    lower=0.10,
    upper=0.60,
    quality_tolerance=0.005,
):
    return candidate_record(
        lower,
        upper,
        metrics_for(
            documents_per_sentence=documents_per_sentence,
            documents_per_subclaim=documents_per_subclaim,
            nonfactual=nonfactual,
            factual=factual,
        ),
        BASELINE,
        quality_tolerance=quality_tolerance,
        retrieval_penalty=retrieval_penalty,
        max_docs=max_docs,
    )


# --------------------------------------------------------------------------
# D-11: the fallback cost is a genuine fraction of the per-subclaim budget
# --------------------------------------------------------------------------


class TestFallbackCostNormalization(unittest.TestCase):
    def test_the_cost_uses_the_per_subclaim_count_not_the_per_sentence_one(self):
        # The exact case in the finding: 8 documents/sentence, 4
        # documents/subclaim, a budget of 10 documents/subclaim.
        record = record_for(
            documents_per_sentence=8.0, documents_per_subclaim=4.0, max_docs=10
        )
        self.assertEqual(record["normalized_document_cost"], 0.4)
        self.assertNotEqual(record["normalized_document_cost"], 0.8)

    def test_the_cost_is_invariant_to_the_supplied_per_sentence_number(self):
        # UNIT / CROSS-DATASET property, NOT two candidates from one fixed
        # validation split. Within a single tuning run every candidate sees the
        # same records, so subclaims-per-sentence is constant and this pair is
        # not realizable there; the realizable consequence of D-11 is in
        # TestRealisticFallbackRegression. What this pins is narrower and still
        # worth pinning: the cost reads the per-subclaim count and nothing else,
        # so a per-sentence number supplied alongside it cannot influence it.
        cheap_sentences = record_for(
            documents_per_sentence=4.0, documents_per_subclaim=2.0, max_docs=10
        )
        dense_sentences = record_for(
            documents_per_sentence=8.0, documents_per_subclaim=2.0, max_docs=10
        )
        self.assertEqual(cheap_sentences["normalized_document_cost"], 0.2)
        self.assertEqual(dense_sentences["normalized_document_cost"], 0.2)
        self.assertEqual(
            cheap_sentences["fallback_objective"],
            dense_sentences["fallback_objective"],
        )
        # And the per-sentence counts they were built from really do differ, so
        # the equality above is not vacuous.
        self.assertNotEqual(
            cheap_sentences["avg_documents"], dense_sentences["avg_documents"]
        )

    def test_a_full_budget_candidate_costs_exactly_one(self):
        record = record_for(
            documents_per_sentence=15.0, documents_per_subclaim=10.0, max_docs=10
        )
        self.assertEqual(record["normalized_document_cost"], 1.0)

    def test_zero_retrieval_costs_exactly_zero(self):
        record = record_for(
            documents_per_sentence=0.0, documents_per_subclaim=0.0, max_docs=10
        )
        self.assertEqual(record["normalized_document_cost"], 0.0)
        self.assertEqual(record["fallback_objective"], record["balanced_pr_auc"])

    def test_the_cost_equals_total_documents_over_subclaims_times_budget(self):
        # The equivalent closed form, stated so the definition is unambiguous.
        total_documents, total_subclaims, max_docs = 900.0, 300.0, 10
        record = record_for(
            documents_per_sentence=4.7,  # irrelevant to the fallback cost
            documents_per_subclaim=total_documents / total_subclaims,
            max_docs=max_docs,
        )
        self.assertAlmostEqual(
            record["normalized_document_cost"],
            total_documents / (total_subclaims * max_docs),
        )

    def test_the_fallback_objective_uses_the_corrected_value_exactly(self):
        record = record_for(
            documents_per_sentence=8.0,
            documents_per_subclaim=4.0,
            max_docs=10,
            retrieval_penalty=0.05,
        )
        self.assertAlmostEqual(
            record["fallback_objective"],
            record["balanced_pr_auc"] - 0.05 * 0.4,
        )
        # Not the old per-sentence normalization.
        self.assertNotAlmostEqual(
            record["fallback_objective"],
            record["balanced_pr_auc"] - 0.05 * 0.8,
        )

    def test_the_penalty_value_itself_is_recorded_not_rescaled(self):
        record = record_for(
            documents_per_sentence=3.0,
            documents_per_subclaim=3.0,
            retrieval_penalty=0.05,
        )
        self.assertEqual(record["fallback_retrieval_penalty"], 0.05)


class TestFallbackCostValidation(unittest.TestCase):
    """Impossible costs are refused, never clamped."""

    def test_a_zero_budget_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            record_for(
                documents_per_sentence=1.0, documents_per_subclaim=1.0, max_docs=0
            )
        self.assertIn("must be positive", str(caught.exception))

    def test_a_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            record_for(
                documents_per_sentence=1.0, documents_per_subclaim=1.0, max_docs=-5
            )

    def test_more_documents_than_the_budget_is_rejected_not_clamped(self):
        # A subclaim cannot consume more than max_docs documents, so this is
        # either a unit inconsistency or a corrupted count. Clamping it to 1.0
        # would hide exactly the error this check exists to catch.
        with self.assertRaises(ValueError) as caught:
            record_for(
                documents_per_sentence=15.0,
                documents_per_subclaim=12.0,
                max_docs=10,
            )
        message = str(caught.exception)
        self.assertIn("cannot consume more than", message)
        self.assertIn("NOT clamped", message)

    def test_a_per_sentence_count_supplied_by_mistake_is_caught(self):
        # The concrete regression: passing the per-sentence count (which can
        # exceed the per-subclaim budget once sentences carry ~1.57 subclaims)
        # where the per-subclaim count belongs.
        with self.assertRaises(ValueError) as caught:
            record_for(
                documents_per_sentence=15.7,
                documents_per_subclaim=15.7,
                max_docs=10,
            )
        self.assertIn("unit inconsistency", str(caught.exception))

    def test_a_negative_retrieval_count_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            record_for(
                documents_per_sentence=1.0,
                documents_per_subclaim=-0.5,
                max_docs=10,
            )
        self.assertIn("negative", str(caught.exception))

    def test_exactly_the_budget_is_accepted(self):
        # The boundary is legal: it is a full budget, not an impossible one.
        record = record_for(
            documents_per_sentence=10.0, documents_per_subclaim=10.0, max_docs=10
        )
        self.assertEqual(record["normalized_document_cost"], 1.0)


class TestFallbackProvenanceIsExplicit(unittest.TestCase):
    """A reviewer must not have to infer the units."""

    def test_every_candidate_records_both_counts_and_the_normalization(self):
        record = record_for(
            documents_per_sentence=8.0, documents_per_subclaim=4.0, max_docs=10
        )
        self.assertEqual(record["avg_documents"], 8.0)
        self.assertEqual(record["avg_documents_per_subclaim"], 4.0)
        self.assertEqual(record["normalized_document_cost"], 0.4)
        self.assertEqual(record["max_documents_per_subclaim"], 10.0)
        self.assertEqual(record["fallback_retrieval_penalty"], 0.05)
        self.assertIn("fallback_objective", record)

    def test_the_normalization_is_stated_in_words_on_every_candidate(self):
        record = record_for(
            documents_per_sentence=2.0, documents_per_subclaim=2.0
        )
        self.assertEqual(
            record["fallback_cost_normalization"],
            "avg_retrieved_documents_per_subclaim / max_documents_per_subclaim",
        )
        self.assertEqual(record["fallback_cost_normalization"],
                         FALLBACK_COST_NORMALIZATION)
        self.assertIn("documents/subclaim", record["fallback_cost_units"])
        self.assertEqual(record["fallback_cost_units"], FALLBACK_COST_UNITS)

    def test_the_selection_rules_state_the_normalization(self):
        self.assertIn(FALLBACK_COST_NORMALIZATION, FALLBACK_SELECTION_RULE)
        self.assertIn(FALLBACK_COST_NORMALIZATION, SAFEGUARD_NOTE)
        self.assertIn("PER SENTENCE", SAFEGUARD_NOTE)


class TestFeasibleSelectionIsUnchanged(unittest.TestCase):
    """D-11 corrects the FALLBACK only."""

    def test_feasible_selection_still_minimises_documents_per_sentence(self):
        # The candidate with the lower normalized (per-subclaim) cost is the
        # one with MORE documents per sentence. Feasible selection must still
        # pick the per-sentence minimum, so the corrected fallback scale cannot
        # leak into the primary objective.
        fewer_per_sentence = record_for(
            documents_per_sentence=2.0,
            documents_per_subclaim=8.0,
            lower=0.10, upper=0.60,
        )
        lower_normalized_cost = record_for(
            documents_per_sentence=9.0,
            documents_per_subclaim=1.0,
            lower=0.15, upper=0.65,
        )
        self.assertTrue(fewer_per_sentence["preserves_baseline_quality"])
        self.assertTrue(lower_normalized_cost["preserves_baseline_quality"])
        self.assertGreater(
            fewer_per_sentence["normalized_document_cost"],
            lower_normalized_cost["normalized_document_cost"],
        )
        selected, rule, confirmatory = select_threshold_configuration(
            [lower_normalized_cost, fewer_per_sentence]
        )
        self.assertEqual(selected["avg_documents"], 2.0)
        self.assertTrue(confirmatory)
        self.assertIn("minimum retrieval cost", rule)

    def test_the_feasible_tie_break_order_is_unchanged(self):
        # Exact document tie -> balanced, then nonfactual, then factual.
        worse_balanced = record_for(
            documents_per_sentence=3.0, documents_per_subclaim=3.0,
            nonfactual=0.80, factual=0.60, lower=0.10, upper=0.60,
        )
        better_balanced = record_for(
            documents_per_sentence=3.0, documents_per_subclaim=9.0,
            nonfactual=0.82, factual=0.62, lower=0.15, upper=0.65,
        )
        selected, _, _ = select_threshold_configuration(
            [worse_balanced, better_balanced]
        )
        self.assertEqual(selected["lower"], 0.15)
        # Chosen on quality despite the HIGHER normalized fallback cost.
        self.assertGreater(
            selected["normalized_document_cost"],
            worse_balanced["normalized_document_cost"],
        )


class TestRealisticFallbackRegression(unittest.TestCase):
    """The consequence of D-11 under the tuner's real invariant.

    Every threshold candidate in one tuning run is scored on the SAME
    ``validation_records``, so the sentence and subclaim counts are fixed and
    every candidate satisfies

        avg_docs_per_sentence / avg_docs_per_subclaim
            == N_subclaims / N_sentences   (a constant, ~1.57 here)

    That invariant is what makes the pair below realizable, and it is asserted
    directly. Because the old cost was the correct cost multiplied by that
    constant while balanced PR-AUC was NOT, the two objectives are not
    order-equivalent and can choose differently -- which is the actual defect,
    not any claim about candidates seeing different data.
    """

    SUBCLAIMS_PER_SENTENCE = 1.5
    MAX_DOCS = 10
    PENALTY = 0.05  # the production default, unchanged

    def candidates(self):
        # Chosen so 0.2 * PENALTY < (balanced_b - balanced_a) < 0.3 * PENALTY,
        # the window in which the corrected and the old objectives disagree.
        cheap = record_for(
            documents_per_subclaim=2.0,
            documents_per_sentence=2.0 * self.SUBCLAIMS_PER_SENTENCE,
            nonfactual=0.3000, factual=0.3000,   # balanced 0.3000
            max_docs=self.MAX_DOCS, retrieval_penalty=self.PENALTY,
            lower=0.10, upper=0.60,
        )
        better_but_dearer = record_for(
            documents_per_subclaim=4.0,
            documents_per_sentence=4.0 * self.SUBCLAIMS_PER_SENTENCE,
            nonfactual=0.3125, factual=0.3125,   # balanced 0.3125
            max_docs=self.MAX_DOCS, retrieval_penalty=self.PENALTY,
            lower=0.15, upper=0.65,
        )
        return cheap, better_but_dearer

    def old_objective(self, record):
        """What the pre-D-11 code computed: the PER-SENTENCE count over max_docs."""
        return record["balanced_pr_auc"] - self.PENALTY * (
            record["avg_documents"] / self.MAX_DOCS
        )

    def test_every_candidate_shares_one_subclaims_per_sentence_constant(self):
        # Without this the pair would not be realizable, and the regression
        # below would be testing a situation the tuner cannot produce.
        ratios = {
            record["avg_documents"] / record["avg_documents_per_subclaim"]
            for record in self.candidates()
        }
        self.assertEqual(ratios, {self.SUBCLAIMS_PER_SENTENCE})

    def test_the_old_and_corrected_objectives_disagree(self):
        cheap, better_but_dearer = self.candidates()
        self.assertGreater(
            self.old_objective(cheap), self.old_objective(better_but_dearer),
            "the old per-sentence objective should prefer the cheap candidate",
        )
        self.assertGreater(
            better_but_dearer["fallback_objective"], cheap["fallback_objective"],
            "the corrected objective should prefer the dearer, better candidate",
        )

    def test_the_tuner_now_selects_the_configuration_the_corrected_scale_prefers(self):
        cheap, better_but_dearer = self.candidates()
        # Neither preserves baseline quality, so the fallback decides.
        self.assertFalse(cheap["preserves_baseline_quality"])
        self.assertFalse(better_but_dearer["preserves_baseline_quality"])

        selected, rule, confirmatory = select_threshold_configuration(
            [cheap, better_but_dearer]
        )
        self.assertEqual((selected["lower"], selected["upper"]), (0.15, 0.65))
        # And it is NOT what the old normalization would have picked.
        self.assertNotEqual(
            (selected["lower"], selected["upper"]),
            (cheap["lower"], cheap["upper"]),
        )
        self.assertFalse(confirmatory)
        self.assertIs(rule, FALLBACK_SELECTION_RULE)

    def test_the_old_cost_was_the_correct_cost_times_the_constant(self):
        # The relationship stated as an identity, since it is the whole of the
        # finding: the old objective used an exchange rate inflated by the
        # subclaims-per-sentence factor.
        for record in self.candidates():
            old_cost = record["avg_documents"] / self.MAX_DOCS
            self.assertAlmostEqual(
                old_cost,
                record["normalized_document_cost"] * self.SUBCLAIMS_PER_SENTENCE,
            )


class TestFallbackRemainsNonConfirmatory(unittest.TestCase):
    def test_a_fallback_selection_is_never_confirmatory(self):
        candidates = [
            record_for(
                documents_per_sentence=2.0, documents_per_subclaim=2.0,
                nonfactual=0.10, factual=0.10, lower=0.10, upper=0.60,
            )
        ]
        _, rule, confirmatory = select_threshold_configuration(candidates)
        self.assertFalse(confirmatory)
        self.assertIn("NOT ELIGIBLE", rule)


# --------------------------------------------------------------------------
# D-12: publishing is opt-in, and never touches a protected branch
# --------------------------------------------------------------------------


class FakeCompleted:
    def __init__(self, returncode=0):
        self.returncode = returncode


class RecordingGit:
    """Records every git invocation so ordering can be asserted, not just outcome.

    ``git diff --cached --quiet`` is now consulted twice -- once BEFORE staging,
    to refuse a pre-existing dirty index, and once after, to detect that the
    result files were unchanged. The two are answered separately so a test can
    say which one it is exercising.

    Return codes follow git: 0 means no staged differences, 1 means there are
    some, anything else means the check itself failed.
    """

    def __init__(self, branch, index_before_add=0, index_after_add=1):
        self.branch = branch
        self.index_before_add = index_before_add
        self.index_after_add = index_after_add
        self.calls = []
        self.added = False

    def check_output(self, argv, text=True):
        self.calls.append(list(argv))
        return self.branch + "\n"

    def run(self, argv, check=False):
        self.calls.append(list(argv))
        if argv[:2] == ["git", "add"]:
            self.added = True
            return FakeCompleted(0)
        if argv[:3] == ["git", "diff", "--cached"]:
            return FakeCompleted(
                self.index_after_add if self.added else self.index_before_add
            )
        return FakeCompleted(0)

    def subcommands(self):
        return [c[1] for c in self.calls if len(c) > 1]


def load_auto_push_results():
    """``auto_push_results`` without importing torch-dependent ``main``.

    ``main.py`` imports torch at module scope, which CI does not install. The
    function under test is pure subprocess plumbing, so its source is executed
    in an isolated namespace with a recording stand-in for ``subprocess``.
    """
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"auto_push_results"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ] + [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "PROTECTED_BRANCHES"
            for t in node.targets
        )
    ]
    assert len(nodes) == 2, f"expected the function and the constant, got {nodes}"
    module = ast.Module(body=sorted(nodes, key=lambda n: n.lineno), type_ignores=[])
    namespace = {}
    exec(compile(module, "main.py", "exec"), namespace)
    return namespace


class TestPublicationIsOptIn(unittest.TestCase):
    """The CLI default must not resolve to pushing."""

    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

    def parser(self):
        # The real flag definitions, lifted out of parse_args so the test does
        # not need torch. Kept in step with main.py by
        # test_the_flags_are_defined_in_main_as_a_mutually_exclusive_group.
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--push-results", action="store_true", default=False)
        group.add_argument("--no-push-results", action="store_true", default=False)
        parser.add_argument("--smoke-test", action="store_true")
        return parser

    def test_the_default_does_not_push(self):
        args = self.parser().parse_args([])
        self.assertFalse(args.push_results)

    def test_the_explicit_flag_requests_a_push(self):
        args = self.parser().parse_args(["--push-results"])
        self.assertTrue(args.push_results)

    def test_the_deprecated_flag_leaves_pushing_disabled(self):
        args = self.parser().parse_args(["--no-push-results"])
        self.assertFalse(args.push_results)
        self.assertTrue(args.no_push_results)

    def test_the_two_flags_together_are_rejected(self):
        parser = self.parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--push-results", "--no-push-results"])

    def test_the_flags_are_defined_in_main_as_a_mutually_exclusive_group(self):
        self.assertIn("add_mutually_exclusive_group()", self.source)
        self.assertIn('"--push-results"', self.source)
        self.assertIn('"--no-push-results"', self.source)

    def test_mains_own_flag_definitions_default_to_not_pushing(self):
        # Read main.py's REAL argparse call rather than a copy of it. A test
        # that rebuilds the parser itself cannot see main.py's default drift.
        defaults = {}
        for node in ast.walk(ast.parse(self.source)):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ("--push-results", "--no-push-results")
            ):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            self.assertIn(
                "default", keywords, f"{node.args[0].value} has no explicit default"
            )
            default = keywords["default"]
            self.assertIsInstance(default, ast.Constant)
            defaults[node.args[0].value] = default.value
            self.assertEqual(keywords["action"].value, "store_true")

        self.assertEqual(
            defaults, {"--push-results": False, "--no-push-results": False},
            "neither flag may default to a value that resolves to pushing",
        )

    def test_the_help_text_says_publishing_is_explicit(self):
        self.assertIn("Explicitly allow committing and pushing", self.source)
        self.assertIn("Off by default", self.source)

    def test_the_old_opt_out_condition_is_gone(self):
        self.assertNotIn("not args.no_push_results", self.source)


class TestPublicationControlFlow(unittest.TestCase):
    """Asserted on main.py's AST; running it needs torch and the model."""

    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_the_push_is_guarded_by_the_explicit_request(self):
        self.assertIn(
            "if not args.smoke_test and args.push_results:\n"
            "            push_status = auto_push_results(",
            self.source,
        )

    def test_a_smoke_test_never_pushes_even_when_asked(self):
        # The guard requires `not args.smoke_test` regardless of the flag.
        guard = "if not args.smoke_test and args.push_results:"
        self.assertIn(guard, self.source)
        self.assertIn("elif args.smoke_test and args.push_results:", self.source)
        self.assertIn("are never auto-pushed", collapsed(self.source))

    def test_declining_to_push_is_not_reported_as_a_failure(self):
        self.assertIn(
            "Result artifacts written locally; automatic push not requested.",
            self.source,
        )
        for alarming in ("ERROR", "WARNING: results were not pushed", "FAILED"):
            self.assertNotIn(
                f'"{alarming}: automatic push not requested', self.source
            )

    def test_only_one_call_site_can_push(self):
        self.assertEqual(self.source.count("auto_push_results("), 2)  # def + call

    def test_the_summary_records_the_publication_intent(self):
        self.assertIn('"result_publication": {', self.source)
        self.assertIn(
            '"automatic_push_requested": bool(args.push_results)', self.source
        )
        self.assertIn(
            '"policy": "opt-in; protected branches are never auto-pushed"',
            self.source,
        )
        # Intent, not outcome: the summary is not rewritten after a push.
        self.assertIn(
            "not whether a push was later accepted by the remote",
            collapsed(self.source),
        )


class TestProtectedBranchSafety(unittest.TestCase):
    """An experiment helper must not bypass branch -> PR -> review -> merge."""

    def setUp(self):
        self.namespace = load_auto_push_results()
        self.auto_push_results = self.namespace["auto_push_results"]

    def push(self, branch, index_before_add=0, index_after_add=1):
        git = RecordingGit(branch, index_before_add, index_after_add)
        self.namespace["subprocess"] = git
        self.namespace["datetime"] = __import__("datetime").datetime
        self.namespace["timezone"] = __import__("datetime").timezone
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            status = self.auto_push_results(["results/summary.json"])
        self.printed = captured.getvalue()
        return status, git

    def test_main_is_refused_before_anything_is_staged(self):
        status, git = self.push("main")
        self.assertEqual(status, {"pushed": False, "reason": "protected branch",
                                  "branch": "main"})
        self.assertNotIn("add", git.subcommands())
        self.assertNotIn("commit", git.subcommands())
        self.assertNotIn("push", git.subcommands())

    def test_master_is_refused_before_anything_is_staged(self):
        status, git = self.push("master")
        self.assertEqual(status["reason"], "protected branch")
        self.assertNotIn("add", git.subcommands())

    def test_a_detached_head_is_refused_before_anything_is_staged(self):
        status, git = self.push("HEAD")
        self.assertEqual(status["reason"], "detached HEAD")
        self.assertNotIn("add", git.subcommands())

    def test_an_empty_branch_name_is_refused(self):
        status, git = self.push("")
        self.assertEqual(status["reason"], "detached HEAD")
        self.assertNotIn("add", git.subcommands())

    def test_the_branch_is_resolved_before_git_add(self):
        # Ordering is the point: staging first and refusing afterwards would
        # still leave the index dirty on a protected branch.
        _, git = self.push("claude/some-research-branch")
        subcommands = git.subcommands()
        self.assertLess(subcommands.index("rev-parse"), subcommands.index("add"))

    def test_a_feature_branch_still_stages_commits_and_pushes(self):
        status, git = self.push("claude/some-research-branch")
        self.assertTrue(status["pushed"])
        self.assertEqual(status["branch"], "claude/some-research-branch")
        for subcommand in ("add", "commit", "push"):
            self.assertIn(subcommand, git.subcommands())

    def test_no_result_changes_remains_a_clean_no_op(self):
        # Clean index to begin with, and staging the result files changed
        # nothing: the existing no-op behaviour, unaffected by the new guard.
        status, git = self.push(
            "claude/research", index_before_add=0, index_after_add=0
        )
        self.assertEqual(status, {"pushed": False, "reason": "no changes"})
        self.assertIn("add", git.subcommands())
        self.assertNotIn("commit", git.subcommands())
        self.assertNotIn("push", git.subcommands())

    def test_a_pre_existing_staged_change_refuses_the_push(self):
        # `git commit -m` commits EVERYTHING staged. Someone who ran
        # `git add src/unrelated_work.py` before starting the experiment would
        # otherwise find that file swept into a commit labelled "update full
        # experiment results".
        status, _ = self.push("claude/research", index_before_add=1)
        self.assertFalse(status["pushed"])
        self.assertEqual(status["reason"], "pre-existing staged changes")
        self.assertEqual(status["branch"], "claude/research")

    def test_a_dirty_index_is_never_staged_committed_or_pushed(self):
        _, git = self.push("claude/research", index_before_add=1)
        subcommands = git.subcommands()
        self.assertNotIn("add", subcommands)
        self.assertNotIn("commit", subcommands)
        self.assertNotIn("push", subcommands)

    def test_a_dirty_index_is_not_unstaged_or_partially_committed(self):
        # Fail closed: the index belongs to whoever staged it. Nothing is
        # reset, restored or stashed to work around the refusal.
        _, git = self.push("claude/research", index_before_add=1)
        for repair in ("reset", "restore", "stash", "checkout", "rm"):
            self.assertNotIn(repair, git.subcommands())

    def test_the_refusal_message_says_the_results_are_still_on_disk(self):
        self.push("claude/research", index_before_add=1)
        self.assertIn("Result artifacts remain on disk", self.printed)
        self.assertIn("reviewed and committed separately", self.printed)
        self.assertIn("existing index was not modified", self.printed)

    def test_an_index_check_error_fails_closed(self):
        # Any return code but 0 or 1 means the check itself did not work, which
        # is not a licence to proceed.
        status, git = self.push("claude/research", index_before_add=128)
        self.assertFalse(status["pushed"])
        self.assertEqual(status["reason"], "index check failed")
        self.assertIn("check itself failed", self.printed)
        for subcommand in ("add", "commit", "push"):
            self.assertNotIn(subcommand, git.subcommands())

    def test_the_index_is_checked_before_git_add(self):
        # Ordering again: a guard that runs after staging has already done the
        # damage it exists to prevent.
        _, git = self.push("claude/research")
        subcommands = git.subcommands()
        self.assertIn("add", subcommands)
        self.assertLess(subcommands.index("diff"), subcommands.index("add"))

    def test_a_protected_branch_is_refused_before_the_index_is_even_inspected(self):
        # Branch safety comes first, so a protected-branch run does not depend
        # on the index check working at all.
        _, git = self.push("main", index_before_add=1)
        self.assertNotIn("diff", git.subcommands())
        self.assertNotIn("add", git.subcommands())

    def test_a_detached_head_is_refused_before_the_index_is_even_inspected(self):
        _, git = self.push("HEAD", index_before_add=1)
        self.assertNotIn("diff", git.subcommands())
        self.assertNotIn("add", git.subcommands())

    def test_a_clean_index_on_a_feature_branch_still_publishes(self):
        # The guard must not break the case it is guarding.
        status, git = self.push(
            "claude/research", index_before_add=0, index_after_add=1
        )
        self.assertTrue(status["pushed"])
        subcommands = git.subcommands()
        for subcommand in ("add", "commit", "push"):
            self.assertIn(subcommand, subcommands)
        self.assertLess(subcommands.index("add"), subcommands.index("commit"))
        self.assertLess(subcommands.index("commit"), subcommands.index("push"))

    def test_the_full_refusal_order_is_branch_then_index_then_staging(self):
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        positions = [
            source.index("if branch in PROTECTED_BRANCHES:"),
            source.index('if not branch or branch == "HEAD":'),
            source.index("preexisting = subprocess.run("),
            source.index('subprocess.run(["git", "add"'),
            source.index('["git", "commit", "-m"'),
            source.index('subprocess.run(["git", "push"'),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_the_protected_set_is_the_default_branches(self):
        self.assertEqual(self.namespace["PROTECTED_BRANCHES"], {"main", "master"})

    def test_there_is_no_protected_branch_override(self):
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for escape_hatch in ("--force-push", "allow_protected", "--allow-main"):
            self.assertNotIn(escape_hatch, source)


if __name__ == "__main__":
    unittest.main()
