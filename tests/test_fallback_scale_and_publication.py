"""Tests for audit findings D-11 and D-12.

**D-11:** the fallback objective's retrieval penalty must be *dimensionless*.
It divided a per-sentence document count by a per-subclaim budget, so the same
retrieval behaviour scored differently purely because sentences in one
configuration happened to carry more subclaims. The corrected cost is
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

    def test_the_same_per_subclaim_retrieval_costs_the_same(self):
        # This is the unit correction, stated as a property. Two configurations
        # retrieving 2 documents per subclaim differ only in how many subclaims
        # their sentences carry, which is a property of the DATA, not of the
        # retrieval policy. Under the old normalization they scored differently.
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


class TestFallbackRemainsNonConfirmatory(unittest.TestCase):
    def test_the_corrected_fallback_can_pick_a_different_configuration(self):
        # Nothing preserves baseline quality, so the fallback decides. Candidate
        # B is cheaper per subclaim but more expensive per sentence; the
        # corrected objective prefers B, the old per-sentence one preferred A.
        # That change is exactly what D-11 asks for.
        a = record_for(
            documents_per_sentence=2.0, documents_per_subclaim=9.0,
            nonfactual=0.50, factual=0.40, lower=0.10, upper=0.60,
            retrieval_penalty=0.50,
        )
        b = record_for(
            documents_per_sentence=9.0, documents_per_subclaim=1.0,
            nonfactual=0.50, factual=0.40, lower=0.15, upper=0.65,
            retrieval_penalty=0.50,
        )
        self.assertFalse(a["preserves_baseline_quality"])
        self.assertFalse(b["preserves_baseline_quality"])
        selected, rule, confirmatory = select_threshold_configuration([a, b])
        self.assertEqual(selected["lower"], 0.15)
        self.assertFalse(confirmatory)
        self.assertIs(rule, FALLBACK_SELECTION_RULE)

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
    """Records every git invocation so 'before git add' can be asserted."""

    def __init__(self, branch, staged_returncode=1):
        self.branch = branch
        self.staged_returncode = staged_returncode
        self.calls = []

    def check_output(self, argv, text=True):
        self.calls.append(list(argv))
        return self.branch + "\n"

    def run(self, argv, check=False):
        self.calls.append(list(argv))
        if argv[:3] == ["git", "diff", "--cached"]:
            return FakeCompleted(self.staged_returncode)
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

    def push(self, branch, staged_returncode=1):
        git = RecordingGit(branch, staged_returncode)
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
        status, git = self.push("claude/research", staged_returncode=0)
        self.assertEqual(status, {"pushed": False, "reason": "no changes"})
        self.assertIn("add", git.subcommands())
        self.assertNotIn("commit", git.subcommands())
        self.assertNotIn("push", git.subcommands())

    def test_the_protected_set_is_the_default_branches(self):
        self.assertEqual(self.namespace["PROTECTED_BRANCHES"], {"main", "master"})

    def test_there_is_no_protected_branch_override(self):
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for escape_hatch in ("--force-push", "allow_protected", "--allow-main"):
            self.assertNotIn(escape_hatch, source)


if __name__ == "__main__":
    unittest.main()
