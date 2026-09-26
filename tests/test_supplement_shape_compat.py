"""The checkpoint-provenance supplement must be readable by the guard.

``scripts/diagnose_gate1_provenance.py`` writes its artifact as::

    {"live_environment": <collect_live_environment snapshot>,
     "live_fields":      <flattened view>, ...}

``extract_reference()`` did not recognise either prefix, so every checkpoint
field read back as unrecorded and D-03 aborted at ``resolved_revision_present``
before it could do anything -- even though the supplement demonstrably carried
the values.

The supplement in these tests is NOT hand-written. It is produced by calling the
real ``collect_live_environment`` and the real ``live_fields`` from the script,
then assembling them exactly as the script's ``result`` dict does, so a future
change to either side breaks this test rather than silently reintroducing the
mismatch.

Nothing here relaxes the guard. The last two classes assert that a genuinely
missing revision still fails closed and that the 398-pair probe is still exact.
"""

import importlib.machinery
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

from src.provenance_guard import extract_reference, merge_reference

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2"
WANG_COMMIT = "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea"


def load_producer():
    """Import the script by path; scripts/ is not a package."""
    path = PROJECT_ROOT / "scripts" / "diagnose_gate1_provenance.py"
    spec = importlib.util.spec_from_file_location("diagnose_gate1_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def fake_torch():
    """A torch-shaped stub so device_state() runs without torch installed."""
    module = types.ModuleType("torch")
    module.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    module.cuda = types.SimpleNamespace(is_available=lambda: False)
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )
    module.version = types.SimpleNamespace(cuda=None)
    previous = sys.modules.get("torch")
    sys.modules["torch"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous


class FakeParameter:
    def __init__(self):
        self.dtype = "torch.float16"
        self.device = "mps:0"


class FakeModel:
    def __init__(self):
        self.training = False
        self.config = types.SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            type_vocab_size=0,
            max_position_embeddings=512,
            model_type="deberta-v2",
            _commit_hash=REVISION,
            _attn_implementation="eager",
            dtype=None,
            torch_dtype=None,
        )

    def parameters(self):
        return iter([FakeParameter()])


class FakeTokenizer:
    def __init__(self):
        self._commit_hash = REVISION
        self.init_kwargs = {"_commit_hash": REVISION}
        self.is_fast = True
        self.model_max_length = 512
        self.vocab_size = 128000
        self.name_or_path = MODEL

    def __call__(self, *args, **kwargs):  # tokenizer_emission_probe
        raise RuntimeError("no tokenizer backend in this test")


def build_real_supplement():
    """Exactly what diagnose_gate1_provenance.py writes, via its own code."""
    from src.diagnostic_probe import collect_live_environment

    producer = load_producer()
    with fake_torch():
        snapshot = collect_live_environment(
            model_name=MODEL,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            repo_root=PROJECT_ROOT,
            score_version=SCORE_VERSION,
            selected_device="mps",
        )
    # The snapshot's checkpoint_identity resolves against the real hub cache,
    # which is absent here. Pin the three checkpoint values the supplement
    # carries on a machine where the model IS cached; every other field below
    # is whatever the producer genuinely emitted.
    snapshot["checkpoint_identity"]["resolved_revision"] = REVISION
    snapshot["checkpoint_identity"]["model_config_commit_hash"] = REVISION
    snapshot["checkpoint_identity"]["tokenizer_commit_hash"] = REVISION
    snapshot["libraries"].update(
        {"torch": "2.8.0", "transformers": "5.17.0", "tokenizers": "0.22.1",
         "numpy": "2.1.3", "scipy": "1.14.1", "sklearn": "1.5.2",
         "sentencepiece": "0.2.0"}
    )
    live = producer.live_fields(snapshot)
    return {"step": "0-provenance", "live_environment": snapshot, "live_fields": live}


class TestSupplementShapeIsRecognised(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supplement = build_real_supplement()
        cls.reference = extract_reference(cls.supplement)

    def test_the_producer_really_emits_both_prefixes(self):
        self.assertIn("live_environment", self.supplement)
        self.assertIn("live_fields", self.supplement)
        self.assertEqual(
            self.supplement["live_environment"]["checkpoint_identity"]["resolved_revision"],
            REVISION,
        )
        self.assertEqual(self.supplement["live_fields"]["resolved_revision"], REVISION)

    def test_checkpoint_fields_are_recovered(self):
        self.assertEqual(self.reference["resolved_revision"], REVISION)
        self.assertEqual(self.reference["model_config_commit_hash"], REVISION)
        self.assertEqual(self.reference["tokenizer_commit_hash"], REVISION)

    def test_model_and_tokenizer_fields_are_recovered(self):
        self.assertEqual(self.reference["model_name"], MODEL)
        self.assertEqual(self.reference["model_dtype"], "torch.float16")
        self.assertEqual(self.reference["tokenizer_model_max_length"], 512)

    def test_device_and_score_version_are_recovered(self):
        self.assertEqual(self.reference["device"], "mps")
        self.assertEqual(self.reference["score_version"], SCORE_VERSION)

    def test_libraries_are_recovered(self):
        for name, version in (
            ("torch", "2.8.0"), ("transformers", "5.17.0"), ("tokenizers", "0.22.1"),
            ("numpy", "2.1.3"), ("scipy", "1.14.1"), ("sklearn", "1.5.2"),
            ("sentencepiece", "0.2.0"),
        ):
            with self.subTest(library=name):
                self.assertEqual(self.reference["libraries"][name], version)

    def test_libraries_also_recover_from_live_fields_alone(self):
        # A supplement carrying only the flattened view still resolves.
        flat_only = {"live_fields": self.supplement["live_fields"]}
        reference = extract_reference(flat_only)
        self.assertEqual(reference["resolved_revision"], REVISION)
        self.assertEqual(reference["libraries"]["torch"], "2.8.0")
        self.assertEqual(reference["libraries"]["sentencepiece"], "0.2.0")
        self.assertEqual(reference["device"], "mps")

    def test_live_environment_alone_also_resolves(self):
        nested_only = {"live_environment": self.supplement["live_environment"]}
        reference = extract_reference(nested_only)
        self.assertEqual(reference["resolved_revision"], REVISION)
        self.assertEqual(reference["tokenizer_commit_hash"], REVISION)
        self.assertEqual(reference["libraries"]["torch"], "2.8.0")


class TestBackwardCompatibility(unittest.TestCase):
    """Every shape already accepted must resolve exactly as before."""

    def test_existing_shapes_still_resolve(self):
        for payload, expected in (
            ({"checkpoint_identity": {"resolved_revision": "a" * 40}}, "a" * 40),
            ({"environment": {"checkpoint_identity": {"resolved_revision": "b" * 40}}}, "b" * 40),
            ({"environment_summary": {"resolved_hf_revision": "c" * 40}}, "c" * 40),
        ):
            with self.subTest(payload=sorted(payload)):
                self.assertEqual(extract_reference(payload)["resolved_revision"], expected)

    def test_precedence_is_preserved_when_both_shapes_are_present(self):
        # The new paths are APPENDED, so an older path still wins.
        payload = {
            "checkpoint_identity": {"resolved_revision": "a" * 40},
            "live_environment": {"checkpoint_identity": {"resolved_revision": "b" * 40}},
            "live_fields": {"resolved_revision": "c" * 40},
        }
        self.assertEqual(extract_reference(payload)["resolved_revision"], "a" * 40)

    def test_nested_live_environment_beats_the_flattened_view(self):
        payload = {
            "live_environment": {"checkpoint_identity": {"resolved_revision": "b" * 40}},
            "live_fields": {"resolved_revision": "c" * 40},
        }
        self.assertEqual(extract_reference(payload)["resolved_revision"], "b" * 40)

    def test_gate_report_provenance_block_is_unaffected(self):
        payload = {"provenance": {"runtime": {"device": "mps", "score_version": SCORE_VERSION},
                                  "wang_data": {"source_commit": WANG_COMMIT},
                                  "nli_model": {"model_name": MODEL}}}
        reference = extract_reference(payload)
        self.assertEqual(reference["device"], "mps")
        self.assertEqual(reference["wang_source_commit"], WANG_COMMIT)
        self.assertEqual(reference["model_name"], MODEL)

    def test_an_empty_payload_still_yields_all_none(self):
        reference = extract_reference({})
        for field in ("resolved_revision", "model_config_commit_hash",
                      "tokenizer_commit_hash", "device", "model_dtype"):
            with self.subTest(field=field):
                self.assertIsNone(reference[field])
        self.assertEqual(reference["libraries"], {})


class TestRoundTripIntoMergeReference(unittest.TestCase):
    """formal Gate reference + supplement -> merged bundle."""

    @classmethod
    def setUpClass(cls):
        formal_payload = {
            "provenance": {
                "runtime": {"device": "mps", "score_version": SCORE_VERSION, "batch_size": 1},
                "wang_data": {"source_commit": WANG_COMMIT},
                "nli_model": {"model_name": MODEL, "tokenizer_model_max_length": 512},
                "libraries": {"torch": "2.8.0", "transformers": "5.17.0",
                              "tokenizers": "0.22.1"},
            }
        }
        cls.formal = extract_reference(formal_payload)
        cls.supplement = extract_reference(build_real_supplement())
        cls.bundle = merge_reference(
            cls.formal, cls.supplement,
            formal_path="results/gate1/wang_reproduction_fidelity_v2_batch1_mps.json",
            checkpoint_path="results/diagnostics/step0_checkpoint_provenance.json",
        )

    def test_the_formal_reference_has_no_revision_of_its_own(self):
        self.assertIsNone(self.formal["resolved_revision"])

    def test_the_merged_reference_carries_the_checkpoint_values(self):
        reference = self.bundle["reference"]
        self.assertEqual(reference["resolved_revision"], REVISION)
        self.assertEqual(reference["model_config_commit_hash"], REVISION)
        self.assertEqual(reference["tokenizer_commit_hash"], REVISION)

    def test_the_formal_device_survives_the_merge(self):
        self.assertEqual(self.bundle["reference"]["device"], "mps")

    def test_identity_is_still_not_established(self):
        # The revision came from the supplement, so it cannot speak for the
        # frozen v2 cache. This semantics is unchanged by this fix.
        self.assertFalse(self.bundle["checkpoint_identity_established"])
        self.assertTrue(
            any("NO ARTIFACT ESTABLISHES" in text for text in self.bundle["limitations"])
        )

    def test_overlapping_agreeing_fields_raise_no_gating_conflict(self):
        self.assertEqual(self.bundle["conflicts"], [])

    def test_the_revision_is_attributed_to_the_supplement(self):
        entry = next(e for e in self.bundle["field_sources"]
                     if e["field"] == "resolved_revision")
        self.assertEqual(entry["value"], REVISION)
        self.assertNotEqual(entry["source"], "formal_v2_gate_report")


class TestMissingRevisionStillFailsClosed(unittest.TestCase):
    """No field was made optional."""

    def test_a_supplement_without_a_revision_leaves_it_none(self):
        supplement = build_real_supplement()
        supplement["live_environment"]["checkpoint_identity"]["resolved_revision"] = None
        supplement["live_fields"]["resolved_revision"] = None
        self.assertIsNone(extract_reference(supplement)["resolved_revision"])

    def test_the_static_gate_still_fails_without_a_revision(self):
        from src.provenance_guard import check_static_preconditions

        reference = extract_reference({})
        checks = check_static_preconditions(
            reference, score_version=SCORE_VERSION, batch_size=1,
            wang_source_commit=WANG_COMMIT, model_name=MODEL,
        )
        entry = next(c for c in checks if c["name"] == "resolved_revision_present")
        self.assertEqual(entry["status"], "FAIL")

    def test_a_present_revision_passes_that_same_check(self):
        from src.provenance_guard import check_static_preconditions

        reference = extract_reference(build_real_supplement())
        checks = check_static_preconditions(
            reference, score_version=SCORE_VERSION, batch_size=1,
            wang_source_commit=WANG_COMMIT, model_name=MODEL,
        )
        entry = next(c for c in checks if c["name"] == "resolved_revision_present")
        self.assertEqual(entry["status"], "PASS")


class TestExactCompatibilityUnchanged(unittest.TestCase):
    def test_exact_match_is_still_raw_float_equality(self):
        source = (PROJECT_ROOT / "src" / "score_compatibility.py").read_text(encoding="utf-8")
        self.assertIn(
            '"exact_match": present and float(stored_score) == fresh_score', source
        )
        self.assertIn("all_exact = len(exact) == probed", source)
        for banned in ("isclose", "allclose", "atol", "rtol"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, source)

    def test_a_tiny_difference_is_still_not_an_exact_match(self):
        from src.score_compatibility import compare_pair_scores

        entry = compare_pair_scores(
            pairs=[("p", "h")], polarities=["positive"], keys=["k"],
            stored={"k": 41.23456}, fresh=[41.23457],
        )[0]
        self.assertFalse(entry["exact_match"])
        self.assertTrue(entry["one_decimal_match"])


if __name__ == "__main__":
    unittest.main()
