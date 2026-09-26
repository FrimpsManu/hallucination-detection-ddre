"""D-03 / checkpoint-provenance execution path: MPS-safe and truthful.

Two things are established here.

**The device the provenance records is the device the weights are on.** The
D-03 runner and the checkpoint-provenance diagnostic previously selected
``cuda if available else cpu``. On Apple silicon that placed the model on the
CPU, and -- because the formal v2 Gate report records ``mps`` -- it also made
``check_runtime_preconditions`` compare a reference of ``mps`` against an
observed ``cpu``, so D-03 would abort at the runtime provenance gate before
scoring anything. Both now select through the shared CUDA -> MPS -> CPU helper
and pass the string they actually used into the snapshot.

**A live revision is a supplement, never evidence about the frozen cache.** No
historical artifact records a resolved Hugging Face revision, so checkpoint
identity with the v2 cache is unestablishable rather than merely unestablished.
``checkpoint_identity_established`` must stay ``False`` whenever the revision
comes only from the supplement.

Nothing here touches the scoring mathematics, and the last class asserts that.
``torch`` is stubbed into ``sys.modules`` so every branch runs in CI.
"""

import ast
import importlib.machinery
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

from src.diagnostic_probe import (
    checkpoint_identity,
    collect_live_environment,
    device_matches,
    device_state,
    hub_repo_dir_name,
    infer_tokenizer_commit_from_paths,
    snapshot_revision_from_path,
    tokenizer_source_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# A torch-shaped stub, good enough for device_state and model_state.
# --------------------------------------------------------------------------

class _Cuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        return "FakeGPU"

    def get_device_capability(self, index):
        return (8, 0)


class _Mps:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _Backends:
    def __init__(self, mps_available):
        self.mps = _Mps(mps_available)


@contextmanager
def fake_torch(*, cuda=False, mps=False):
    """Install a torch-shaped module so device_state's ``import torch`` finds it."""
    module = types.ModuleType("torch")
    module.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    module.cuda = _Cuda(cuda)
    module.backends = _Backends(mps)
    module.version = types.SimpleNamespace(cuda="12.1" if cuda else None)
    previous = sys.modules.get("torch")
    sys.modules["torch"] = module
    try:
        yield module
    finally:
        if previous is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous


class FakeParameter:
    def __init__(self, device):
        self.dtype = "torch.float32"
        self.device = device


class FakeModel:
    """Enough surface for model_state()."""

    def __init__(self, device):
        self._parameter = FakeParameter(device)
        self.training = False
        self.config = types.SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            type_vocab_size=0,
            max_position_embeddings=512,
            model_type="deberta-v2",
            _commit_hash="cfg0123456789",
            _attn_implementation="eager",
            dtype=None,
            torch_dtype=None,
        )

    def parameters(self):
        return iter([self._parameter])


# --------------------------------------------------------------------------

class TestDeviceMatches(unittest.TestCase):
    """torch reports an indexed device; the selector returns a bare backend."""

    def test_indexed_device_matches_bare_backend(self):
        self.assertTrue(device_matches("mps", "mps:0"))
        self.assertTrue(device_matches("cuda", "cuda:0"))
        self.assertTrue(device_matches("cpu", "cpu"))

    def test_a_real_mismatch_is_false(self):
        self.assertFalse(device_matches("mps", "cpu"))
        self.assertFalse(device_matches("cuda", "cpu"))
        self.assertFalse(device_matches("cpu", "mps:0"))

    def test_not_applicable_is_none_not_false(self):
        # No model loaded, or no explicit selection: the question does not
        # apply, and that is different from a failed check.
        self.assertIsNone(device_matches(None, "mps:0"))
        self.assertIsNone(device_matches("mps", None))
        self.assertIsNone(device_matches(None, None))


class TestDeviceStateOverride(unittest.TestCase):
    def test_default_is_the_historical_cuda_or_cpu_expression(self):
        # Unrelated callers pass nothing and must be unaffected.
        with fake_torch(cuda=False, mps=True):
            state = device_state()
        self.assertEqual(state["selected_device"], "cpu")
        self.assertEqual(state["selected_device_source"], "device_state_default")

    def test_default_on_a_cuda_host_is_still_cuda(self):
        with fake_torch(cuda=True, mps=False):
            self.assertEqual(device_state()["selected_device"], "cuda")

    def test_an_explicit_device_is_recorded_verbatim(self):
        with fake_torch(cuda=False, mps=True):
            state = device_state("mps")
        self.assertEqual(state["selected_device"], "mps")
        self.assertEqual(state["selected_device_source"], "caller")
        # The default is kept alongside, so the artifact shows what the old
        # expression would have claimed.
        self.assertEqual(state["device_state_default"], "cpu")

    def test_mps_availability_is_reported_either_way(self):
        with fake_torch(cuda=False, mps=True):
            self.assertTrue(device_state()["mps_available"])
        with fake_torch(cuda=False, mps=False):
            self.assertFalse(device_state()["mps_available"])

    def test_cuda_priority_is_unchanged_when_both_exist(self):
        with fake_torch(cuda=True, mps=True):
            self.assertEqual(device_state()["selected_device"], "cuda")

    def test_absent_torch_still_degrades_rather_than_raising(self):
        previous = sys.modules.get("torch")
        sys.modules["torch"] = None  # import torch raises ImportError
        try:
            state = device_state("mps")
        finally:
            if previous is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous
        self.assertFalse(state["available"])


class TestTruthfulProvenanceOnMps(unittest.TestCase):
    def test_snapshot_records_mps_and_confirms_placement(self):
        with fake_torch(cuda=False, mps=True):
            snapshot = collect_live_environment(
                model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                model=FakeModel("mps:0"),
                selected_device="mps",
            )
        self.assertEqual(snapshot["device"]["selected_device"], "mps")
        self.assertEqual(snapshot["model"]["device"], "mps:0")
        self.assertTrue(snapshot["device_placement"]["matches"])

    def test_a_model_left_on_the_cpu_is_caught(self):
        # The failure this guards: recording "mps" while the weights are on the
        # CPU would put a device into provenance that nothing ran on.
        with fake_torch(cuda=False, mps=True):
            snapshot = collect_live_environment(
                model_name="m", model=FakeModel("cpu"), selected_device="mps"
            )
        self.assertFalse(snapshot["device_placement"]["matches"])

    def test_placement_is_not_applicable_without_a_model(self):
        with fake_torch(cuda=False, mps=True):
            snapshot = collect_live_environment(model_name="m", selected_device="mps")
        self.assertIsNone(snapshot["device_placement"]["matches"])

    def test_omitting_the_override_leaves_existing_callers_untouched(self):
        with fake_torch(cuda=False, mps=True):
            snapshot = collect_live_environment(
                model_name="m", model=FakeModel("cpu")
            )
        self.assertEqual(snapshot["device"]["selected_device"], "cpu")
        self.assertEqual(
            snapshot["device"]["selected_device_source"], "device_state_default"
        )
        self.assertIsNone(snapshot["device_placement"]["matches"])

    def test_the_recorded_device_is_what_the_runtime_guard_reads(self):
        # provenance_guard reads observed via "device.selected_device". The
        # formal v2 Gate report records device "mps", so this is the exact path
        # that decides whether D-03 clears the runtime gate on a Mac.
        from src.provenance_guard import check_runtime_preconditions

        with fake_torch(cuda=False, mps=True):
            observed = collect_live_environment(
                model_name="m", model=FakeModel("mps:0"), selected_device="mps"
            )
        checks = check_runtime_preconditions({"device": "mps"}, observed)
        device_check = next(c for c in checks if c["name"] == "device")
        self.assertEqual(device_check["observed"], "mps")
        self.assertEqual(device_check["status"], "PASS")

    def test_without_the_override_that_same_guard_check_fails_on_a_mac(self):
        from src.provenance_guard import check_runtime_preconditions

        with fake_torch(cuda=False, mps=True):
            observed = collect_live_environment(
                model_name="m", model=FakeModel("cpu")
            )
        checks = check_runtime_preconditions({"device": "mps"}, observed)
        device_check = next(c for c in checks if c["name"] == "device")
        self.assertEqual(device_check["observed"], "cpu")
        self.assertEqual(device_check["status"], "FAIL")


class TestRepointedScripts(unittest.TestCase):
    """Asserted on source: both scripts import torch, which CI does not install."""

    @classmethod
    def setUpClass(cls):
        cls.d03 = (PROJECT_ROOT / "scripts" / "diagnose_ddre_ratio_support.py").read_text(
            encoding="utf-8"
        )
        cls.prov = (PROJECT_ROOT / "scripts" / "diagnose_gate1_provenance.py").read_text(
            encoding="utf-8"
        )
        ast.parse(cls.d03)
        ast.parse(cls.prov)

    def test_d03_selects_through_the_shared_helper(self):
        self.assertIn("select_device", self.d03)
        self.assertIn("selected_device = select_device(torch)", self.d03)
        self.assertIn("device = torch.device(selected_device)", self.d03)

    def test_d03_no_longer_uses_the_cuda_or_cpu_expression(self):
        self.assertNotIn(
            'torch.device("cuda" if torch.cuda.is_available() else "cpu")', self.d03
        )

    def test_d03_records_the_device_it_selected(self):
        self.assertIn("selected_device=selected_device,", self.d03)

    def test_d03_aborts_when_placement_disagrees_with_the_record(self):
        self.assertIn('placement.get("matches") is not True', self.d03)
        self.assertIn("Nothing scored.", self.d03)

    def test_checkpoint_provenance_selects_through_the_shared_helper(self):
        self.assertIn("select_device", self.prov)
        self.assertIn("selected_device = select_device(torch)", self.prov)

    def test_checkpoint_provenance_no_longer_uses_the_cuda_or_cpu_expression(self):
        self.assertNotIn(
            'torch.device("cuda" if torch.cuda.is_available() else "cpu")', self.prov
        )

    def test_checkpoint_provenance_records_the_device_it_selected(self):
        self.assertIn("selected_device=selected_device,", self.prov)

    def test_checkpoint_provenance_records_the_supplement_fields(self):
        for field in (
            "resolved_revision",
            "model_config_commit_hash",
            "tokenizer_commit_hash",
            "sentencepiece_version",
            "tokenizers_version",
            "model_dtype",
        ):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', self.prov)


class TestSupplementCannotEstablishCheckpointIdentity(unittest.TestCase):
    def test_a_supplement_supplied_revision_leaves_identity_false(self):
        from src.provenance_guard import merge_reference

        # The real situation: the formal v2 Gate report records a device but no
        # revision; the supplement carries the revision.
        bundle = merge_reference(
            {"device": "mps", "resolved_revision": None},
            {"resolved_revision": "abc123def456",
             "model_config_commit_hash": "abc123def456"},
            formal_path="results/gate1/wang_reproduction_fidelity_v2_batch1_mps.json",
            checkpoint_path="results/diagnostics/step0_provenance.json",
        )
        self.assertEqual(bundle["reference"]["resolved_revision"], "abc123def456")
        self.assertFalse(bundle["checkpoint_identity_established"])
        self.assertTrue(
            any("NO ARTIFACT ESTABLISHES" in text for text in bundle["limitations"])
        )

    def test_identity_is_only_established_when_the_formal_report_records_it(self):
        from src.provenance_guard import merge_reference

        bundle = merge_reference(
            {"resolved_revision": "abc123def456"},
            {"resolved_revision": "abc123def456"},
            formal_path="formal.json",
            checkpoint_path="checkpoint.json",
        )
        self.assertTrue(bundle["checkpoint_identity_established"])

    def test_the_diagnostic_states_the_limitation_in_its_own_artifact(self):
        source = (PROJECT_ROOT / "scripts" / "diagnose_gate1_provenance.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("checkpoint_identity_limitation", source)
        self.assertIn('"checkpoint_identity_established": False', source)
        self.assertIn('"supplement_only": True', source)
        self.assertIn("NO ARTIFACT ESTABLISHES", source)


class TestExactCompatibilityRemainsMandatory(unittest.TestCase):
    """The empirical bridge. Exact float equality, or D-03 writes nothing."""

    def test_exact_match_is_raw_float_equality(self):
        source = (PROJECT_ROOT / "src" / "score_compatibility.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"exact_match": present and float(stored_score) == fresh_score', source
        )

    def test_establishment_requires_every_probed_pair_to_match_exactly(self):
        from src.score_compatibility import compare_pair_scores, compatibility_report

        source = (PROJECT_ROOT / "src" / "score_compatibility.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("all_exact = len(exact) == probed", source)
        self.assertIn("and all_exact", source)
        self.assertTrue(callable(compare_pair_scores))
        self.assertTrue(callable(compatibility_report))

    def test_a_rounded_but_unequal_score_is_not_an_exact_match(self):
        from src.score_compatibility import compare_pair_scores

        entries = compare_pair_scores(
            pairs=[("p", "h")],
            polarities=["positive"],
            keys=["k"],
            stored={"k": 41.23456},
            fresh=[41.23457],
        )
        entry = entries[0]
        self.assertFalse(entry["exact_match"])
        # The rounded/bucket agreements exist as diagnostics and must never
        # substitute for exact equality in the verdict.
        self.assertTrue(entry["one_decimal_match"])
        self.assertTrue(entry["nbc_bucket_match"])

    def test_d03_aborts_before_writing_when_compatibility_is_not_established(self):
        source = (PROJECT_ROOT / "scripts" / "diagnose_ddre_ratio_support.py").read_text(
            encoding="utf-8"
        )
        # Order matters: compatibility is checked and aborted on BEFORE the
        # derived cache is prepared.
        abort = source.index('if not compatibility["score_compatibility_established"]')
        prepare = source.index("prepare_derived_cache(")
        self.assertLess(abort, prepare)
        self.assertIn("ABORTED: score compatibility not established", source)

    def test_no_tolerance_was_introduced_into_the_probe(self):
        source = (PROJECT_ROOT / "src" / "score_compatibility.py").read_text(
            encoding="utf-8"
        )
        for banned in ("isclose", "allclose", "atol", "rtol"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, source)

    def test_absolute_delta_is_diagnostic_and_never_gates_the_verdict(self):
        from src.score_compatibility import compare_pair_scores

        entry = compare_pair_scores(
            pairs=[("p", "h")], polarities=["positive"], keys=["k"],
            stored={"k": 41.23456}, fresh=[41.23457],
        )[0]
        # The delta is recorded for the reader...
        self.assertGreater(entry["absolute_delta"], 0.0)
        # ...and the verdict is still driven by exact equality alone.
        self.assertFalse(entry["exact_match"])
        source = (PROJECT_ROOT / "src" / "score_compatibility.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("all_exact = len(exact) == probed", source)


class TestNoScientificConstantChanged(unittest.TestCase):
    """The backend is not part of the estimand. Prove the change stayed there."""

    def test_score_version_untouched(self):
        source = (PROJECT_ROOT / "src" / "utils.py").read_text(encoding="utf-8")
        self.assertIn(
            'SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2"',
            source,
        )

    def test_d03_protocol_constants_untouched(self):
        source = (PROJECT_ROOT / "scripts" / "diagnose_ddre_ratio_support.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("WANG_BATCH_SIZE", source)
        self.assertIn("batch_size=WANG_BATCH_SIZE", source)

    def test_confirmatory_constants_untouched(self):
        from src import paired_bootstrap as pb

        self.assertEqual(pb.CONFIRMATORY_PR_AUC_MARGIN, 0.005)
        self.assertEqual(pb.CONFIRMATORY_BOOTSTRAP_RESAMPLES, 10_000)
        self.assertEqual(pb.CONFIRMATORY_BOOTSTRAP_SEED, 42)
        self.assertEqual(pb.CONFIRMATORY_C_MISS, 28)
        self.assertEqual(pb.CONFIRMATORY_C_FALSE_ALARM, 96)
        self.assertEqual(pb.CONFIRMATORY_C_RETRIEVE, 1)
        self.assertEqual(pb.CONFIRMATORY_P0, 0.5)
        self.assertEqual(pb.CONFIRMATORY_MAX_DOCS, 10)
        self.assertEqual(pb.CONFIRMATORY_VALIDATION_FRACTION, 0.20)
        self.assertEqual(pb.CONFIRMATORY_SPLIT_SEED, 42)

    def test_gate_tolerances_untouched(self):
        from src.reproduction_gate import TOLERANCES

        self.assertEqual(
            TOLERANCES["evidence_num_per_sentence"],
            {"kind": "relative", "pass": 0.05, "warn": 0.10},
        )
        self.assertEqual(
            TOLERANCES["factual_auc_pr"], {"kind": "absolute", "pass": 0.01, "warn": 0.03}
        )

    def test_ratio_clipping_untouched(self):
        source = (PROJECT_ROOT / "src" / "ddre_core.py").read_text(encoding="utf-8")
        self.assertIn("1e-6, 1e6", source)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# Tokenizer commit hash: object first, snapshot path only as a fallback.
# --------------------------------------------------------------------------

REV_A = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
REV_B = "0123456789abcdef0123456789abcdef01234567"
MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
REPO_DIR = "models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class FakeTokenizer:
    """Only the surface checkpoint_identity and tokenizer_state read."""

    def __init__(self, commit_hash=None, init_kwargs=None, **file_attributes):
        self._commit_hash = commit_hash
        self.init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
        for name, value in file_attributes.items():
            setattr(self, name, value)


def build_hub_cache(tmp, revisions, repo_dir=REPO_DIR, filename="spm.model"):
    """A realistic hub cache: snapshot entries are SYMLINKS into blobs/.

    This layout is the reason the revision must be read from the path as
    given. Resolving a snapshot entry lands in ``blobs/<sha256>``, which
    carries no revision at all.
    """
    root = Path(tmp) / repo_dir
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    made = {}
    for index, revision in enumerate(revisions):
        blob = root / "blobs" / f"deadbeef{index:04d}"
        blob.write_bytes(b"spm")
        snapshot = root / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        link = snapshot / filename
        if not link.exists():
            os.symlink(blob, link)
        made[revision] = str(link)
    return made


class TestSnapshotRevisionFromPath(unittest.TestCase):
    def test_extracts_the_revision(self):
        path = f"/home/u/.cache/huggingface/hub/{REPO_DIR}/snapshots/{REV_A}/spm.model"
        self.assertEqual(snapshot_revision_from_path(path, REPO_DIR), REV_A)

    def test_requires_the_repo_directory_to_match(self):
        # A file from a DIFFERENT model's snapshot must not establish this
        # model's revision.
        path = f"/hub/models--other--model/snapshots/{REV_A}/spm.model"
        self.assertIsNone(snapshot_revision_from_path(path, REPO_DIR))
        self.assertEqual(snapshot_revision_from_path(path), REV_A)

    def test_rejects_anything_that_is_not_a_40_hex_component(self):
        for bad in ("main", "refs", REV_A[:39], REV_A + "0", "z" * 40, ""):
            with self.subTest(bad=bad):
                path = f"/hub/{REPO_DIR}/snapshots/{bad}/spm.model"
                self.assertIsNone(snapshot_revision_from_path(path, REPO_DIR))

    def test_ordinary_paths_establish_nothing(self):
        for path in (
            "/home/u/models/deberta/spm.model",
            "./spm.model",
            "/tmp/spm.model",
            f"/hub/{REPO_DIR}/blobs/abcdef",
            "",
            None,
        ):
            with self.subTest(path=path):
                self.assertIsNone(snapshot_revision_from_path(path, REPO_DIR))

    def test_revision_is_normalised_to_lowercase(self):
        path = f"/hub/{REPO_DIR}/snapshots/{REV_A.upper()}/spm.model"
        self.assertEqual(snapshot_revision_from_path(path, REPO_DIR), REV_A)

    def test_repo_dir_name_is_derived_from_the_repo_id(self):
        self.assertEqual(hub_repo_dir_name(MODEL), REPO_DIR)


class TestTokenizerSourcePaths(unittest.TestCase):
    def test_named_file_attributes_are_collected(self):
        tok = FakeTokenizer(vocab_file="/a/spm.model", tokenizer_file="/a/tokenizer.json")
        self.assertEqual(
            tokenizer_source_paths(tok), ["/a/spm.model", "/a/tokenizer.json"]
        )

    def test_path_like_init_kwargs_are_collected(self):
        tok = FakeTokenizer(init_kwargs={"vocab_file": "/a/spm.model", "do_lower_case": "False"})
        self.assertIn("/a/spm.model", tokenizer_source_paths(tok))
        self.assertNotIn("False", tokenizer_source_paths(tok))

    def test_duplicates_are_collapsed_and_order_kept(self):
        tok = FakeTokenizer(vocab_file="/a/spm.model", init_kwargs={"vocab_file": "/a/spm.model"})
        self.assertEqual(tokenizer_source_paths(tok), ["/a/spm.model"])

    def test_a_tokenizer_with_no_paths_yields_nothing(self):
        self.assertEqual(tokenizer_source_paths(FakeTokenizer()), [])


class TestInferTokenizerCommitFromPaths(unittest.TestCase):
    def test_one_existing_snapshot_file_establishes_that_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A])
            tok = FakeTokenizer(vocab_file=made[REV_A])
            revision, evidence, note = infer_tokenizer_commit_from_paths(tok, REPO_DIR)
        self.assertEqual(revision, REV_A)
        self.assertEqual(evidence, [made[REV_A]])
        self.assertIsNone(note)

    def test_the_snapshot_entry_is_a_symlink_into_blobs(self):
        # Guards the core reason this is path-based: resolving the entry
        # discards the revision entirely.
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A])
            link = Path(made[REV_A])
            self.assertTrue(link.is_symlink())
            self.assertIn("blobs", str(link.resolve()))
            self.assertIsNone(snapshot_revision_from_path(link.resolve(), REPO_DIR))
            self.assertEqual(snapshot_revision_from_path(link, REPO_DIR), REV_A)

    def test_conflicting_revisions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A, REV_B])
            tok = FakeTokenizer(vocab_file=made[REV_A], tokenizer_file=made[REV_B])
            revision, evidence, note = infer_tokenizer_commit_from_paths(tok, REPO_DIR)
        self.assertIsNone(revision)
        self.assertEqual(evidence, [])
        self.assertIn("disagree", note)

    def test_a_path_that_does_not_exist_establishes_nothing(self):
        tok = FakeTokenizer(vocab_file=f"/nope/{REPO_DIR}/snapshots/{REV_A}/spm.model")
        revision, evidence, note = infer_tokenizer_commit_from_paths(tok, REPO_DIR)
        self.assertIsNone(revision)
        self.assertIn("no existing tokenizer source file", note)

    def test_a_non_snapshot_path_establishes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "spm.model"
            local.write_bytes(b"spm")
            tok = FakeTokenizer(vocab_file=str(local))
            revision, _, note = infer_tokenizer_commit_from_paths(tok, REPO_DIR)
        self.assertIsNone(revision)
        self.assertIn("no existing tokenizer source file", note)

    def test_another_models_snapshot_establishes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A], repo_dir="models--someone--else")
            tok = FakeTokenizer(vocab_file=made[REV_A])
            revision, _, note = infer_tokenizer_commit_from_paths(tok, REPO_DIR)
        self.assertIsNone(revision)
        self.assertIn("no existing tokenizer source file", note)

    def test_no_paths_at_all_is_reported(self):
        revision, _, note = infer_tokenizer_commit_from_paths(FakeTokenizer(), REPO_DIR)
        self.assertIsNone(revision)
        self.assertIn("no source file paths", note)


class TestCheckpointIdentityTokenizerHash(unittest.TestCase):
    def test_the_tokenizer_object_hash_wins_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_B])
            tok = FakeTokenizer(commit_hash=REV_A, vocab_file=made[REV_B])
            identity = checkpoint_identity(MODEL, tokenizer=tok)
        self.assertEqual(identity["tokenizer_commit_hash"], REV_A)
        self.assertEqual(identity["tokenizer_commit_hash_source"], "tokenizer_object")
        self.assertEqual(identity["tokenizer_commit_hash_evidence"], [])

    def test_init_kwargs_commit_hash_is_used_before_any_path(self):
        tok = FakeTokenizer(init_kwargs={"_commit_hash": REV_A})
        identity = checkpoint_identity(MODEL, tokenizer=tok)
        self.assertEqual(identity["tokenizer_commit_hash"], REV_A)
        self.assertEqual(identity["tokenizer_commit_hash_source"], "tokenizer_object")

    def test_absent_hash_falls_back_to_the_snapshot_path(self):
        # The reported transformers 5.17.0 behaviour.
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A])
            tok = FakeTokenizer(
                commit_hash=None,
                init_kwargs={"_commit_hash": None, "vocab_file": made[REV_A]},
                vocab_file=made[REV_A],
            )
            identity = checkpoint_identity(MODEL, tokenizer=tok)
        self.assertEqual(identity["tokenizer_commit_hash"], REV_A)
        self.assertEqual(
            identity["tokenizer_commit_hash_source"], "huggingface_snapshot_path"
        )
        self.assertEqual(identity["tokenizer_commit_hash_evidence"], [made[REV_A]])
        self.assertTrue(
            any("path component" in n for n in identity["notes"]),
            identity["notes"],
        )

    def test_the_inferred_value_equals_the_snapshot_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A])
            tok = FakeTokenizer(vocab_file=made[REV_A])
            identity = checkpoint_identity(MODEL, tokenizer=tok)
        self.assertEqual(identity["tokenizer_commit_hash"], Path(made[REV_A]).parent.name)

    def test_ambiguity_leaves_the_hash_none_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A, REV_B])
            tok = FakeTokenizer(vocab_file=made[REV_A], tokenizer_file=made[REV_B])
            identity = checkpoint_identity(MODEL, tokenizer=tok)
        self.assertIsNone(identity["tokenizer_commit_hash"])
        self.assertIsNone(identity["tokenizer_commit_hash_source"])
        self.assertTrue(any("fails closed" in n for n in identity["notes"]))

    def test_a_local_checkout_still_establishes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "spm.model"
            local.write_bytes(b"spm")
            identity = checkpoint_identity(MODEL, tokenizer=FakeTokenizer(vocab_file=str(local)))
        self.assertIsNone(identity["tokenizer_commit_hash"])
        self.assertIsNone(identity["tokenizer_commit_hash_source"])

    def test_no_tokenizer_means_no_claim_either_way(self):
        identity = checkpoint_identity(MODEL)
        self.assertIsNone(identity["tokenizer_commit_hash"])
        self.assertIsNone(identity["tokenizer_commit_hash_source"])
        self.assertEqual(identity["tokenizer_commit_hash_evidence"], [])


class TestRuntimeGuardStillCompares(unittest.TestCase):
    """The guard is untouched: it still demands an exact match, closed."""

    def test_guard_source_has_no_tokenizer_exemption(self):
        source = (PROJECT_ROOT / "src" / "provenance_guard.py").read_text(encoding="utf-8")
        self.assertIn('"tokenizer_commit_hash"', source)
        for banned in ("advisory=True  # tokenizer", "skip_tokenizer", "tokenizer_optional"):
            self.assertNotIn(banned, source)

    def test_a_none_tokenizer_hash_still_fails_the_guard(self):
        from src.provenance_guard import check_runtime_preconditions

        observed = {"checkpoint_identity": {"tokenizer_commit_hash": None},
                    "tokenizer": {"commit_hash": None}}
        checks = check_runtime_preconditions({"tokenizer_commit_hash": REV_A}, observed)
        entry = next(c for c in checks if c["name"] == "tokenizer_commit_hash")
        self.assertEqual(entry["status"], "FAIL")

    def test_a_path_inferred_hash_satisfies_the_guard_without_relaxing_it(self):
        from src.provenance_guard import check_runtime_preconditions

        with tempfile.TemporaryDirectory() as tmp:
            made = build_hub_cache(tmp, [REV_A])
            identity = checkpoint_identity(MODEL, tokenizer=FakeTokenizer(vocab_file=made[REV_A]))
        observed = {"checkpoint_identity": identity, "tokenizer": {"commit_hash": None}}
        checks = check_runtime_preconditions({"tokenizer_commit_hash": REV_A}, observed)
        entry = next(c for c in checks if c["name"] == "tokenizer_commit_hash")
        self.assertEqual(entry["status"], "PASS")
        self.assertEqual(entry["observed"], REV_A)
