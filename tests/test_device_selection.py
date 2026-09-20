"""Device selection for the Gate 1 reproduction: CUDA -> MPS -> CPU.

Apple Metal was added so the corrected-v2 Gate 1 reproduction can use the GPU
on Apple silicon instead of silently falling back to CPU. The priority order is
the whole contract, and it is pinned here in both directions: CUDA must keep
absolute priority so every existing CUDA run selects exactly what it selected
before, and MPS must be consulted only where CUDA is absent.

``select_device`` takes an injected torch-like module, so all three branches are
exercised deterministically without a GPU and without torch installed -- which
is what lets these run in CI.

Nothing here concerns the scoring mathematics. The backend is not part of the
estimand, and the tests below additionally assert that this change did not
reach any frozen quantity.
"""

import ast
import unittest
from pathlib import Path

from src.diagnostic_probe import mps_available, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeMps:
    def __init__(self, available=None, raises=False):
        self._available = available
        self._raises = raises

    def is_available(self):
        if self._raises:
            raise RuntimeError("Metal probe exploded")
        return self._available


class FakeBackends:
    def __init__(self, mps=None):
        if mps is not None:
            self.mps = mps


class FakeTorch:
    """A torch-shaped stub. Omitted attributes model older torch builds."""

    def __init__(self, cuda=None, backends=None):
        if cuda is not None:
            self.cuda = cuda
        if backends is not None:
            self.backends = backends


def torch_like(*, cuda, mps):
    """cuda/mps: True, False, or None meaning 'this attribute does not exist'."""
    return FakeTorch(
        cuda=None if cuda is None else FakeCuda(cuda),
        backends=FakeBackends(mps=None if mps is None else FakeMps(mps)),
    )


class TestDevicePriority(unittest.TestCase):
    def test_cuda_available_selects_cuda(self):
        self.assertEqual(select_device(torch_like(cuda=True, mps=False)), "cuda")

    def test_cuda_wins_even_when_mps_is_also_available(self):
        # The priority order is not "whichever accelerator exists". CUDA keeps
        # absolute priority so no existing CUDA run changes backend.
        self.assertEqual(select_device(torch_like(cuda=True, mps=True)), "cuda")

    def test_no_cuda_but_mps_available_selects_mps(self):
        self.assertEqual(select_device(torch_like(cuda=False, mps=True)), "mps")

    def test_neither_available_selects_cpu(self):
        self.assertEqual(select_device(torch_like(cuda=False, mps=False)), "cpu")

    def test_all_four_combinations(self):
        for cuda, mps, expected in (
            (True, True, "cuda"),
            (True, False, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ):
            with self.subTest(cuda=cuda, mps=mps):
                self.assertEqual(
                    select_device(torch_like(cuda=cuda, mps=mps)), expected
                )


class TestDegradedTorchBuilds(unittest.TestCase):
    """An absent or broken probe is an unusable backend, never a crash."""

    def test_torch_without_a_cuda_attribute_falls_through(self):
        self.assertEqual(select_device(torch_like(cuda=None, mps=True)), "mps")

    def test_torch_without_backends_mps_selects_cpu(self):
        # Older torch has no backends.mps at all.
        self.assertEqual(select_device(FakeTorch(cuda=FakeCuda(False))), "cpu")

    def test_backends_present_but_no_mps_attribute_selects_cpu(self):
        torch_module = FakeTorch(cuda=FakeCuda(False), backends=FakeBackends())
        self.assertEqual(select_device(torch_module), "cpu")

    def test_an_mps_probe_that_raises_is_treated_as_unavailable(self):
        torch_module = FakeTorch(
            cuda=FakeCuda(False), backends=FakeBackends(mps=FakeMps(raises=True))
        )
        self.assertFalse(mps_available(torch_module))
        self.assertEqual(select_device(torch_module), "cpu")

    def test_mps_available_is_strictly_boolean(self):
        self.assertIs(mps_available(torch_like(cuda=False, mps=True)), True)
        self.assertIs(mps_available(torch_like(cuda=False, mps=False)), False)


class TestGate1UsesTheSelectorAndRecordsItHonestly(unittest.TestCase):
    """Asserted on source: the script imports torch, which CI does not install."""

    @classmethod
    def setUpClass(cls):
        cls.path = PROJECT_ROOT / "scripts" / "reproduce_wang_baseline.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_the_scorer_selects_its_device_through_the_shared_helper(self):
        self.assertIn("from src.diagnostic_probe import select_device", self.source)
        self.assertIn("device = torch.device(select_device(torch))", self.source)

    def test_the_old_cuda_or_cpu_device_expression_is_gone(self):
        self.assertNotIn(
            'torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            self.source,
        )

    def test_the_selected_device_is_what_provenance_records(self):
        # collect_provenance stores str(device) at provenance.runtime.device, so
        # the artifact states the backend actually used. One source of truth:
        # the device is selected once and passed straight through.
        self.assertIn("device=device,", self.source)
        self.assertEqual(self.source.count("torch.device("), 1)

    def test_the_batch_size_policy_is_unchanged(self):
        # No new scientific batch-size policy is invented for MPS. The canonical
        # Wang-fidelity path is batch size 1, passed explicitly by the formal
        # run; CUDA's existing default of 8 is untouched.
        self.assertIn(
            "batch_size = args.batch_size or (8 if torch.cuda.is_available() else 2)",
            self.source,
        )


class TestNothingFrozenWasTouched(unittest.TestCase):
    """The backend is not part of the estimand. Prove the change stayed there.

    These assert repository invariants that hold on any commit, so they are
    meaningful on main as well as on a branch. Two earlier tests here compared
    ``git diff --name-only origin/main`` against an expected file list; they
    were removed because that is a property of one pull request, not of the
    repository. On main after merge the diff is empty and such a test fails for
    a reason that has nothing to do with the code. The PR diff is the right
    place to establish which files a change touched.
    """

    def test_the_score_version_is_untouched(self):
        # Read from source rather than imported: src/utils.py imports torch at
        # module scope and CI does not install it. The literal is what matters
        # anyway -- it is the cache key and the v2 corrected-scorer marker.
        source = (PROJECT_ROOT / "src" / "utils.py").read_text(encoding="utf-8")
        self.assertIn(
            'SCORE_VERSION = "wang-emnlp23-temp5-seg400-overlap100-hostscale-v2"',
            source,
        )

    def test_no_other_inference_script_was_repointed_yet(self):
        # D-03 and the cache-completion runner keep their existing cuda/cpu
        # selection, so they continue to RECORD the backend they actually use.
        # Repointing them is a separate, later change.
        for script in (
            "diagnose_ddre_ratio_support.py",
            "complete_nbc_cache.py",
        ):
            source = (PROJECT_ROOT / "scripts" / script).read_text(encoding="utf-8")
            with self.subTest(script=script):
                self.assertNotIn("select_device", source)
                self.assertIn(
                    'torch.device("cuda" if torch.cuda.is_available() else "cpu")',
                    source,
                )

    def test_device_state_still_reports_what_those_scripts_select(self):
        # device_state() feeds collect_live_environment for six scripts. It is
        # deliberately NOT changed: reporting "mps" there while those scripts
        # still select cpu would put a false device into their provenance.
        source = (PROJECT_ROOT / "src" / "diagnostic_probe.py").read_text(
            encoding="utf-8"
        )
        state = source[source.index("def device_state("):]
        self.assertIn(
            '"selected_device": "cuda" if torch.cuda.is_available() else "cpu"',
            state,
        )


if __name__ == "__main__":
    unittest.main()
