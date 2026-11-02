"""
Hardware tests for NKI quasi-random sequence kernels (closes #11, #12).

All tests require Neuron hardware (trn1/trn2) and are marked @pytest.mark.neuron.
They are not run in CI — execute manually on a Neuron instance via:

    pytest tests/test_nki_quasi.py -v -m neuron

Or via the run_neuron_tests.sh helper:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn2
"""

import pytest
import torch

import trnrand
from trnrand import HAS_NKI, get_backend, set_backend
from trnrand.quasi import halton, sobol


@pytest.fixture(autouse=True)
def reset_backend():
    """Restore backend to auto after each test."""
    yield
    set_backend("auto")


# ---------------------------------------------------------------------------
# Sobol NKI (#11)
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestSobolNKI:
    """NKI Sobol kernel: shape, range, uniformity, seed determinism."""

    def test_shape_1d(self):
        set_backend("nki")
        out = sobol(128, 1)
        assert out.shape == (128, 1)

    def test_shape_multidim(self):
        set_backend("nki")
        out = sobol(256, 5)
        assert out.shape == (256, 5)

    def test_range_unit_interval(self):
        """All coordinates in [0, 1)."""
        set_backend("nki")
        out = sobol(512, 4)
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 1.0

    def test_dtype_float32(self):
        set_backend("nki")
        out = sobol(64, 2)
        assert out.dtype == torch.float32

    def test_max_dims(self):
        """10 dimensions supported (the maximum)."""
        set_backend("nki")
        out = sobol(128, 10)
        assert out.shape == (128, 10)
        assert float(out.min()) >= 0.0 and float(out.max()) < 1.0

    def test_marginal_uniformity_dim0(self):
        """First-dimension marginal is approximately uniform on [0,1)."""
        set_backend("nki")
        out = sobol(1024, 2)
        # Mean of U[0,1) = 0.5; Sobol in dim 0 is Van der Corput — very uniform
        assert abs(float(out[:, 0].mean()) - 0.5) < 0.02

    def test_marginal_uniformity_dim1(self):
        """Second-dimension marginal is approximately uniform on [0,1)."""
        set_backend("nki")
        out = sobol(1024, 2)
        assert abs(float(out[:, 1].mean()) - 0.5) < 0.05

    def test_deterministic_no_seed(self):
        """Same call (seed=0) produces the same result."""
        set_backend("nki")
        a = sobol(256, 3, seed=0)
        b = sobol(256, 3, seed=0)
        assert torch.equal(a, b)

    def test_deterministic_with_seed(self):
        """Same nonzero seed → same scrambled result."""
        set_backend("nki")
        a = sobol(256, 3, seed=42)
        b = sobol(256, 3, seed=42)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        """Different nonzero seeds → different outputs."""
        set_backend("nki")
        a = sobol(256, 3, seed=1)
        b = sobol(256, 3, seed=2)
        assert not torch.equal(a, b)

    def test_start_index_advances_sequence(self):
        """start_index shifts the Sobol sequence forward.

        start_index is a parameter of the NKI wrapper ``sobol_nki``; the
        public ``sobol()`` does not take it (#59).
        """
        from trnrand.nki.dispatch import sobol_nki

        full = sobol_nki(512, 2, seed=0)
        tail = sobol_nki(256, 2, seed=0, start_index=256)
        assert torch.equal(full[256:], tail)

    def test_conformance_vs_cpu(self):
        """NKI output matches CPU Sobol output (without scrambling), all dims."""
        set_backend("nki")
        nki_out = sobol(256, 10, scramble=False)
        set_backend("pytorch")
        cpu_out = sobol(256, 10, scramble=False)
        # NKI uses Van der Corput / Joe-Kuo; CPU uses PyTorch SobolEngine.
        # Without scrambling, both should produce identical sequences.
        assert torch.allclose(nki_out, cpu_out, atol=1e-5)

    def test_non_power_of_two_n_points(self):
        """n_points not divisible by 128 (tile size) handled correctly."""
        set_backend("nki")
        out = sobol(100, 2)
        assert out.shape == (100, 2)
        assert float(out.min()) >= 0.0 and float(out.max()) < 1.0

    def test_large_batch(self):
        """Large batch (multiple tiles) produces correct shape and range."""
        set_backend("nki")
        out = sobol(2048, 4)
        assert out.shape == (2048, 4)
        assert float(out.min()) >= 0.0 and float(out.max()) < 1.0


# ---------------------------------------------------------------------------
# Owen-scrambled Sobol NKI (#100)
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.needs_hw_validation
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestSobolOwenNKI:
    """`sobol_owen_kernel`. Written and checked on CPU against
    `sobol_owen_reference` only; these tests are its first hardware run."""

    @pytest.mark.parametrize("seed", [0, 1, 123456789])
    @pytest.mark.parametrize("n", [1, 127, 129, 1000])
    def test_matches_reference(self, n, seed):
        from trnrand.nki.dispatch import sobol_owen_nki, sobol_owen_reference

        got = sobol_owen_nki(n, 10, seed=seed)
        assert torch.equal(got.cpu(), sobol_owen_reference(n, 10, seed))

    def test_matches_cpu_path(self):
        set_backend("nki")
        dev = sobol(1000, 10, seed=42)
        set_backend("pytorch")
        assert torch.equal(dev.cpu(), sobol(1000, 10, seed=42))

    def test_start_index(self):
        from trnrand.nki.dispatch import sobol_owen_nki

        full = sobol_owen_nki(512, 3, seed=8)
        assert torch.equal(full[256:], sobol_owen_nki(256, 3, seed=8, start_index=256))

    def test_one_point_per_stratum(self):
        set_backend("nki")
        x = sobol(256, 10, seed=5).cpu().double()
        for d in range(10):
            cells = torch.floor(x[:, d] * 256).long()
            assert torch.equal(torch.sort(cells).values, torch.arange(256)), d


# ---------------------------------------------------------------------------
# Halton NKI (#12) — placeholder, activated when halton_nki lands in PR5
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestHaltonNKI:
    """NKI Halton kernel: shape, range, uniformity.

    These tests require PR5 (feat/nki-halton) to land. Until then, halton()
    falls through to the CPU path when backend="nki" (same as the CPU tests).
    """

    def test_shape(self):
        set_backend("nki")
        out = halton(128, 3)
        assert out.shape == (128, 3)

    def test_range_unit_interval(self):
        set_backend("nki")
        out = halton(256, 4)
        assert float(out.min()) > 0.0
        assert float(out.max()) < 1.0

    def test_dtype_float32(self):
        set_backend("nki")
        out = halton(64, 2)
        assert out.dtype == torch.float32

    def test_marginal_uniformity_base2(self):
        """First dimension (base 2) is approximately uniform."""
        set_backend("nki")
        out = halton(1024, 3)
        assert abs(float(out[:, 0].mean()) - 0.5) < 0.05

    def test_marginal_uniformity_base3(self):
        """Second dimension (base 3) is approximately uniform."""
        set_backend("nki")
        out = halton(1024, 3)
        assert abs(float(out[:, 1].mean()) - 0.5) < 0.05

    def test_deterministic(self):
        """Halton is deterministic — same call → same result."""
        set_backend("nki")
        a = halton(256, 3)
        b = halton(256, 3)
        assert torch.equal(a, b)

    def test_non_power_of_two(self):
        """Non-power-of-two n_points handled correctly."""
        set_backend("nki")
        out = halton(100, 3)
        assert out.shape == (100, 3)
