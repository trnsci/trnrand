"""
Tests for NKI backend dispatch wiring in distributions.py and quasi.py.

These tests verify that set_backend("nki") routes calls to the NKI path,
and set_backend("pytorch") routes to the PyTorch path. They run on the
NKI simulator (no hardware required) when TRNRAND_USE_SIMULATOR=1 is set,
or are skipped when NKI is unavailable.
"""

import pytest
import torch

import trnrand
from trnrand import HAS_NKI, get_backend, set_backend
from trnrand.distributions import exponential, normal, uniform
from trnrand.quasi import halton, sobol


@pytest.fixture(autouse=True)
def reset_backend():
    """Restore backend to auto after each test."""
    yield
    set_backend("auto")


class TestBackendRoutingCPU:
    """Verify backend routing without NKI — pytorch path always wins."""

    def test_set_backend_auto_is_default(self):
        assert get_backend() == "auto"

    def test_set_backend_pytorch_roundtrip(self):
        set_backend("pytorch")
        assert get_backend() == "pytorch"

    def test_set_backend_invalid_raises(self):
        with pytest.raises(AssertionError):
            set_backend("bad_backend")

    def test_set_backend_nki_raises_without_nki(self):
        if HAS_NKI:
            pytest.skip("NKI is available; test only applies when NKI is absent")
        with pytest.raises(RuntimeError, match="requires neuronxcc"):
            set_backend("nki")

    def test_pytorch_backend_uniform_shape(self):
        set_backend("pytorch")
        out = uniform(100)
        assert out.shape == (100,)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0 and out.max() < 1.0

    def test_pytorch_backend_normal_shape(self):
        set_backend("pytorch")
        out = normal(200)
        assert out.shape == (200,)
        assert out.dtype == torch.float32

    def test_pytorch_backend_sobol_shape(self):
        set_backend("pytorch")
        out = sobol(50, 3)
        assert out.shape == (50, 3)

    def test_pytorch_backend_halton_shape(self):
        set_backend("pytorch")
        out = halton(50, 3)
        assert out.shape == (50, 3)

    def test_pytorch_backend_exponential_shape(self):
        set_backend("pytorch")
        out = exponential(100, rate=2.0)
        assert out.shape == (100,)
        assert (out > 0).all()

    def test_generator_seeding_reproducible_pytorch(self):
        """Same generator seed → same output on pytorch path."""
        set_backend("pytorch")
        g1 = trnrand.Generator()
        g1.manual_seed(42)
        g2 = trnrand.Generator()
        g2.manual_seed(42)
        assert torch.equal(normal(100, generator=g1), normal(100, generator=g2))

    def test_generator_seeding_different_seeds_differ_pytorch(self):
        """Different seeds → different output on pytorch path."""
        set_backend("pytorch")
        g1 = trnrand.Generator()
        g1.manual_seed(1)
        g2 = trnrand.Generator()
        g2.manual_seed(2)
        assert not torch.equal(normal(100, generator=g1), normal(100, generator=g2))


@pytest.mark.nki_simulator
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestNKIDispatchSimulator:
    """Verify NKI routing via the CPU simulator (TRNRAND_USE_SIMULATOR=1)."""

    def test_nki_uniform_shape_and_range(self):
        set_backend("nki")
        out = uniform(256)
        assert out.shape == (256,)
        assert out.dtype == torch.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) < 1.0

    def test_nki_normal_shape(self):
        set_backend("nki")
        out = normal(256)
        assert out.shape == (256,)
        assert out.dtype == torch.float32

    def test_nki_exponential_shape_and_positivity(self):
        set_backend("nki")
        out = exponential(256, rate=1.0)
        assert out.shape == (256,)
        assert (out > 0).all()

    def test_nki_uniform_multidim(self):
        set_backend("nki")
        out = uniform(4, 64)
        assert out.shape == (4, 64)
        assert float(out.min()) >= 0.0 and float(out.max()) < 1.0

    def test_nki_normal_mean_std(self):
        set_backend("nki")
        out = normal(2048, mean=5.0, std=2.0)
        assert abs(float(out.mean()) - 5.0) < 0.3
        assert abs(float(out.std()) - 2.0) < 0.3

    def test_nki_reproducible_with_same_seed(self):
        """Same generator seed → same NKI output."""
        set_backend("nki")
        g1 = trnrand.Generator()
        g1.manual_seed(7)
        g2 = trnrand.Generator()
        g2.manual_seed(7)
        assert torch.equal(normal(256, generator=g1), normal(256, generator=g2))

    def test_nki_successive_calls_differ(self):
        """Successive calls with same generator produce different streams."""
        set_backend("nki")
        g = trnrand.Generator()
        g.manual_seed(99)
        a = normal(256, generator=g)
        b = normal(256, generator=g)
        assert not torch.equal(a, b)

    def test_nki_sobol_falls_through_to_cpu(self):
        """sobol() falls through to CPU path when sobol_nki is not yet available."""
        set_backend("nki")
        out = sobol(64, 3)
        assert out.shape == (64, 3)
        assert float(out.min()) >= 0.0 and float(out.max()) < 1.0

    def test_nki_halton_falls_through_to_cpu(self):
        """halton() falls through to CPU path when halton_nki is not yet available."""
        set_backend("nki")
        out = halton(64, 3)
        assert out.shape == (64, 3)
