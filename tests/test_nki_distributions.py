"""
Hardware tests for NKI distribution kernels (closes #14, #13, #16, #17).

All tests require Neuron hardware (trn1/trn2) and are marked @pytest.mark.neuron.
They are not run in CI — execute manually on a Neuron instance via:

    pytest tests/test_nki_distributions.py -v -m neuron

Or via the run_neuron_tests.sh helper:

    AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn2
"""

import math

import pytest
import torch

import trnrand
from trnrand import HAS_NKI, get_backend, set_backend
from trnrand.distributions import beta, chi_squared, gamma, truncated_normal


@pytest.fixture(autouse=True)
def reset_backend():
    """Restore backend to auto after each test."""
    yield
    set_backend("auto")


# ---------------------------------------------------------------------------
# Truncated Normal (#17)
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestTruncatedNormalNKI:
    """NKI truncated-normal kernel: bounds, moments, reproducibility."""

    def test_bounds_default(self):
        """All samples in [-2σ, +2σ] (default bounds in std units)."""
        set_backend("nki")
        out = truncated_normal(1024)
        assert out.shape == (1024,)
        assert float(out.min()) >= -2.0
        assert float(out.max()) <= 2.0

    def test_bounds_custom(self):
        """Samples respect asymmetric custom bounds."""
        set_backend("nki")
        out = truncated_normal(1024, low=-1.0, high=3.0)
        assert float(out.min()) >= -1.0
        assert float(out.max()) <= 3.0

    def test_moments_default_bounds(self):
        """Mean and std approximate N(0,1) restricted to [-2, +2]."""
        set_backend("nki")
        out = truncated_normal(4096)
        # Truncated normal N(0,1) in [-2,2]: mean ≈ 0, std ≈ 0.877
        assert abs(float(out.mean())) < 0.05
        assert 0.7 < float(out.std()) < 1.0

    def test_output_mean_std_shift(self):
        """mean= and std= parameters shift the output distribution."""
        set_backend("nki")
        out = truncated_normal(4096, mean=3.0, std=0.5)
        assert abs(float(out.mean()) - 3.0) < 0.15
        assert 0.35 < float(out.std()) < 0.55

    def test_seed_deterministic(self):
        """Same generator seed → identical output."""
        set_backend("nki")
        g1 = trnrand.Generator()
        g1.manual_seed(42)
        g2 = trnrand.Generator()
        g2.manual_seed(42)
        a = truncated_normal(512, generator=g1)
        b = truncated_normal(512, generator=g2)
        assert torch.equal(a, b)

    def test_successive_calls_differ(self):
        """Successive calls with the same generator produce different streams."""
        set_backend("nki")
        g = trnrand.Generator()
        g.manual_seed(7)
        a = truncated_normal(256, generator=g)
        b = truncated_normal(256, generator=g)
        assert not torch.equal(a, b)

    def test_dtype_float32(self):
        """Output dtype is float32."""
        set_backend("nki")
        out = truncated_normal(256)
        assert out.dtype == torch.float32

    def test_multidim_shape(self):
        """Reshape to multi-dimensional output works correctly."""
        set_backend("nki")
        out = truncated_normal(4, 64)
        assert out.shape == (4, 64)
        assert float(out.min()) >= -2.0
        assert float(out.max()) <= 2.0


# ---------------------------------------------------------------------------
# Gamma (#14)
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestGammaNKI:
    """NKI gamma kernel: moments, shape < 1 boost identity, reproducibility."""

    def test_shape_ge_1_mean(self):
        """Mean ≈ shape * scale for shape ≥ 1."""
        set_backend("nki")
        shape, scale = 3.0, 2.0
        out = gamma(4096, shape=shape, scale=scale)
        expected_mean = shape * scale
        assert abs(float(out.mean()) - expected_mean) < 0.15 * expected_mean

    def test_shape_ge_1_variance(self):
        """Variance ≈ shape * scale² for shape ≥ 1."""
        set_backend("nki")
        shape, scale = 3.0, 2.0
        out = gamma(4096, shape=shape, scale=scale)
        expected_var = shape * scale**2
        assert abs(float(out.var()) - expected_var) < 0.2 * expected_var

    def test_shape_lt_1_mean(self):
        """Mean ≈ shape * scale for shape < 1 (boost identity path)."""
        set_backend("nki")
        shape, scale = 0.5, 1.0
        out = gamma(4096, shape=shape, scale=scale)
        expected_mean = shape * scale
        assert abs(float(out.mean()) - expected_mean) < 0.2 * expected_mean

    def test_positivity(self):
        """All gamma samples are strictly positive."""
        set_backend("nki")
        out = gamma(1024, shape=2.0, scale=1.0)
        assert (out > 0).all()

    def test_seed_deterministic(self):
        """Same seed → same output."""
        set_backend("nki")
        g1 = trnrand.Generator()
        g1.manual_seed(13)
        g2 = trnrand.Generator()
        g2.manual_seed(13)
        a = gamma(512, shape=2.0, scale=1.0, generator=g1)
        b = gamma(512, shape=2.0, scale=1.0, generator=g2)
        assert torch.equal(a, b)

    def test_successive_calls_differ(self):
        """Successive calls with the same generator differ."""
        set_backend("nki")
        g = trnrand.Generator()
        g.manual_seed(99)
        a = gamma(256, shape=2.0, scale=1.0, generator=g)
        b = gamma(256, shape=2.0, scale=1.0, generator=g)
        assert not torch.equal(a, b)

    def test_shape_one_special_case(self):
        """Gamma(1, scale) is Exponential(1/scale): mean ≈ scale."""
        set_backend("nki")
        out = gamma(4096, shape=1.0, scale=2.0)
        assert abs(float(out.mean()) - 2.0) < 0.15

    def test_dtype_float32(self):
        set_backend("nki")
        out = gamma(256, shape=2.0)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Chi-Squared (#16)
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestChiSquaredNKI:
    """NKI chi-squared kernel: moments and positivity."""

    def test_mean(self):
        """Mean ≈ df."""
        set_backend("nki")
        df = 4.0
        out = chi_squared(4096, df=df)
        assert abs(float(out.mean()) - df) < 0.2 * df

    def test_variance(self):
        """Variance ≈ 2*df."""
        set_backend("nki")
        df = 4.0
        out = chi_squared(4096, df=df)
        expected_var = 2.0 * df
        assert abs(float(out.var()) - expected_var) < 0.3 * expected_var

    def test_positivity(self):
        """All chi-squared samples are positive."""
        set_backend("nki")
        out = chi_squared(1024, df=3.0)
        assert (out > 0).all()

    def test_df_1_equivalent_to_squared_normal(self):
        """Chi-sq(1): mean ≈ 1, variance ≈ 2."""
        set_backend("nki")
        out = chi_squared(4096, df=1.0)
        assert abs(float(out.mean()) - 1.0) < 0.15
        assert abs(float(out.var()) - 2.0) < 0.5

    def test_seed_deterministic(self):
        set_backend("nki")
        g1 = trnrand.Generator()
        g1.manual_seed(21)
        g2 = trnrand.Generator()
        g2.manual_seed(21)
        a = chi_squared(256, df=4.0, generator=g1)
        b = chi_squared(256, df=4.0, generator=g2)
        assert torch.equal(a, b)

    def test_dtype_float32(self):
        set_backend("nki")
        out = chi_squared(256, df=4.0)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Beta (#13)
# ---------------------------------------------------------------------------


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestBetaNKI:
    """NKI beta kernel: bounds, moments, reproducibility."""

    def test_bounds(self):
        """All beta samples in (0, 1)."""
        set_backend("nki")
        out = beta(2048, alpha=2.0, beta=3.0)
        assert (out > 0).all()
        assert (out < 1).all()

    def test_mean(self):
        """Mean ≈ α/(α+β)."""
        set_backend("nki")
        alpha, beta_val = 2.0, 3.0
        out = beta(4096, alpha=alpha, beta=beta_val)
        expected = alpha / (alpha + beta_val)
        assert abs(float(out.mean()) - expected) < 0.05

    def test_symmetric_mean_half(self):
        """Beta(a, a) has mean 0.5."""
        set_backend("nki")
        out = beta(4096, alpha=3.0, beta=3.0)
        assert abs(float(out.mean()) - 0.5) < 0.03

    def test_seed_deterministic(self):
        set_backend("nki")
        g1 = trnrand.Generator()
        g1.manual_seed(77)
        g2 = trnrand.Generator()
        g2.manual_seed(77)
        a = beta(512, alpha=2.0, beta=5.0, generator=g1)
        b = beta(512, alpha=2.0, beta=5.0, generator=g2)
        assert torch.equal(a, b)

    def test_successive_calls_differ(self):
        set_backend("nki")
        g = trnrand.Generator()
        g.manual_seed(33)
        a = beta(256, alpha=2.0, beta=3.0, generator=g)
        b = beta(256, alpha=2.0, beta=3.0, generator=g)
        assert not torch.equal(a, b)

    def test_dtype_float32(self):
        set_backend("nki")
        out = beta(256, alpha=2.0, beta=2.0)
        assert out.dtype == torch.float32

    def test_multidim_shape(self):
        set_backend("nki")
        out = beta(4, 64, alpha=1.0, beta=1.0)
        assert out.shape == (4, 64)
        assert (out > 0).all() and (out < 1).all()
