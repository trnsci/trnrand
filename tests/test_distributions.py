"""Test random number generation."""

import pytest
import torch
import numpy as np
import trnrand
from trnrand import Generator


class TestGenerator:

    def test_seeded_reproducibility(self):
        g1 = Generator(seed=42)
        g2 = Generator(seed=42)
        a = trnrand.normal(1000, generator=g1)
        b = trnrand.normal(1000, generator=g2)
        np.testing.assert_allclose(a.numpy(), b.numpy())

    def test_different_seeds(self):
        g1 = Generator(seed=42)
        g2 = Generator(seed=99)
        a = trnrand.normal(1000, generator=g1)
        b = trnrand.normal(1000, generator=g2)
        assert not torch.allclose(a, b)

    def test_manual_seed(self):
        g = Generator(seed=42)
        a = trnrand.uniform(100, generator=g)
        g.manual_seed(42)
        b = trnrand.uniform(100, generator=g)
        np.testing.assert_allclose(a.numpy(), b.numpy())

    def test_module_level_seed(self):
        trnrand.manual_seed(123)
        a = trnrand.normal(100)
        trnrand.manual_seed(123)
        b = trnrand.normal(100)
        np.testing.assert_allclose(a.numpy(), b.numpy())


class TestUniform:

    def test_range(self):
        g = Generator(seed=42)
        x = trnrand.uniform(10000, low=2.0, high=5.0, generator=g)
        assert x.min().item() >= 2.0
        assert x.max().item() < 5.0

    def test_shape(self):
        x = trnrand.uniform(3, 4, 5)
        assert x.shape == (3, 4, 5)

    def test_mean(self):
        g = Generator(seed=42)
        x = trnrand.uniform(100000, low=0.0, high=1.0, generator=g)
        np.testing.assert_allclose(x.mean().item(), 0.5, atol=0.01)


class TestNormal:

    def test_shape(self):
        x = trnrand.normal(3, 4)
        assert x.shape == (3, 4)

    def test_statistics(self):
        g = Generator(seed=42)
        x = trnrand.normal(100000, mean=5.0, std=2.0, generator=g)
        np.testing.assert_allclose(x.mean().item(), 5.0, atol=0.05)
        np.testing.assert_allclose(x.std().item(), 2.0, atol=0.05)

    def test_standard_normal(self):
        g = Generator(seed=42)
        x = trnrand.standard_normal(100000, generator=g)
        np.testing.assert_allclose(x.mean().item(), 0.0, atol=0.02)
        np.testing.assert_allclose(x.std().item(), 1.0, atol=0.02)


class TestExponential:

    def test_positive(self):
        g = Generator(seed=42)
        x = trnrand.exponential(10000, rate=1.0, generator=g)
        assert x.min().item() > 0

    def test_mean(self):
        g = Generator(seed=42)
        rate = 2.0
        x = trnrand.exponential(100000, rate=rate, generator=g)
        np.testing.assert_allclose(x.mean().item(), 1.0 / rate, atol=0.02)


class TestBernoulli:

    def test_binary(self):
        g = Generator(seed=42)
        x = trnrand.bernoulli(1000, p=0.5, generator=g)
        unique = torch.unique(x)
        assert len(unique) == 2
        assert 0.0 in unique
        assert 1.0 in unique

    def test_probability(self):
        g = Generator(seed=42)
        p = 0.3
        x = trnrand.bernoulli(100000, p=p, generator=g)
        np.testing.assert_allclose(x.mean().item(), p, atol=0.01)


class TestRandint:

    def test_range(self):
        g = Generator(seed=42)
        x = trnrand.randint(10000, low=5, high=10, generator=g)
        assert x.min().item() >= 5
        assert x.max().item() < 10

    def test_dtype(self):
        x = trnrand.randint(10, low=0, high=5)
        assert x.dtype == torch.long


class TestRandperm:

    def test_is_permutation(self):
        g = Generator(seed=42)
        x = trnrand.randperm(100, generator=g)
        assert len(torch.unique(x)) == 100
        assert x.min().item() == 0
        assert x.max().item() == 99


class TestTruncatedNormal:

    def test_bounds(self):
        g = Generator(seed=42)
        x = trnrand.truncated_normal(10000, mean=0.0, std=1.0, low=-2.0, high=2.0, generator=g)
        assert x.min().item() >= -2.0
        assert x.max().item() <= 2.0

    def test_shape(self):
        x = trnrand.truncated_normal(5, 3)
        assert x.shape == (5, 3)


class TestGamma:

    def test_shape(self):
        x = trnrand.gamma(7, 5, shape=2.0)
        assert x.shape == (7, 5)

    def test_positive(self):
        g = Generator(seed=42)
        x = trnrand.gamma(10000, shape=2.0, scale=1.0, generator=g)
        assert (x > 0).all()

    def test_statistics_shape_gt_1(self):
        # Gamma(k=3, θ=2): mean=6, var=12.
        g = Generator(seed=42)
        x = trnrand.gamma(200000, shape=3.0, scale=2.0, generator=g).double()
        np.testing.assert_allclose(x.mean().item(), 6.0, atol=0.1)
        np.testing.assert_allclose(x.var().item(), 12.0, atol=0.5)

    def test_statistics_shape_lt_1(self):
        # Gamma(k=0.5, θ=1): mean=0.5, var=0.5. Exercises the boost path.
        g = Generator(seed=42)
        x = trnrand.gamma(200000, shape=0.5, scale=1.0, generator=g).double()
        np.testing.assert_allclose(x.mean().item(), 0.5, atol=0.02)
        np.testing.assert_allclose(x.var().item(), 0.5, atol=0.05)

    def test_reproducibility(self):
        g1 = Generator(seed=42)
        g2 = Generator(seed=42)
        a = trnrand.gamma(1000, shape=2.5, generator=g1)
        b = trnrand.gamma(1000, shape=2.5, generator=g2)
        np.testing.assert_allclose(a.numpy(), b.numpy())


class TestChiSquared:

    def test_shape(self):
        x = trnrand.chi_squared(4, 6, df=5)
        assert x.shape == (4, 6)

    def test_positive(self):
        x = trnrand.chi_squared(10000, df=3)
        assert (x > 0).all()

    def test_statistics(self):
        # χ²(df=8): mean=8, var=16.
        g = Generator(seed=42)
        x = trnrand.chi_squared(200000, df=8, generator=g).double()
        np.testing.assert_allclose(x.mean().item(), 8.0, atol=0.1)
        np.testing.assert_allclose(x.var().item(), 16.0, atol=0.8)

    def test_reproducibility(self):
        g1 = Generator(seed=7)
        g2 = Generator(seed=7)
        a = trnrand.chi_squared(1000, df=4, generator=g1)
        b = trnrand.chi_squared(1000, df=4, generator=g2)
        np.testing.assert_allclose(a.numpy(), b.numpy())


class TestBeta:

    def test_shape(self):
        x = trnrand.beta(3, 4, alpha=2.0, beta=3.0)
        assert x.shape == (3, 4)

    def test_range(self):
        g = Generator(seed=42)
        x = trnrand.beta(10000, alpha=2.0, beta=5.0, generator=g)
        assert (x > 0).all()
        assert (x < 1).all()

    def test_statistics(self):
        # Beta(α=2, β=3): mean=0.4, var=6/(25·6) = 0.04.
        g = Generator(seed=42)
        x = trnrand.beta(200000, alpha=2.0, beta=3.0, generator=g).double()
        np.testing.assert_allclose(x.mean().item(), 0.4, atol=0.01)
        np.testing.assert_allclose(x.var().item(), 0.04, atol=0.005)

    def test_reproducibility(self):
        g1 = Generator(seed=99)
        g2 = Generator(seed=99)
        a = trnrand.beta(1000, alpha=1.5, beta=2.5, generator=g1)
        b = trnrand.beta(1000, alpha=1.5, beta=2.5, generator=g2)
        np.testing.assert_allclose(a.numpy(), b.numpy())


class TestPoisson:

    def test_shape(self):
        x = trnrand.poisson(5, 6, lam=3.0)
        assert x.shape == (5, 6)

    def test_nonnegative_integer(self):
        g = Generator(seed=42)
        x = trnrand.poisson(10000, lam=5.0, generator=g)
        assert (x >= 0).all()
        # Poisson output is integer-valued even when stored as float.
        assert torch.equal(x, x.round())

    def test_statistics(self):
        # Poisson(λ=7): mean=7, var=7.
        g = Generator(seed=42)
        x = trnrand.poisson(100000, lam=7.0, generator=g).double()
        np.testing.assert_allclose(x.mean().item(), 7.0, atol=0.05)
        np.testing.assert_allclose(x.var().item(), 7.0, atol=0.1)

    def test_reproducibility(self):
        g1 = Generator(seed=11)
        g2 = Generator(seed=11)
        a = trnrand.poisson(1000, lam=4.0, generator=g1)
        b = trnrand.poisson(1000, lam=4.0, generator=g2)
        np.testing.assert_allclose(a.numpy(), b.numpy())
