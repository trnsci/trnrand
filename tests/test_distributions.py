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
