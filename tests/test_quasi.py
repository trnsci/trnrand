"""Test quasi-random sequences."""

import pytest
import torch
import numpy as np
import trnrand


class TestSobol:

    def test_shape(self):
        x = trnrand.sobol(100, 5)
        assert x.shape == (100, 5)

    def test_range(self):
        x = trnrand.sobol(1000, 3)
        assert x.min().item() >= 0.0
        assert x.max().item() < 1.0

    def test_deterministic(self):
        a = trnrand.sobol(100, 4, seed=42)
        b = trnrand.sobol(100, 4, seed=42)
        np.testing.assert_allclose(a.numpy(), b.numpy())

    def test_uniformity(self):
        """Sobol points should be more uniform than random."""
        x = trnrand.sobol(1024, 2, seed=42)
        # Check that each quadrant has roughly 25% of points
        q1 = ((x[:, 0] < 0.5) & (x[:, 1] < 0.5)).sum().item()
        q2 = ((x[:, 0] >= 0.5) & (x[:, 1] < 0.5)).sum().item()
        q3 = ((x[:, 0] < 0.5) & (x[:, 1] >= 0.5)).sum().item()
        q4 = ((x[:, 0] >= 0.5) & (x[:, 1] >= 0.5)).sum().item()
        for q in [q1, q2, q3, q4]:
            assert abs(q - 256) < 30  # Within ~12% of ideal


class TestHalton:

    def test_shape(self):
        x = trnrand.halton(100, 3)
        assert x.shape == (100, 3)

    def test_range(self):
        x = trnrand.halton(1000, 5)
        assert x.min().item() > 0.0
        assert x.max().item() < 1.0

    def test_deterministic(self):
        a = trnrand.halton(50, 3)
        b = trnrand.halton(50, 3)
        np.testing.assert_allclose(a.numpy(), b.numpy())

    def test_first_dim_is_halton_base2(self):
        """First dimension should be van der Corput in base 2."""
        x = trnrand.halton(4, 1)
        # Base 2: 1/2, 1/4, 3/4, 1/8
        expected = [0.5, 0.25, 0.75, 0.125]
        np.testing.assert_allclose(x[:, 0].numpy(), expected, atol=1e-6)


class TestLatinHypercube:

    def test_shape(self):
        x = trnrand.latin_hypercube(50, 3)
        assert x.shape == (50, 3)

    def test_range(self):
        x = trnrand.latin_hypercube(100, 4)
        assert x.min().item() >= 0.0
        assert x.max().item() < 1.0

    def test_stratification(self):
        """Each row of n bins should contain exactly one sample."""
        n = 100
        x = trnrand.latin_hypercube(n, 2)
        for d in range(2):
            bins = (x[:, d] * n).long()
            assert len(torch.unique(bins)) == n
