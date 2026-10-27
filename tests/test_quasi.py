"""Test quasi-random sequences."""

import numpy as np
import pytest
import torch

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


class TestSobolNKITables:
    """CPU checks of the NKI Sobol direction vectors against SobolEngine (#59).

    The NKI kernel uses 24-bit direction vectors; SobolEngine uses 30-bit ones
    (``sobolstate``). The 24-bit vectors must equal the top 24 bits of
    SobolEngine's for every supported dimension, so the unscrambled sequences
    agree bit-for-bit in those 24 bits.
    """

    def test_direction_vectors_match_sobol_engine(self):
        from trnrand.nki.dispatch import _SOBOL_BITS, _SOBOL_DIR_VECS, _SOBOL_MAX_DIMS

        engine = torch.quasirandom.SobolEngine(_SOBOL_MAX_DIMS, scramble=False)
        state = engine.sobolstate  # (dims, 30) int64
        shift = state.shape[1] - _SOBOL_BITS
        for d in range(_SOBOL_MAX_DIMS):
            expected = [int(v) >> shift for v in state[d, :_SOBOL_BITS]]
            assert list(_SOBOL_DIR_VECS[d]) == expected, f"dimension {d} differs"

    @pytest.mark.parametrize("start_index", [0, 1, 256, 1000])
    def test_reference_matches_sobol_engine(self, start_index):
        from trnrand.nki.dispatch import _SOBOL_BITS, _SOBOL_MAX_DIMS, sobol_reference

        n = 512
        engine = torch.quasirandom.SobolEngine(_SOBOL_MAX_DIMS, scramble=False)
        engine.fast_forward(start_index)
        # float64 draw is exact: 30-bit integers scaled by 2^-30.
        exact = engine.draw(n, dtype=torch.float64)
        expected = (exact * 2.0**30).round().long() >> (30 - _SOBOL_BITS)
        got = sobol_reference(n, _SOBOL_MAX_DIMS, start_index=start_index)
        got_int = (got.double() * 2.0**_SOBOL_BITS).round().long()
        assert torch.equal(got_int, expected)
