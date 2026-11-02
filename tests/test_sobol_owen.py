"""Owen-scrambled Sobol, `sobol(scramble=True)` (#100). CPU path and the
host reference of the NKI kernel."""

import itertools

import numpy as np
import pytest
import torch

import trnrand
from trnrand.nki import dispatch


def _counts(x: torch.Tensor, dims, bits) -> list[int]:
    """Sorted point counts over the elementary intervals with sides 2^-bits[i]."""
    cell = torch.zeros(x.shape[0], dtype=torch.int64)
    for d, b in zip(dims, bits, strict=True):
        cell = cell * 2**b + torch.floor(x[:, d] * 2**b).long()
    c = torch.bincount(cell, minlength=2 ** sum(bits))
    return sorted(c.tolist())


class TestNetProperty:
    @pytest.mark.parametrize("seed", [0, 1, 2**40 + 3])
    def test_one_point_per_stratum_each_dim(self, seed):
        x = trnrand.sobol(1024, 10, seed=seed, dtype=torch.float64)
        for d in range(10):
            cells = torch.floor(x[:, d] * 1024).long()
            assert torch.equal(torch.sort(cells).values, torch.arange(1024)), d

    @pytest.mark.parametrize("seed", [0, 7])
    def test_elementary_interval_counts_match_unscrambled(self, seed):
        # Nested scrambling maps each elementary interval onto one of the same
        # shape, so for every 2-D projection and every shape 2^-a x 2^-b the
        # multiset of counts equals that of the plain Sobol set.
        m = 9
        n = 2**m
        x = trnrand.sobol(n, 6, seed=seed, dtype=torch.float64)
        plain = trnrand.sobol(n, 6, scramble=False, dtype=torch.float64)
        for d0, d1 in itertools.combinations(range(6), 2):
            for a in range(m + 1):
                for b in range(m + 1 - a):
                    want = _counts(plain, (d0, d1), (a, b))
                    assert _counts(x, (d0, d1), (a, b)) == want, (d0, d1, a, b)

    def test_zero_net_in_first_two_dims(self):
        # Dims 0-1 of Sobol are a (0, m, 2)-net: exactly one point per interval.
        m = 10
        x = trnrand.sobol(2**m, 2, seed=11, dtype=torch.float64)
        for a in range(m + 1):
            assert _counts(x, (0, 1), (a, m - a)) == [1] * 2**m, a


class TestDeterminism:
    def test_same_seed_same_points(self):
        assert torch.equal(trnrand.sobol(300, 5, seed=4), trnrand.sobol(300, 5, seed=4))

    def test_seeds_differ(self):
        a = trnrand.sobol(300, 5, seed=4)
        for s in (0, 5, -4, 2**63):
            assert not torch.equal(a, trnrand.sobol(300, 5, seed=s)), s

    def test_dimensions_are_prefix_stable(self):
        full = trnrand.sobol(256, 8, seed=9)
        assert torch.equal(full[:, :3], trnrand.sobol(256, 3, seed=9))

    def test_seed_none_uses_global_rng(self):
        torch.manual_seed(5)
        a = trnrand.sobol(128, 3)
        torch.manual_seed(5)
        assert torch.equal(a, trnrand.sobol(128, 3))
        assert not torch.equal(a, trnrand.sobol(128, 3))

    def test_owen_alias(self):
        assert torch.equal(
            trnrand.sobol(64, 3, seed=2), trnrand.sobol(64, 3, seed=2, scramble="owen")
        )

    def test_float32_is_top_24_bits_of_float64(self):
        x64 = trnrand.sobol(1000, 4, seed=3, dtype=torch.float64)
        x32 = trnrand.sobol(1000, 4, seed=3, dtype=torch.float32)
        assert torch.equal(x32.double(), torch.floor(x64 * 2**24) / 2**24)


class TestShift:
    @pytest.mark.parametrize("seed", [None, 0, 42])
    def test_shift_is_the_old_cpu_scramble(self, seed):
        torch.manual_seed(1)
        got = trnrand.sobol(200, 4, seed=seed, scramble="shift")
        torch.manual_seed(1)
        ref = torch.quasirandom.SobolEngine(4, scramble=True, seed=seed).draw(200)
        assert torch.equal(got, ref)

    def test_owen_stream_differs_from_shift(self):
        assert not torch.equal(
            trnrand.sobol(64, 3, seed=42), trnrand.sobol(64, 3, seed=42, scramble="shift")
        )


class TestNkiReference:
    """`sobol_owen_reference` mirrors the NKI kernel's 24-bit limb arithmetic."""

    def test_limb_arithmetic(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2**24, 100_000, dtype=np.int64)
        b = rng.integers(0, 2**24, 100_000, dtype=np.int64)
        edge = np.array([0, 1, 2**24 - 1, 2**23, 0xFFF, 0x1000], dtype=np.int64)
        a = np.concatenate([a, *[np.full(6, e) for e in edge]])
        b = np.concatenate([b, *[edge for _ in edge]])
        np.testing.assert_array_equal(dispatch._mul24_np(a, b), (a * b) & 0xFFFFFF)
        np.testing.assert_array_equal(dispatch._add24_np(a, b), (a + b) & 0xFFFFFF)

    @pytest.mark.parametrize("seed", [0, 1, 123456789, 2**64 - 1])
    @pytest.mark.parametrize("n", [1, 127, 129, 1000])
    def test_reference_equals_cpu_path(self, n, seed):
        ref = dispatch.sobol_owen_reference(n, 10, seed)
        assert torch.equal(ref, trnrand.sobol(n, 10, seed=seed, scramble=True))

    def test_reference_start_index(self):
        full = dispatch.sobol_owen_reference(512, 4, 6)
        assert torch.equal(full[256:], dispatch.sobol_owen_reference(256, 4, 6, start_index=256))

    def test_reversed_direction_vectors(self):
        for fwd, rev in zip(dispatch._SOBOL_DIR_VECS, dispatch._SOBOL_REV_DIR_VECS, strict=True):
            for v, r in zip(fwd, rev, strict=True):
                assert int(f"{v:024b}"[::-1], 2) == r
