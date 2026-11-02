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


class TestLatinHypercubeReducedPrecision:
    """float16 / bfloat16 LHS must keep one point per stratum (#96)."""

    @staticmethod
    def _stratified(x, n):
        if bool((x >= 1).any()) or bool((x < 0).any()):
            return False
        return all(
            torch.equal(
                torch.sort(torch.floor(x[:, d].double() * n).long()).values, torch.arange(n)
            )
            for d in range(x.shape[1])
        )

    @pytest.mark.parametrize(
        ("dtype", "n"),
        [(torch.float16, 64), (torch.float16, 1000), (torch.bfloat16, 32), (torch.bfloat16, 200)],
    )
    def test_one_point_per_stratum(self, dtype, n):
        for seed in range(5):
            x = trnrand.latin_hypercube(
                n, 3, dtype=dtype, generator=torch.Generator().manual_seed(seed)
            )
            assert x.dtype == dtype
            assert self._stratified(x, n), f"seed {seed}"

    def test_valid_points_unchanged(self):
        n = 64
        x = trnrand.latin_hypercube(
            n, 2, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1)
        )
        g = torch.Generator().manual_seed(1)
        for d in range(2):
            perm = torch.randperm(n, generator=g)
            u = torch.rand(n, generator=g, dtype=torch.bfloat16)
            old = (perm.to(torch.bfloat16) + u) / n
            ok = (old.double() >= perm.double() / n) & (old.double() < (perm.double() + 1) / n)
            assert ok.any()
            assert torch.equal(x[ok, d], old[ok])

    def test_unrepresentable_raises(self):
        with pytest.raises(ValueError, match="strata"):
            trnrand.latin_hypercube(
                5000, 1, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
            )

    def test_float32_unchanged(self):
        x = trnrand.latin_hypercube(100, 2, generator=torch.Generator().manual_seed(3))
        g = torch.Generator().manual_seed(3)
        for d in range(2):
            perm = torch.randperm(100, generator=g)
            u = torch.rand(100, generator=g)
            assert torch.equal(x[:, d], (perm.float() + u) / 100)


class TestQuasiReducedPrecisionRange:
    """float16 / bfloat16 sobol / halton stay inside the unit interval (#101)."""

    @pytest.fixture
    def pytorch_backend(self):
        # The SobolEngine references below are the PyTorch path; on a Neuron
        # host the auto backend takes the NKI path (a random shift, #100).
        prev = trnrand.get_backend()
        trnrand.set_backend("pytorch")
        yield
        trnrand.set_backend(prev)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "fn",
        [
            lambda n, dt: trnrand.sobol(n, 4, seed=3, dtype=dt),
            lambda n, dt: trnrand.sobol(n, 4, scramble=False, dtype=dt),
            lambda n, dt: trnrand.halton(n, 4, dtype=dt),
        ],
        ids=["sobol", "sobol_unscrambled", "halton"],
    )
    def test_below_one(self, fn, dtype):
        x = fn(4096, dtype)
        assert x.dtype == dtype
        assert bool((x >= 0).all() and (x < 1).all()), f"{int((x >= 1).sum())} points >= 1"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_only_points_rounding_to_one_change(self, pytorch_backend, dtype):
        ref = torch.quasirandom.SobolEngine(4, scramble=True, seed=3).draw(4096)
        cast = ref.to(dtype)
        x = trnrand.sobol(4096, 4, seed=3, dtype=dtype)
        hit = cast == 1
        assert bool(hit.any())  # the case this guards actually occurs
        assert torch.equal(x[~hit], cast[~hit])
        top = torch.nextafter(torch.ones((), dtype=dtype), torch.zeros((), dtype=dtype))
        assert bool((x[hit] == top).all())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_full_precision_bit_identical(self, pytorch_backend, dtype):
        ref = torch.quasirandom.SobolEngine(4, scramble=True, seed=3).draw(4096).to(dtype)
        assert torch.equal(trnrand.sobol(4096, 4, seed=3, dtype=dtype), ref)

    @pytest.mark.neuron
    @pytest.mark.skipif(not trnrand.HAS_NKI, reason="requires the NKI path")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "fn",
        [
            lambda dt: trnrand.sobol(4096, 4, seed=3, dtype=dt),
            lambda dt: trnrand.sobol(4096, 4, scramble=False, dtype=dt),
            lambda dt: trnrand.halton(4096, 4, dtype=dt),
        ],
        ids=["sobol", "sobol_unscrambled", "halton"],
    )
    def test_nki_only_points_rounding_to_one_change(self, fn, dtype):
        # NKI return path: the reduced-precision output is the float32 NKI
        # output cast, except points that round to 1.0.
        prev = trnrand.get_backend()
        trnrand.set_backend("nki")
        try:
            cast = fn(torch.float32).to(dtype)
            x = fn(dtype)
        finally:
            trnrand.set_backend(prev)
        hit = cast == 1
        assert bool(hit.any())  # the case this guards actually occurs
        assert torch.equal(x[~hit], cast[~hit])
        top = torch.nextafter(torch.ones((), dtype=dtype), torch.zeros((), dtype=dtype))
        assert bool((x[hit] == top).all())
