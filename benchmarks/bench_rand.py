"""RNG benchmarks.

Run with:

    pytest benchmarks/ --benchmark-only

Three variants per operation:

- `_nki`             — trnrand with `set_backend("nki")` on Tensor/GpSimd engine
- `_trnrand_pytorch` — trnrand with `set_backend("pytorch")` (host CPU fallback)
- `_torch`           — vanilla `torch.empty(...).uniform_()` etc. (host CPU)

Comparison 1 (nki vs trnrand_pytorch): does the NKI kernel beat our own
PyTorch fallback on the same code path?
Comparison 2 (nki vs torch): user-visible difference vs vanilla PyTorch.

`_nki` tests skip when `HAS_NKI` is False (no neuronxcc).
"""

from __future__ import annotations

import pytest
import torch

import trnrand
from trnrand.nki import HAS_NKI


@pytest.fixture
def gen():
    return trnrand.Generator(seed=0)


def _with_nki():
    trnrand.set_backend("nki")


def _with_pytorch():
    trnrand.set_backend("pytorch")


nki_only = pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")


# ── Uniform ───────────────────────────────────────────────────────────────────


class TestUniform:
    @nki_only
    def test_uniform_nki(self, benchmark, square_size, gen):
        _with_nki()
        n = square_size
        benchmark(lambda: trnrand.uniform(n, n, generator=gen))

    def test_uniform_trnrand_pytorch(self, benchmark, square_size, gen):
        _with_pytorch()
        n = square_size
        benchmark(lambda: trnrand.uniform(n, n, generator=gen))

    def test_uniform_torch(self, benchmark, square_size):
        n = square_size
        g = torch.Generator().manual_seed(0)
        benchmark(lambda: torch.empty(n, n).uniform_(0, 1, generator=g))


# ── Normal ────────────────────────────────────────────────────────────────────


class TestNormal:
    @nki_only
    def test_normal_nki(self, benchmark, square_size, gen):
        _with_nki()
        n = square_size
        benchmark(lambda: trnrand.normal(n, n, generator=gen))

    def test_normal_trnrand_pytorch(self, benchmark, square_size, gen):
        _with_pytorch()
        n = square_size
        benchmark(lambda: trnrand.normal(n, n, generator=gen))

    def test_normal_torch(self, benchmark, square_size):
        n = square_size
        g = torch.Generator().manual_seed(0)
        benchmark(lambda: torch.empty(n, n).normal_(0.0, 1.0, generator=g))


# ── Truncated normal ──────────────────────────────────────────────────────────


class TestTruncatedNormal:
    def test_truncated_normal_trnrand_pytorch(self, benchmark, square_size, gen):
        _with_pytorch()
        n = square_size
        benchmark(lambda: trnrand.truncated_normal(n, n, generator=gen))


# ── Sobol ─────────────────────────────────────────────────────────────────────


class TestSobol:
    def test_sobol_trnrand_pytorch(self, benchmark, square_size):
        _with_pytorch()
        n = square_size
        benchmark(lambda: trnrand.sobol(n, n_dims=8, seed=0))

    def test_sobol_torch(self, benchmark, square_size):
        n = square_size
        benchmark(
            lambda: torch.quasirandom.SobolEngine(dimension=8, scramble=True, seed=0).draw(n)
        )


# ── Latin Hypercube ───────────────────────────────────────────────────────────


class TestLatinHypercube:
    def test_latin_hypercube_trnrand_pytorch(self, benchmark, square_size):
        _with_pytorch()
        n = square_size
        benchmark(lambda: trnrand.latin_hypercube(n, n_dims=8))
