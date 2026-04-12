"""RNG benchmarks.

Run with:

    pytest benchmarks/ --benchmark-only

Baseline is PyTorch / SobolEngine. When validated, these should be re-run
on trn1/trn2 with `set_backend("nki")` for the comparison vs cuRAND
write-up.
"""

import trnrand


def test_uniform(benchmark, square_size):
    n = square_size
    g = trnrand.Generator(seed=0)
    benchmark(lambda: trnrand.uniform(n, n, generator=g))


def test_normal(benchmark, square_size):
    n = square_size
    g = trnrand.Generator(seed=0)
    benchmark(lambda: trnrand.normal(n, n, generator=g))


def test_truncated_normal(benchmark, square_size):
    n = square_size
    g = trnrand.Generator(seed=0)
    benchmark(lambda: trnrand.truncated_normal(n, n, generator=g))


def test_sobol(benchmark, square_size):
    n = square_size
    benchmark(lambda: trnrand.sobol(n, n_dims=8, seed=0))


def test_halton(benchmark):
    benchmark(lambda: trnrand.halton(1024, n_dims=8))


def test_latin_hypercube(benchmark, square_size):
    n = square_size
    benchmark(lambda: trnrand.latin_hypercube(n, n_dims=8))
