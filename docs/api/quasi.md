# Quasi-random sequences

Low-discrepancy sequences for quasi-Monte Carlo integration. Convergence
is `O(1/N)` instead of pseudo-random's `O(1/√N)`, which is decisive for
moderate-dimensional integrals.

## `sobol(n_points, n_dims, seed=None, scramble=True, dtype=torch.float32)`

Sobol sequence. Returns a `(n_points, n_dims)` tensor in `[0, 1)`.

- PyTorch path: `torch.quasirandom.SobolEngine`, with Owen scrambling on by
  default for better equidistribution.
- NKI path: the GpSimd Gray-code kernel, up to 10 dimensions and 2^24
  points. A nonzero `seed` applies a per-dimension random shift (not Owen
  scrambling); `seed=None` or `0` gives the unscrambled sequence, and
  `scramble` is not consulted. Use `set_backend("pytorch")` for more
  dimensions or Owen scrambling.

## `halton(n_points, n_dims, dtype=torch.float32)`

Halton sequence using the first `n_dims` primes as bases. Simpler than
Sobol but degrades above ~20 dimensions — use Sobol for `d > 10`.
Returns a `(n_points, n_dims)` tensor in `(0, 1)`. The NKI kernel covers
up to 10 dimensions and indices below 2^22.

## `latin_hypercube(n_points, n_dims, dtype=torch.float32, generator=None)`

Latin Hypercube Sampling. Stratified — each row and column of the
hypercube contains exactly one sample. Better space-filling than pure
random for small sample sizes.
