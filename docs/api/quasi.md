# Quasi-random sequences

Low-discrepancy sequences for quasi-Monte Carlo integration. Convergence
is `O(1/N)` instead of pseudo-random's `O(1/√N)`, which is decisive for
moderate-dimensional integrals.

## `sobol(n_points, n_dims, seed=None, scramble=True, dtype=torch.float32)`

Sobol sequence. Returns a `(n_points, n_dims)` tensor in `[0, 1)`.

`scramble` selects the randomisation:

- `True` (default, also `"owen"`): Owen (nested uniform) scrambling, hash
  based (Laine–Karras, with the constants of Burley 2020). It keeps the
  (t, m, s)-net property of the plain sequence: for `n = 2^m`, every
  elementary interval holds the same number of points as before scrambling.
  Its centred discrepancy matches `scipy.stats.qmc.Sobol(scramble=True)`.
  Each integer `seed` (including 0) gives a fixed scramble; `seed=None`
  draws one from torch's global RNG.
- `False`: the plain sequence; `seed` is ignored.
- `"shift"`: the scramble of releases up to 0.13, kept to reproduce old
  streams. PyTorch path: `SobolEngine(scramble=True, seed=seed)`. NKI path:
  a per-dimension random shift mod 1 (Cranley–Patterson), which breaks the
  net property, and no scrambling at all for `seed=None` or `0`.

Paths:

- PyTorch: `torch.quasirandom.SobolEngine` for the plain sequence (up to
  21201 dimensions), with the Owen hash on 32-bit integers. float32 output is
  the top 24 bits; float64 output keeps 32.
- NKI: the GpSimd Gray-code kernel, up to 10 dimensions and 2^24 points.
  With `scramble=True` the `sobol_owen` kernel applies the same hash on 24
  bits, so its points equal the PyTorch path's float32 points for the same
  seed. Use `set_backend("pytorch")` for more dimensions or points.

## `halton(n_points, n_dims, dtype=torch.float32)`

Halton sequence using the first `n_dims` primes as bases. Simpler than
Sobol but degrades above ~20 dimensions — use Sobol for `d > 10`.
Returns a `(n_points, n_dims)` tensor in `(0, 1)`. The NKI kernel covers
up to 10 dimensions and indices below 2^22.

## `latin_hypercube(n_points, n_dims, dtype=torch.float32, generator=None)`

Latin Hypercube Sampling. Stratified — each row and column of the
hypercube contains exactly one sample. Better space-filling than pure
random for small sample sizes.
