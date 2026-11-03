"""
Quasi-random (low-discrepancy) sequences for Trainium.

Sobol and Halton sequences for quasi-Monte Carlo integration.
These provide better convergence than pseudo-random sampling for
high-dimensional integrals — O(1/N) vs O(1/√N).

Use cases:
- Numerical integration in quantum chemistry (electron integrals)
- Stochastic trace estimation (Hutchinson with QMC)
- Hyperparameter search (Sobol beats random for small budgets)
- Importance sampling with low-discrepancy proposals
"""

from __future__ import annotations

import math
import warnings

import torch

from .nki import HAS_NKI, get_backend


def _nki_active() -> bool:
    """True when the NKI path should be used for this call."""
    backend = get_backend()
    if backend == "pytorch":
        return False
    return HAS_NKI


def _halton_fp64_to_torch(dtype: torch.dtype) -> bool:
    """The float64 policy (trnrand#105) for halton: a float64 request takes the
    float64 torch path under backend "auto", the float32 kernel (cast, warned
    once) under "nki", and raises under TRNRAND_REQUIRE_NKI.

    sobol has no such routing: the pytorch backend's float64 Sobol is a
    float32 SobolEngine draw cast to float64, so there is no float64-precision
    torch Sobol to route to; float64 Sobol keeps the kernel on every backend.
    """
    from .nki.dispatch import _float64_to_torch

    return _float64_to_torch("halton", dtype)


def _outside_kernel_limits(name: str, reason: str) -> None:
    """Fall back to the CPU path for a call the NKI kernel cannot take (#89, #104).

    Warns with `NkiFallbackWarning`, or raises `ValueError` when
    `TRNRAND_REQUIRE_NKI` is set (the caller asked for the kernel or nothing).
    """
    from .nki import dispatch

    msg = f"{name}: {reason}, outside the NKI kernel's limits"
    if dispatch._REQUIRE_NKI:
        raise ValueError(f"{msg} (TRNRAND_REQUIRE_NKI is set, so no CPU fallback)")
    warnings.warn(f"{msg}; using the CPU path.", dispatch.NkiFallbackWarning, stacklevel=3)


def _sobol_fits_kernel(n_points: int, n_dims: int) -> bool:
    """True if `sobol_nki` can take the call; warns / raises otherwise (#89)."""
    from .nki.dispatch import _SOBOL_BITS, _SOBOL_MAX_DIMS

    if n_points <= 0:
        return False  # n = 0: the CPU path returns (0, n_dims); n < 0: it raises
    if n_dims > _SOBOL_MAX_DIMS:
        _outside_kernel_limits("sobol", f"n_dims={n_dims} > {_SOBOL_MAX_DIMS}")
        return False
    if n_points > 1 << _SOBOL_BITS:
        _outside_kernel_limits("sobol", f"n_points={n_points} > 2^{_SOBOL_BITS}")
        return False
    return True


# halton_nki's limits. The constants live inside dispatch's `if HAS_NKI:` block,
# so they are restated here; test_halton_nki_limits checks they agree.
_HALTON_MAX_DIMS = 10
_HALTON_MAX_POINTS = 1 << 22  # halton_nki needs the last index, n_points, < 2^22


def _halton_fits_kernel(n_points: int, n_dims: int) -> bool:
    """True if `halton_nki` can take the call; warns / raises otherwise (#104)."""
    if n_points <= 0:
        return False  # n = 0: the CPU path returns (0, n_dims)
    if n_dims > _HALTON_MAX_DIMS:
        _outside_kernel_limits("halton", f"n_dims={n_dims} > {_HALTON_MAX_DIMS}")
        return False
    if n_points + 1 > _HALTON_MAX_POINTS:
        _outside_kernel_limits("halton", f"n_points={n_points} >= 2^22")
        return False
    return True


def sobol(
    n_points: int,
    n_dims: int,
    seed: int | None = None,
    scramble: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sobol quasi-random sequence.

    Args:
        n_points: Number of points to generate
        n_dims: Dimensionality
        seed: Random seed for scrambling
        scramble: Apply Owen scrambling for better equidistribution

    Returns:
        Tensor of shape (n_points, n_dims) with values in [0, 1)

    On the NKI path, calls outside the kernel's limits (``n_dims > 10`` or
    ``n_points > 2^24``) use the CPU path with an ``NkiFallbackWarning``, or
    raise ``ValueError`` under ``TRNRAND_REQUIRE_NKI=1``.
    """
    if _nki_active() and _sobol_fits_kernel(n_points, n_dims):
        try:
            from .nki.dispatch import sobol_nki

            # sobol_nki shifts whenever seed != 0, so scramble=False must pass 0 (#90).
            nki_seed = (seed or 0) if scramble else 0
            return _below_one(sobol_nki(n_points, n_dims, seed=nki_seed).to(dtype))
        except ImportError:
            pass  # sobol_nki not yet available; fall through to CPU path
    engine = torch.quasirandom.SobolEngine(
        dimension=n_dims,
        scramble=scramble,
        seed=seed,
    )
    if n_points == 0:
        return torch.empty(0, n_dims, dtype=dtype)  # SobolEngine.draw(0) raises
    return _below_one(engine.draw(n_points).to(dtype))


def halton(
    n_points: int,
    n_dims: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Halton quasi-random sequence.

    Uses the first n_dims primes as bases.
    Simpler than Sobol but degrades in high dimensions (>~20).

    Args:
        n_points: Number of points
        n_dims: Dimensionality (recommend ≤ 20)

    Returns:
        Tensor of shape (n_points, n_dims) with values in (0, 1)

    On the NKI path, calls outside the kernel's limits (``n_dims > 10`` or
    ``n_points >= 2^22``) use the CPU path with an ``NkiFallbackWarning``, or
    raise ``ValueError`` under ``TRNRAND_REQUIRE_NKI=1``.
    """
    if _nki_active() and not _halton_fp64_to_torch(dtype) and _halton_fits_kernel(n_points, n_dims):
        try:
            from .nki.dispatch import halton_nki

            return _below_one(halton_nki(n_points, n_dims).to(dtype), open_low=True)
        except ImportError:
            pass  # halton_nki not yet available; fall through to CPU path
    primes = _first_n_primes(n_dims)
    result = torch.zeros(n_points, n_dims, dtype=dtype)
    for d in range(n_dims):
        result[:, d] = _halton_sequence(n_points, primes[d], dtype)
    return _below_one(result, open_low=True)


_REDUCED = (torch.float16, torch.bfloat16)


def _below_one(x: torch.Tensor, open_low: bool = False) -> torch.Tensor:
    """Keep reduced-precision QMC points inside [0, 1) (or (0, 1)).

    float16 / bfloat16 rounding sends points within half an ulp of 1 to
    exactly 1.0 (#101). Those step back to the largest value below 1 in the
    dtype; with `open_low`, points that round to 0 step up to the smallest
    positive value. Every other point is unchanged, and float32 / float64
    output is returned as is (bit-identical).
    """
    if x.dtype not in _REDUCED:
        return x
    one = torch.ones((), dtype=x.dtype, device=x.device)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    x = torch.where(x >= one, torch.nextafter(one, zero), x)
    if open_low:
        x = torch.where(x <= zero, torch.nextafter(zero, one), x)
    return x


def _halton_sequence(n: int, base: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate Halton sequence for a single base."""
    result = torch.zeros(n, dtype=dtype)
    for i in range(n):
        f = 1.0
        r = 0.0
        idx = i + 1  # Skip 0
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result[i] = r
    return result


def _first_n_primes(n: int) -> list[int]:
    """Return first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


def latin_hypercube(
    n_points: int,
    n_dims: int,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Latin Hypercube Sampling.

    Stratified sampling that ensures each row and column of the
    hypercube contains exactly one sample. Better space-filling
    than pure random for small sample sizes.
    """
    result = torch.zeros(n_points, n_dims, dtype=dtype)
    for d in range(n_dims):
        perm = torch.randperm(n_points, generator=generator)
        u = torch.rand(n_points, generator=generator, dtype=dtype)
        col = (perm.to(dtype) + u) / n_points
        if dtype in (torch.float16, torch.bfloat16):
            col = _into_strata(col, perm, u, n_points)
        result[:, d] = col
    return result


def _into_strata(col: torch.Tensor, perm: torch.Tensor, u: torch.Tensor, n: int) -> torch.Tensor:
    """Move reduced-precision LHS points back into their stratum [perm/n, (perm+1)/n).

    float16 / bfloat16 rounding of `perm + u` and of the division can push a
    point into the next stratum or onto 1.0 (#96). Points already inside their
    stratum are unchanged; the others get the correctly rounded value, stepped
    one ulp back inside. Raises if the dtype has no value in some stratum.
    """
    lo = perm.double() / n
    hi = (perm.double() + 1) / n
    bad = (col.double() < lo) | (col.double() >= hi)
    if not bool(bad.any()):
        return col
    v = ((perm.double() + u.double()) / n).to(col.dtype)
    down = torch.nextafter(v, torch.full_like(v, -math.inf))
    up = torch.nextafter(v, torch.full_like(v, math.inf))
    v = torch.where(v.double() >= hi, down, v)
    v = torch.where(v.double() < lo, up, v)
    if bool(((v.double() < lo) | (v.double() >= hi))[bad].any()):
        raise ValueError(
            f"latin_hypercube: {col.dtype} cannot represent a point in each of "
            f"{n} strata; use float32 or fewer points"
        )
    return torch.where(bad, v, col)
