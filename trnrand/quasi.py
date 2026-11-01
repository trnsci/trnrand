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

import torch

from .nki import HAS_NKI, get_backend


def _nki_active() -> bool:
    """True when the NKI path should be used for this call."""
    backend = get_backend()
    if backend == "pytorch":
        return False
    return HAS_NKI


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
    """
    if _nki_active():
        try:
            from .nki.dispatch import sobol_nki

            return sobol_nki(n_points, n_dims, seed=seed or 0).to(dtype)
        except ImportError:
            pass  # sobol_nki not yet available; fall through to CPU path
    engine = torch.quasirandom.SobolEngine(
        dimension=n_dims,
        scramble=scramble,
        seed=seed,
    )
    return engine.draw(n_points).to(dtype)


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
    """
    if _nki_active():
        try:
            from .nki.dispatch import halton_nki

            return halton_nki(n_points, n_dims).to(dtype)
        except ImportError:
            pass  # halton_nki not yet available; fall through to CPU path
    primes = _first_n_primes(n_dims)
    result = torch.zeros(n_points, n_dims, dtype=dtype)
    for d in range(n_dims):
        result[:, d] = _halton_sequence(n_points, primes[d], dtype)
    return result


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
