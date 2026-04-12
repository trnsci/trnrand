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
from typing import Optional


def sobol(
    n_points: int,
    n_dims: int,
    seed: Optional[int] = None,
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
    generator: Optional[torch.Generator] = None,
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
        result[:, d] = (perm.to(dtype) + u) / n_points
    return result
