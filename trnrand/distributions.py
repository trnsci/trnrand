"""
Random number distributions for Trainium.

Standard distributions with reproducible seeding via Generator.
All functions accept an optional Generator for deterministic streams.

For scientific computing:

- uniform / normal / standard_normal — Monte Carlo sampling, weight init
- exponential — Poisson process simulation, radioactive decay
- bernoulli — coin-flip sampling, dropout masks
- randint / randperm — integer sampling, permutations
- truncated_normal — bounded weight initialization
- gamma — Bayesian priors, waiting-time distributions
- chi_squared — variance / goodness-of-fit tests
- beta — A/B testing, Bayesian conjugate priors
- poisson — event-count / queuing simulation
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Tuple, Union

from .generator import Generator, get_default_generator


def uniform(
    *size: int,
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Uniform distribution on [low, high)."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.empty(*size, dtype=dtype).uniform_(low, high, generator=gen)


def normal(
    *size: int,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Normal (Gaussian) distribution."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.empty(*size, dtype=dtype).normal_(mean, std, generator=gen)


def standard_normal(
    *size: int,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Standard normal N(0, 1)."""
    return normal(*size, mean=0.0, std=1.0, dtype=dtype, generator=generator)


def exponential(
    *size: int,
    rate: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Exponential distribution with given rate (λ).

    PDF: f(x) = λ exp(-λx)
    Mean: 1/λ
    """
    gen = (generator or get_default_generator()).torch_generator
    u = torch.empty(*size, dtype=dtype).uniform_(0, 1, generator=gen)
    # Clamp to avoid log(0)
    u = torch.clamp(u, min=1e-10)
    return -torch.log(u) / rate


def bernoulli(
    *size: int,
    p: float = 0.5,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Bernoulli distribution: 1 with probability p, 0 otherwise."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.empty(*size, dtype=dtype).bernoulli_(p, generator=gen)


def randint(
    *size: int,
    low: int = 0,
    high: int = 1,
    dtype: torch.dtype = torch.long,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Random integers from [low, high)."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.randint(low, high, size, dtype=dtype, generator=gen)


def randperm(
    n: int,
    dtype: torch.dtype = torch.long,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Random permutation of 0..n-1."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.randperm(n, dtype=dtype, generator=gen)


def gamma(
    *size: int,
    shape: float,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Gamma distribution with given shape (k) and scale (θ).

    PDF: f(x) = x^(k-1) exp(-x/θ) / (Γ(k) θ^k)
    Mean: k·θ; Variance: k·θ²

    Uses Marsaglia-Tsang rejection for shape ≥ 1. For shape < 1, samples
    Gamma(shape+1) and multiplies by U^(1/shape) (the "boost" identity).
    """
    assert shape > 0 and scale > 0, "gamma requires shape > 0 and scale > 0"
    gen = (generator or get_default_generator()).torch_generator

    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    n = 1
    for d in size:
        n *= d

    if shape < 1.0:
        boost = torch.empty(n, dtype=torch.float64).uniform_(0.0, 1.0, generator=gen)
        boost = boost ** (1.0 / shape)
        shape_eff = shape + 1.0
    else:
        boost = None
        shape_eff = shape

    d = shape_eff - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    result = torch.empty(n, dtype=torch.float64)
    idx = 0
    while idx < n:
        remaining = n - idx
        # Oversample ~30% so most iterations finish in one pass.
        draw = max(32, int(remaining * 1.3) + 32)
        z = torch.empty(draw, dtype=torch.float64).normal_(0.0, 1.0, generator=gen)
        u = torch.empty(draw, dtype=torch.float64).uniform_(0.0, 1.0, generator=gen)
        v = (1.0 + c * z) ** 3
        # v > 0 guaranteed when z > -1/c; log needs strictly-positive v.
        valid = z > -1.0 / c
        # log(v) safe where valid; clamp on the masked-out entries to avoid nan.
        v_safe = torch.where(valid, v, torch.ones_like(v))
        accept = (
            valid
            & (u.log() < 0.5 * z * z + d - d * v + d * v_safe.log())
        )
        draws = (d * v)[accept]
        take = min(draws.numel(), remaining)
        result[idx:idx + take] = draws[:take]
        idx += take

    if boost is not None:
        result = result * boost
    return (result * scale).to(dtype).reshape(size)


def chi_squared(
    *size: int,
    df: float,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Chi-squared distribution with `df` degrees of freedom.

    Equivalent to Gamma(df/2, scale=2). Mean: df; Variance: 2·df.
    """
    assert df > 0, "chi_squared requires df > 0"
    return gamma(*size, shape=df / 2.0, scale=2.0, dtype=dtype, generator=generator)


def beta(
    *size: int,
    alpha: float,
    beta: float,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Beta distribution on (0, 1) with shape parameters α, β > 0.

    Sampled via the gamma-ratio identity: X ~ Gamma(α, 1), Y ~ Gamma(β, 1),
    then Z = X / (X + Y) ~ Beta(α, β).

    Mean: α/(α+β); Variance: αβ/((α+β)²(α+β+1)).
    """
    assert alpha > 0 and beta > 0, "beta requires alpha > 0 and beta > 0"
    x = gamma(*size, shape=alpha, scale=1.0, dtype=torch.float64, generator=generator)
    y = gamma(*size, shape=beta, scale=1.0, dtype=torch.float64, generator=generator)
    # Both x, y > 0 with probability 1; the sum never underflows for reasonable shapes.
    return (x / (x + y)).to(dtype)


def poisson(
    *size: int,
    lam: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Poisson distribution with rate λ.

    PMF: P(k) = λ^k exp(-λ) / k!
    Mean: λ; Variance: λ
    """
    assert lam >= 0, "poisson requires lam >= 0"
    gen = (generator or get_default_generator()).torch_generator
    rates = torch.full(size, float(lam), dtype=torch.float32)
    return torch.poisson(rates, generator=gen).to(dtype)


def truncated_normal(
    *size: int,
    mean: float = 0.0,
    std: float = 1.0,
    low: float = -2.0,
    high: float = 2.0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[Generator] = None,
) -> torch.Tensor:
    """Truncated normal distribution clipped to [low, high] (in std units).

    Uses rejection sampling. Useful for bounded weight initialization.
    """
    result = torch.empty(*size, dtype=dtype)
    gen = (generator or get_default_generator()).torch_generator

    # Rejection sampling
    remaining = result.reshape(-1)
    n_remaining = remaining.numel()
    idx = 0
    while idx < n_remaining:
        batch_size = min(n_remaining - idx, n_remaining * 2)
        samples = torch.empty(batch_size, dtype=dtype).normal_(mean, std, generator=gen)
        valid = samples[(samples >= mean + low * std) & (samples <= mean + high * std)]
        take = min(len(valid), n_remaining - idx)
        remaining[idx:idx + take] = valid[:take]
        idx += take

    return result
