"""
Random number distributions for Trainium.

Standard distributions with reproducible seeding via Generator.
All functions accept an optional Generator for deterministic streams.

For scientific computing:
- uniform/normal: Monte Carlo sampling, weight initialization
- exponential: Poisson process simulation, radioactive decay
- multinomial: importance sampling
- truncated_normal: bounded parameter initialization
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
