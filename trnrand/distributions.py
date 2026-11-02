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

from .generator import Generator, get_default_generator
from .nki import HAS_NKI, get_backend


def _nki_active() -> bool:
    """True when the NKI path should be used for this call."""
    backend = get_backend()
    if backend == "pytorch":
        return False
    return HAS_NKI


def _nki_seed(gen_obj: Generator) -> int:
    """Advance generator state and return a 24-bit seed for NKI dispatch.

    Drawing one random int from the torch generator advances its internal
    state so successive NKI calls with the same generator produce independent
    streams — the same guarantee the PyTorch path provides.
    """
    return int(torch.randint(0, 2**24, (1,), generator=gen_obj.torch_generator).item())


def _compiling() -> bool:
    """True while Dynamo traces and the custom-op route of #61 is available."""
    if not torch.compiler.is_compiling():
        return False
    from . import _compile

    return _compile.AVAILABLE


def _nki_draw_eager(kind: str, gen: Generator, n: int) -> torch.Tensor:
    """Eager NKI draw: seed + counter step, kernel launches, counter advance."""
    from .nki.dispatch import threefry_normal_nki, threefry_uniform_nki

    fn = threefry_uniform_nki if kind == "uniform" else threefry_normal_nki
    # key_mode="counter": fixed key from the seed, only the counter moves (#61).
    seed = gen._key48 if gen._key_mode == "counter" else _nki_seed(gen)
    out = fn(n, seed=seed, counter_offset=gen._chip_counter_offset(n))
    gen._advance_by_elements(n)
    return out


def _nki_draw(kind: str, gen: Generator, n: int) -> torch.Tensor:
    if _compiling():
        from . import _compile

        return _compile.nki_draw(gen, kind, n)
    return _nki_draw_eager(kind, gen, n)


def _affine(out: torch.Tensor, scale: float, shift: float, nonneg: bool = False) -> torch.Tensor:
    """out * scale + shift. Eager NKI draws are fresh host tensors, so this runs
    in place: the same two float32 ops, without allocating two more n-element
    tensors (out-of-place, this was ~9 ms of a 4M `normal()` on Trn2, #60).

    `nonneg`: `out` holds no -0.0 and no NaN (the uniform kernels emit k · 2^-24,
    k >= 0). Then with scale > 0 the product holds no -0.0 either, and
    `x + 0.0 == x` bit for bit, so the add is skipped (~11 ms of a 64M
    `uniform()` on Trn2).
    """
    if _compiling():
        return out * scale + shift
    if scale != 1.0:  # x * 1.0 == x exactly
        out.mul_(scale)
    if nonneg and scale > 0.0 and shift == 0.0:
        return out
    return out.add_(shift)


def _torch_fill(kind, size, a, b, dtype, generator):
    gen = generator or get_default_generator()
    if _compiling():
        from . import _compile

        return _compile.fill(gen, size, kind, a, b, dtype)
    out = torch.empty(*size, dtype=dtype)
    if kind == "uniform":
        return out.uniform_(a, b, generator=gen.torch_generator)
    return out.normal_(a, b, generator=gen.torch_generator)


def uniform(
    *size: int,
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Uniform distribution on [low, high)."""
    if _nki_active():
        gen = generator or get_default_generator()
        out = _nki_draw("uniform", gen, math.prod(size))
        return _affine(out, high - low, low, nonneg=True).to(dtype).reshape(size)
    return _torch_fill("uniform", size, low, high, dtype, generator)


def normal(
    *size: int,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Normal (Gaussian) distribution."""
    if _nki_active():
        gen = generator or get_default_generator()
        out = _nki_draw("normal", gen, math.prod(size))
        return _affine(out, std, mean).to(dtype).reshape(size)
    return _torch_fill("normal", size, mean, std, dtype, generator)


def standard_normal(
    *size: int,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Standard normal N(0, 1)."""
    return normal(*size, mean=0.0, std=1.0, dtype=dtype, generator=generator)


def exponential(
    *size: int,
    rate: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Exponential distribution with given rate (λ).

    PDF: f(x) = λ exp(-λx)
    Mean: 1/λ
    """
    if _nki_active():
        from .nki.dispatch import threefry_uniform_nki

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = math.prod(size)
        u = threefry_uniform_nki(n, seed=seed, counter_offset=gen._chip_counter_offset(n))
        gen._advance_by_elements(n)
        u = torch.clamp(u, min=1e-10)
        return (-torch.log(u) / rate).to(dtype).reshape(size)
    gen = (generator or get_default_generator()).torch_generator
    # Draw the uniform in at least float32: a float16 / bfloat16 uniform has only
    # ~11 / ~8 random bits, so u == 0 is common, and 1e-10 underflows to 0 in
    # float16 (-log(0) = inf). float32 / float64 are unchanged (#80).
    work = torch.promote_types(dtype, torch.float32)
    u = torch.empty(*size, dtype=work).uniform_(0, 1, generator=gen)
    # Clamp to avoid log(0)
    u = torch.clamp(u, min=1e-10)
    return (-torch.log(u) / rate).to(dtype)


def bernoulli(
    *size: int,
    p: float = 0.5,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Bernoulli distribution: 1 with probability p, 0 otherwise."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.empty(*size, dtype=dtype).bernoulli_(p, generator=gen)


def randint(
    *size: int,
    low: int = 0,
    high: int = 1,
    dtype: torch.dtype = torch.long,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Random integers from [low, high)."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.randint(low, high, size, dtype=dtype, generator=gen)


def randperm(
    n: int,
    dtype: torch.dtype = torch.long,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Random permutation of 0..n-1."""
    gen = (generator or get_default_generator()).torch_generator
    return torch.randperm(n, dtype=dtype, generator=gen)


def gamma(
    *size: int,
    shape: float,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Gamma distribution with given shape (k) and scale (θ).

    PDF: f(x) = x^(k-1) exp(-x/θ) / (Γ(k) θ^k)
    Mean: k·θ; Variance: k·θ²

    Uses Marsaglia-Tsang rejection for shape ≥ 1. For shape < 1, samples
    Gamma(shape+1) and multiplies by U^(1/shape) (the "boost" identity).
    """
    assert shape > 0 and scale > 0, "gamma requires shape > 0 and scale > 0"
    if _nki_active():
        from .nki.dispatch import gamma_nki

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = math.prod(size)
        out = gamma_nki(n, shape=shape, scale=scale, seed=seed)
        return out.to(dtype).reshape(size)
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
        accept = valid & (u.log() < 0.5 * z * z + d - d * v + d * v_safe.log())
        draws = (d * v)[accept]
        take = min(draws.numel(), remaining)
        result[idx : idx + take] = draws[:take]
        idx += take

    if boost is not None:
        result = result * boost
    return (result * scale).to(dtype).reshape(size)


def chi_squared(
    *size: int,
    df: float,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Chi-squared distribution with `df` degrees of freedom.

    Equivalent to Gamma(df/2, scale=2). Mean: df; Variance: 2·df.
    """
    assert df > 0, "chi_squared requires df > 0"
    if _nki_active():
        from .nki.dispatch import chi_squared_nki

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = math.prod(size)
        out = chi_squared_nki(n, df=df, seed=seed)
        return out.to(dtype).reshape(size)
    return gamma(*size, shape=df / 2.0, scale=2.0, dtype=dtype, generator=generator)


def beta(
    *size: int,
    alpha: float,
    beta: float,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Beta distribution on (0, 1) with shape parameters α, β > 0.

    Sampled via the gamma-ratio identity: X ~ Gamma(α, 1), Y ~ Gamma(β, 1),
    then Z = X / (X + Y) ~ Beta(α, β).

    Mean: α/(α+β); Variance: αβ/((α+β)²(α+β+1)).
    """
    assert alpha > 0 and beta > 0, "beta requires alpha > 0 and beta > 0"
    if _nki_active():
        from .nki.dispatch import beta_nki

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = math.prod(size)
        out = beta_nki(n, alpha=alpha, beta_param=beta, seed=seed)
        return out.to(dtype).reshape(size)
    x = gamma(*size, shape=alpha, scale=1.0, dtype=torch.float64, generator=generator)
    y = gamma(*size, shape=beta, scale=1.0, dtype=torch.float64, generator=generator)
    # Both x, y > 0 with probability 1; the sum never underflows for reasonable shapes.
    return (x / (x + y)).to(dtype)


def poisson(
    *size: int,
    lam: float = 1.0,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Poisson distribution with rate λ.

    PMF: P(k) = λ^k exp(-λ) / k!
    Mean: λ; Variance: λ

    NKI path (lam ≥ 20): normal approximation round(N(λ, √λ)) clamped to ≥ 0.
    CPU path (lam < 20 or NKI inactive): torch.poisson (exact, Knuth algorithm).
    """
    assert lam >= 0, "poisson requires lam >= 0"
    if _nki_active():
        from .nki.dispatch import _POISSON_NORMAL_THRESHOLD, poisson_nki

        if lam >= _POISSON_NORMAL_THRESHOLD:
            gen = generator or get_default_generator()
            seed = _nki_seed(gen)
            n = math.prod(size)
            out = poisson_nki(n, lam=lam, seed=seed, counter_offset=gen._chip_counter_offset(n))
            gen._advance_by_elements(n)
            return out.to(dtype).reshape(size)
    gen = (generator or get_default_generator()).torch_generator
    rates = torch.full(size, float(lam), dtype=torch.float32)
    return torch.poisson(rates, generator=gen).to(dtype)


def normal_into(
    buf: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: Generator | None = None,
) -> None:
    """Write N(mean, std) samples into pre-allocated buf in-place.

    Zero-allocation variant of `normal`. On the NKI path, uses the streaming
    Threefry+Box-Muller kernel (16,384 samples per XLA call). The NEFF is
    cached by the XLA runtime for repeated calls with identical buffer sizes.
    On PyTorch fallback, calls buf.normal_() directly.

    Args:
        buf:       Pre-allocated float32 tensor; filled in-place.
        mean:      Distribution mean (default 0.0).
        std:       Distribution standard deviation (default 1.0).
        generator: Optional Generator for reproducible seeding.
    """
    if _nki_active():
        from .nki.dispatch import _PROGRAM_TILES, threefry_stream_normal

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = buf.numel()
        n_launches = math.ceil(n / (_PROGRAM_TILES * 128 * 4))
        streaming_batches = n_launches * _PROGRAM_TILES
        counter_offset = gen._partition_rank * streaming_batches + gen._counter
        raw = threefry_stream_normal(n, seed=seed, counter_offset=counter_offset)
        gen._counter += streaming_batches * gen._partition_size  # every chip's slice
        if mean != 0.0 or std != 1.0:
            raw = raw * std + mean
        buf.copy_(raw.reshape(buf.shape))
        return
    gen_t = (generator or get_default_generator()).torch_generator
    buf.normal_(mean, std, generator=gen_t)


def uniform_into(
    buf: torch.Tensor,
    low: float = 0.0,
    high: float = 1.0,
    generator: Generator | None = None,
) -> None:
    """Write U(low, high) samples into pre-allocated buf in-place.

    Zero-allocation variant of `uniform`. On the NKI path, uses the streaming
    Threefry kernel; NEFF is cached by the XLA runtime for same-size buffers.
    On PyTorch fallback, calls buf.uniform_() directly.

    Args:
        buf:       Pre-allocated float32 tensor; filled in-place.
        low:       Lower bound (inclusive, default 0.0).
        high:      Upper bound (exclusive, default 1.0).
        generator: Optional Generator for reproducible seeding.
    """
    if _nki_active():
        from .nki.dispatch import _PROGRAM_TILES, threefry_stream_uniform

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = buf.numel()
        n_launches = math.ceil(n / (_PROGRAM_TILES * 128 * 4))
        streaming_batches = n_launches * _PROGRAM_TILES
        counter_offset = gen._partition_rank * streaming_batches + gen._counter
        raw = threefry_stream_uniform(n, seed=seed, counter_offset=counter_offset)
        gen._counter += streaming_batches * gen._partition_size  # every chip's slice
        if low != 0.0 or high != 1.0:
            raw = raw * (high - low) + low
        buf.copy_(raw.reshape(buf.shape))
        return
    gen_t = (generator or get_default_generator()).torch_generator
    buf.uniform_(low, high, generator=gen_t)


def exponential_into(
    buf: torch.Tensor,
    rate: float = 1.0,
    generator: Generator | None = None,
) -> None:
    """Write Exp(rate) samples into pre-allocated buf in-place.

    Zero-allocation variant of `exponential`. On the NKI path, uses the
    streaming Threefry uniform kernel with inverse-CDF transform; NEFF is
    cached for same-size buffers. On PyTorch fallback, calls
    buf.exponential_() directly.

    Args:
        buf:       Pre-allocated float32 tensor; filled in-place.
        rate:      Rate parameter λ (default 1.0). Mean = 1/λ.
        generator: Optional Generator for reproducible seeding.
    """
    if _nki_active():
        from .nki.dispatch import _PROGRAM_TILES, threefry_stream_uniform

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = buf.numel()
        n_launches = math.ceil(n / (_PROGRAM_TILES * 128 * 4))
        streaming_batches = n_launches * _PROGRAM_TILES
        counter_offset = gen._partition_rank * streaming_batches + gen._counter
        u = threefry_stream_uniform(n, seed=seed, counter_offset=counter_offset)
        gen._counter += streaming_batches * gen._partition_size  # every chip's slice
        # The kernel emits k * 2^-24, k >= 0: clamp as exponential() does, or
        # u == 0 (p = 2^-24 per sample) gives +inf (#82).
        raw = -torch.log(torch.clamp(u, min=1e-10)) / rate
        buf.copy_(raw.reshape(buf.shape))
        return
    gen_t = (generator or get_default_generator()).torch_generator
    buf.exponential_(lambd=rate, generator=gen_t)


def truncated_normal(
    *size: int,
    mean: float = 0.0,
    std: float = 1.0,
    low: float = -2.0,
    high: float = 2.0,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> torch.Tensor:
    """Truncated normal distribution clipped to [low, high] (in std units).

    Uses rejection sampling. Useful for bounded weight initialization.
    """
    if _nki_active():
        from .nki.dispatch import truncated_normal_nki

        gen = generator or get_default_generator()
        seed = _nki_seed(gen)
        n = math.prod(size)
        out = truncated_normal_nki(n, low=low, high=high, mean=mean, std=std, seed=seed)
        return out.to(dtype).reshape(size)
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
        remaining[idx : idx + take] = valid[:take]
        idx += take

    return result
