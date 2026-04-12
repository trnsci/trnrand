# Distributions

All distribution functions accept a positional `*size` argument and an
optional `generator=` for deterministic streams. When `generator` is
omitted, the module-level default generator is used.

## `uniform(*size, low=0.0, high=1.0, dtype=torch.float32, generator=None)`

Uniform distribution on `[low, high)`.

## `normal(*size, mean=0.0, std=1.0, dtype=torch.float32, generator=None)`

Normal (Gaussian) distribution `N(μ, σ²)`.

## `standard_normal(*size, dtype=torch.float32, generator=None)`

Shorthand for `normal(*size, mean=0.0, std=1.0, ...)`.

## `exponential(*size, rate=1.0, dtype=torch.float32, generator=None)`

Exponential distribution with rate `λ` (mean `1/λ`). Samples via
`-log(U) / λ` with clamping to avoid `log(0)`.

## `bernoulli(*size, p=0.5, dtype=torch.float32, generator=None)`

Bernoulli distribution: 1 with probability `p`, 0 otherwise.

## `randint(*size, low=0, high=1, dtype=torch.long, generator=None)`

Random integers from `[low, high)`.

## `randperm(n, dtype=torch.long, generator=None)`

Random permutation of `0..n-1`.

## `truncated_normal(*size, mean=0.0, std=1.0, low=-2.0, high=2.0, dtype=torch.float32, generator=None)`

Truncated normal clipped to `[mean + low·std, mean + high·std]` via
rejection sampling. Useful for bounded weight initialization.
