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

## `gamma(*size, shape, scale=1.0, dtype=torch.float32, generator=None)`

Gamma distribution with shape `k` and scale `θ`. Mean `k·θ`, variance
`k·θ²`. Samples via Marsaglia-Tsang rejection for `shape ≥ 1`; falls back
to the boost identity (`Gamma(k) = Gamma(k+1) · U^(1/k)`) for `shape < 1`.

## `chi_squared(*size, df, dtype=torch.float32, generator=None)`

Chi-squared distribution with `df` degrees of freedom. Mean `df`, variance
`2·df`. Equivalent to `Gamma(df/2, scale=2)`.

## `beta(*size, alpha, beta, dtype=torch.float32, generator=None)`

Beta distribution on `(0, 1)` with shape parameters `α, β > 0`. Mean
`α/(α+β)`, variance `αβ / ((α+β)²(α+β+1))`. Sampled via the gamma-ratio
identity `X / (X + Y)` where `X ~ Gamma(α, 1)`, `Y ~ Gamma(β, 1)`.

## `poisson(*size, lam=1.0, dtype=torch.float32, generator=None)`

Poisson distribution with rate `λ ≥ 0`. Mean and variance both `λ`. Wraps
`torch.poisson` with a generator-bound rates tensor for reproducibility.
