# trnrand

[![CI](https://github.com/trnsci/trnrand/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnrand/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trnrand)](https://pypi.org/project/trnrand/)
[![Python](https://img.shields.io/pypi/pyversions/trnrand)](https://pypi.org/project/trnrand/)
[![License](https://img.shields.io/github/license/trnsci/trnrand)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/trnrand/)
[![codecov](https://codecov.io/gh/trnsci/trnrand/graph/badge.svg)](https://codecov.io/gh/trnsci/trnrand)

Random number generation for AWS Trainium via NKI.

Seeded pseudo-random distributions, quasi-random sequences for quasi-Monte Carlo, and on-device Philox RNG targeting the GpSimd engine.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trnrand follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trnrand/issues/18)** (active): Philox 4×32 and Box-Muller NKI kernels scaffolded with CPU-reference oracles; awaiting hardware validation on trn1 / trn2.
- **[Phase 3 — perf](https://github.com/trnsci/trnrand/issues/19)**: batched-tile RNG streaming, NEFF cache reuse.
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnrand/issues/20)**: stream-partitioned RNG across NeuronCores.
- **[Phase 5 — generation](https://github.com/trnsci/trnrand/issues/21)**: trn2 PSUM sizing for larger batched tile generation.

_(No Phase 2 for trnrand — the precision story is inherited where relevant.)_

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Install

```bash
pip install trnrand

# With Neuron hardware support
pip install trnrand[neuron]
```

## Usage

```python
import trnrand

# Seeded, reproducible generation
g = trnrand.Generator(seed=42)

# Standard distributions
x = trnrand.normal(1000, mean=0.0, std=1.0, generator=g)
u = trnrand.uniform(1000, low=-1.0, high=1.0, generator=g)
e = trnrand.exponential(1000, rate=2.0, generator=g)

# Quasi-random sequences (better convergence for MC integration)
sobol_pts = trnrand.sobol(1024, n_dims=5, seed=42)
halton_pts = trnrand.halton(1024, n_dims=3)
lhs_pts = trnrand.latin_hypercube(100, n_dims=4)

# Module-level seeding
trnrand.manual_seed(42)
x = trnrand.standard_normal(256)
```

## Operations

| Category | Function | Description |
|----------|----------|-------------|
| Distributions | `uniform` | U[low, high) |
| | `normal` | N(μ, σ²) |
| | `standard_normal` | N(0, 1) |
| | `exponential` | Exp(λ) |
| | `bernoulli` | Bernoulli(p) |
| | `randint` | Uniform integers [low, high) |
| | `randperm` | Random permutation |
| | `truncated_normal` | Bounded normal (rejection sampling) |
| Quasi-random | `sobol` | Sobol sequence (scrambled) |
| | `halton` | Halton sequence |
| | `latin_hypercube` | Latin Hypercube Sampling |

## MC vs QMC Example

```bash
python examples/mc_integration.py
```

Compares pseudo-random vs Sobol quasi-random for estimating the volume of a 5-D hypersphere. QMC converges O(1/N) vs O(1/√N).

## Status

- [x] Seeded Generator with state management
- [x] Standard distributions (uniform, normal, exponential, Bernoulli, etc.)
- [x] Sobol, Halton, Latin Hypercube sequences
- [x] MC vs QMC integration example
- [ ] NKI Philox kernel on GpSimd
- [ ] On-device Box-Muller (uniform → normal)
- [ ] Benchmarks vs cuRAND

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/trnsci/trnfft) | FFT + complex ops |
| [trnblas](https://github.com/trnsci/trnblas) | BLAS operations |
| [trnsolver](https://github.com/trnsci/trnsolver) | Linear solvers |

## License

Apache 2.0 — Copyright 2026 Scott Friedman
