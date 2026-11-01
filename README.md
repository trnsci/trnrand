# trnrand

[![CI](https://github.com/trnsci/trnrand/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnrand/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/trnsci/trnrand/graph/badge.svg)](https://codecov.io/gh/trnsci/trnrand)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/trnrand)](https://pypi.org/project/trnrand/)
[![Python](https://img.shields.io/pypi/pyversions/trnrand)](https://pypi.org/project/trnrand/)
[![License](https://img.shields.io/github/license/trnsci/trnrand)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/trnrand/)

Random number generation for AWS Trainium via NKI.

Seeded pseudo-random distributions, quasi-random sequences for quasi-Monte Carlo, and on-device counter-based RNG (Threefry4x32-20, with a Philox 4x32-10 kernel alongside) on the GpSimd engine.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Current phase

trnrand follows the [trnsci 5-phase roadmap](https://trnsci.dev/roadmap/). Active work is tracked in phase-labeled GitHub issues:

- **[Phase 1 — correctness](https://github.com/trnsci/trnrand/issues/18)** (done): Threefry, Box-Muller, Philox, Sobol and Halton NKI kernels are bit-exact against their CPU references on Trn2 (Neuron SDK 2.33, native TorchNeuron).
- **[Phase 3 — perf](https://github.com/trnsci/trnrand/issues/19)** (done): wide-tile bulk Threefry kernels (v0.8.0–v0.9.0), opt-in multi-core bulk draws with `set_cores` (v0.10.0), pinned host output (v0.11.0).
- **[Phase 4 — multi-chip](https://github.com/trnsci/trnrand/issues/20)** (done): counter-partitioned generators (`partition_rank` / `partition_size`); the counter reuse on P > 1 in 0.7.0–0.11.0 is fixed in v0.12.0.
- **[Phase 5 — generation](https://github.com/trnsci/trnrand/issues/21)** (open): trn2 PSUM sizing for larger batched tile generation.

_(No Phase 2 for trnrand — the precision story is inherited where relevant.)_

Suite-wide tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Install

```bash
pip install trnrand

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnrand[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnrand[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Kernels dispatch on whichever Neuron PyTorch lineage is installed:

- **TorchNeuron (native)** — Trn2/Trn3, Neuron SDK 2.33+, Python ≥ 3.12,
  `torch` 2.14, `torch-neuronx` 2.14.*. Each kernel is a PyTorch custom op
  (`torch.ops.trnrand.*`) on the `neuron` device; no `torch_xla`.
- **PyTorch/XLA** — Trn1 (and any host on the `torch-neuronx` 2.9 lineage).
  Kernels dispatch through `torch_xla`, unchanged.

trnrand picks the native path when the native `torch_neuronx` and a `neuron`
device are present, and the XLA path when `torch_xla` is installed; without
`nki` it uses `torch.Generator`. Set `TRNSCI_NKI_DISPATCH=native` or `xla`
(shared by all trnsci libraries) to force a path. With an editable install
(`pip install -e`), set `NKI_ENABLE_TRACE_CACHE=0`: the NKI kernel cache only
versions installed wheels and can serve stale kernels. See
[installation](docs/installation.md).

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
| | `gamma`, `beta`, `chi_squared` | Gamma(k, θ), Beta(α, β), χ²(df) |
| | `poisson` | Poisson(λ) |
| | `normal_into`, `uniform_into`, `exponential_into` | In-place fills of a preallocated tensor |
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
- [x] NKI Threefry and Philox kernels on GpSimd (bit-exact on Trn2)
- [x] On-device Box-Muller (uniform → normal)
- [x] Counter-partitioned multi-chip generators and multi-core bulk draws (`set_cores`)
- [ ] Benchmarks vs cuRAND

## Related Projects

| Project | What |
|---------|------|
| [trnfft](https://github.com/trnsci/trnfft) | FFT + complex ops |
| [trnblas](https://github.com/trnsci/trnblas) | BLAS operations |
| [trnsolver](https://github.com/trnsci/trnsolver) | Linear solvers |
| [trnsparse](https://github.com/trnsci/trnsparse) | Sparse formats + SpMM |
| [trntensor](https://github.com/trnsci/trntensor) | Einsum + tensor decompositions |

## License

Apache 2.0 — Copyright 2026 Scott Friedman


## Disclaimer

trnsci is an **independent open-source project**. It is not sponsored by, endorsed by, or affiliated with Amazon.com, Inc., Amazon Web Services, Inc., or Annapurna Labs Ltd.

"AWS", "Amazon", "Trainium", "Inferentia", "NeuronCore", "Neuron SDK", and related identifiers are trademarks of their respective owners and are used here solely for descriptive and interoperability purposes. Use does not imply endorsement, partnership, or any other relationship.

All work, opinions, analyses, benchmark results, architectural commentary, and editorial judgments in this repository and on [trnsci.dev](https://trnsci.dev) are those of the project's contributors. They do not represent the views, positions, or commitments of Amazon, AWS, or Annapurna Labs.

Feedback directed at the Neuron SDK or Trainium hardware is good-faith ecosystem commentary from independent users. It is not privileged information, is not pre-reviewed by AWS, and should not be read as authoritative about product roadmap, behavior, or quality.

For official AWS guidance, see [aws-neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/).
