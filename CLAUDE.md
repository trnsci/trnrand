# trnrand

Random number generation for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

## What This Is

Pseudo-random and quasi-random number generation targeting Trainium.
Seeded generators, standard distributions, and low-discrepancy sequences
for Monte Carlo and quasi-Monte Carlo integration.

The NKI target is a Philox counter-based RNG on the GpSimd engine —
on-device generation avoids host→device transfer for large random tensors.

## Architecture

```
trnrand/
├── trnrand/
│   ├── __init__.py          # Re-exports all RNG operations
│   ├── generator.py         # Generator class, seeding, state management
│   ├── distributions.py     # uniform, normal, exponential, bernoulli, etc.
│   ├── quasi.py             # sobol, halton, latin_hypercube
│   └── nki/
│       ├── __init__.py
│       └── dispatch.py      # Philox kernel stub for on-device RNG
├── tests/
│   ├── conftest.py
│   ├── test_distributions.py  # Reproducibility, statistics, bounds
│   └── test_quasi.py          # Uniformity, determinism, stratification
├── examples/
│   └── mc_integration.py    # MC vs QMC hypersphere volume estimation
├── pyproject.toml
├── README.md
├── LICENSE                  # Apache 2.0
└── CLAUDE.md                # This file
```

## Use Cases Across the Suite

| Use Case | trnrand Function | Consumer |
|----------|-----------------|----------|
| Noise injection for speech training | `normal()` | trnfft |
| Stochastic trace estimation | `normal()`, `sobol()` | trnsolver |
| Weight initialization | `truncated_normal()` | trnfft/nn.py |
| Monte Carlo integration | `sobol()`, `halton()` | trnblas (DF-MP2) |
| Hyperparameter sweeps | `sobol()` | Ablation studies |
| Data augmentation | `uniform()`, `bernoulli()` | General |

## NKI Strategy

The Philox 4×32 counter-based RNG maps to Trainium:
- **GpSimd engine** for the integer multiply-XOR rounds (not Tensor Engine)
- **Parallel generation**: each tile gets disjoint counter range
- **Deterministic**: (counter, key) → output, no state to synchronize

Philox is preferred over Mersenne Twister because it's stateless and
trivially parallelizable. Same engine used by cuRAND and JAX.

## Known Gaps

- **NKI Philox kernel is a stub.** All generation uses torch.Generator
  until validated on hardware.

- **Halton degrades above ~20 dimensions.** Known limitation of the
  algorithm — correlations appear in high-dimensional projections.
  Sobol is preferred for d > 10.

- **No on-device normal distribution yet.** Box-Muller transform from
  uniform samples would run on Vector Engine (cos, sin, log, sqrt).

## Dependencies

- `torch>=2.1` — tensor operations and CPU fallback
- `numpy>=1.24` — test reference
- `neuronxcc` — NKI kernels (optional)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/mc_integration.py
```

## Naming Convention

Sibling repos in the trn-* suite:
- `trnfft` — FFT + complex ops (https://github.com/trnsci/trnfft)
- `trnblas` — BLAS operations (https://github.com/trnsci/trnblas)
- `trnsolver` — Linear solvers (https://github.com/trnsci/trnsolver)
- `trnrand` — Random number generation (this repo)

All repos: Python/NKI, Apache 2.0.
