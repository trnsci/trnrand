# Architecture

## Layout

```
trnrand/
├── trnrand/
│   ├── __init__.py        # Re-exports all RNG operations
│   ├── generator.py       # Generator class, seeding, state management
│   ├── distributions.py   # uniform, normal, exponential, bernoulli, etc.
│   ├── quasi.py           # sobol, halton, latin_hypercube
│   └── nki/
│       ├── __init__.py    # Backend dispatch (set_backend / HAS_NKI)
│       └── dispatch.py    # Philox kernel scaffold for on-device RNG
├── tests/
├── examples/mc_integration.py
└── benchmarks/
```

## Use cases across the suite

| Use case                          | trnrand function     | Consumer              |
|-----------------------------------|----------------------|-----------------------|
| Noise injection (speech training) | `normal()`           | trnfft                |
| Stochastic trace estimation       | `normal()`, `sobol()`| trnsolver             |
| Weight initialization             | `truncated_normal()` | trnfft/nn.py          |
| Monte Carlo integration           | `sobol()`, `halton()`| trnblas (DF-MP2)      |
| Hyperparameter sweeps             | `sobol()`            | Ablation studies      |
| Data augmentation                 | `uniform()`, `bernoulli()` | General         |

## NKI strategy

The Philox 4×32 counter-based RNG maps cleanly to Trainium:

- **GpSimd engine** runs the integer multiply-XOR rounds (the Tensor Engine
  is wasted on this).
- **Parallel generation:** each tile gets a disjoint counter range, no
  cross-tile coordination required.
- **Deterministic:** `(counter, key) → output` — no state to synchronize
  across cores.

Philox is preferred over Mersenne Twister precisely because it's stateless
and trivially parallelizable. It's the same engine used by cuRAND and JAX.

## Box-Muller for `normal()`

The on-device normal path is a Box-Muller transform layered on the Philox
uniform stream:

- Pairs of uniforms `(u1, u2)` → standard-normal pairs `(z1, z2)` via
  `r = √(-2 ln u1)`, `θ = 2π u2`, `z1 = r cos θ`, `z2 = r sin θ`.
- Runs on the Vector Engine, which has hardware `cos`/`sin`/`log`/`sqrt`.
- Box-Muller is preferred over Marsaglia polar here: Marsaglia avoids the
  trig calls but uses rejection sampling, which serializes branch-divergent
  lanes and kills SIMD throughput. Box-Muller has constant work per pair.

## Known gaps

- **NKI Philox and Box-Muller kernels await on-hardware validation.** Both
  landed in v0.1.0 with CPU conformance oracles (the three canonical Salmon
  et al. SC'11 test vectors pass in
  `tests/test_nki_philox.py::TestPhiloxReference::test_spec_vectors`). The
  hardware-gated `TestPhiloxNKI` suite runs only with `neuronxcc` available.
  Tracked on the [v0.2.0 milestone](https://github.com/trnsci/trnrand/milestone/1)
  (#1 Philox, #2 Box-Muller).
- **Halton degrades above ~20 dimensions** — known algorithmic limitation.
  Sobol is preferred for `d > 10`.
- **Quasi-random sequences are host-only.** NKI scrambling for
  Sobol/Halton is scoped for [v0.3.0](roadmap.md) (#11, #12).
- **FP32 throughout.** BF16 / FP16 output paths and multi-NeuronCore
  sharding are v0.4+ candidates (see [roadmap](roadmap.md)).
