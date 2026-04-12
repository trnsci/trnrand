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
| Noise injection (speech training) | `normal()`           | trnfft (Williamson)   |
| Stochastic trace estimation       | `normal()`, `sobol()`| trnsolver             |
| Weight initialization             | `truncated_normal()` | trnfft/nn.py          |
| Monte Carlo integration           | `sobol()`, `halton()`| trnblas (Janesko)     |
| Hyperparameter sweeps             | `sobol()`            | Ephemeron ablation    |
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

## Known gaps

- **NKI Philox kernel is a stub.** All generation currently routes through
  `torch.Generator`. The kernel scaffold lives in `trnrand/nki/dispatch.py`.
- **Halton degrades above ~20 dimensions** — known algorithmic limitation.
  Sobol is preferred for `d > 10`.
- **No on-device normal distribution yet.** A Box-Muller transform from
  uniform samples (cos/sin/log/sqrt on the Vector Engine) is the path
  forward.
