# trnrand

Random number generation for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

## What This Is

Pseudo-random and quasi-random number generation targeting Trainium.
Seeded generators, standard distributions, and low-discrepancy sequences
for Monte Carlo and quasi-Monte Carlo integration.

The NKI generator is Threefry4x32-20, a counter-based RNG on the GpSimd
engine (a Philox 4x32-10 kernel ships alongside it) — on-device generation
avoids host→device transfer for large random tensors.

## Architecture

```
trnrand/
├── trnrand/
│   ├── __init__.py          # Re-exports all RNG operations
│   ├── _compile.py          # torch.compile custom ops for generator state (#61)
│   ├── generator.py         # Generator class, seeding, state management
│   ├── distributions.py     # uniform, normal, exponential, bernoulli, etc.
│   ├── quasi.py             # sobol, halton, latin_hypercube
│   └── nki/
│       ├── __init__.py
│       ├── dispatch.py      # NKI kernels, host wrappers, dispatch-path selection
│       ├── _native.py       # TorchNeuron custom ops (nki_op), loaded lazily
│       └── program.py       # GeneratorProgram streaming builder
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

The counter-based RNGs (Threefry4x32-20 for every distribution, Philox 4×32-10
as a second kernel) map to Trainium:
- **GpSimd engine** for the integer multiply-XOR rounds (not Tensor Engine)
- **Parallel generation**: each tile gets disjoint counter range
- **Deterministic**: (counter, key) → output, no state to synchronize

Counter-based RNGs are preferred over Mersenne Twister because they're
stateless and trivially parallelizable. Threefry needs no 32-bit integer
multiply (JAX's default); Philox is cuRAND's and PyTorch CUDA's.

## Known Gaps

- **`advance` / `advance_to` resume the stream only with
  `key_mode="counter"`** (and only for `uniform` / `normal`). With the default `key_mode="draw"` each NKI draw
  takes a fresh key from the host `torch.Generator`, which they don't move;
  checkpoint with `get_state` / `set_state` instead (#76).

- **Halton degrades above ~20 dimensions.** Known limitation of the
  algorithm — correlations appear in high-dimensional projections.
  Sobol is preferred for d > 10.

- **NKI Sobol is limited to 10 dimensions and 2^24 points**, and its
  scrambling is a per-dimension random shift, not Owen scrambling.

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

## Blog posts

When asked to draft a blog post for this library for the [trnsci blog](https://trnsci.dev/blog/):

1. Read the editorial brief at [`docs/blog/AUTHOR_BRIEF.md`](https://github.com/trnsci/trnsci/blob/main/docs/blog/AUTHOR_BRIEF.md) in the umbrella repo (trnsci/trnsci). It defines voice (authorless, library-as-subject), stance (architecture-first, transparency-always), and the nine required section headings.

2. Find the prompt block for this library in [`BLOG_PROMPTS.md`](https://github.com/trnsci/trnsci/blob/main/BLOG_PROMPTS.md) at the umbrella repo root. It carries library-specific context and suggested architectural angles.

3. Draft the post following the brief. Open a PR against `trnsci/trnsci` at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`. Scott (suite director) reviews before merge.

The umbrella repo — not this one — owns the blog. Per-library retrospective posts are unsigned; library is the subject, no byline. See the brief for the full set of rules.
