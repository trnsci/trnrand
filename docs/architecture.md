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
| Bayesian inference priors         | `gamma()`, `beta()`  | General scientific    |
| Event-count / queuing simulation  | `poisson()`          | General scientific    |
| Variance / goodness-of-fit tests  | `chi_squared()`      | General scientific    |

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

## Multi-core bulk draws (Trn2, opt-in)

On the native TorchNeuron path, a large `uniform()` / `normal()` (and anything
built on them) can be spread over several logical NeuronCores of one chip:

```python
import trnrand
trnrand.set_cores("auto")      # or an int; or TRNRAND_CORES=auto|<n>
x = trnrand.normal(64 * 2**20)  # same values as on one core
```

- The batch range is split the same way `Generator(partition_rank=r,
  partition_size=P)` splits it across chips: core r draws one contiguous run
  of batches, and its output is that run of the single-core stream. The parts
  are copied into one host tensor in order, so the output is bit-identical for
  every core count (`tests/test_bulk_kernel.py::TestMultiCore`, and on
  hardware `tests/test_nki_native.py::TestMultiCore`).
- The cores are the logical devices `neuron:0..` visible to the process
  (`NEURON_RT_VISIBLE_CORES`), starting at the current device.
- Requests under `TRNRAND_CORES_MIN_ELEMENTS` (default 8,388,608) stay on one
  core. The kernels scale from about 2M values, but the result goes to the
  host, and the device-to-host copy and host page faults do not get faster
  with more cores. End to end, splitting only pays off from about 8M values.
- Measured on trn2.3xlarge (#60): kernel time at 64M values is 36 ms on one
  core, 19 ms on two and 12 ms on four. End to end to the host it is about
  136, 126 and 120 ms. For large host draws, `THP_MEM_ALLOC_ENABLE=1`
  (transparent huge pages for PyTorch CPU allocations) cuts the page-fault
  share: 64M on four cores goes from about 120 to 71 ms.

## Known gaps

- **NKI Philox and Box-Muller kernels await validation.** Both landed in
  v0.1.0 with CPU conformance oracles (the three canonical Salmon et al.
  SC'11 test vectors pass in
  `tests/test_nki_philox.py::TestPhiloxReference::test_spec_vectors`).
  Tracked on the [v0.3.0 milestone](https://github.com/trnsci/trnrand/milestones)
  (#1 Philox, #2 Box-Muller, #18 Phase 1).

**Phase 1 iteration is no longer hardware-gated.** Neuron SDK 2.29
(April 2026) brought NKI 0.3.0 Stable with a CPU simulator
(`nki.simulate_kernel`); the `test-simulator` CI job exercises kernel
correctness on `ubuntu-latest` without a trn1 instance. Use
`pytest tests/ -v -m simulator` as the **inner loop** for NKI kernel
development — seconds per iteration, no AWS round-trip. Hardware runs
via `scripts/run_neuron_tests.sh` are reserved for final real-device
validation, not inner-loop iteration.
- **Halton degrades above ~20 dimensions** — known algorithmic limitation.
  Sobol is preferred for `d > 10`.
- **Quasi-random sequences are host-only.** NKI scrambling for
  Sobol/Halton is scoped for [v0.3.0](roadmap.md) (#11, #12).
- **FP32 throughout.** BF16 / FP16 output paths and multi-NeuronCore
  sharding are v0.4+ candidates (see [roadmap](roadmap.md)).
