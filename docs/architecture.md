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
│       ├── dispatch.py    # NKI kernels, host wrappers, dispatch-path selection
│       ├── _native.py     # TorchNeuron custom ops (nki_op), loaded lazily
│       └── program.py     # GeneratorProgram streaming builder
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

Counter-based RNGs map cleanly to Trainium. Every NKI distribution draws
from Threefry4x32-20, which needs no 32-bit integer multiply; a Philox
4×32-10 kernel ships alongside it (`torch.ops.trnrand.philox4x32`). Both are
bit-exact against their CPU references on Trn2.

- **GpSimd engine** runs the integer multiply-XOR rounds (the Tensor Engine
  is wasted on this).
- **Parallel generation:** each tile gets a disjoint counter range, no
  cross-tile coordination required.
- **Deterministic:** `(counter, key) → output` — no state to synchronize
  across cores.

Counter-based RNGs are preferred over Mersenne Twister precisely because
they're stateless and trivially parallelizable. Philox is cuRAND's default
engine; Threefry is JAX's.

## Box-Muller for `normal()`

The on-device normal path is a Box-Muller transform layered on the Threefry
uniform stream, fused into the same kernel:

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

trnrand.set_cores("auto")  # or an int; or TRNRAND_CORES=auto|<n>
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
  core, 19 ms on two and 12 ms on four. End to end to the host it was about
  136, 126 and 120 ms, because the device-to-host copy into fresh pageable
  memory page-faults at about 4 GB/s.
- The host result now comes from the accelerator's caching host allocator
  (`torch.empty(..., pin_memory=True)`, as in trnblas#64): a fresh tensor per
  call, already faulted in, so the copy runs at the 13-14 GB/s link rate.
  `normal(64M)` end to end is about 76 ms on one core and 55 ms on four.
  `TRNRAND_PINNED_OUTPUT=0` turns it off.

## Known gaps

- **Checkpoint/resume with `advance_to` needs `key_mode="counter"`.** With
  the default `key_mode="draw"`, each NKI draw takes a fresh key from the
  host `torch.Generator`, and `advance` / `advance_to` move only the
  Threefry counter, so a new generator advanced to `position()` draws
  different values. On the PyTorch path they have no effect. Use
  `get_state` / `set_state` (which carry the counter since v0.12.1), or
  `key_mode="counter"`, which fixes the key per seed for `uniform`,
  `normal` and `standard_normal` (#76).

**Kernel iteration is not hardware-gated.** Neuron SDK 2.29
(April 2026) brought NKI 0.3.0 Stable with a CPU simulator
(`nki.simulate_kernel`); the `test-simulator` CI job exercises kernel
correctness on `ubuntu-latest` without a trn1 instance. Use
`pytest tests/ -v -m simulator` as the **inner loop** for NKI kernel
development — seconds per iteration, no AWS round-trip. Hardware runs
via `scripts/run_neuron_tests.sh` are reserved for final real-device
validation, not inner-loop iteration.
- **Halton degrades above ~20 dimensions** — known algorithmic limitation.
  Sobol is preferred for `d > 10`.
- **NKI Sobol covers 10 dimensions and 2^24 points** (it raises past
  2^24), and its scrambling is a per-dimension random shift, not the Owen
  scrambling of the CPU `SobolEngine` path. NKI Halton covers 10 dimensions
  and indices below 2^22. Past those limits the NKI path raises; use
  `set_backend("pytorch")`.
- **FP32 throughout.** BF16 / FP16 output paths are future work.
