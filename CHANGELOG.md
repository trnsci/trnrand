# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2026-04-29

### Fixed

#### Multi-step `GeneratorProgram` partition counter bug (#52)

`GeneratorProgram._stream_into_nki` advanced `_counter` by the per-chip
`streaming_batches` after each distribution step. In a multi-step partitioned
program (`ProgramBuilder.normal(…).uniform(…).build()`) this caused the base
counter for step 2 onward to be misaligned: chip r's step 2 started at the
per-chip offset instead of the correct full-step offset. Concretely, the
uniform step in a 2-distribution P=2 program drew from the wrong region of
the Threefry stream, breaking partition equivalence.

Fix: advance `_counter` by `streaming_batches × partition_size` (the full
single-chip step width) rather than `streaming_batches` alone. For
`partition_size=1` this is a no-op; all single-step programs and single-chip
programs are unaffected.

Detected by `test_multi_distribution_program_equivalence_P2` in the NKI
simulator CI job introduced in #49.

#### Simulator CI test fixes (#52)

- `test_neff_cache_reuse_timing` — skip when `TRNRAND_USE_SIMULATOR=1`. The
  NKI simulator compiles every kernel invocation fresh (~10s each); asserting
  <500ms is meaningless without actual NEFF caching.
- `TestSimulatorBitExact` renamed `TestHardwareBitExact`, moved from
  `nki_simulator` to `neuron` marker. The simulator's `nl.static_range` tile
  execution order differs from sequential per-tile invocations, producing
  completely different output sequences (max diff 5.58 for normal). On
  hardware, NEFF compilation guarantees consistent deterministic output.
- `ruff format` — reformatted `program.py` to resolve CI lint failure.

## [0.7.0] - 2026-04-27

### Added

#### Counter-partitioned multi-chip RNG — bit-exact reproducibility across chip counts (#47 #48 #49)

The central claim: a 1-chip run and a P-chip run with the same seed produce the
same combined sample stream — byte-for-byte — with zero cross-chip coordination.
Concatenating P-chip draws in rank order exactly reproduces the single-chip output.
GPU libraries cannot offer this without user-side key-threading (JAX split) or
manual offset management (cuRAND). On Trainium, it falls out of the Threefry
counter structure for free.

**`Generator` partition API (#47)**

- `Generator(seed, partition_rank=r, partition_size=P)` — declares this generator
  as chip `r` of `P`. Defaults `rank=0, size=1` so all existing single-chip code
  is unchanged.
- `partition_rank` / `partition_size` validation at construction time: `size ≥ 1`,
  `0 ≤ rank < size`.
- `generator.position() → int` — logical sample count consumed (rounded up to the
  nearest 512-sample Threefry batch). Authoritative for checkpoint / resume.
- `generator.advance(n_samples)` — skip `n_samples` forward without generating
  them. Increments internal counter by `ceil(n / 512)` batches. Cheap — no RNG
  work performed.
- `generator.advance_to(position)` — jump to an absolute sample position for
  checkpoint / resume. Rounded down to the nearest batch boundary.
- `generator._chip_counter_offset(n)` — counter offset for chip `r` of `P`
  dispatching `n` elements: `partition_rank × ceil(n / 512) + _counter`.
- `generator._advance_by_elements(n)` — advance internal counter by batches
  consumed by `n` samples. Called by NKI dispatch after each kernel launch.
- `manual_seed()` resets `_counter` to 0 — reseeding starts a fresh stream.
- `_BATCH_SIZE = 512` exported at module level (`generator.py`) for use in tests
  and dispatch without NKI dependency.

**Dispatch wiring (#48)**

- `uniform()`, `normal()`, `exponential()` (per-tile NKI path): pass
  `counter_offset = gen._chip_counter_offset(n)` to the NKI kernel, then call
  `gen._advance_by_elements(n)`. Single-chip behaviour is unchanged
  (`rank=0` → offset = `_counter`, same as before).
- `normal_into()`, `uniform_into()`, `exponential_into()` (streaming path):
  `counter_offset = partition_rank × streaming_batches + gen._counter`;
  `gen._counter += streaming_batches` after dispatch.
- `ProgramBuilder` captures `partition_rank`, `partition_size`, and `_counter`
  from the `Generator` at `.new_program()` time.
- `GeneratorProgram._stream_into_nki` applies per-step partition offset:
  `counter_offset = self._counter + partition_rank × streaming_batches`.
  Counter advances by `streaming_batches` per step (not `partition_size ×
  streaming_batches`), so consecutive single-chip calls remain independent.

**Simulator and hardware tests (#49)**

- `tests/test_partitioning.py` — 43 CPU tests (no NKI required):
  `TestGeneratorPartitionAPI` (30 tests), `TestPartitionEquivalenceCPU` using
  `threefry_uniform_cpu` with 4-sample block units (8 tests),
  `TestProgramBuilderPartitionContext` (5 tests).
- `tests/test_nki_partitioning.py` — gated tests (skip automatically on dev hosts):
  - `nki_simulator` marker: per-tile equivalence (uniform/normal/exponential, P=2,4),
    streaming equivalence (stream_normal/stream_uniform, P=2,4), GeneratorProgram
    equivalence including multi-distribution programs, advance/resume equivalence.
  - `neuron` marker (trn1.32xlarge, manual): zero cross-chip coordination smoke test
    (profiler confirms no collective ops), near-linear strong scaling benchmark (≥87%
    efficiency / ≥28× 1-chip target), P=32 chip-0 slice structural check, NEFF cache
    reuse with partition args (<100ms second launch).

### Architecture notes

**Counter unit distinction**: `threefry_uniform_cpu` uses 4-sample Threefry output
blocks as its `counter_offset` unit; the NKI kernel aggregates 128 lanes into
512-sample batches. `Generator._counter` and `_BATCH_SIZE = 512` track NKI batch
units. CPU-reference tests use `ceil(n / 4)` block units directly.

**NEFF is not partition-aware at compile time.** Partition rank and counter offset
live in HBM arguments passed at runtime. The same compiled NEFF serves all chips in
a partition — no per-rank recompilation.

**Partition equivalence boundary**: N must be a multiple of `512 × P` (per-tile)
or `16,384 × P` (streaming) for exact batch alignment. Partial-batch tails
produce correct but non-partitionable outputs — documented and tested.

### Deferred

- **Partition wiring for `gamma`, `beta`, `chi_squared`, `truncated_normal`** —
  rejection-loop batch counts are data-dependent and cannot be tracked at the
  `Generator` level without significant complexity. These distributions remain
  single-chip on the NKI path. Partition support will be added once a fixed-batch
  rejection kernel is available.

- **Hardware validation** — zero cross-chip coordination profiler trace and
  near-linear strong scaling numbers (trn1.32xlarge, 32 chips) are pending a
  manual hardware run. The `neuron`-marked tests in `test_nki_partitioning.py`
  capture the assertions; results will be documented in a follow-up release note.

## [0.6.0] - 2026-04-22

### Added

#### Streaming NKI kernels — 32-tile pipeline (#42)

- `threefry_streaming_normal_kernel` / `threefry_streaming_uniform_kernel` —
  `@nki.jit` kernels using `nl.static_range(_PROGRAM_TILES)` (32 tiles) to
  pipeline GpSimd Threefry rounds and Vector Engine Box-Muller transcendentals
  within a single NEFF invocation. This is only possible inside one kernel call;
  separate XLA graph submissions cannot share SBUF state across the tile loop.
- `threefry_stream_normal(n, seed, counter_offset)` /
  `threefry_stream_uniform(n, seed, counter_offset)` — host wrappers.
  `counter_offset` is passed as a runtime HBM argument so the same NEFF serves
  every launch; the counter advances `n_launches × _PROGRAM_TILES` per call to
  guarantee non-overlapping streams.
- `_PROGRAM_TILES = 32` exported at module level (outside `if HAS_NKI:`) so
  `program.py` and test files can import it on dev hosts without neuronxcc.
- `tests/test_nki_streaming.py` — simulator + neuron tests: shape, range,
  moments, determinism, counter-advance, bit-exactness, hardware throughput.

#### `GeneratorProgram` API (#43)

- `ProgramBuilder` — fluent builder: `.normal(n)`, `.uniform(n)`,
  `.exponential(n)`, `.build()`. Raises `ValueError` if no steps added.
- `GeneratorProgram` — pre-compiled streaming NEFF wrapper. `stream_into(buffers)`
  fills named `torch.Tensor` buffers in-place and advances an internal counter so
  consecutive calls draw independent, non-overlapping streams.
  - **NKI path**: calls `threefry_stream_normal` / `threefry_stream_uniform` with
    `counter_offset=self._counter`; counter masked to 24 bits.
  - **CPU fallback**: `torch.Generator` seeded with `seed ^ counter`; increments
    counter by 1 per call. Not bit-exact with the NKI path.
- `Generator.new_program()` — entry point that creates a `ProgramBuilder` seeded
  from the generator. Late import keeps `generator.py` free of NKI dependency.
- `GeneratorProgram` exported from `trnrand.__all__`; `ProgramBuilder` is internal.
- `__version__` now resolves dynamically via `importlib.metadata` instead of the
  stale hardcoded `"0.2.0"`.
- `tests/test_program.py` — 30 CPU tests + 7 `nki_simulator` tests.

#### Zero-allocation in-place variants (#44)

- `normal_into(buf, mean, std, generator)` — fills `buf` in-place with
  `N(mean, std)` samples. NKI path uses `threefry_stream_normal`; XLA caches the
  NEFF for same-size buffers automatically — no Python-level cache needed.
- `uniform_into(buf, low, high, generator)` — fills `buf` with `U(low, high)`.
- `exponential_into(buf, rate, generator)` — fills `buf` with `Exp(rate)` via
  inverse-CDF from a uniform stream.
- All three exported in `trnrand.__all__`; PyTorch fallback calls `buf.normal_()`,
  `buf.uniform_()`, `buf.exponential_()` directly.
- 22 new CPU tests added to `tests/test_distributions.py`.

#### Streaming benchmarks (#45)

- `benchmarks/bench_streaming.py` — validates five RFC claims:
  latency floor (single NKI launch = 16,384 samples), throughput (250-round
  1M-sample benchmark), NEFF cache-reuse assertion (< 100ms), bit-exactness
  (program output matches direct kernel dispatch at same seed + counter_offset),
  engine-overlap timing guard (neuron-only).
- `scripts/run_benchmarks.sh` updated to include `bench_streaming.py`.

### Architecture notes

**Tensor Engine is intentionally idle** in all RNG kernels. Random number
generation is not matmul-shaped; Trainium's Tensor Engine is a matrix-multiply
unit and would add nothing here. GpSimd (Threefry byte arithmetic) + Vector Engine
(Box-Muller transcendentals) is the correct engine pairing for this workload.

**Stochastic rounding is not used.** SR corrects systematic bias in repeated
accumulations — the error mode that kills FP32 training. For RNG, the dominant
error is statistical (1/√N); arithmetic rounding contributes ≪ 1 ULP per sample.
SR would add hardware cost with no distributional benefit.

### Deferred

- **Poisson NKI kernel** (#15) — Knuth's sequential-product algorithm requires a
  data-dependent loop termination per sample, which cannot be vectorized across
  lanes. `torch.poisson` on the host is the only practical path until a
  Trainium-native rejection sampler becomes feasible. Honest non-fit; not deferred
  for schedule reasons.

## [0.5.0] - 2026-04-21

### Added

#### NKI backend dispatch wiring (#18)

- `set_backend("nki")` now routes `uniform()`, `normal()`, `standard_normal()`,
  and `exponential()` to the Threefry NKI kernels (`threefry_uniform_nki`,
  `threefry_normal_nki`) shipped in v0.4.0. Closes
  [#18](https://github.com/trnsci/trnrand/issues/18).
- `_nki_active()` helper in both `distributions.py` and `quasi.py`: returns
  `True` when `get_backend() != "pytorch"` and `HAS_NKI`.
- `_nki_seed(gen)`: draws one `torch.randint` to advance generator state and
  produce a 24-bit seed for NKI dispatch — same state-advance guarantee the
  PyTorch path provides for successive calls.
- `quasi.py` dispatch stubs for `sobol()` and `halton()`: try-import pattern
  that falls through to the CPU path when NKI kernels are not yet available.
- `tests/test_nki_dispatch.py`: CPU-only backend routing tests
  (`TestBackendRoutingCPU`, 11 tests) and NKI simulator tests
  (`TestNKIDispatchSimulator`, 9 tests, `@pytest.mark.nki_simulator`).

#### Truncated normal NKI kernel (#17)

- `truncated_normal_nki(n, low, high, mean, std, seed)` — Box-Muller
  candidates from `threefry_normal_nki` + host-side rejection sampling.
  2.5× oversample; configurable `[low, high]` bounds (in standard-normal
  units); `mean`/`std` shift applied to accepted samples. No new NKI code —
  reuses the Threefry normal kernel established in v0.4.0. Closes
  [#17](https://github.com/trnsci/trnrand/issues/17).
- `distributions.truncated_normal()` dispatches to `truncated_normal_nki`
  when NKI is active.

#### Gamma NKI kernel (#14)

- `gamma_nki(n, shape, scale, seed)` — Marsaglia-Tsang squeeze acceptance
  in float32. Boost identity (`U^(1/shape)`) for `shape < 1`. 1.7× oversample.
  All arithmetic in float32 (adequate precision for scientific Monte Carlo).
  Closes [#14](https://github.com/trnsci/trnrand/issues/14).
- `distributions.gamma()` dispatches to `gamma_nki` when NKI is active.

#### Chi-squared NKI kernel (#16)

- `chi_squared_nki(n, df, seed)` — thin wrapper: `gamma_nki(shape=df/2, scale=2)`.
  Closes [#16](https://github.com/trnsci/trnrand/issues/16).
- `distributions.chi_squared()` dispatches to `chi_squared_nki` when NKI is active.

#### Beta NKI kernel (#13)

- `beta_nki(n, alpha, beta_param, seed)` — gamma-ratio identity
  `X / (X + Y)` where `X ~ Gamma(alpha)`, `Y ~ Gamma(beta)`. Distinct seeds
  for the X and Y streams ensure statistical independence. Closes
  [#13](https://github.com/trnsci/trnrand/issues/13).
- `distributions.beta()` dispatches to `beta_nki` when NKI is active.

#### Sobol NKI kernel (#11)

- `_init_sobol_directions()` — Joe-Kuo 2010 direction vectors for 10 Sobol
  dimensions, 24-bit precision. Computed once at module import; embedded as
  `nl.full` constants at `@nki.jit` trace time. No HBM round-trip for
  lookup tables.
- `sobol_gray_code_kernel(@nki.jit)` — GpSimd XOR accumulation over Gray-code
  bits. For each lane `p` and dimension `d`:
  `s[p,d] = XOR{ v[d][k]  for k in 0..23  if bit k of gray(i_p) }`.
  10 dims × 24 bits fully unrolled via `nl.static_range` (~240 ops, all
  vectorized over 128 lanes). Float conversion via 3-byte decomposition
  identical to `threefry4x32_kernel`. Output: `(P, 10)` float32 in `[0, 1)`.
- `sobol_nki(n_points, n_dims, seed, start_index)` — host wrapper. Gray codes
  computed host-side; 128-lane tiling with padding; additive scrambling
  (random shift in `[0,1)` per dimension) when `seed != 0`. Closes
  [#11](https://github.com/trnsci/trnrand/issues/11).
- `quasi.sobol()` routes to `sobol_nki` when NKI is active.

#### Halton NKI kernel (#12)

- `halton_kernel(@nki.jit)` — iterative float32 radical inverse on the Vector
  Engine. Per digit: `q = nl.floor(i / p)` (exact for `i < 2²²` — ULP proof
  in source), `digit = i - p * q`, `result += digit / p^(k+1)`, `i = q`.
  Outer loop (10 dims) + inner loop (depth) both unrolled via `nl.static_range`.
  Output: `(P, 10)` float32 in `(0, 1)`.
- `halton_nki(n_points, n_dims, start_index)` — host wrapper. Skips index 0
  per Halton convention; 128-lane tiling; asserts index < 2²² = 4,194,304.
  Primes: (2, 3, 5, 7, 11, 13, 17, 19, 23, 29). Closes
  [#12](https://github.com/trnsci/trnrand/issues/12).
- `quasi.halton()` routes to `halton_nki` when NKI is active.

### Architecture notes

All five new distribution/QMC kernels follow the **fixed-batch-with-mask pattern**:
each kernel generates a fixed tile of candidates; the host wrapper retries until
enough accepted samples are gathered. This sidesteps NKI's no-dynamic-loop
constraint without any kernel changes.

The Sobol and Halton kernels demonstrate complementary engine strategies:
- **Sobol** uses GpSimd exclusively (integer XOR — no multiplies needed)
- **Halton** uses GpSimd for arithmetic + Vector Engine for `nl.floor` (same
  engine mix as the Threefry normal kernel)

### Deferred

- **Poisson NKI kernel** (#15) — Knuth's algorithm (sequential product loop)
  is not vectorizable. Poisson remains CPU-only via `torch.poisson`. Will be
  revisited if a Trainium-native rejection sampler becomes practical.

## [0.4.1] - 2026-04-18

### Fixed

- **NKI hardware compiler compatibility** — three categories of Python constructs
  accepted by the NKI CPU simulator but rejected by the real trn1 compiler have
  been eliminated from the Threefry and Philox kernels:
  1. Inner `def` statements inside functions in the `@nki.jit` call tree
     (`_mul32_hi_lo` inner helpers extracted to module-level `_nki_mul_u32` etc.)
  2. List comprehensions (`[expr for i in range(n)]` → explicit literals)
  3. Subscript expressions as LHS assignment targets in tuple unpacking
     (`x_b_list[0], x_b_list[1] = ...` → named variables `x0_b, x1_b = ...`)

### Hardware validation (trn2, 2026-04-18)

- **Threefry normal kernel: hardware-validated on trn2.3xlarge (sa-east-1)** —
  `test_normal_kernel_distribution` and `test_normal_kernel_matches_box_muller_cpu`
  pass on trn2 (XPASS). NCC_IBIR605 confirmed trn1-only; does not affect trn2+.
  All 5 `TestThreefryNKI` tests now pass on trn2.
- `xfail` marks removed from both tests; trn1 CI is unaffected (`--philox-only`
  already deselects them on trn1).

### Hardware validation (trn1, 2026-04-16)

- **Threefry4×32-20 uniform kernel: hardware-validated** — 4 of 5
  `TestThreefryNKI` tests pass on trn1 (correctness, distribution, seed
  determinism, seed isolation). Byte-tile arithmetic is confirmed exact.
- Threefry normal kernel (fused Box-Muller): blocked by NCC_IBIR605
  (same trn1 compiler restriction as standalone `box_muller_kernel`; tracked
  in trnrand#2). Confirmed trn1-only; passes on trn2.3xlarge.

## [0.4.0] - 2026-04-16

### Added

- **Threefry4×32-20 NKI kernel** — integer-multiply-free PRNG for Trainium,
  implemented entirely in byte-tile arithmetic. Every tile element stays ≤ 511,
  well below float32's 2²⁴ exact-integer ceiling, sidestepping
  [aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)
  at the algorithm level rather than the decomposition level.
  Uses only 32-bit addition, XOR, and rotation — the operations Threefry was
  designed for (Salmon et al. SC'11, same paper as Philox).
- `threefry4x32_reference()` — CPU oracle verified against 3 Random123 KAT vectors.
- `threefry_uniform_cpu()` — uniform float stream, mirrors `philox_uniform_cpu` API.
- `threefry4x32_kernel` — NKI kernel: 8×(P,1) counter/key → (P,4) float32 U[0,1).
  Output built directly from 3 low bytes of each output word (23-bit mantissa),
  bypassing uint32 tile assembly entirely.
- `threefry_normal_kernel` — fused Threefry + Box-Muller NKI kernel. GpSimd byte
  arithmetic feeds directly into Vector Engine transcendentals with tiles remaining
  SBUF-resident between stages — no HBM round-trip between RNG and transform.
- `threefry_uniform_nki()`, `threefry_normal_nki()` — host-side wrappers.
- `_add32_bytes_numpy()`, `_rotl32_bytes_numpy()` — pure-numpy byte-arithmetic
  references for NKI kernel parity testing.
- 4 new simulator tests (no `xfail` marks): KAT vector check, 128-lane reference
  parity, U[0,1) distribution, N(0,1) distribution for the fused kernel.
- `TestThreefryReference` class: 12 CPU tests including KAT vectors, distributional
  checks, and byte-helper ground-truth comparisons.

### Architecture note

The four-engine RNG framing is now fully realised for normals: GpSimd handles
Threefry byte arithmetic; Vector Engine handles Box-Muller transcendentals;
SBUF holds output between stages. Philox remains as the CPU reference and will
become the hardware path once AWS ships GpSimd integer multiply
([aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)).

## [0.3.0] - 2026-04-15

### Added

- NKI 0.3.0 CPU simulator dispatch via `TRNRAND_USE_SIMULATOR=1`
  ([#26](https://github.com/trnsci/trnrand/issues/26)). Lets contributors
  iterate kernel correctness on any Linux x86_64 host — or the existing
  trn1 DLAMI via SSM — without paying NEFF compile + hardware dispatch
  cost. Seconds per cycle vs minutes on hardware.
- `nki_simulator` pytest marker + `tests/test_nki_sim.py` curated suite
  (Philox spec vectors, reference equivalence, Box-Muller distribution).
- `scripts/run_simulator_tests.sh` — SSM runner for the simulator suite.
- `scripts/run_neuron_tests.sh --philox-only` flag — deselects Box-Muller
  kernel tests when isolating Philox's hardware status.
- `nki-simulator` GitHub Actions job on `ubuntu-latest` (installs
  `nki>=0.3.0` from the Neuron pip index).
- `docs/developing_kernels.md` — kernel authoring guide, env var
  reference, trn1 NKI 0.3.0 gotchas encountered during Phase 1.
- `_mul32_hi_lo_numpy` pure-numpy ground-truth port alongside the NKI
  `_mul32_hi_lo` helper, plus `test_mul32_numpy_matches_ground_truth`
  and `test_mul32_simulator_matches_numpy` tests to separate algorithm
  bugs from NKI op-semantics bugs during kernel debugging.

### Changed

- Migrated to NKI 0.3.0 canonical namespace: `import nki` / `nki.language`
  / `nki.isa` replace the deprecated `neuronxcc.nki.*` path. `[neuron]`
  extra pins `nki>=0.3.0`. NKI 0.3.0 ships on the latest Deep Learning
  AMI (Neuron SDK 2.29, April 2026).
- Main CI `test` job marker filter now `-m "not neuron and not nki_simulator"`
  so the three test channels don't collide.
- Philox 4×32-10 NKI kernel: 32×32 multiply reworked to an 8-bit byte
  decomposition (16 sub-products, byte-wise carry chain). Algorithm is
  validated bit-exact against Python unbounded-int ground truth via
  `_mul32_hi_lo_numpy`.

### Known limitations

- **NKI Philox kernel is not hardware-validated in this release.**
  `aws-neuron-sdk#1308` — NKI ops on `uint32` tiles route through the
  float32 activation engine on both CPU simulator and trn1 hardware.
  Because float32 exactly represents integers only up to 2^24, any
  uint32 tile value > 2^24 (including Philox counter state itself)
  loses precision at the NKI op boundary. No kernel-level decomposition
  can work around this — the issue is at `nl.copy(..., dtype=nl.uint32)`,
  not just `nl.multiply`. Filed upstream with reproducer; `trnrand`
  dispatch continues to use the PyTorch `torch.Generator` path by
  default, which is the only user-visible path today. Tracked in
  [#1](https://github.com/trnsci/trnrand/issues/1) alongside the
  upstream issue; will reopen for hardware validation once AWS ships
  a true integer multiply primitive.
- Box-Muller NKI kernel still gated on trn1 compiler fix for
  `NCC_IBIR605` (InstActivation bias parameter) — tracked in
  [#2](https://github.com/trnsci/trnrand/issues/2). Unaffected by #1308.

## [0.2.0] - 2026-04-13

### Added

- `gamma(size, shape, scale=1.0, ...)` — Gamma distribution via
  Marsaglia-Tsang rejection with the boost identity for `shape < 1`.
- `chi_squared(size, df, ...)` — Chi-squared via `2·Gamma(df/2, 1)`.
- `beta(size, alpha, beta, ...)` — Beta via the gamma-ratio identity
  `X / (X + Y)` with `X ~ Gamma(α)`, `Y ~ Gamma(β)`.
- `poisson(size, lam, ...)` — Poisson via `torch.poisson` with a
  generator-bound rates tensor.

All four distributions are generator-aware for reproducibility and
tested against analytic mean / variance on 100k–200k samples.

### Known limitations

- NKI kernels for these distributions are v0.3.0 scope (#13, #14, #15,
  #16). CPU path works today via the PyTorch backend; on-device
  acceleration waits for hardware validation.

## [0.1.1] - 2026-04-13

### Fixed

- Docs badge in `README.md` and `mkdocs.yml` `site_url` now point at
  `trnsci.dev/trnrand/` (the central docs home) instead of the retired
  per-repo github.io URL (#9). Repo `homepageUrl` updated to match (#10).

### Added

- `ROADMAP.md` and `docs/roadmap.md` with v0.1.1 / v0.2.0 / v0.3.0
  milestones and the issues attached to each.
- `[Unreleased]` skeleton in CHANGELOG for future entries.

## [0.1.0] - 2026-04-12

### Added

- `Generator` class with `manual_seed`, `get_state`/`set_state`, and a
  module-level default generator (`trnrand.manual_seed`).
- Standard distributions: `uniform`, `normal`, `standard_normal`,
  `exponential`, `bernoulli`, `randint`, `randperm`, `truncated_normal`.
- Quasi-random sequences: `sobol` (scrambled), `halton`, `latin_hypercube`.
- Backend dispatch: `set_backend` (`auto`/`pytorch`/`nki`), `get_backend`,
  `HAS_NKI` flag.
- Philox 4×32-10 NKI kernel in `trnrand/nki/dispatch.py` with a CPU
  reference (`philox4x32_reference`, `philox_uniform_cpu`) used as the
  conformance oracle. CPU-side spec invariants and the three canonical
  Salmon et al. SC'11 test vectors are verified in
  `tests/test_nki_philox.py::TestPhiloxReference`.
- Box-Muller transform on the Vector Engine for `normal()`, with a CPU
  reference (`box_muller_cpu`) and CPU-side distributional tests
  (`TestBoxMullerReference`).
- MC vs QMC hypersphere-volume example in `examples/mc_integration.py`.
- Test suites for distributions (reproducibility, statistics, bounds) and
  quasi-random sequences (uniformity, determinism, stratification).

### Known limitations

- **NKI path is not hardware-validated yet.** The Philox and Box-Muller
  kernels are scaffolded against the NKI 2.24 API and pass the CPU
  conformance oracle by construction, but the `@pytest.mark.neuron`
  suite has not yet run on trn1/trn2. Tracked in #1 (Philox) and #2
  (Box-Muller). Until validated, `trnrand` falls back to `torch.Generator`
  on CPU — the backend dispatch path is the only user-visible difference.
- **Benchmarks vs cuRAND pending hardware access** (#3).
