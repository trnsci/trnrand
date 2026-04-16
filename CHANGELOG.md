# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **NKI hardware compiler compatibility** — three categories of Python constructs
  accepted by the NKI CPU simulator but rejected by the real trn1 compiler have
  been eliminated from the Threefry and Philox kernels:
  1. Inner `def` statements inside functions in the `@nki.jit` call tree
     (`_mul32_hi_lo` inner helpers extracted to module-level `_nki_mul_u32` etc.)
  2. List comprehensions (`[expr for i in range(n)]` → explicit literals)
  3. Subscript expressions as LHS assignment targets in tuple unpacking
     (`x_b_list[0], x_b_list[1] = ...` → named variables `x0_b, x1_b = ...`)

### Hardware validation (trn1, 2026-04-16)

- **Threefry4×32-20 uniform kernel: hardware-validated** — 4 of 5
  `TestThreefryNKI` tests pass on trn1 (correctness, distribution, seed
  determinism, seed isolation). Byte-tile arithmetic is confirmed exact.
- Threefry normal kernel (fused Box-Muller): blocked by NCC_IBIR605
  (same trn1 compiler restriction as standalone `box_muller_kernel`; tracked
  in trnrand#2). Marked `xfail` on trn1; does not affect trn2+ or the
  Threefry algorithm itself.

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
