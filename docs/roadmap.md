# Roadmap

Forward-looking plan for trnrand. Tracked via [GitHub
milestones](https://github.com/trnsci/trnrand/milestones) — this page is a
browsable summary.

## v0.1.0 — Foundation _(shipped 2026-04-12)_

`Generator` class with `manual_seed` / `get_state` / `set_state`, module-level
default generator, standard distributions (`uniform`, `normal`, `exponential`,
`bernoulli`, `randint`, `randperm`, `truncated_normal`), quasi-random sequences
(`sobol`, `halton`, `latin_hypercube`), Philox 4×32 NKI stub, Box-Muller NKI
stub, MC/QMC hypersphere-volume example.

## v0.1.1 — Docs and housekeeping _(shipped 2026-04-13)_

Docs badge and `mkdocs.yml` `site_url` pointing to `trnsci.dev/trnrand/`,
repo `homepageUrl` updated, ROADMAP.md skeleton.

## v0.2.0 — CPU distributions _(shipped 2026-04-13)_

Gamma, chi-squared, beta, Poisson CPU implementations; all generator-aware
for reproducibility. CPU path works today; NKI acceleration deferred.

## v0.3.0 — NKI simulator + Philox byte-tile arithmetic _(shipped 2026-04-15)_

NKI 0.3.0 simulator dispatch via `TRNRAND_USE_SIMULATOR=1` — contributors
can iterate kernel correctness on any Linux x86_64 host without hardware
dispatch cost. Philox 4×32 32-bit multiply reworked to 8-bit byte
decomposition (16 sub-products, byte-wise carry chain), validated bit-exact
against Python unbounded-int ground truth. `nki-simulator` CI job on
`ubuntu-latest`.

## v0.4.0 — Threefry4×32-20 NKI kernel _(shipped 2026-04-18)_

Integer-multiply-free PRNG implemented entirely in GpSimd byte-tile arithmetic,
sidestepping [aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)
at the algorithm level. Fused `threefry_normal_kernel` routes GpSimd output
directly into Vector Engine transcendentals with tiles SBUF-resident — no HBM
round-trip between RNG and Box-Muller transform. Three categories of
trn1-compiler-rejected constructs eliminated from the kernel call trees. trn1
hardware validation: 4/5 `TestThreefryNKI` pass; Threefry normal kernel blocked
by NCC_IBIR605 (trn1-only). Self-contained trn2 Terraform root; TMPDIR fix.

## v0.4.1 — trn2 hardware validation _(pending — pre-staged in `feat/trn2-validated`)_

Both `test_normal_kernel_distribution` and `test_normal_kernel_matches_box_muller_cpu`
pass on trn2.3xlarge (XPASS). NCC_IBIR605 confirmed trn1-only; does not affect
trn2+. `xfail` marks removed. Closes
[#2](https://github.com/trnsci/trnrand/issues/2).

## v0.5.0 — QMC on-device + NKI distributions _(planned)_

Extend the stateless GpSimd infra to quasi-random sequences and the distribution
kernels that currently use CPU fallbacks.

**QMC on-device (GpSimd):**

- [#11](https://github.com/trnsci/trnrand/issues/11) — NKI Sobol scrambling
  kernel on GpSimd.
- [#12](https://github.com/trnsci/trnrand/issues/12) — NKI Halton on-device
  generation (low-dim only).

**Distribution breadth (Vector Engine):**

- [#13](https://github.com/trnsci/trnrand/issues/13) — Beta distribution
  (gamma-ratio method). _CPU path shipped in v0.2.0; NKI pending._
- [#14](https://github.com/trnsci/trnrand/issues/14) — Gamma distribution
  (Marsaglia-Tsang + boost). _CPU path shipped in v0.2.0; NKI pending._
- [#15](https://github.com/trnsci/trnrand/issues/15) — Poisson distribution
  (Knuth / Atkinson rejection). _CPU path shipped in v0.2.0; NKI pending._
- [#16](https://github.com/trnsci/trnrand/issues/16) — Chi-squared
  distribution (sum of squared normals). _CPU path shipped in v0.2.0;
  NKI pending._
- [#17](https://github.com/trnsci/trnrand/issues/17) — Truncated normal on
  the Vector Engine (currently host-side rejection).

## v0.6.0 — Phase 3: single-chip streaming performance _(planned)_

Batched-tile RNG streaming, NEFF compile-cache reuse, per-kernel tuning
so the NKI path is meaningfully faster than the PyTorch fallback.

**Design:** [SBUF-resident streaming Generator RFC](design/sbuf_resident_generator.md)
— pre-compiled streaming kernel that keeps Generator state SBUF-resident
across multiple distribution calls, pipelining GpSimd / Vector / Scalar
engines concurrently. This is qualitatively different from cuRAND's
per-call dispatch model, not a perf tweak.

- [#19](https://github.com/trnsci/trnrand/issues/19) — Phase 3 tracker:
  `trnrand.normal_into(buf)` streaming API, Sobol/Halton perf parity,
  published tokens/sec + GB/s benchmarks.

## v0.7.0 — Phase 4: multi-chip counter partitioning _(planned)_

Philox's counter-based design makes cross-chip sharding trivial — each
NeuronCore gets a disjoint counter subrange, outputs are bit-exact vs
single-chip.

**Design:** [Counter-partitioned multi-chip RNG RFC](design/counter_partitioned_multichip.md)
— the bit-exactness thesis: a 1-chip run and a 32-chip run with the
same seed produce the same combined stream, byte-for-byte. GPU RNG
libraries typically can't guarantee this; for MCMC / replication
studies / cluster-reshape debugging, that's a qualitatively different
property.

- [#20](https://github.com/trnsci/trnrand/issues/20) — Phase 4 tracker:
  `Generator` accepts `partition_rank` / `partition_size`; near-linear
  strong scaling on `trn1.32xlarge`.

## v0.8.0 — Phase 5: trn2 wider-PSUM fast path _(planned)_

Exploit trn2's larger partition count without maintaining two separately
tuned codebases; runtime capability detection picks the right kernel.

- [#21](https://github.com/trnsci/trnrand/issues/21) — Phase 5 tracker:
  trn2-specific Philox kernel + runtime hardware detection in dispatch.

## Suite phase mapping

trnrand's roadmap aligns with the
[trnsci suite-wide phase plan](https://github.com/trnsci/trnsci/blob/main/ROADMAP.md):

| Suite Phase | trnrand Milestone | Tracker |
|---|---|---|
| Phase 1 — correctness on hardware | v0.4.0 / v0.4.1 | [#18](https://github.com/trnsci/trnrand/issues/18) |
| Phase 2 — precision | *(N/A — trnrand is precision-neutral)* | — |
| Phase 3 — single-chip perf | v0.6.0 | [#19](https://github.com/trnsci/trnrand/issues/19) |
| Phase 4 — multi-chip | v0.7.0 | [#20](https://github.com/trnsci/trnrand/issues/20) |
| Phase 5 — generation-specific | v0.8.0 | [#21](https://github.com/trnsci/trnrand/issues/21) |
