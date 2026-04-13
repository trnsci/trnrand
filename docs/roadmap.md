# Roadmap

Forward-looking plan for trnrand. Tracked via [GitHub
milestones](https://github.com/trnsci/trnrand/milestones) — this page is a
browsable summary.

## v0.1.1 — post-transfer hygiene _(in progress)_

Housekeeping after the `scttfrdmn/trnrand → trnsci/trnrand` transfer. No
new features.

- [#9](https://github.com/trnsci/trnrand/issues/9) — Point Docs badge and
  `mkdocs.yml` `site_url` at `trnsci.dev`.
- [#10](https://github.com/trnsci/trnrand/issues/10) — Update GitHub repo
  `homepageUrl` to `trnsci.dev/trnrand/`.

## v0.2.0 — CPU distributions + NKI hardware validation _(CPU distributions shipped; hardware validation gated on trn1)_

CPU implementations of the v0.3.0 distributions ship here so users can
pick them up today without waiting for NKI hardware validation. The
Philox and Box-Muller NKI scaffolds from v0.1.0 still need proving on
real Trainium silicon — tracked in the same milestone.

**Shipped (v0.2.0):**

- Gamma, chi-squared, beta, poisson CPU implementations (see
  [Distributions API](api/distributions.md)).

**Hardware-gated (open on this milestone):**

- [#1](https://github.com/trnsci/trnrand/issues/1) — Validate NKI
  Philox 4×32-10 kernel on trn1/trn2.
- [#2](https://github.com/trnsci/trnrand/issues/2) — On-device Box-Muller
  transform (uniform → normal) on the Vector Engine.
- [#3](https://github.com/trnsci/trnrand/issues/3) — Benchmarks vs cuRAND
  on trn1.2xlarge.

## v0.3.0 — QMC on-device + distribution breadth _(planned)_

Extend the stateless-Philox infra to quasi-random sequences on GpSimd; add
the distributions that close the gap vs cuRAND / NumPy.

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

## v0.4.0 — Phase 3: single-chip performance _(planned)_

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

## v0.5.0 — Phase 4: multi-chip counter partitioning _(planned)_

Philox's counter-based design makes cross-chip sharding trivial — each
NeuronCore gets a disjoint counter subrange, outputs are bit-exact vs
single-chip.

- [#20](https://github.com/trnsci/trnrand/issues/20) — Phase 4 tracker:
  `Generator` accepts `partition_rank` / `partition_size`; near-linear
  strong scaling on `trn1.32xlarge`.

## v0.6.0 — Phase 5: trn2 wider-PSUM fast path _(planned)_

Exploit trn2's larger partition count without maintaining two separately
tuned codebases; runtime capability detection picks the right kernel.

- [#21](https://github.com/trnsci/trnrand/issues/21) — Phase 5 tracker:
  trn2-specific Philox kernel + runtime hardware detection in dispatch.

## Suite phase mapping

trnrand's roadmap aligns with the
[trnsci suite-wide phase plan](https://github.com/trnsci/trnsci/blob/main/ROADMAP.md):

| Suite Phase | trnrand Milestone | Tracker |
|---|---|---|
| Phase 1 — correctness on hardware | v0.3.0 | [#18](https://github.com/trnsci/trnrand/issues/18) |
| Phase 2 — precision | *(N/A — trnrand is precision-neutral)* | — |
| Phase 3 — single-chip perf | v0.4.0 | [#19](https://github.com/trnsci/trnrand/issues/19) |
| Phase 4 — multi-chip | v0.5.0 | [#20](https://github.com/trnsci/trnrand/issues/20) |
| Phase 5 — generation-specific | v0.6.0 | [#21](https://github.com/trnsci/trnrand/issues/21) |
