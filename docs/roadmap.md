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

## v0.2.0 — NKI hardware validation _(blocked on trn1 access)_

Prove the Philox and Box-Muller NKI scaffolds from v0.1.0 work on real
Trainium silicon. Nothing new to build — these issues drive the
hardware-in-the-loop validation cycle.

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
  (gamma-ratio method).
- [#14](https://github.com/trnsci/trnrand/issues/14) — Gamma distribution
  (Marsaglia-Tsang + boost).
- [#15](https://github.com/trnsci/trnrand/issues/15) — Poisson distribution
  (Knuth / Atkinson rejection).
- [#16](https://github.com/trnsci/trnrand/issues/16) — Chi-squared
  distribution (sum of squared normals).
- [#17](https://github.com/trnsci/trnrand/issues/17) — Truncated normal on
  the Vector Engine (currently host-side rejection).

## v0.4.0+ candidates _(not yet issue-tracked)_

Directions that didn't make v0.3.0 but are on the radar:

- **Multi-NeuronCore counter-space sharding.** Throughput scaling beyond
  one NeuronCore by partitioning Philox counter space across cores.
- **BF16 / FP16 output dtypes.** Today trnrand emits FP32 only. Low-
  precision paths save bandwidth in weight-init and noise-injection
  workloads.
- **`torch.compile` / inductor integration.** Make `Generator` state
  graph-friendly so trnrand slots into compiled model graphs without
  tracing breaks.

Open an issue if any of these turn into priorities.
