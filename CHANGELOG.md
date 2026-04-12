# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Philox 4×32-10 NKI kernel scaffold in `trnrand/nki/dispatch.py` with a
  matching CPU reference (`philox4x32_reference`, `philox_uniform_cpu`)
  used as the conformance oracle. CPU-side spec invariants tested in
  `tests/test_nki_philox.py::TestPhiloxReference`. NKI kernel awaits
  on-hardware validation per #1 — the `TestPhiloxNKI` class is gated by
  the `neuron` marker.
- Box-Muller transform on the Vector Engine for `normal()`, with a CPU
  reference (`box_muller_cpu`) and CPU-side distributional tests
  (`TestBoxMullerReference`). Vector Engine kernel awaits on-hardware
  validation per #2.

## [0.1.0] - 2026-04-11

### Added

- `Generator` class with `manual_seed`, `get_state`/`set_state`, and a
  module-level default generator (`trnrand.manual_seed`).
- Standard distributions: `uniform`, `normal`, `standard_normal`,
  `exponential`, `bernoulli`, `randint`, `randperm`, `truncated_normal`.
- Quasi-random sequences: `sobol` (scrambled), `halton`, `latin_hypercube`.
- Backend dispatch: `set_backend` (`auto`/`pytorch`/`nki`), `get_backend`,
  `HAS_NKI` flag.
- NKI Philox kernel scaffold in `trnrand/nki/dispatch.py` (stub — falls
  back to `torch.Generator` until validated on hardware).
- MC vs QMC hypersphere-volume example in `examples/mc_integration.py`.
- Test suites for distributions (reproducibility, statistics, bounds) and
  quasi-random sequences (uniformity, determinism, stratification).
