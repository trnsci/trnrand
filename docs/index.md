# trnrand

Random number generation for AWS Trainium via NKI (Neuron Kernel Interface).

Seeded pseudo-random distributions, low-discrepancy quasi-random sequences for
quasi-Monte Carlo, and an on-device Philox RNG targeting the GpSimd engine —
on-device generation avoids host→device transfer for large random tensors.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Why

Reproducible, parallel-friendly RNG is a prerequisite for scientific
computing on Trainium: weight initialization, noise injection, stochastic
trace estimation, and Monte Carlo integration all depend on it.
PyTorch's host-side `torch.Generator` works but pays the host→device
transfer cost for every sample. trnrand targets a stateless, counter-based
RNG (Philox 4×32) that runs on the GpSimd engine so large tensors are
generated where they're consumed.

## Primary use cases

- Noise injection for speech training (consumed by `trnfft`).
- Stochastic trace estimation and Monte Carlo integration (consumed by
  `trnblas` for DF-MP2 quantum chemistry workloads).
- Truncated-normal weight initialization for neural-net layers.

See [Architecture](architecture.md) for the Philox/GpSimd story.

## Related projects

- [trnfft](https://github.com/trnsci/trnfft) — FFT + complex ops for Trainium.
- [trnblas](https://github.com/trnsci/trnblas) — BLAS operations for Trainium.
- `trnsolver` *(planned)* — Linear solvers and eigendecomposition.
