# trnrand

Random number generation for AWS Trainium via NKI (Neuron Kernel Interface).

Seeded pseudo-random distributions, low-discrepancy quasi-random sequences for
quasi-Monte Carlo, and an on-device counter-based RNG (Threefry4x32-20) on the GpSimd engine —
on-device generation avoids host→device transfer for large random tensors.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Why

Reproducible, parallel-friendly RNG is a prerequisite for scientific
computing on Trainium: weight initialization, noise injection, stochastic
trace estimation, and Monte Carlo integration all depend on it.
PyTorch's host-side `torch.Generator` works but pays the host→device
transfer cost for every sample. trnrand uses a stateless, counter-based
RNG (Threefry4x32-20; a Philox 4×32-10 kernel ships alongside it) that runs
on the GpSimd engine so large tensors are generated where they're consumed.

## Primary use cases

- Noise injection for speech training (consumed by `trnfft`).
- Stochastic trace estimation and Monte Carlo integration (consumed by
  `trnblas` for DF-MP2 quantum chemistry workloads).
- Truncated-normal weight initialization for neural-net layers.

See [Architecture](architecture.md) for the counter-based RNG / GpSimd story.

## Related projects

- [trnfft](https://github.com/trnsci/trnfft) — FFT + complex ops for Trainium.
- [trnblas](https://github.com/trnsci/trnblas) — BLAS operations for Trainium.
- [trnsolver](https://github.com/trnsci/trnsolver) — Linear solvers and eigendecomposition.
- [trnsparse](https://github.com/trnsci/trnsparse) — Sparse formats and SpMM.
- [trntensor](https://github.com/trnsci/trntensor) — Einsum and tensor decompositions.
