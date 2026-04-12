# Benchmarks

trnrand kernels measured against two baselines on the same Trainium instance:

1. **NKI** — trnrand with `set_backend("nki")` (Philox on GpSimd; Box-Muller
   on Vector Engine for `normal()`).
2. **trnrand-PyTorch** — trnrand with `set_backend("pytorch")` running on
   the host CPU (`torch.Generator` + `torch.empty(...).uniform_()` etc.).
3. **torch.\*** — vanilla `torch.empty(n).uniform_(generator=g)` /
   `torch.quasirandom.SobolEngine(...)` on the host CPU.

The first comparison (1 vs 2) answers *"did the NKI Philox kernel actually
help vs our own PyTorch fallback?"* — it uses the same trnrand call site
on both sides; only the backend dispatch differs.

The second comparison (1 vs 3) answers *"what's the user-visible
difference between trnrand and reaching for raw torch?"*

## Methodology

- **Hardware**: AWS `trn1.2xlarge` (1 NeuronCore-v2, 32 GB SBUF, AMD EPYC host CPU)
- **Image**: Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)
- **Neuron SDK**: `neuronxcc 2.24.5133.0`
- **Tool**: [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) with default settings (calibration + 5+ rounds, reports median)
- **NKI warmup**: each NKI test does an explicit warmup call before the timed loop so kernel compilation cost is excluded from steady-state numbers
- **Reproduce**: `AWS_PROFILE=aws ./scripts/run_benchmarks.sh`

## Caveats

- **Small sizes are dispatch-bound.** NKI kernel invocation has fixed overhead. Below some per-op threshold the host CPU wins for RNG, where each call already amortizes to a few hundred nanoseconds per element.
- **Philox is integer-heavy.** GpSimd does the multiply-XOR rounds; the Tensor Engine isn't useful here. Don't expect Tensor-Engine-class speedups — the win comes from avoiding host→device transfer of the random tensor.
- **Quasi-random is host-only today.** Sobol/Halton/LHS run via `torch.quasirandom` on the host. NKI scrambling is a v0.3 follow-up.
- **FP32 throughout.** BF16/FP16 paths are future work.
- Numbers vary 5-15% run-to-run; treat the table as approximate.

## Results

<!-- BENCH_TABLE_START -->

_Pending first run on trn1.2xlarge — see issue #3._

<!-- BENCH_TABLE_END -->
