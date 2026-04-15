# Developing NKI kernels

trnrand ships two NKI kernels — `philox4x32_kernel` (GpSimd) and
`box_muller_kernel` (Vector Engine) — in `trnrand/nki/dispatch.py`.
This page is for contributors designing / debugging them.

For the suite-wide pattern (shared across trnfft, trnblas, trnrand,
trnsolver, trnsparse, trntensor), see
[`trnsci/docs/developing_kernels.md`](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md).
That guide covers the NKI 0.3.0 migration, simulator workflow, and
architectural-exploitation design discipline. This page adds only
what's trnrand-specific.

## Three dispatch modes

| Mode | Trigger | When to use |
|------|---------|-------------|
| **PyTorch fallback** | `HAS_NKI = False` (e.g. macOS), or an `_nki_*` exception caught by the wrapper | Laptops, CI's `ubuntu-latest` main `test` job — the default for anyone who doesn't have `nki>=0.3.0` installed. |
| **NKI hardware** | `HAS_NKI = True` + default env. Kernel runs through `torch_xla` → NEFF compile → Trainium dispatch | Real perf numbers, final validation (`AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1`). |
| **NKI simulator** | `TRNRAND_USE_SIMULATOR=1` + `HAS_NKI = True`. Kernel runs through `nki.simulate(kernel)(numpy_args)` on CPU | Fast correctness iteration during kernel design — seconds per cycle vs minutes on hardware. |

## Environment variables

| Env var | Effect |
|---|---|
| `TRNRAND_USE_SIMULATOR=1` | Dispatch bypasses `torch_xla` and runs kernels through `nki.simulate(kernel)(numpy_args)` on CPU. |
| `TRNRAND_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to the PyTorch reference. Used by the validation suite to catch silent kernel breakage during iteration. |

## Simulator workflow

Fastest inner loop for kernel correctness work:

```bash
# On any Linux x86_64 host with nki>=0.3.0 installed:
TRNRAND_USE_SIMULATOR=1 pytest tests/ -m nki_simulator -v

# Against the provisioned trn1 DLAMI via SSM (no local nki install needed):
AWS_PROFILE=aws ./scripts/run_simulator_tests.sh
```

Both routes run the same `tests/test_nki_sim.py` suite (Philox spec
vectors, reference equivalence, Box-Muller distribution).

## Hardware workflow

Final validation only. Simulator is the inner loop.

```bash
# All neuron-marked tests on trn1:
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn1

# Just Philox (skip Box-Muller — useful when its compile is blocked):
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh --philox-only trn1

# Warm pass to expose NEFF cache reuse:
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh --warm trn1
```

The script tags match `trnrand-ci-trn1`; terraform at `infra/terraform/`
provisions. Instance auto-stops on exit.

## NKI 0.3.0 gotchas specific to trnrand

- **No int64 / uint64.** Max integer dtype is 32 bits (`uint32`, `int32`,
  etc.). The 32×32→64 Philox multiply decomposes into four 16×16→32
  sub-multiplies that each stay within uint32 range; see
  `_mul32_hi_lo` in `trnrand/nki/dispatch.py`.
- **`nl.multiply` output dtype promotion.** Products that can exceed
  `INT32_MAX` (e.g., `0xFFFF * 0xD251 = 0xD2503DAF`) need explicit
  `dtype=nl.uint32` on the multiply to avoid sign extension.
- **`nl.copy(x, dtype=...)` is the cast primitive.** NKI 0.3.0's `nl.*`
  has neither `cast` nor `static_cast`.
- **trn1-specific**: `InstActivation` bias cannot be a scalar immediate
  for `Ln` — must be a vector-immediate `(P, 1)` tensor. Affects
  Box-Muller's `nl.log + scalar` fusion on trn1. trn2+ unaffected.
  See [aws-neuron-sdk NCC_IBIR605](https://github.com/aws-neuron/aws-neuron-sdk/issues?q=NCC_IBIR605).

## See also

- [Suite-wide kernel development guide](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md)
- [trnblas developing_kernels.md](https://github.com/trnsci/trnblas/blob/main/docs/developing_kernels.md) — reference implementation for GEMM kernel patterns.
- [Phase 1 trnrand tracker (#18)](https://github.com/trnsci/trnrand/issues/18), [NKI 0.3.0 migration (#26)](https://github.com/trnsci/trnrand/issues/26).
