# NKI Backend

The NKI dispatch layer controls whether RNG operations run on the native
Trainium GpSimd engine (Threefry4x32-20) or fall back to `torch.Generator`.

## Backend selection

```python
import trnrand

trnrand.set_backend("auto")  # NKI on Trainium, PyTorch elsewhere (default)
trnrand.set_backend("pytorch")  # force PyTorch fallback
trnrand.set_backend("nki")  # force NKI (requires neuronxcc)
```

`trnrand.HAS_NKI` is `True` when `nki>=0.3.0` is importable.
`trnrand.get_backend()` returns the active backend name.

## Precision and dtypes

The NKI kernels draw float32 values (24-bit uniforms; every kernel operand is
an int32 word, or float32 for Box-Muller). A float64 request
(`dtype=torch.float64`, or a float64 buffer for `*_into` /
`GeneratorProgram.stream_into`) follows the trnsci suite policy (trnrand#105,
from 0.14.0):

| Mode | float64 request |
|---|---|
| `set_backend("auto")` | Drawn in torch at float64 precision: the same values as `set_backend("pytorch")`, no kernel launch. |
| `set_backend("nki")` | The float32 kernels run and the result is cast to float64; one `UserWarning` per op per process says the precision is float32. |
| `TRNRAND_REQUIRE_NKI=1` | `NotImplementedError` (either backend). |

The result dtype always matches the PyTorch backend: the requested dtype (or
the buffer's) for float32, float64, float16 and bfloat16. float16 / bfloat16
requests run on the kernels and are rounded from float32, staying inside the
documented support. Integer, bool and complex requests take the PyTorch path
(the same values or the same error as `set_backend("pytorch")`), or raise
`NotImplementedError` under `TRNRAND_REQUIRE_NKI=1`.

**Float64 buffers on the neuron device.** trnrand's allocating calls
(`uniform`, `normal`, `halton`, ...) return host tensors, so the only
device-resident float64 is a `*_into` / `GeneratorProgram.stream_into` buffer
on the `neuron` device. Device arithmetic on Trainium is float32 (measured on
Trn2: device aten computes float64 at float32 precision), so under
`set_backend("auto")` such a buffer is filled on the host with the host
`torch.Generator` at float64 precision and the values are copied into the
device buffer, which keeps its float64 dtype. The values are the same as for
a host buffer with the same generator state (and, for `*_into`, the same as
`set_backend("pytorch")`). Each op emits one `UserWarning` per process:
"float64 on the neuron device is computed on the host for float64 accuracy
(device arithmetic is fp32); pass float32 to stay on device". A float32 buffer
stays on the device kernels, and host float64 buffers are filled in place as
before, without a warning.

`TRNRAND_FP64_ON_DEVICE=1` opts out for callers who prefer speed: a float64
buffer on the neuron device keeps the on-device float32 kernels under
`"auto"` (float32 precision, cast to float64; no host round trip, no warning).
Without the NKI kernels there is no on-device route (the generator is a host
`torch.Generator`), so the buffer is filled on the host either way.
`TRNRAND_REQUIRE_NKI=1` still raises for float64.

Owen-scrambled `sobol` (the default, `scramble=True`) follows this policy:
the PyTorch path holds its 32-bit points in float64, the `sobol_owen` kernel
the top 24 bits in float32. Unscrambled and `scramble="shift"` `sobol` are
the exception: the PyTorch backend's float64 Sobol there is a float32
`SobolEngine` draw cast to float64, so such a float64 request keeps the NKI
kernel on every backend, without a warning or a `TRNRAND_REQUIRE_NKI` error.

Before 0.14.0, float64 on the NKI path (including `"auto"` on a Neuron host)
was always the cast float32 kernel output, with no warning.

## Environment variables

| Env var | Effect |
|---|---|
| `TRNRAND_USE_SIMULATOR=1` | Dispatch routes kernels through `nki.simulate(kernel)(numpy_args)` on CPU — bypasses `torch_xla` and hardware. Use for fast correctness iteration; no NEFF compile. |
| `TRNRAND_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to PyTorch, and requests the kernels cannot serve at full precision (float64, integer, bool, complex) raise `NotImplementedError` (see [Precision and dtypes](#precision-and-dtypes)). Used by the validation suite to catch silent kernel breakage. |
| `TRNRAND_FP64_ON_DEVICE=1` | Under `set_backend("auto")`, a float64 `*_into` / `GeneratorProgram` buffer on the neuron device keeps the on-device float32 kernels (float32 precision) instead of being filled on the host at float64 precision (see [Precision and dtypes](#precision-and-dtypes)). |
| `TRNRAND_CORES=<n>\|auto` | Default for `set_cores` (below). Off (1) by default. |
| `TRNRAND_CORES_MIN_ELEMENTS` | Requests below this (default 8,388,608) stay on one core. |
| `TRNRAND_PINNED_OUTPUT=0` | Turns off pinned host output for large native-path draws (on by default since v0.11.0). |
| `TRNRAND_KEY_MODE=draw\|counter` | Default `Generator` key mode. |
| `TRNSCI_NKI_DISPATCH=auto\|native\|xla` | Suite-wide. Forces the native or XLA dispatch path; `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | Suite-wide, off by default. The XLA path emits one `DeprecationWarning`. |

## Multi-core bulk draws

`trnrand.set_cores(n | "auto" | None)` / `trnrand.get_cores()` (v0.10.0,
opt-in) spread large native-path `uniform()` / `normal()` draws over
several logical NeuronCores of one chip. Output is bit-identical for every
core count. See [Architecture](../architecture.md#multi-core-bulk-draws-trn2-opt-in).

## Dispatch path

On Neuron hardware, kernels run on one of two `torch-neuronx` lineages,
detected on the first kernel launch:

- **Native (TorchNeuron, Trn2/Trn3)** — each kernel is a PyTorch custom op
  on `device="neuron"`, also callable directly and inside
  `torch.compile(backend="neuron")`:
  `torch.ops.trnrand.philox4x32`, `box_muller`, `threefry4x32`,
  `threefry_normal`, `sobol_gray_code`, `sobol_owen`, `halton`,
  `threefry_streaming_normal`, `threefry_streaming_uniform`, and the bulk
  kernels `threefry_bulk_normal` / `threefry_bulk_uniform` and their
  full-word `threefry_bulk32_*` variants. The ops are registered only once
  this path is selected.
- **XLA (Trn1)** — kernels run on the `torch_xla` device, as before.

`set_backend` is unaffected: it still chooses NKI vs `torch.Generator`.
See [Installation](../installation.md#two-neuron-pytorch-lineages).

See [Developing NKI kernels](../developing_kernels.md) for the simulator
vs hardware workflow.

## Threefry and Philox kernels

The kernels live in `trnrand/nki/dispatch.py`. The strategy:

- Counter-based — `(counter, key) → output`, no shared state across tiles.
- Each tile gets a disjoint counter range and runs its rounds on the
  GpSimd engine.
- Every NKI distribution draws from Threefry4x32-20 (add-rotate-xor, no
  integer multiply), with Box-Muller fused into the same kernel for
  `normal()`. Philox 4×32-10 (cuRAND's default engine) ships as a separate
  kernel, `torch.ops.trnrand.philox4x32`.

**Status:** both are bit-exact against their CPU references on Trn2
(Neuron SDK 2.33, native TorchNeuron). The Philox output corruption was
saturating int32/uint32 casts, fixed in v0.8.0 (#56). On Trn1 the fused
Box-Muller kernel hits a compile restriction (`NCC_IBIR605`) that doesn't
apply to trn2+.
