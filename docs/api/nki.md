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

## Environment variables

| Env var | Effect |
|---|---|
| `TRNRAND_USE_SIMULATOR=1` | Dispatch routes kernels through `nki.simulate(kernel)(numpy_args)` on CPU — bypasses `torch_xla` and hardware. Use for fast correctness iteration; no NEFF compile. |
| `TRNRAND_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to PyTorch. Used by the validation suite to catch silent kernel breakage. |
| `TRNRAND_CORES=<n>\|auto` | Default for `set_cores` (below). Off (1) by default. |
| `TRNRAND_CORES_MIN_ELEMENTS` | Requests below this (default 8,388,608) stay on one core. |
| `TRNRAND_PINNED_OUTPUT=0` | Turns off pinned host output for large native-path draws (on by default since v0.11.0). |
| `TRNRAND_KEY_MODE=draw\|counter` | Default `Generator` key mode. |

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
  `threefry_normal`, `sobol_gray_code`, `halton`,
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
