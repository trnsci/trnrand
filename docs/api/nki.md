# NKI Backend

The NKI dispatch layer controls whether RNG operations run on the native
Trainium GpSimd engine (Philox 4×32) or fall back to `torch.Generator`.

## Backend selection

```python
import trnrand

trnrand.set_backend("auto")     # NKI on Trainium, PyTorch elsewhere (default)
trnrand.set_backend("pytorch")  # force PyTorch fallback
trnrand.set_backend("nki")      # force NKI (requires neuronxcc)
```

`trnrand.HAS_NKI` is `True` when `nki>=0.3.0` is importable.
`trnrand.get_backend()` returns the active backend name.

## Environment variables

| Env var | Effect |
|---|---|
| `TRNRAND_USE_SIMULATOR=1` | Dispatch routes kernels through `nki.simulate(kernel)(numpy_args)` on CPU — bypasses `torch_xla` and hardware. Use for fast correctness iteration; no NEFF compile. |
| `TRNRAND_REQUIRE_NKI=1` | Kernel-path failures re-raise instead of silently falling back to PyTorch. Used by the validation suite to catch silent kernel breakage. |

See [Developing NKI kernels](../developing_kernels.md) for the simulator
vs hardware workflow.

## Philox kernel

The NKI Philox kernel lives in `trnrand/nki/dispatch.py`. The strategy:

- Counter-based — `(counter, key) → output`, no shared state across tiles.
- Each tile gets a disjoint counter range and runs the multiply-XOR rounds
  on the GpSimd engine.
- Same engine used by cuRAND and JAX.

**Status:** migrated to NKI 0.3.0 namespace. Philox compiles + executes
on trn1 hardware but output has an algorithmic bug under investigation
(see [#1](https://github.com/trnsci/trnrand/issues/1) /
[#26](https://github.com/trnsci/trnrand/issues/26)). Box-Muller passes
the simulator; hits a trn1 compile restriction (`NCC_IBIR605`) that
doesn't apply to trn2+. All `trnrand.*` generation falls back to
`torch.Generator` by default until the kernel ships.
