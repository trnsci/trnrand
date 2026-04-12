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

`trnrand.HAS_NKI` is `True` when `neuronxcc` is importable.
`trnrand.get_backend()` returns the active backend name.

## Philox kernel

The NKI Philox kernel lives in `trnrand/nki/dispatch.py`. The strategy:

- Counter-based — `(counter, key) → output`, no shared state across tiles.
- Each tile gets a disjoint counter range and runs the multiply-XOR rounds
  on the GpSimd engine.
- Same engine used by cuRAND and JAX.

**Status:** scaffolded but not yet validated on trn1/trn2 hardware. All
generation falls back to `torch.Generator` until the kernel ships. See the
roadmap issues for on-hardware validation work.
