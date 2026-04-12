"""
NKI dispatch for random number generation.

The Philox counter-based RNG is the primary NKI target:
- Stateless: (counter, key) → deterministic output
- Parallel: each tile generates independently with disjoint counters
- Used by PyTorch (cuRAND) and JAX as default engine

On Trainium, Philox runs on the GpSimd engine (general-purpose SIMD)
since it's integer arithmetic, not matmul. The Tensor Engine isn't
useful here — RNG is compute-light, bandwidth-heavy.

For scientific workloads, on-device RNG avoids the host→device transfer
of random tensors, which can be significant for large Monte Carlo runs.
"""

from __future__ import annotations

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

_backend = "auto"


def set_backend(backend: str):
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires neuronxcc")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


if HAS_NKI:

    @nki.jit
    def philox_kernel(counter_ref, key_ref, out_ref, n: int):
        """Philox 4×32 counter-based RNG kernel.

        Each invocation: (counter, key) → 4 uint32 outputs
        Outputs converted to float32 in [0, 1) via multiply by 2^{-32}.

        STUB: Scaffolded for on-hardware validation.

        The Philox round function:
            1. Multiply: hi, lo = mulhilo32(counter[0], PHILOX_M0)
            2. XOR: counter[0] = hi ^ key[0] ^ counter[1]
            3. Swap and increment key
            4. Repeat for 10 rounds
        """
        TILE = 128
        for tile_start in nl.affine_range(n // TILE):
            offset = tile_start * TILE
            ctr = nl.load(counter_ref[offset:offset + TILE])
            # Placeholder: actual Philox rounds would go here
            # For now, just pass through (NOT random — stub only)
            nl.store(out_ref[offset:offset + TILE], ctr)
