"""
NKI dispatch for random number generation.

The Philox 4×32 counter-based RNG is the primary NKI target:
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

import math

import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

TWO_PI = 2.0 * math.pi

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


# Philox 4×32-10 constants (Salmon et al., SC'11; matches cuRAND, JAX).
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_ROUNDS = 10
UINT32_MASK = 0xFFFFFFFF


def philox4x32_reference(counter: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """CPU reference implementation of Philox 4×32-10.

    Used as the conformance oracle for the NKI kernel.

    Args:
        counter: int64 tensor of shape (..., 4) — the 4 counter words per stream.
        key:     int64 tensor of shape (..., 2) — the 2 key words per stream.

    Returns:
        int64 tensor of shape (..., 4) holding the 4 output uint32 words
        (stored in the low 32 bits of int64 for arithmetic safety).
    """
    assert counter.shape[-1] == 4 and key.shape[-1] == 2
    c0 = counter[..., 0] & UINT32_MASK
    c1 = counter[..., 1] & UINT32_MASK
    c2 = counter[..., 2] & UINT32_MASK
    c3 = counter[..., 3] & UINT32_MASK
    k0 = key[..., 0] & UINT32_MASK
    k1 = key[..., 1] & UINT32_MASK

    for _ in range(PHILOX_ROUNDS):
        prod0 = c0 * PHILOX_M0
        prod1 = c2 * PHILOX_M1
        hi0 = (prod0 >> 32) & UINT32_MASK
        lo0 = prod0 & UINT32_MASK
        hi1 = (prod1 >> 32) & UINT32_MASK
        lo1 = prod1 & UINT32_MASK
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & UINT32_MASK,
            lo1 & UINT32_MASK,
            (hi0 ^ c3 ^ k1) & UINT32_MASK,
            lo0 & UINT32_MASK,
        )
        k0 = (k0 + PHILOX_W0) & UINT32_MASK
        k1 = (k1 + PHILOX_W1) & UINT32_MASK

    return torch.stack([c0, c1, c2, c3], dim=-1)


def box_muller_cpu(uniforms: torch.Tensor) -> torch.Tensor:
    """Box-Muller transform: pairs of uniforms → standard-normal pairs.

    Same algorithm as the Vector Engine kernel — kept on CPU so the
    `normal_nki` path can be tested for distributional correctness without
    Trainium hardware.

    Args:
        uniforms: even-length 1-D tensor of U(0, 1) samples.

    Returns:
        Tensor of the same shape, holding standard-normal samples.
    """
    assert uniforms.numel() % 2 == 0, "Box-Muller needs pairs"
    u = uniforms.view(-1, 2)
    # Clamp u1 away from 0 to avoid log(0) = -inf.
    u1 = torch.clamp(u[:, 0], min=1e-10)
    u2 = u[:, 1]
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = TWO_PI * u2
    z = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    return z.reshape(uniforms.shape)


def philox_uniform_cpu(
    n_elements: int,
    seed: int,
    counter_offset: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CPU Philox uniform stream — used as fallback and as the oracle for tests.

    Generates `n_elements` floats in [0, 1) by emitting `ceil(n/4)` Philox
    blocks with sequential counters starting at `counter_offset`.
    """
    n_blocks = (n_elements + 3) // 4
    counters = torch.zeros(n_blocks, 4, dtype=torch.int64)
    counters[:, 0] = torch.arange(counter_offset, counter_offset + n_blocks, dtype=torch.int64)
    key = torch.tensor([seed & UINT32_MASK, (seed >> 32) & UINT32_MASK], dtype=torch.int64)
    key = key.expand(n_blocks, 2)
    out_u32 = philox4x32_reference(counters, key).reshape(-1)[:n_elements]
    # uint32 → float32 in [0, 1) via mantissa-only conversion (cuRAND convention).
    return (out_u32.to(torch.float64) * (1.0 / 2**32)).to(dtype)


if HAS_NKI:

    @nki.jit
    def philox4x32_kernel(counter_lo_ref, key_lo_ref, key_hi_ref, out_ref):
        """Philox 4×32-10 NKI kernel running on GpSimd.

        Each lane processes one independent Philox stream:
        - `counter_lo` (per-lane) is the 32-bit counter increment; the
          remaining counter words start at zero. This is enough to give
          each tile a disjoint 2^96-element stream.
        - `key_lo`, `key_hi` are the two key words (broadcast across lanes).

        Output: 4 uint32 words per lane, written contiguously to out_ref.

        Status: scaffolded against the Philox 4×32-10 spec but not yet
        validated on trn1/trn2. The CPU reference in `philox4x32_reference`
        is the conformance oracle (see tests/test_nki_philox.py).
        """
        # Load one tile of counter_lo values (one per lane); other counter
        # words start at 0 — they get written by the round function.
        c0 = nl.load(counter_lo_ref)
        c1 = nl.zeros_like(c0)
        c2 = nl.zeros_like(c0)
        c3 = nl.zeros_like(c0)
        k0 = nl.load(key_lo_ref)
        k1 = nl.load(key_hi_ref)

        # 10 rounds of the Philox round function.
        # NKI mul-hi-lo: result is i64; high 32 / low 32 isolated by shift+mask.
        for _ in nl.static_range(PHILOX_ROUNDS):
            prod0 = nl.multiply(c0, PHILOX_M0)  # i64
            prod1 = nl.multiply(c2, PHILOX_M1)  # i64
            hi0 = nl.bitwise_and(nl.right_shift(prod0, 32), UINT32_MASK)
            lo0 = nl.bitwise_and(prod0, UINT32_MASK)
            hi1 = nl.bitwise_and(nl.right_shift(prod1, 32), UINT32_MASK)
            lo1 = nl.bitwise_and(prod1, UINT32_MASK)

            new_c0 = nl.bitwise_xor(nl.bitwise_xor(hi1, c1), k0)
            new_c1 = lo1
            new_c2 = nl.bitwise_xor(nl.bitwise_xor(hi0, c3), k1)
            new_c3 = lo0
            c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3

            k0 = nl.bitwise_and(nl.add(k0, PHILOX_W0), UINT32_MASK)
            k1 = nl.bitwise_and(nl.add(k1, PHILOX_W1), UINT32_MASK)

        # Write 4 output words per lane interleaved.
        nl.store(out_ref[0::4], c0)
        nl.store(out_ref[1::4], c1)
        nl.store(out_ref[2::4], c2)
        nl.store(out_ref[3::4], c3)

    @nki.jit
    def box_muller_kernel(uniforms_ref, out_ref):
        """Box-Muller transform on the Vector Engine: U(0,1) pairs → N(0,1) pairs.

        Pairs of uniform inputs (u1, u2) → (z1, z2) with
            r = sqrt(-2 ln u1),  θ = 2π u2,
            z1 = r cos θ,  z2 = r sin θ.

        Chosen over Marsaglia polar because the Vector Engine has hardware
        cos/sin/log/sqrt; Marsaglia's rejection step kills SIMD throughput.

        Status: scaffolded but awaits on-hardware validation per #2.
        Iterate after #1 (Philox kernel) lands cleanly on trn1.
        """
        u1 = nl.load(uniforms_ref[0::2])
        u2 = nl.load(uniforms_ref[1::2])
        # Clamp u1 away from 0 to avoid log(0) = -inf.
        u1_safe = nl.maximum(u1, 1e-10)
        r = nl.sqrt(nl.multiply(nl.log(u1_safe), -2.0))
        theta = nl.multiply(u2, TWO_PI)
        z1 = nl.multiply(r, nl.cos(theta))
        z2 = nl.multiply(r, nl.sin(theta))
        nl.store(out_ref[0::2], z1)
        nl.store(out_ref[1::2], z2)
