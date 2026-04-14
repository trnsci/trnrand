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
import os
import warnings

import numpy as np
import torch

try:
    import nki
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

# When set, kernel-path failures re-raise instead of falling back to the
# PyTorch reference path. Used by the validation suite to catch silent
# kernel breakage during iteration — mirrors TRNBLAS_REQUIRE_NKI etc.
_REQUIRE_NKI = os.environ.get("TRNRAND_REQUIRE_NKI", "").lower() in ("1", "true", "yes")

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


class NkiFallbackWarning(UserWarning):
    """Emitted when an NKI kernel path fails and we fall back to PyTorch."""


def _warn_fallback(exc: Exception) -> None:
    warnings.warn(
        f"NKI path failed ({type(exc).__name__}: {exc}); falling back to PyTorch. "
        "Set TRNRAND_REQUIRE_NKI=1 to re-raise instead.",
        NkiFallbackWarning,
        stacklevel=3,
    )


def _to_xla(*tensors):
    """Move tensors to the XLA device for NKI kernel dispatch.

    Importantly, importing `torch_xla` here also fully registers
    `torch_neuronx` in `sys.modules` — without that,
    `neuronxcc.nki._torch_xla` raises `KeyError: 'torch_neuronx'` on the
    first kernel call.
    """
    import torch_xla

    device = torch_xla.device()
    orig = tensors[0].device
    return [t.to(device) for t in tensors], orig


# NKI partition axis is limited to 128 lanes on trn1/trn2. The host wrapper
# tiles inputs into 128-lane chunks before dispatch.
_PHILOX_LANES_PER_TILE = 128


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
# Canonical uint32 values — used by both the CPU reference and the NKI
# kernel. All multiplies in the kernel promote operands to int64 so these
# constants fit cleanly (they exceed INT32_MAX but are well under
# INT64_MAX).
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
    def philox4x32_kernel(counter_lo_ref, key_lo_ref, key_hi_ref):
        """Philox 4×32-10 NKI kernel on GpSimd.

        Each of the P partition-axis lanes runs one independent Philox
        stream. Input shape is `(P, 1)` per @nki.jit / NKI convention
        (partition, free). Output is `(P, 4)` — one int32 per round-output
        word per lane. The wrapper flattens to `(P*4,)` interleaved.

        Status: scaffolded against the Philox 4×32-10 spec. CPU reference
        in `philox4x32_reference` is the conformance oracle.
        """
        c0 = nl.load(counter_lo_ref)
        c1 = nl.zeros_like(c0)
        c2 = nl.zeros_like(c0)
        c3 = nl.zeros_like(c0)
        k0 = nl.load(key_lo_ref)
        k1 = nl.load(key_hi_ref)

        # 10 rounds of Philox. The multiply needs 32×32→64-bit semantics;
        # NKI int32 multiplies sign-extend the result, so we promote
        # operands to int64 and mask to uint32-range before the multiply.
        # That gives a non-negative int64 product whose high/low 32 bits
        # we can extract cleanly with shift+mask. Cast back to int32 for
        # the XOR/store path.
        for _ in nl.static_range(PHILOX_ROUNDS):
            c0_u = nl.bitwise_and(nl.cast(c0, dtype=nl.int64), UINT32_MASK)
            c2_u = nl.bitwise_and(nl.cast(c2, dtype=nl.int64), UINT32_MASK)

            prod0 = nl.multiply(c0_u, PHILOX_M0)
            prod1 = nl.multiply(c2_u, PHILOX_M1)

            hi0_i64 = nl.bitwise_and(nl.right_shift(prod0, 32), UINT32_MASK)
            lo0_i64 = nl.bitwise_and(prod0, UINT32_MASK)
            hi1_i64 = nl.bitwise_and(nl.right_shift(prod1, 32), UINT32_MASK)
            lo1_i64 = nl.bitwise_and(prod1, UINT32_MASK)

            hi0 = nl.cast(hi0_i64, dtype=nl.int32)
            lo0 = nl.cast(lo0_i64, dtype=nl.int32)
            hi1 = nl.cast(hi1_i64, dtype=nl.int32)
            lo1 = nl.cast(lo1_i64, dtype=nl.int32)

            # Bitwise XOR is sign-agnostic; int32 two's complement is fine.
            new_c0 = nl.bitwise_xor(nl.bitwise_xor(hi1, c1), k0)
            new_c1 = lo1
            new_c2 = nl.bitwise_xor(nl.bitwise_xor(hi0, c3), k1)
            new_c3 = lo0
            c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3

            # Key bump in int64 so the W0/W1 constants (which exceed
            # INT32_MAX) don't overflow the int32 add.
            k0_u = nl.bitwise_and(nl.add(nl.cast(k0, dtype=nl.int64), PHILOX_W0), UINT32_MASK)
            k1_u = nl.bitwise_and(nl.add(nl.cast(k1, dtype=nl.int64), PHILOX_W1), UINT32_MASK)
            k0 = nl.cast(k0_u, dtype=nl.int32)
            k1 = nl.cast(k1_u, dtype=nl.int32)

        P = counter_lo_ref.shape[0]
        out = nl.ndarray((P, 4), dtype=counter_lo_ref.dtype, buffer=nl.shared_hbm)
        out[:, 0:1] = c0
        out[:, 1:2] = c1
        out[:, 2:3] = c2
        out[:, 3:4] = c3
        return out

    @nki.jit
    def box_muller_kernel(uniforms_ref):
        """Box-Muller transform on the Vector Engine: U(0,1) pairs → N(0,1).

        Input shape `(P, 2)` — P lanes, each holding (u1, u2). Output
        shape `(P, 2)` — paired standard normals (z1, z2). Runs
        SBUF-resident on Vector Engine using hardware cos/sin/log/sqrt.

        Box-Muller chosen over Marsaglia polar: Marsaglia's rejection
        step serializes branch-divergent lanes, killing SIMD throughput.
        Box-Muller has constant work per pair.
        """
        pairs = nl.load(uniforms_ref)
        u1 = pairs[:, 0:1]
        u2 = pairs[:, 1:2]
        # Clamp u1 away from 0 to avoid log(0) = -inf.
        u1_safe = nl.maximum(u1, 1e-10)
        r = nl.sqrt(nl.multiply(nl.log(u1_safe), -2.0))
        theta = nl.multiply(u2, TWO_PI)
        z1 = nl.multiply(r, nl.cos(theta))
        z2 = nl.multiply(r, nl.sin(theta))

        P = uniforms_ref.shape[0]
        out = nl.ndarray((P, 2), dtype=uniforms_ref.dtype, buffer=nl.shared_hbm)
        out[:, 0:1] = z1
        out[:, 1:2] = z2
        return out

    def philox4x32_nki(
        counter_lo: torch.Tensor,
        key_lo: torch.Tensor,
        key_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Host-side wrapper for the Philox NKI kernel.

        NKI's partition axis is capped at 128 lanes on trn1/trn2. For
        larger lane counts we tile the input into 128-lane chunks,
        dispatch the kernel per chunk, and concatenate outputs. The
        output layout per chunk is `[c0, c1, c2, c3]` interleaved per
        lane (4 uint32 words → 4 * lanes int32 values per chunk).

        Importing `torch_xla` inside `_to_xla` also populates
        `sys.modules['torch_neuronx']` before the first kernel invocation.

        Args:
            counter_lo: int32 per-lane counter increment, shape `(lanes,)`.
            key_lo:     int32 key low word, shape `(lanes,)`.
            key_hi:     int32 key high word, shape `(lanes,)`.

        Returns:
            int32 tensor of shape `(lanes * 4,)` holding the Philox output.
        """
        n_lanes = counter_lo.shape[0]
        counter_lo = counter_lo.to(torch.int32).contiguous()
        key_lo = key_lo.to(torch.int32).contiguous()
        key_hi = key_hi.to(torch.int32).contiguous()

        chunks = []
        for start in range(0, n_lanes, _PHILOX_LANES_PER_TILE):
            end = min(start + _PHILOX_LANES_PER_TILE, n_lanes)
            cl_tile = counter_lo[start:end].reshape(-1, 1).contiguous()
            kl_tile = key_lo[start:end].reshape(-1, 1).contiguous()
            kh_tile = key_hi[start:end].reshape(-1, 1).contiguous()

            if _use_simulator():
                # CPU path: feed NumPy directly to nki.simulate(kernel).
                out_np = nki.simulate(philox4x32_kernel)(
                    cl_tile.cpu().numpy(),
                    kl_tile.cpu().numpy(),
                    kh_tile.cpu().numpy(),
                )
                out_tile = torch.from_numpy(np.asarray(out_np)).reshape(-1)
            else:
                (cl, kl, kh), orig = _to_xla(cl_tile, kl_tile, kh_tile)
                # Kernel returns (tile_len, 4); flatten to interleaved
                # [c0_0, c1_0, c2_0, c3_0, c0_1, c1_1, ...].
                out_tile = philox4x32_kernel(cl, kl, kh).reshape(-1).to(orig)
            chunks.append(out_tile)
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

    def box_muller_nki(uniforms: torch.Tensor) -> torch.Tensor:
        """Host-side wrapper for the Box-Muller NKI kernel.

        Same 128-lane partition-axis constraint as Philox. The input is
        tiled into chunks of 2 * 128 uniforms per dispatch (each
        emitting 2 * 128 normals via Box-Muller pairing).
        """
        assert uniforms.numel() % 2 == 0, "Box-Muller needs pairs"
        uniforms = uniforms.to(torch.float32).contiguous()
        n_pairs = uniforms.numel() // 2
        tile_pairs = _PHILOX_LANES_PER_TILE  # one pair per "lane" for symmetry

        chunks = []
        for start_pair in range(0, n_pairs, tile_pairs):
            end_pair = min(start_pair + tile_pairs, n_pairs)
            # Reshape flat uniforms to (tile_len, 2) — partition axis is lanes,
            # free axis is the (u1, u2) pair.
            tile_slice = uniforms[start_pair * 2 : end_pair * 2].reshape(-1, 2).contiguous()

            if _use_simulator():
                out_np = nki.simulate(box_muller_kernel)(tile_slice.cpu().numpy())
                out_tile = torch.from_numpy(np.asarray(out_np)).reshape(-1)
            else:
                (u,), orig = _to_xla(tile_slice)
                # Kernel returns (tile_len, 2); flatten to interleaved [z1, z2, z1, z2, ...].
                out_tile = box_muller_kernel(u).reshape(-1).to(orig)
            chunks.append(out_tile)
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]
