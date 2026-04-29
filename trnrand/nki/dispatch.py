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


def _mul32_hi_lo_numpy(a, b_l: int, b_h: int):
    """Pure-numpy reimplementation of `_mul32_hi_lo` — used for debugging.

    Mirrors the NKI kernel line-by-line with an 8-bit byte decomposition.
    Every intermediate is ≤ 2^18, staying within float32's 2^24 exact-
    integer ceiling — which matters because NKI's `nl.multiply` routes
    through the activation engine's float path on both simulator and
    hardware.

    Input `a` is a uint32 numpy array of any shape. `b_l`, `b_h` are
    the low/high 16 bits of the full uint32 multiplier. Returns
    `(hi32, lo32)` int32 numpy arrays with the same shape as `a`.
    """
    import numpy as np

    # Split b = b_h<<16 | b_l into four 8-bit bytes b0..b3 (low → high).
    b0 = np.uint32(b_l & 0xFF)
    b1 = np.uint32((b_l >> 8) & 0xFF)
    b2 = np.uint32(b_h & 0xFF)
    b3 = np.uint32((b_h >> 8) & 0xFF)

    a_u = a.astype(np.uint32)
    a0 = np.bitwise_and(a_u, np.uint32(0xFF))
    a1 = np.bitwise_and(np.right_shift(a_u, np.uint32(8)), np.uint32(0xFF))
    a2 = np.bitwise_and(np.right_shift(a_u, np.uint32(16)), np.uint32(0xFF))
    a3 = np.right_shift(a_u, np.uint32(24))  # a3 already ≤ 0xFF

    # 16 sub-products, each ≤ 0xFF·0xFF = 65025.
    def mul(x, y):
        return np.multiply(x, y, dtype=np.uint32)

    def add(*args):
        out = args[0]
        for x in args[1:]:
            out = np.add(out, x, dtype=np.uint32)
        return out

    # Column sums at shift 8k, k = 0..6. Each ≤ 4 × 65025 ≈ 2^18.
    c0 = mul(a0, b0)
    c1 = add(mul(a0, b1), mul(a1, b0))
    c2 = add(mul(a0, b2), mul(a1, b1), mul(a2, b0))
    c3 = add(mul(a0, b3), mul(a1, b2), mul(a2, b1), mul(a3, b0))
    c4 = add(mul(a1, b3), mul(a2, b2), mul(a3, b1))
    c5 = add(mul(a2, b3), mul(a3, b2))
    c6 = mul(a3, b3)

    # Byte-wise carry propagation. `acc` stays ≤ 2^18 + 2^10 ≈ 2^18.
    def step(acc, col):
        byte = np.bitwise_and(acc, np.uint32(0xFF))
        carry = np.right_shift(acc, np.uint32(8))
        return byte, np.add(carry, col, dtype=np.uint32)

    byte0, acc = step(c0, c1)
    byte1, acc = step(acc, c2)
    byte2, acc = step(acc, c3)
    byte3, acc = step(acc, c4)
    byte4, acc = step(acc, c5)
    byte5, acc = step(acc, c6)
    byte6 = np.bitwise_and(acc, np.uint32(0xFF))
    byte7 = np.bitwise_and(np.right_shift(acc, np.uint32(8)), np.uint32(0xFF))

    lo32_u = np.bitwise_or(
        np.bitwise_or(byte0, np.left_shift(byte1, np.uint32(8))),
        np.bitwise_or(
            np.left_shift(byte2, np.uint32(16)),
            np.left_shift(byte3, np.uint32(24)),
        ),
    )
    hi32_u = np.bitwise_or(
        np.bitwise_or(byte4, np.left_shift(byte5, np.uint32(8))),
        np.bitwise_or(
            np.left_shift(byte6, np.uint32(16)),
            np.left_shift(byte7, np.uint32(24)),
        ),
    )

    # Same bit pattern reinterpreted as int32 (matches NKI's nl.copy cast).
    return hi32_u.astype(np.int32), lo32_u.astype(np.int32)


# ── Threefry 4×32-20 ──────────────────────────────────────────────────────────
#
# Threefry4×32-20 (Salmon et al. SC'11, same paper as Philox) is the correct
# algorithm when fast integer multiply is unavailable — designed explicitly for
# FPGAs and embedded processors. Uses only: 32-bit addition, XOR, rotation.
# All three operations decompose into byte arithmetic where every intermediate
# stays ≤ 511 < 2^10, far below float32's 2^24 exact-integer ceiling.
# This is the architecturally correct choice for Trainium while aws-neuron-sdk#1308
# (NKI has no true GpSimd integer ops yet) remains open.
#
# Reference: Salmon, Moraes, Dror, Shaw — "Parallel Random Numbers: As Easy as
# 1, 2, 3", SC'11, https://doi.org/10.1145/2063384.2063405
# Test vectors: Random123 library (DE Shaw Research)

THREEFRY_SKEIN_KS_PARITY = 0x1BD11BDA  # Skein key schedule parity constant
THREEFRY_ROUNDS = 20
# Rotation constants for Threefry4×32-20 — 4 pairs, cycling every 4 rounds.
# From Table 4, Salmon et al. SC'11 (Skein hash variant used by Random123).
THREEFRY_ROTATIONS = [(10, 26), (11, 21), (13, 27), (23, 5)]

# Streaming kernel tile count — exported at module level so non-NKI hosts can
# import it for test parameterisation without requiring neuronxcc.
_PROGRAM_TILES = 32  # tiles per streaming launch; 32 × 128 × 4 = 16,384 samples/call

# Normal-approximation threshold for Poisson NKI path.
# For lam >= 20, skewness 1/sqrt(lam) <= 0.22 and the normal approximation
# round(N(lam, sqrt(lam))) produces tail probabilities within ~1% of exact.
# Exported at module level so distributions.py can check lam without NKI.
_POISSON_NORMAL_THRESHOLD = 20.0


def threefry4x32_reference(counter: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """CPU reference implementation of Threefry4×32-20.

    Used as the conformance oracle for the NKI kernel. Algorithm uses only
    32-bit addition, XOR, and rotation — no multiply. All ops are exact at
    any uint32 value in Python (unbounded integers, then masked to 32 bits).

    Args:
        counter: int64 tensor of shape (..., 4) — the 4 counter words.
        key:     int64 tensor of shape (..., 4) — the 4 key words.

    Returns:
        int64 tensor of shape (..., 4) holding the 4 output uint32 words
        (stored in the low 32 bits of int64 for arithmetic safety).
    """
    assert counter.shape[-1] == 4 and key.shape[-1] == 4

    M = UINT32_MASK
    x0 = counter[..., 0] & M
    x1 = counter[..., 1] & M
    x2 = counter[..., 2] & M
    x3 = counter[..., 3] & M
    k0 = key[..., 0] & M
    k1 = key[..., 1] & M
    k2 = key[..., 2] & M
    k3 = key[..., 3] & M
    k4 = (k0 ^ k1 ^ k2 ^ k3 ^ THREEFRY_SKEIN_KS_PARITY) & M

    ks = [k0, k1, k2, k3, k4]  # 5-word key schedule

    def rotl32(v, r):
        return ((v << r) | (v >> (32 - r))) & M

    def mix(a, b, rot):
        a = (a + b) & M
        b = rotl32(b, rot) ^ a
        return a, b

    def inject(x0, x1, x2, x3, step):
        x0 = (x0 + ks[(step + 0) % 5]) & M
        x1 = (x1 + ks[(step + 1) % 5]) & M
        x2 = (x2 + ks[(step + 2) % 5]) & M
        x3 = (x3 + ks[(step + 3) % 5] + step) & M
        return x0, x1, x2, x3

    # Key injection before round 0.
    x0, x1, x2, x3 = inject(x0, x1, x2, x3, 0)

    for r in range(THREEFRY_ROUNDS):
        rot_pair = THREEFRY_ROTATIONS[r % 4]
        if r % 2 == 0:
            # Even rounds: MIX(x0,x1) then MIX(x2,x3)
            x0, x1 = mix(x0, x1, rot_pair[0])
            x2, x3 = mix(x2, x3, rot_pair[1])
        else:
            # Odd rounds: MIX(x0,x3) then MIX(x2,x1)
            x0, x3 = mix(x0, x3, rot_pair[0])
            x2, x1 = mix(x2, x1, rot_pair[1])
        # Key injection after every 4th round (at rounds 3, 7, 11, 15, 19).
        if (r + 1) % 4 == 0:
            x0, x1, x2, x3 = inject(x0, x1, x2, x3, (r + 1) // 4)

    return torch.stack([x0, x1, x2, x3], dim=-1)


def threefry_uniform_cpu(
    n_elements: int,
    seed: int,
    counter_offset: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CPU Threefry uniform stream — used as fallback and oracle for tests.

    Generates `n_elements` floats in [0, 1) using Threefry4×32-20 with
    sequential counters. Seed maps to the first two key words; k2=k3=0.
    Counter: (block_index, 0, 0, 0) starting at `counter_offset`.
    """
    n_blocks = (n_elements + 3) // 4
    counters = torch.zeros(n_blocks, 4, dtype=torch.int64)
    counters[:, 0] = torch.arange(counter_offset, counter_offset + n_blocks, dtype=torch.int64)
    k0 = seed & UINT32_MASK
    k1 = (seed >> 32) & UINT32_MASK
    key = torch.tensor([k0, k1, 0, 0], dtype=torch.int64).expand(n_blocks, 4)
    out_u32 = threefry4x32_reference(counters, key).reshape(-1)[:n_elements]
    # uint32 → float32 in [0, 1) via mantissa-only conversion (cuRAND convention).
    return (out_u32.to(torch.float64) * (1.0 / 2**32)).to(dtype)


def _add32_bytes_numpy(a_u32, b_u32):
    """Pure-numpy carry-propagating 32-bit addition via byte decomposition.

    Inputs are uint32 numpy arrays of the same shape. Every intermediate
    is ≤ 511 < 2^10, within float32's exact-integer envelope.
    Returns a uint32 numpy array with the same shape as inputs.
    """
    a0 = a_u32 & np.uint32(0xFF)
    a1 = (a_u32 >> np.uint32(8)) & np.uint32(0xFF)
    a2 = (a_u32 >> np.uint32(16)) & np.uint32(0xFF)
    a3 = (a_u32 >> np.uint32(24)) & np.uint32(0xFF)
    b0 = b_u32 & np.uint32(0xFF)
    b1 = (b_u32 >> np.uint32(8)) & np.uint32(0xFF)
    b2 = (b_u32 >> np.uint32(16)) & np.uint32(0xFF)
    b3 = (b_u32 >> np.uint32(24)) & np.uint32(0xFF)

    s0 = a0.astype(np.uint32) + b0.astype(np.uint32)  # ≤ 510
    c0 = s0 >> np.uint32(8)
    r0 = s0 & np.uint32(0xFF)
    s1 = a1 + b1 + c0
    c1 = s1 >> np.uint32(8)
    r1 = s1 & np.uint32(0xFF)
    s2 = a2 + b2 + c1
    c2 = s2 >> np.uint32(8)
    r2 = s2 & np.uint32(0xFF)
    s3 = a3 + b3 + c2
    r3 = s3 & np.uint32(0xFF)  # carry out discarded (mod 2^32)

    return (r0 | (r1 << np.uint32(8)) | (r2 << np.uint32(16)) | (r3 << np.uint32(24))).astype(
        np.uint32
    )


def _rotl32_bytes_numpy(a_u32, R: int):
    """Pure-numpy 32-bit rotate-left via byte decomposition.

    Decomposed as byte-shift (q = R//8) + sub-byte rotation (r = R%8).
    The sub-byte step: hi = byte << r (≤ 32640 < 2^15), lo = byte >> (8-r).
    Every intermediate stays well below 2^24.
    Returns a uint32 numpy array with the same shape as `a_u32`.
    """
    assert 0 < R < 32, "rotation must be in (0, 32)"
    q = R // 8  # byte-level shift
    r = R % 8  # sub-byte bit shift

    # Split into 4 bytes (byte 0 = LSB).
    bytes_ = [(a_u32 >> np.uint32(8 * i)) & np.uint32(0xFF) for i in range(4)]

    if r == 0:
        # Pure byte rotation: no sub-byte work needed.
        out_bytes = [bytes_[(i - q) % 4] for i in range(4)]
    else:
        out_bytes = []
        for i in range(4):
            hi = (bytes_[(i - q) % 4].astype(np.uint32) << np.uint32(r)) & np.uint32(0xFF)
            lo = (bytes_[(i - q - 1) % 4].astype(np.uint32) >> np.uint32(8 - r)) & np.uint32(0xFF)
            out_bytes.append((hi | lo).astype(np.uint32))

    return (
        out_bytes[0]
        | (out_bytes[1] << np.uint32(8))
        | (out_bytes[2] << np.uint32(16))
        | (out_bytes[3] << np.uint32(24))
    ).astype(np.uint32)


if HAS_NKI:
    # NKI's `nl.multiply` routes uint32 operands through the activation
    # engine's float32 path on both simulator and hardware. float32
    # exactly represents integers only up to 2^24 (≈ 1.67e7), so the
    # 32×32 multiply must decompose into chunks small enough that no
    # intermediate product or sum exceeds 2^24. Byte-level (8-bit)
    # decomposition keeps sub-products ≤ 0xFF·0xFF = 65025 ≈ 2^16
    # and column sums ≤ 2^18.
    #
    # Helper must be module-level (inside the `if HAS_NKI:` block):
    # NKI rejects inner function definitions inside @nki.jit kernels.
    _PHILOX_M0_L = PHILOX_M0 & 0xFFFF  # 0x1F53
    _PHILOX_M0_H = (PHILOX_M0 >> 16) & 0xFFFF  # 0xD251
    _PHILOX_M1_L = PHILOX_M1 & 0xFFFF  # 0x8D57
    _PHILOX_M1_H = (PHILOX_M1 >> 16) & 0xFFFF  # 0xCD9E

    # Module-level NKI arithmetic helpers for `_mul32_hi_lo`.
    # NKI rejects inner function definitions inside functions called from
    # @nki.jit kernels, so all helpers must live at module scope.

    def _nki_mul_u32(x, y):
        return nl.multiply(x, y, dtype=nl.uint32)

    def _nki_add_u32(x, y):
        return nl.add(x, y, dtype=nl.uint32)

    def _nki_carry_step(acc, col):
        """Extract low byte and carry, add carry to next column."""
        byte = nl.bitwise_and(acc, 0xFF, dtype=nl.uint32)
        carry = nl.right_shift(acc, 8, dtype=nl.uint32)
        return byte, nl.add(carry, col, dtype=nl.uint32)

    def _nki_pack4_u32(b0, b1, b2, b3):
        """Pack 4 byte tiles into a single uint32 tile."""
        return nl.bitwise_or(
            nl.bitwise_or(b0, nl.left_shift(b1, 8, dtype=nl.uint32)),
            nl.bitwise_or(
                nl.left_shift(b2, 16, dtype=nl.uint32),
                nl.left_shift(b3, 24, dtype=nl.uint32),
            ),
        )

    def _mul32_hi_lo(a, b_l, b_h):
        """32×32→64 multiply returning (hi32, lo32) int32 tensors.

        8-bit byte decomposition:

            a = a3·2^24 + a2·2^16 + a1·2^8 + a0   (ai ∈ [0,255])
            b = b3·2^24 + b2·2^16 + b1·2^8 + b0

            p_ij = ai · bj                       ≤ 65025   (2^16)
            c_k  = Σ{i+j=k} p_ij                 ≤ 2^18    (k=0..6)

        Byte-wise carry propagation over c0..c6 yields eight output
        bytes; low four pack lo32, high four pack hi32. Every
        intermediate stays well below 2^24, avoiding the float32
        precision loss on the activation-engine multiply path.

        `a` is a uint32-valued int32 tile of shape (P, 1). `b_l`, `b_h`
        are Python-int constants (low/high 16 bits of the full 32-bit
        multiplier).
        """
        # Cast `a` to uint32 up front so the dst/src dtype match in
        # MLIR tensor_scalar_bitvec ops. Hardware verifier rejects
        # src=i32/dst=ui32 mismatches.
        a_u = nl.copy(a, dtype=nl.uint32)
        a0 = nl.bitwise_and(a_u, 0xFF, dtype=nl.uint32)
        a1 = nl.bitwise_and(nl.right_shift(a_u, 8, dtype=nl.uint32), 0xFF, dtype=nl.uint32)
        a2 = nl.bitwise_and(nl.right_shift(a_u, 16, dtype=nl.uint32), 0xFF, dtype=nl.uint32)
        a3 = nl.right_shift(a_u, 24, dtype=nl.uint32)  # already ≤ 0xFF

        # Materialize the four b-bytes as uint32 tiles. Passing Python-int
        # scalars to nl.multiply makes the compiler see (uint32, int32) —
        # a mixed-dtype pair that promotes to float32 internally. Both
        # operands as uint32 tiles keeps the path integer-typed (though
        # the multiply itself still goes through float activation, now
        # within the 2^24 exact-integer envelope).
        P = a.shape[0]
        b0_vec = nl.full((P, 1), b_l & 0xFF, dtype=nl.uint32)
        b1_vec = nl.full((P, 1), (b_l >> 8) & 0xFF, dtype=nl.uint32)
        b2_vec = nl.full((P, 1), b_h & 0xFF, dtype=nl.uint32)
        b3_vec = nl.full((P, 1), (b_h >> 8) & 0xFF, dtype=nl.uint32)

        # 16 sub-products → 7 column sums. Each column sum ≤ 2^18.
        c0 = _nki_mul_u32(a0, b0_vec)
        c1 = _nki_add_u32(_nki_mul_u32(a0, b1_vec), _nki_mul_u32(a1, b0_vec))
        c2 = _nki_add_u32(
            _nki_add_u32(_nki_mul_u32(a0, b2_vec), _nki_mul_u32(a1, b1_vec)),
            _nki_mul_u32(a2, b0_vec),
        )
        c3 = _nki_add_u32(
            _nki_add_u32(_nki_mul_u32(a0, b3_vec), _nki_mul_u32(a1, b2_vec)),
            _nki_add_u32(_nki_mul_u32(a2, b1_vec), _nki_mul_u32(a3, b0_vec)),
        )
        c4 = _nki_add_u32(
            _nki_add_u32(_nki_mul_u32(a1, b3_vec), _nki_mul_u32(a2, b2_vec)),
            _nki_mul_u32(a3, b1_vec),
        )
        c5 = _nki_add_u32(_nki_mul_u32(a2, b3_vec), _nki_mul_u32(a3, b2_vec))
        c6 = _nki_mul_u32(a3, b3_vec)

        # Byte-wise carry propagation. `acc` stays ≤ 2^18 + 2^10 ≈ 2^18.
        byte0, acc = _nki_carry_step(c0, c1)
        byte1, acc = _nki_carry_step(acc, c2)
        byte2, acc = _nki_carry_step(acc, c3)
        byte3, acc = _nki_carry_step(acc, c4)
        byte4, acc = _nki_carry_step(acc, c5)
        byte5, acc = _nki_carry_step(acc, c6)
        byte6 = nl.bitwise_and(acc, 0xFF, dtype=nl.uint32)
        byte7 = nl.bitwise_and(nl.right_shift(acc, 8, dtype=nl.uint32), 0xFF, dtype=nl.uint32)

        lo32_u = _nki_pack4_u32(byte0, byte1, byte2, byte3)
        hi32_u = _nki_pack4_u32(byte4, byte5, byte6, byte7)

        # Return int32 for downstream XOR — same bit pattern.
        return nl.copy(hi32_u, dtype=nl.int32), nl.copy(lo32_u, dtype=nl.int32)

    @nki.jit
    def philox4x32_kernel(counter_lo_ref, key_lo_ref, key_hi_ref):
        """Philox 4×32-10 NKI kernel on GpSimd.

        Each of the P partition-axis lanes runs one independent Philox
        stream. Input shape is `(P, 1)` per NKI convention (partition,
        free). Output is `(P, 4)` — one int32 per round-output word per
        lane. The wrapper flattens to `(P*4,)` interleaved.
        """
        P = counter_lo_ref.shape[0]
        c0 = nl.load(counter_lo_ref)
        c1 = nl.zeros_like(c0)
        c2 = nl.zeros_like(c0)
        c3 = nl.zeros_like(c0)
        k0 = nl.load(key_lo_ref)
        k1 = nl.load(key_hi_ref)

        # Materialize the key-bump constants as uint32 vector-immediates
        # up front. PHILOX_W0 = 0x9E3779B9 and PHILOX_W1 = 0xBB67AE85
        # both exceed INT32_MAX, so passing them as Python-scalar args to
        # `nl.add(..., dtype=nl.uint32)` may trip signed-int32 scalar-path
        # handling before the uint32 cast takes effect. Materialized
        # vector-immediates avoid the ambiguity entirely.
        w0_vec = nl.full((P, 1), PHILOX_W0, dtype=nl.uint32)
        w1_vec = nl.full((P, 1), PHILOX_W1, dtype=nl.uint32)

        # 10 rounds of Philox — multiply via the carry-free 16-bit
        # decomposition helper (module-level; NKI forbids inner defs).
        for _ in nl.static_range(PHILOX_ROUNDS):
            hi0, lo0 = _mul32_hi_lo(c0, _PHILOX_M0_L, _PHILOX_M0_H)
            hi1, lo1 = _mul32_hi_lo(c2, _PHILOX_M1_L, _PHILOX_M1_H)

            new_c0 = nl.bitwise_xor(nl.bitwise_xor(hi1, c1), k0)
            new_c1 = lo1
            new_c2 = nl.bitwise_xor(nl.bitwise_xor(hi0, c3), k1)
            new_c3 = lo0
            c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3

            # Key bump: (k + W) mod 2^32 via the vector-immediate W.
            k0_u = nl.add(nl.copy(k0, dtype=nl.uint32), w0_vec, dtype=nl.uint32)
            k1_u = nl.add(nl.copy(k1, dtype=nl.uint32), w1_vec, dtype=nl.uint32)
            k0 = nl.copy(k0_u, dtype=nl.int32)
            k1 = nl.copy(k1_u, dtype=nl.int32)

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
        P = uniforms_ref.shape[0]
        pairs = nl.load(uniforms_ref)
        u1 = pairs[:, 0:1]
        u2 = pairs[:, 1:2]
        # trn1 compiler restriction (NCC_IBIR605): InstActivation with Ln
        # rejects scalar-immediate bias parameters. Materialize any
        # scalar feeding into or out of nl.log as a vector-immediate
        # (P, 1) tensor. This avoids the compiler fusing the clamp /
        # scale into a Log activation with a scalar bias.
        clamp_eps = nl.full((P, 1), 1e-10, dtype=uniforms_ref.dtype)
        u1_safe = nl.maximum(u1, clamp_eps)
        neg_two = nl.full((P, 1), -2.0, dtype=uniforms_ref.dtype)
        r = nl.sqrt(nl.multiply(nl.log(u1_safe), neg_two))
        two_pi = nl.full((P, 1), TWO_PI, dtype=uniforms_ref.dtype)
        theta = nl.multiply(u2, two_pi)
        z1 = nl.multiply(r, nl.cos(theta))
        z2 = nl.multiply(r, nl.sin(theta))

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

    # ── Threefry 4×32-20 NKI kernel (byte-tile representation) ───────────────
    #
    # Core invariant: EVERY 32-bit word is stored as 4 separate (P, 1) tiles
    # (b3, b2, b1, b0) with values in [0, 255]. All NKI operations receive
    # inputs ≤ 255 (or ≤ 511 for addition with carry), well within float32's
    # 2^24 exact-integer ceiling. This sidesteps aws-neuron-sdk#1308 entirely:
    # we never form a uint32 tile element, so the float32 activation path
    # is exact for all our inputs.
    #
    # Rotation constants precomputed as (q=R//8, r=R%8) pairs at module level.
    # NKI kernels must not have inner function definitions, so all helpers are
    # defined here at module scope inside `if HAS_NKI:`.

    # Precompute (q, r) for each of the 8 rotation constants.
    # THREEFRY_ROTATIONS = [(10,26),(11,21),(13,27),(23,5)]
    # Flattened: [10, 26, 11, 21, 13, 27, 23, 5]
    _THREEFRY_ROT_QR = tuple((rot // 8, rot % 8) for pair in THREEFRY_ROTATIONS for rot in pair)
    # 128 lanes per tile (Trainium partition-axis limit).
    _THREEFRY_LANES = 128

    def _b_split(v_ref):
        """Load a (P,1) int32 tile and split into 4 byte tiles, each in [0,255].

        Returns [b0, b1, b2, b3] where b0 is the least-significant byte.
        All returned tiles are uint32-typed with values ≤ 255.
        """
        v = nl.load(v_ref)
        v_u = nl.copy(v, dtype=nl.uint32)
        b0 = nl.bitwise_and(v_u, 0xFF, dtype=nl.uint32)
        b1 = nl.bitwise_and(nl.right_shift(v_u, 8, dtype=nl.uint32), 0xFF, dtype=nl.uint32)
        b2 = nl.bitwise_and(nl.right_shift(v_u, 16, dtype=nl.uint32), 0xFF, dtype=nl.uint32)
        b3 = nl.right_shift(v_u, 24, dtype=nl.uint32)
        return [b0, b1, b2, b3]

    def _b_from_scalar(P, val):
        """Materialize a Python int (< 2^32) as 4 byte tiles."""
        val = int(val) & 0xFFFFFFFF
        b0 = nl.full((P, 1), val & 0xFF, dtype=nl.uint32)
        b1 = nl.full((P, 1), (val >> 8) & 0xFF, dtype=nl.uint32)
        b2 = nl.full((P, 1), (val >> 16) & 0xFF, dtype=nl.uint32)
        b3 = nl.full((P, 1), (val >> 24) & 0xFF, dtype=nl.uint32)
        return [b0, b1, b2, b3]

    def _add32_b(a_b, b_b):
        """Carry-propagating 32-bit addition in byte-tile representation.

        Each byte tile is in [0, 255]; sums ≤ 511 < 2^10 (exact in float32).
        Returns 4 byte tiles representing (a + b) mod 2^32.
        """
        s0 = nl.add(a_b[0], b_b[0], dtype=nl.uint32)  # ≤ 510
        c0 = nl.right_shift(s0, 8, dtype=nl.uint32)
        r0 = nl.bitwise_and(s0, 0xFF, dtype=nl.uint32)
        s1 = nl.add(nl.add(a_b[1], b_b[1], dtype=nl.uint32), c0, dtype=nl.uint32)
        c1 = nl.right_shift(s1, 8, dtype=nl.uint32)
        r1 = nl.bitwise_and(s1, 0xFF, dtype=nl.uint32)
        s2 = nl.add(nl.add(a_b[2], b_b[2], dtype=nl.uint32), c1, dtype=nl.uint32)
        c2 = nl.right_shift(s2, 8, dtype=nl.uint32)
        r2 = nl.bitwise_and(s2, 0xFF, dtype=nl.uint32)
        s3 = nl.add(nl.add(a_b[3], b_b[3], dtype=nl.uint32), c2, dtype=nl.uint32)
        r3 = nl.bitwise_and(s3, 0xFF, dtype=nl.uint32)  # carry out discarded
        return [r0, r1, r2, r3]

    def _xor32_b(a_b, b_b):
        """Byte-by-byte XOR. Result bytes in [0, 255]."""
        return [
            nl.bitwise_xor(a_b[0], b_b[0], dtype=nl.uint32),
            nl.bitwise_xor(a_b[1], b_b[1], dtype=nl.uint32),
            nl.bitwise_xor(a_b[2], b_b[2], dtype=nl.uint32),
            nl.bitwise_xor(a_b[3], b_b[3], dtype=nl.uint32),
        ]

    def _rotl32_b(x_b, q, r):
        """Rotate-left by (q bytes + r bits) in byte-tile representation.

        q = rotation // 8  (byte-level shift, precomputed)
        r = rotation % 8   (sub-byte bit shift)

        For r == 0: pure byte rotation, result bytes ∈ [0, 255].
        For r > 0:  out_byte = ((h << r) | (l >> (8-r))) & 0xFF
                    where h = x_b[(i-q)%4], l = x_b[(i-q-1)%4].
                    All intermediates < 2^15, exact in float32.

        Fully unrolled (no list comprehensions, no append loops) —
        NKI hardware compiler rejects both constructs inside jit-traced
        functions (list comprehensions raise "unsupported expression";
        inner defs raise their own error).
        """
        if r == 0:
            # Pure byte-level rotation — select source bytes by position.
            # output byte i = input byte (i - q) % 4.
            if q == 0:
                return [x_b[0], x_b[1], x_b[2], x_b[3]]
            elif q == 1:
                return [x_b[3], x_b[0], x_b[1], x_b[2]]
            elif q == 2:
                return [x_b[2], x_b[3], x_b[0], x_b[1]]
            else:  # q == 3
                return [x_b[1], x_b[2], x_b[3], x_b[0]]

        # Sub-byte rotation: for output byte i,
        #   hi_src = x_b[(i - q) % 4],  lo_src = x_b[(i - q - 1) % 4]
        # Select (hi, lo) pairs for output bytes 0..3 per q value.
        r8 = 8 - r
        if q == 0:
            h0, l0 = x_b[0], x_b[3]
            h1, l1 = x_b[1], x_b[0]
            h2, l2 = x_b[2], x_b[1]
            h3, l3 = x_b[3], x_b[2]
        elif q == 1:
            h0, l0 = x_b[3], x_b[2]
            h1, l1 = x_b[0], x_b[3]
            h2, l2 = x_b[1], x_b[0]
            h3, l3 = x_b[2], x_b[1]
        elif q == 2:
            h0, l0 = x_b[2], x_b[1]
            h1, l1 = x_b[3], x_b[2]
            h2, l2 = x_b[0], x_b[3]
            h3, l3 = x_b[1], x_b[0]
        else:  # q == 3
            h0, l0 = x_b[1], x_b[0]
            h1, l1 = x_b[2], x_b[1]
            h2, l2 = x_b[3], x_b[2]
            h3, l3 = x_b[0], x_b[3]
        out0 = nl.bitwise_and(
            nl.bitwise_or(
                nl.left_shift(h0, r, dtype=nl.uint32),
                nl.right_shift(l0, r8, dtype=nl.uint32),
                dtype=nl.uint32,
            ),
            0xFF,
            dtype=nl.uint32,
        )
        out1 = nl.bitwise_and(
            nl.bitwise_or(
                nl.left_shift(h1, r, dtype=nl.uint32),
                nl.right_shift(l1, r8, dtype=nl.uint32),
                dtype=nl.uint32,
            ),
            0xFF,
            dtype=nl.uint32,
        )
        out2 = nl.bitwise_and(
            nl.bitwise_or(
                nl.left_shift(h2, r, dtype=nl.uint32),
                nl.right_shift(l2, r8, dtype=nl.uint32),
                dtype=nl.uint32,
            ),
            0xFF,
            dtype=nl.uint32,
        )
        out3 = nl.bitwise_and(
            nl.bitwise_or(
                nl.left_shift(h3, r, dtype=nl.uint32),
                nl.right_shift(l3, r8, dtype=nl.uint32),
                dtype=nl.uint32,
            ),
            0xFF,
            dtype=nl.uint32,
        )
        return [out0, out1, out2, out3]

    def _mix_b(a_b, b_b, q, r):
        """Threefry MIX operation in byte-tile representation.

        MIX(a, b, rot):
            a = (a + b) mod 2^32
            b = rotl32(b, rot) ^ a
        """
        a_new = _add32_b(a_b, b_b)
        b_rot = _rotl32_b(b_b, q, r)
        b_new = _xor32_b(b_rot, a_new)
        return a_new, b_new

    def _key_inject_b(x_b_list, ks_b, step):
        """Add Threefry key schedule words to state words.

        Threefry4×32-20 key injection at step s:
            x[i] += ks[(s+i) % 5]   for i in 0..2
            x[3] += ks[(s+3) % 5] + s   (step added to last word)

        `x_b_list`: list of 4 state words, each as [b0,b1,b2,b3] byte tiles.
        `ks_b`:     list of 5 key schedule words, each as [b0,b1,b2,b3].
        `step`:     Python int (0..4), small enough to add directly to b0.
        Returns updated x_b_list (4 words).

        Unrolled explicitly — NKI hardware compiler may reject for-loops
        with list.append() inside jit-traced call trees.
        """
        P = x_b_list[0][0].shape[0]
        out0 = _add32_b(x_b_list[0], ks_b[(step + 0) % 5])
        out1 = _add32_b(x_b_list[1], ks_b[(step + 1) % 5])
        out2 = _add32_b(x_b_list[2], ks_b[(step + 2) % 5])
        # x[3] += ks[(step+3) % 5] + step
        step_b = _b_from_scalar(P, step)
        ks_plus_step = _add32_b(ks_b[(step + 3) % 5], step_b)
        out3 = _add32_b(x_b_list[3], ks_plus_step)
        return [out0, out1, out2, out3]

    @nki.jit
    def threefry4x32_kernel(c0_ref, c1_ref, c2_ref, c3_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Threefry4×32-20 NKI kernel using byte-tile arithmetic.

        Every 32-bit word is held as 4 separate (P,1) uint32 tiles with
        values in [0, 255]. All NKI multiply and shift ops receive inputs
        ≤ 511 — well within float32's 2^24 exact-integer ceiling.

        Inputs: 8 × (P, 1) int32 tiles representing counter (c0..c3) and
                key (k0..k3) words. Values must fit in [0, 2^24) — the host
                wrapper ensures this by design (lane indices 0..127 as c0,
                batch index as c1, higher words remain 0).
        Output: (P, 4) float32 uniforms in [0, 1) — assembled from the 3
                least-significant bytes of each output word (23-bit mantissa),
                which avoids uint32 assembly entirely.
        """
        P = c0_ref.shape[0]

        # Load counter words and split into byte tiles.
        x0_b = _b_split(c0_ref)
        x1_b = _b_split(c1_ref)
        x2_b = _b_split(c2_ref)
        x3_b = _b_split(c3_ref)
        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)

        # Compute k4 = k0 ^ k1 ^ k2 ^ k3 ^ SKEIN_KS_PARITY (byte-by-byte XOR).
        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]

        # Initial key injection (step 0) before round 0.
        # Use subscript READS (_ki[i]) to unpack into named variables —
        # NKI hardware compiler rejects subscript WRITES (x_b_list[i] = ...)
        # as LHS assignment targets inside @nki.jit kernels.
        _ki = _key_inject_b([x0_b, x1_b, x2_b, x3_b], ks_b, 0)
        x0_b = _ki[0]
        x1_b = _ki[1]
        x2_b = _ki[2]
        x3_b = _ki[3]

        # 20 rounds, unrolled at compile time via nl.static_range.
        # Named-variable tuple unpacking (simple name LHS) avoids the
        # subscript-assignment restriction.
        for round_num in nl.static_range(THREEFRY_ROUNDS):
            pair_idx = round_num % 4
            if round_num % 2 == 0:
                # Even rounds: MIX(x0,x1) then MIX(x2,x3)
                q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                x0_b, x1_b = _mix_b(x0_b, x1_b, q0, r0)
                x2_b, x3_b = _mix_b(x2_b, x3_b, q1, r1)
            else:
                # Odd rounds: MIX(x0,x3) then MIX(x2,x1)
                q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                x0_b, x3_b = _mix_b(x0_b, x3_b, q0, r0)
                x2_b, x1_b = _mix_b(x2_b, x1_b, q1, r1)
            # Key injection after every 4th round.
            if (round_num + 1) % 4 == 0:
                _ki = _key_inject_b([x0_b, x1_b, x2_b, x3_b], ks_b, (round_num + 1) // 4)
                x0_b = _ki[0]
                x1_b = _ki[1]
                x2_b = _ki[2]
                x3_b = _ki[3]

        # Convert byte tiles → float32 uniforms in [0, 1).
        # Use 3 least-significant bytes: mantissa = b0 + b1*256 + b2*65536
        # which ≤ 16777215 = 2^24 - 1, exactly representable in float32.
        # Then divide by 2^24. This gives 24-bit uniform resolution.
        # Unrolled for all 4 words — avoids nl.static_range + subscript access.
        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        _s256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        _s65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)
        out = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.shared_hbm)

        b = x0_b
        out[:, 0:1] = nl.multiply(
            nl.add(
                nl.add(
                    nl.copy(b[0], dtype=nl.float32),
                    nl.multiply(nl.copy(b[1], dtype=nl.float32), _s256, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                nl.multiply(nl.copy(b[2], dtype=nl.float32), _s65536, dtype=nl.float32),
                dtype=nl.float32,
            ),
            inv24,
            dtype=nl.float32,
        )
        b = x1_b
        out[:, 1:2] = nl.multiply(
            nl.add(
                nl.add(
                    nl.copy(b[0], dtype=nl.float32),
                    nl.multiply(nl.copy(b[1], dtype=nl.float32), _s256, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                nl.multiply(nl.copy(b[2], dtype=nl.float32), _s65536, dtype=nl.float32),
                dtype=nl.float32,
            ),
            inv24,
            dtype=nl.float32,
        )
        b = x2_b
        out[:, 2:3] = nl.multiply(
            nl.add(
                nl.add(
                    nl.copy(b[0], dtype=nl.float32),
                    nl.multiply(nl.copy(b[1], dtype=nl.float32), _s256, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                nl.multiply(nl.copy(b[2], dtype=nl.float32), _s65536, dtype=nl.float32),
                dtype=nl.float32,
            ),
            inv24,
            dtype=nl.float32,
        )
        b = x3_b
        out[:, 3:4] = nl.multiply(
            nl.add(
                nl.add(
                    nl.copy(b[0], dtype=nl.float32),
                    nl.multiply(nl.copy(b[1], dtype=nl.float32), _s256, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                nl.multiply(nl.copy(b[2], dtype=nl.float32), _s65536, dtype=nl.float32),
                dtype=nl.float32,
            ),
            inv24,
            dtype=nl.float32,
        )
        return out

    @nki.jit
    def threefry_normal_kernel(c0_ref, c1_ref, c2_ref, c3_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Fused Threefry4×32-20 + Box-Muller kernel: counter inputs → N(0,1).

        Chains Threefry output directly into Box-Muller on the Vector Engine.
        Output tiles remain SBUF-resident between stages — no HBM round-trip.
        This is the four-engine framing end-to-end: GpSimd (byte arithmetic)
        → Vector Engine (transcendentals) → SBUF → downstream consumer.

        Inputs: same 8 × (P, 1) counter/key tiles as threefry4x32_kernel.
        Output: (P, 4) float32 standard-normal samples.
        """
        P = c0_ref.shape[0]

        # ── Stage 1: Threefry byte-tile RNG → 4 uniform floats per lane ──
        x0_b = _b_split(c0_ref)
        x1_b = _b_split(c1_ref)
        x2_b = _b_split(c2_ref)
        x3_b = _b_split(c3_ref)
        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)

        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]

        # Initial key injection — unpack via subscript reads into named vars.
        _ki = _key_inject_b([x0_b, x1_b, x2_b, x3_b], ks_b, 0)
        x0_b = _ki[0]
        x1_b = _ki[1]
        x2_b = _ki[2]
        x3_b = _ki[3]

        for round_num in nl.static_range(THREEFRY_ROUNDS):
            pair_idx = round_num % 4
            if round_num % 2 == 0:
                q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                x0_b, x1_b = _mix_b(x0_b, x1_b, q0, r0)
                x2_b, x3_b = _mix_b(x2_b, x3_b, q1, r1)
            else:
                q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                x0_b, x3_b = _mix_b(x0_b, x3_b, q0, r0)
                x2_b, x1_b = _mix_b(x2_b, x1_b, q1, r1)
            if (round_num + 1) % 4 == 0:
                _ki = _key_inject_b([x0_b, x1_b, x2_b, x3_b], ks_b, (round_num + 1) // 4)
                x0_b = _ki[0]
                x1_b = _ki[1]
                x2_b = _ki[2]
                x3_b = _ki[3]

        # Convert to 4 uniform floats (SBUF-resident, no HBM write yet).
        # Inlined explicitly for all 4 words — NKI hardware compiler rejects
        # inner function definitions and list.append() loops inside @nki.jit.
        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        scale256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        scale65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)
        clamp_eps = nl.full((P, 1), 1e-7, dtype=nl.float32)  # avoid log(0)

        b = x0_b
        u0 = nl.maximum(
            nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            ),
            clamp_eps,
        )
        b = x1_b
        u1 = nl.maximum(
            nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            ),
            clamp_eps,
        )
        b = x2_b
        u2 = nl.maximum(
            nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            ),
            clamp_eps,
        )
        b = x3_b
        u3 = nl.maximum(
            nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            ),
            clamp_eps,
        )

        # ── Stage 2: Box-Muller pairs (u0,u1) → (z0,z1), (u2,u3) → (z2,z3) ──
        neg_two = nl.full((P, 1), -2.0, dtype=nl.float32)
        two_pi = nl.full((P, 1), TWO_PI, dtype=nl.float32)

        r0 = nl.sqrt(nl.multiply(nl.log(u0), neg_two))
        theta0 = nl.multiply(u1, two_pi)
        z0 = nl.multiply(r0, nl.cos(theta0))
        z1 = nl.multiply(r0, nl.sin(theta0))

        r1 = nl.sqrt(nl.multiply(nl.log(u2), neg_two))
        theta1 = nl.multiply(u3, two_pi)
        z2 = nl.multiply(r1, nl.cos(theta1))
        z3 = nl.multiply(r1, nl.sin(theta1))

        out = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.shared_hbm)
        out[:, 0:1] = z0
        out[:, 1:2] = z1
        out[:, 2:3] = z2
        out[:, 3:4] = z3
        return out

    def threefry_uniform_nki(
        n_elements: int,
        seed: int = 0,
        counter_offset: int = 0,
    ) -> torch.Tensor:
        """Host-side wrapper: Threefry4×32-20 → float32 uniforms in [0, 1).

        Counter design (ensures all tile inputs < 2^24 for the host's
        typical workload up to ~128 * 2^16 = ~8M elements per call):
            c0 = lane index within tile (0..127)
            c1 = batch/tile index (0..n_batches-1)
            c2 = counter_offset high word
            c3 = 0
        Key: k0=seed&0xFFFFFF, k1=(seed>>24)&0xFFFFFF, k2=k3=0
        (trimmed to 24 bits to guarantee < 2^24 at tile load time).
        """
        LANES = _THREEFRY_LANES
        n_words = (n_elements + 3) // 4  # each lane emits 4 words
        n_batches = (n_words + LANES - 1) // LANES

        k0_val = seed & 0xFFFFFF
        k1_val = (seed >> 24) & 0xFFFFFF

        chunks = []
        for batch in range(n_batches):
            tile_lanes = min(LANES, n_words - batch * LANES)
            c0_np = np.arange(tile_lanes, dtype=np.int32).reshape(-1, 1)
            c1_np = np.full((tile_lanes, 1), (batch + counter_offset) & 0xFFFFFF, dtype=np.int32)
            c2_np = np.zeros((tile_lanes, 1), dtype=np.int32)
            c3_np = np.zeros((tile_lanes, 1), dtype=np.int32)
            k0_np = np.full((tile_lanes, 1), k0_val, dtype=np.int32)
            k1_np = np.full((tile_lanes, 1), k1_val, dtype=np.int32)
            k2_np = np.zeros((tile_lanes, 1), dtype=np.int32)
            k3_np = np.zeros((tile_lanes, 1), dtype=np.int32)

            if _use_simulator():
                out_np = nki.simulate(threefry4x32_kernel)(
                    c0_np,
                    c1_np,
                    c2_np,
                    c3_np,
                    k0_np,
                    k1_np,
                    k2_np,
                    k3_np,
                )
                out_tile = torch.from_numpy(np.asarray(out_np).reshape(-1))
            else:

                def _t(arr):
                    return torch.from_numpy(arr)

                (c0t, c1t, c2t, c3t, k0t, k1t, k2t, k3t), orig = _to_xla(
                    _t(c0_np),
                    _t(c1_np),
                    _t(c2_np),
                    _t(c3_np),
                    _t(k0_np),
                    _t(k1_np),
                    _t(k2_np),
                    _t(k3_np),
                )
                out_tile = (
                    threefry4x32_kernel(c0t, c1t, c2t, c3t, k0t, k1t, k2t, k3t).reshape(-1).to(orig)
                )
            chunks.append(out_tile)

        result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        return result.float()[:n_elements]

    def threefry_normal_nki(
        n_elements: int,
        seed: int = 0,
        counter_offset: int = 0,
    ) -> torch.Tensor:
        """Host-side wrapper: Threefry4×32-20 + Box-Muller → N(0,1) floats.

        Uses the fused `threefry_normal_kernel`. Each tile emits 4 normals
        per lane (2 Box-Muller pairs). Counter layout matches
        `threefry_uniform_nki`.
        """
        LANES = _THREEFRY_LANES
        n_words = (n_elements + 3) // 4
        n_batches = (n_words + LANES - 1) // LANES

        k0_val = seed & 0xFFFFFF
        k1_val = (seed >> 24) & 0xFFFFFF

        chunks = []
        for batch in range(n_batches):
            tile_lanes = min(LANES, n_words - batch * LANES)
            c0_np = np.arange(tile_lanes, dtype=np.int32).reshape(-1, 1)
            c1_np = np.full((tile_lanes, 1), (batch + counter_offset) & 0xFFFFFF, dtype=np.int32)
            c2_np = np.zeros((tile_lanes, 1), dtype=np.int32)
            c3_np = np.zeros((tile_lanes, 1), dtype=np.int32)
            k0_np = np.full((tile_lanes, 1), k0_val, dtype=np.int32)
            k1_np = np.full((tile_lanes, 1), k1_val, dtype=np.int32)
            k2_np = np.zeros((tile_lanes, 1), dtype=np.int32)
            k3_np = np.zeros((tile_lanes, 1), dtype=np.int32)

            if _use_simulator():
                out_np = nki.simulate(threefry_normal_kernel)(
                    c0_np,
                    c1_np,
                    c2_np,
                    c3_np,
                    k0_np,
                    k1_np,
                    k2_np,
                    k3_np,
                )
                out_tile = torch.from_numpy(np.asarray(out_np).reshape(-1))
            else:

                def _t(arr):
                    return torch.from_numpy(arr)

                (c0t, c1t, c2t, c3t, k0t, k1t, k2t, k3t), orig = _to_xla(
                    _t(c0_np),
                    _t(c1_np),
                    _t(c2_np),
                    _t(c3_np),
                    _t(k0_np),
                    _t(k1_np),
                    _t(k2_np),
                    _t(k3_np),
                )
                out_tile = (
                    threefry_normal_kernel(c0t, c1t, c2t, c3t, k0t, k1t, k2t, k3t)
                    .reshape(-1)
                    .to(orig)
                )
            chunks.append(out_tile)

        result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        return result.float()[:n_elements]

    def truncated_normal_nki(
        n_elements,
        low=-2.0,
        high=2.0,
        mean=0.0,
        std=1.0,
        seed=0,
        counter_offset=0,
    ):
        """Truncated normal via NKI Box-Muller + host-side rejection.

        Generates standard normal candidates via threefry_normal_nki and
        accepts those in [low, high] (in standard-normal units). Applies
        mean/std shift on accepted samples. Retries until n_elements gathered.

        Args:
            n_elements: Number of samples to return.
            low: Lower bound in standard-normal units (default -2.0).
            high: Upper bound in standard-normal units (default +2.0).
            mean: Output mean (default 0.0).
            std: Output std (default 1.0).
            seed: 24-bit Threefry key seed.
            counter_offset: Starting counter tile offset.

        Returns:
            Float32 tensor of shape (n_elements,).
        """
        LANES = _THREEFRY_LANES
        result = torch.empty(n_elements, dtype=torch.float32)
        idx = 0
        batch_count = counter_offset
        while idx < n_elements:
            remaining = n_elements - idx
            # Oversample by 2.5× — typical acceptance rate for [-2, 2] is ~95%
            draw = max(LANES, int(remaining * 2.5) + LANES)
            z = threefry_normal_nki(draw, seed=seed, counter_offset=batch_count)
            # Advance counter by the number of tiles consumed
            n_words = (draw + 3) // 4
            batch_count += (n_words + LANES - 1) // LANES
            accepted = z[(z >= low) & (z <= high)]
            take = min(accepted.numel(), remaining)
            result[idx : idx + take] = accepted[:take]
            idx += take
        return (result * std + mean).float()

    def gamma_nki(n_elements, shape, scale=1.0, seed=0, counter_offset=0):
        """Gamma distribution via Marsaglia-Tsang + NKI Threefry RNG.

        Uses the squeeze-acceptance form:
            d = shape - 1/3,  c = 1/sqrt(9*d)
            z ~ N(0,1),  v = (1 + c*z)^3,  u ~ U(0,1)
            Accept if z > -1/c and log(u) < 0.5*z^2 + d*(1 - v + log(v))
        For shape < 1 applies the boost identity: multiply by U^(1/shape).

        Args:
            n_elements: Number of samples to return.
            shape: Shape parameter k > 0.
            scale: Scale parameter θ > 0.
            seed: 24-bit Threefry key seed.
            counter_offset: Starting counter tile offset.

        Returns:
            Float32 tensor of shape (n_elements,).
        """
        assert shape > 0 and scale > 0
        LANES = _THREEFRY_LANES

        # Boost identity for shape < 1: sample Gamma(shape+1) then multiply by U^(1/shape)
        if shape < 1.0:
            boost_u = threefry_uniform_nki(n_elements, seed=seed, counter_offset=counter_offset)
            boost = boost_u ** (1.0 / shape)
            shape_eff = shape + 1.0
            n_words_boost = (n_elements + 3) // 4
            counter_offset = counter_offset + (n_words_boost + LANES - 1) // LANES
            seed_eff = (seed + 1) & 0xFFFFFF
        else:
            boost = None
            shape_eff = shape
            seed_eff = seed

        d = shape_eff - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        neg_c_inv = -1.0 / c

        result = torch.empty(n_elements, dtype=torch.float32)
        idx = 0
        batch_count = counter_offset
        while idx < n_elements:
            remaining = n_elements - idx
            # Oversample by 1.7× — typical acceptance rate ~76% for shape ≥ 1
            draw = max(LANES, int(remaining * 1.7) + LANES)
            z = threefry_normal_nki(draw, seed=seed_eff, counter_offset=batch_count)
            u = threefry_uniform_nki(
                draw, seed=(seed_eff + 2) & 0xFFFFFF, counter_offset=batch_count
            )
            n_words = (draw + 3) // 4
            batch_count += (n_words + LANES - 1) // LANES
            v = (1.0 + c * z) ** 3
            valid = z > neg_c_inv
            v_safe = torch.where(valid, v, torch.ones_like(v))
            accept = valid & (u.log() < 0.5 * z * z + d * (1.0 - v + v_safe.log()))
            draws = (d * v)[accept]
            take = min(draws.numel(), remaining)
            result[idx : idx + take] = draws[:take]
            idx += take

        if boost is not None:
            result = result * boost
        return (result * scale).float()

    def chi_squared_nki(n_elements, df, seed=0, counter_offset=0):
        """Chi-squared distribution: Gamma(df/2, scale=2).

        Args:
            n_elements: Number of samples.
            df: Degrees of freedom (> 0).
            seed: 24-bit Threefry key seed.
            counter_offset: Starting counter tile offset.

        Returns:
            Float32 tensor of shape (n_elements,).
        """
        assert df > 0
        return gamma_nki(
            n_elements,
            shape=df / 2.0,
            scale=2.0,
            seed=seed,
            counter_offset=counter_offset,
        )

    def beta_nki(n_elements, alpha, beta_param, seed=0, counter_offset=0):
        """Beta distribution via gamma-ratio identity.

        X ~ Gamma(alpha), Y ~ Gamma(beta) → X/(X+Y) ~ Beta(alpha, beta).
        Uses distinct seeds for X and Y streams to ensure independence.

        Args:
            n_elements: Number of samples.
            alpha: Shape parameter α > 0.
            beta_param: Shape parameter β > 0.
            seed: 24-bit Threefry key seed for the alpha stream.
            counter_offset: Starting counter tile offset.

        Returns:
            Float32 tensor of shape (n_elements,) with values in (0, 1).
        """
        assert alpha > 0 and beta_param > 0
        x = gamma_nki(n_elements, shape=alpha, scale=1.0, seed=seed, counter_offset=counter_offset)
        y = gamma_nki(
            n_elements,
            shape=beta_param,
            scale=1.0,
            seed=(seed + 4) & 0xFFFFFF,
            counter_offset=counter_offset,
        )
        return (x / (x + y)).float()

    def poisson_nki(n_elements, lam, seed=0, counter_offset=0):
        """Poisson distribution via normal approximation for large λ.

        For λ ≥ _POISSON_NORMAL_THRESHOLD (20), Poisson(λ) ≈ round(N(λ, √λ))
        clamped to ≥ 0. This is vectorizable on NKI — no data-dependent loop.

        Accuracy: skewness 1/√λ ≤ 0.22 at the threshold; tail probabilities
        are within ~1% of exact for integer k at λ = 20, improving rapidly
        with larger λ. Suitable for scientific Monte Carlo (photon counting,
        event-rate simulation, queuing models) where λ ≥ 20 is typical.

        Args:
            n_elements: Number of samples.
            lam: Rate parameter λ (≥ _POISSON_NORMAL_THRESHOLD = 20).
            seed: 24-bit Threefry key seed.
            counter_offset: Starting counter tile offset.

        Returns:
            Float32 tensor of shape (n_elements,) with non-negative integer values.
        """
        import math as _math

        assert lam >= _POISSON_NORMAL_THRESHOLD, (
            f"poisson_nki requires lam >= {_POISSON_NORMAL_THRESHOLD}; got {lam}. "
            "Use the CPU path (torch.poisson) for small lam."
        )
        z = threefry_normal_nki(n_elements, seed=seed, counter_offset=counter_offset)
        samples = lam + _math.sqrt(lam) * z
        return torch.clamp(torch.round(samples), min=0.0)

    # ── Sobol quasi-random sequence (GpSimd XOR accumulation) ─────────────────
    #
    # Sobol coordinates are computed via Gray-code-indexed XOR accumulation:
    #   s[p, d] = XOR{ v[d][k]  for k=0..B-1  if bit k of gray(i_p) is set }
    # where v[d][k] is the Joe-Kuo 2010 direction vector for dimension d,
    # bit position k, and gray(i) = i XOR (i >> 1) is the Gray code.
    #
    # Using 24-bit direction vectors (_SOBOL_BITS = 24) keeps all operands
    # well inside float32's 2^24 exact-integer envelope.  The XOR loops are
    # unrolled at @nki.jit trace time via nl.static_range.
    #
    # Reference: Joe & Kuo (2010), "Constructing Sobol sequences with better
    # two-dimensional projections", SIAM J. Sci. Comput. 30(5):2635–2654.

    _SOBOL_BITS = 24  # bits of precision — supports up to 2^24 ≈ 16.7M points
    _SOBOL_MAX_DIMS = 10  # Joe-Kuo table covers dims 2-10; dim 1 is Van der Corput
    _SOBOL_SCALE = 1.0 / 16777216.0  # 1 / 2^24 — same constant as Threefry inv24

    def _init_sobol_directions():
        """Compute 24-bit Joe-Kuo 2010 direction vectors for the first 10 dims.

        Returns a tuple of 10 tuples, each of length _SOBOL_BITS.
        _SOBOL_DIR_VECS[d][k] is the 24-bit integer for dimension d, bit k.
        Computed once at module import; embedded as nl.full constants in the kernel.
        """
        joe_kuo = [
            (1, 0, [1]),  # dim 2
            (2, 1, [1, 1]),  # dim 3
            (3, 1, [1, 1, 1]),  # dim 4
            (3, 2, [1, 1, 3]),  # dim 5
            (4, 1, [1, 1, 1, 1]),  # dim 6
            (4, 4, [1, 3, 5, 13]),  # dim 7
            (5, 2, [1, 1, 5, 5, 17]),  # dim 8
            (5, 4, [1, 1, 5, 5, 5]),  # dim 9
            (5, 7, [1, 1, 7, 11, 19]),  # dim 10
        ]
        B = _SOBOL_BITS
        dirs = []
        # Dim 0: Van der Corput base-2 (v[k] = 2^(B-1-k))
        dirs.append(tuple(1 << (B - 1 - k) for k in range(B)))
        # Dims 1-9: Joe-Kuo recurrence
        #   v[k] = v[k-s] XOR (v[k-s] >> s)
        #          XOR c_{s-1}*v[k-1] XOR ... XOR c_0*v[k-s+1]
        for s, a, m_init in joe_kuo:
            v = [0] * B
            for k in range(s):
                v[k] = m_init[k] << (B - 1 - k)
            for k in range(s, B):
                val = v[k - s] ^ (v[k - s] >> s)
                for j in range(1, s):
                    if (a >> (s - 1 - j)) & 1:
                        val ^= v[k - j]
                v[k] = val
            dirs.append(tuple(v))
        return tuple(dirs)

    _SOBOL_DIR_VECS = _init_sobol_directions()

    @nki.jit
    def sobol_gray_code_kernel(gray_ref):
        """GpSimd Sobol XOR accumulation: Gray codes → float32 Sobol coordinates.

        For each lane p and dimension d (0.._SOBOL_MAX_DIMS-1):
            s[p,d] = XOR{ v[d][k]  for k=0.._SOBOL_BITS-1
                          if bit k of gray_ref[p] is set }
        Converts s to float via 3-byte reconstruction (same as threefry4x32_kernel):
            out[p,d] = (b0 + b1*256 + b2*65536) / 2^24  in [0, 1)

        All direction vector constants v[d][k] < 2^24 are float32-exact.
        bit_k ∈ {0,1}, so multiply(bit_k, v_dk) ≤ v_dk < 2^24 — also exact.

        Args:
            gray_ref: (P, 1) int32 tile — pre-computed Gray codes.

        Returns:
            (P, _SOBOL_MAX_DIMS) float32 tile — Sobol coordinates in [0, 1).
        """
        P = gray_ref.shape[0]
        g = nl.load(gray_ref)
        g = nl.copy(g, dtype=nl.uint32)

        out = nl.ndarray((P, _SOBOL_MAX_DIMS), dtype=nl.float32, buffer=nl.shared_hbm)

        # Shared float constants for byte→float conversion
        inv24 = nl.full((P, 1), _SOBOL_SCALE, dtype=nl.float32)
        _s256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        _s65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)

        for d_idx in nl.static_range(_SOBOL_MAX_DIMS):
            # Accumulate XOR of direction vectors for set bits of the Gray code
            s = nl.full((P, 1), 0, dtype=nl.uint32)
            for k in nl.static_range(_SOBOL_BITS):
                v_dk = _SOBOL_DIR_VECS[d_idx][k]
                if v_dk != 0:
                    # Extract bit k from each lane's Gray code
                    bit_k = nl.bitwise_and(
                        nl.right_shift(g, k, dtype=nl.uint32),
                        1,
                        dtype=nl.uint32,
                    )
                    # XOR direction vector contribution into accumulator.
                    # bit_k ∈ {0,1} and v_dk < 2^24 → product < 2^24 (float32-exact)
                    s = nl.bitwise_xor(
                        s,
                        nl.multiply(bit_k, nl.full((P, 1), v_dk, dtype=nl.uint32), dtype=nl.uint32),
                        dtype=nl.uint32,
                    )
            # Convert s (uint32 in [0, 2^24)) to float32 in [0, 1) via
            # the same 3-byte decomposition used in threefry4x32_kernel.
            b0 = nl.bitwise_and(s, 0xFF, dtype=nl.uint32)
            b1 = nl.bitwise_and(nl.right_shift(s, 8, dtype=nl.uint32), 0xFF, dtype=nl.uint32)
            b2 = nl.bitwise_and(nl.right_shift(s, 16, dtype=nl.uint32), 0xFF, dtype=nl.uint32)
            out[:, d_idx : d_idx + 1] = nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b0, dtype=nl.float32),
                        nl.multiply(nl.copy(b1, dtype=nl.float32), _s256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b2, dtype=nl.float32), _s65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            )

        return out

    def sobol_nki(
        n_points: int,
        n_dims: int,
        seed: int = 0,
        start_index: int = 0,
    ) -> torch.Tensor:
        """Sobol quasi-random sequence via GpSimd Gray-code XOR accumulation.

        Generates an Owen-scrambled (randomly shifted) Sobol sequence.
        Supports up to _SOBOL_MAX_DIMS=10 dimensions and up to 2^24 ≈ 16.7M points.

        Args:
            n_points: Number of points to generate.
            n_dims: Dimensionality (1 ≤ n_dims ≤ _SOBOL_MAX_DIMS).
            seed: Scrambling seed (0 = no scrambling). Uses additive shift in [0,1).
            start_index: First index in the Sobol sequence (default 0).

        Returns:
            Float32 tensor of shape (n_points, n_dims) with values in [0, 1).
        """
        assert 1 <= n_dims <= _SOBOL_MAX_DIMS, (
            f"sobol_nki supports 1 ≤ n_dims ≤ {_SOBOL_MAX_DIMS}, got {n_dims}"
        )
        assert n_points > 0

        LANES = _THREEFRY_LANES  # 128 — same tile size as Threefry wrappers

        # Compute Gray codes host-side: g(i) = i XOR (i >> 1)
        indices = torch.arange(start_index, start_index + n_points, dtype=torch.int32)
        gray_codes = (indices ^ (indices >> 1)).reshape(-1, 1).contiguous()

        # Pad to multiple of LANES for clean tiling
        n_padded = ((n_points + LANES - 1) // LANES) * LANES
        if n_padded > n_points:
            pad = torch.zeros(n_padded - n_points, 1, dtype=torch.int32)
            gray_codes = torch.cat([gray_codes, pad], dim=0)

        chunks = []
        for tile_start in range(0, n_padded, LANES):
            gray_tile = gray_codes[tile_start : tile_start + LANES].contiguous()

            if _use_simulator():
                out_np = nki.simulate(sobol_gray_code_kernel)(gray_tile.numpy())
                chunks.append(torch.from_numpy(np.asarray(out_np)))
            else:
                (gray_xla,), orig = _to_xla(gray_tile)
                out_tile = sobol_gray_code_kernel(gray_xla).to(orig)
                chunks.append(out_tile)

        # (n_padded, _SOBOL_MAX_DIMS) → (n_points, n_dims) float32
        result = torch.cat(chunks, dim=0)[:n_points, :n_dims]

        # Randomly shifted Sobol: add a per-dimension uniform offset and wrap.
        # This breaks the regular structure while preserving low discrepancy.
        if seed != 0:
            rng = torch.Generator()
            rng.manual_seed(seed)
            shifts = torch.rand(n_dims, generator=rng, dtype=torch.float32)
            result = torch.fmod(result + shifts.unsqueeze(0), 1.0)

        return result.float()

    # ── Halton quasi-random sequence (GpSimd + Vector Engine) ─────────────────
    #
    # The Halton radical inverse in prime base p for index i is:
    #   H(i, p) = sum_k d_k * p^(-k-1)
    # where d_k = floor(i / p^k) % p is the k-th digit of i in base p.
    #
    # Iterative digit extraction in float32:
    #   q = floor(i_float / p)   [exact for i < 2^22 — see proof below]
    #   digit = i_float - p * q  [exact since digit is integer ≤ p-1 ≤ 28]
    #   result += digit / p^(k+1)
    #   i_float = q
    #
    # Exactness proof (i < 2^22, p ≤ 29):
    #   ULP(i/p) ≤ ULP(2^22/2) = ULP(2^21) = 2^(21-23) = 2^-2 = 0.25.
    #   Minimum non-zero fractional part of i/p = 1/p ≥ 1/29 ≈ 0.034.
    #   Since 0.5*ULP ≤ 0.125 < 1/29: floor(float32(i/p)) = floor(i/p) exactly.
    #   After floor: b * floor(i/p) is an integer ≤ i < 2^22 < 2^24 (float32-exact).
    #   digit = i - b*floor(i/p) = i%p — an integer in [0, p-1]. ✓
    #
    # All operations route through the Vector Engine (nl.floor) or activation
    # engine (nl.multiply, nl.add). nl.static_range unrolls both loops at
    # @nki.jit trace time.
    #
    # Maximum supported: _HALTON_MAX_DIMS=10 dimensions, _HALTON_MAX_POINTS=2^22 points.

    _HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    _HALTON_DEPTHS = (22, 14, 10, 8, 7, 6, 6, 6, 5, 5)  # digits to cover i < 2^22
    _HALTON_MAX_DIMS = 10
    _HALTON_MAX_POINTS = 1 << 22  # 4,194,304 — float32 digit extraction exact below this

    @nki.jit
    def halton_kernel(indices_ref):
        """GpSimd/VE Halton radical inverse: indices → float32 Halton coordinates.

        For each lane p and dimension d (0.._HALTON_MAX_DIMS-1), computes the
        Van der Corput / Halton radical inverse of the integer index in the
        prime base _HALTON_PRIMES[d] using iterative digit extraction.

        Digit extraction loop per dimension:
            q = nl.floor(i / p)   — Vector Engine floor (exact for i < 2^22)
            digit = i - p * q     — remainder digit in [0, p-1]
            result += digit / p^(k+1)  — accumulate contribution
            i = q                 — advance to next digit position

        Args:
            indices_ref: (P, 1) int32 tile — point indices (must be ≥ 1).

        Returns:
            (P, _HALTON_MAX_DIMS) float32 tile — Halton coordinates in (0, 1).
        """
        P = indices_ref.shape[0]
        i_load = nl.load(indices_ref)
        i_base = nl.copy(i_load, dtype=nl.float32)  # starting index as float32

        out = nl.ndarray((P, _HALTON_MAX_DIMS), dtype=nl.float32, buffer=nl.shared_hbm)

        for d_idx in nl.static_range(_HALTON_MAX_DIMS):
            p = _HALTON_PRIMES[d_idx]  # Python int — compile-time constant
            depth = _HALTON_DEPTHS[d_idx]  # Python int — compile-time constant
            inv_p = nl.full((P, 1), 1.0 / p, dtype=nl.float32)
            p_float = nl.full((P, 1), float(p), dtype=nl.float32)

            # Accumulate radical inverse in float32.
            result = nl.full((P, 1), 0.0, dtype=nl.float32)
            i_float = i_base  # reset to original index for this dimension

            f = 1.0 / p  # Python-level contribution factor (decreases by 1/p each step)
            for k in nl.static_range(depth):
                f_tile = nl.full((P, 1), f, dtype=nl.float32)
                # floor(i_float / p) — exact for i_float < 2^22 (see comment above)
                q = nl.floor(nl.multiply(i_float, inv_p, dtype=nl.float32))
                # digit = i_float - p * q = i_float % p (integer in [0, p-1])
                digit = nl.subtract(
                    i_float,
                    nl.multiply(q, p_float, dtype=nl.float32),
                    dtype=nl.float32,
                )
                # Accumulate digit contribution: digit / p^(k+1)
                result = nl.add(
                    result,
                    nl.multiply(digit, f_tile, dtype=nl.float32),
                    dtype=nl.float32,
                )
                i_float = q  # advance: quotient becomes index for next digit
                f /= p  # Python-level update — compile-time constant next iter

            out[:, d_idx : d_idx + 1] = result

        return out

    def halton_nki(
        n_points: int,
        n_dims: int,
        start_index: int = 0,
    ) -> torch.Tensor:
        """Halton quasi-random sequence via GpSimd/VE radical inverse.

        Generates the Halton sequence for the first _HALTON_MAX_DIMS=10 prime
        bases (2, 3, 5, 7, 11, 13, 17, 19, 23, 29). The sequence is deterministic
        (no scrambling — Halton has no seed parameter unlike Sobol).

        Correctness requires all indices < 2^22 = 4,194,304. An assertion is
        raised if this bound is exceeded.

        Args:
            n_points: Number of points to generate.
            n_dims: Dimensionality (1 ≤ n_dims ≤ 10).
            start_index: First index in the Halton sequence (default 0 → starts at i=1).
                         Halton(0, b) = 0 for all b, so the sequence conventionally
                         starts at index 1. If start_index > 0, uses start_index + 1.

        Returns:
            Float32 tensor of shape (n_points, n_dims) with values in (0, 1).
        """
        assert 1 <= n_dims <= _HALTON_MAX_DIMS, (
            f"halton_nki supports 1 ≤ n_dims ≤ {_HALTON_MAX_DIMS}, got {n_dims}"
        )
        assert n_points > 0
        # All indices must be < 2^22 for exact float32 digit extraction
        first_idx = max(start_index, 1)  # skip index 0 (all-zero point)
        last_idx = first_idx + n_points
        assert last_idx <= _HALTON_MAX_POINTS, (
            f"halton_nki: index {last_idx} exceeds 2^22 = {_HALTON_MAX_POINTS}. "
            "Use the CPU path (set_backend('pytorch')) for larger sequences."
        )

        LANES = _THREEFRY_LANES  # 128

        # Build index tensor starting at first_idx
        indices = torch.arange(first_idx, first_idx + n_points, dtype=torch.int32)

        # Pad to multiple of LANES
        n_padded = ((n_points + LANES - 1) // LANES) * LANES
        idx_col = indices.reshape(-1, 1)
        if n_padded > n_points:
            # Pad with harmless indices (reuse last valid index)
            pad_val = indices[-1].item()
            pad = torch.full((n_padded - n_points, 1), pad_val, dtype=torch.int32)
            idx_col = torch.cat([idx_col, pad], dim=0)

        chunks = []
        for tile_start in range(0, n_padded, LANES):
            tile = idx_col[tile_start : tile_start + LANES].contiguous()

            if _use_simulator():
                out_np = nki.simulate(halton_kernel)(tile.numpy())
                chunks.append(torch.from_numpy(np.asarray(out_np)))
            else:
                (tile_xla,), orig = _to_xla(tile)
                out_tile = halton_kernel(tile_xla).to(orig)
                chunks.append(out_tile)

        result = torch.cat(chunks, dim=0)[:n_points, :n_dims]
        return result.float()

    # ── Streaming generator (v0.6.0 Phase 3) ─────────────────────────────────
    #
    # NEFF-resident pipeline: GpSimd (Threefry) → Vector (Box-Muller) → Scalar.
    # All _PROGRAM_TILES tiles execute within ONE kernel invocation, enabling
    # engine overlap that is impossible across separate XLA graph submissions.
    #
    # Steady-state overlap at tile k≥2:
    #   GpSimd  runs Threefry on tile k
    #   Vector  runs Box-Muller on tile k-1
    #   Scalar  handles scale/shift on tile k-2
    #
    # 31× reduction in Python→XLA round-trips for 1M samples vs per-tile dispatch:
    #   v0.5.0: 7813 XLA calls  (1M / (128 × 4) = 1M / 512)
    #   v0.6.0:  245 XLA calls  (1M / (32 × 128 × 4) = 1M / 16384)

    @nki.jit
    def threefry_streaming_normal_kernel(c0_ref, start_batch_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Streaming Threefry4×32-20 + Box-Muller: _PROGRAM_TILES tiles in one NEFF.

        Eliminates per-tile XLA dispatch overhead. Each invocation generates
        _PROGRAM_TILES × _THREEFRY_LANES × 4 = 16,384 standard-normal floats.

        Engine orchestration:
          GpSimd:  Threefry byte-tile arithmetic (20 rounds per tile)
          Vector:  Box-Muller log/sqrt/cos/sin transcendentals (per tile, pipelined)
          Scalar:  scale+shift (identity here — output is N(0,1))
          Tensor:  intentionally idle — RNG is not matmul-shaped

        Stochastic rounding: NOT used — statistical error 1/√N dominates arithmetic
          error at all practical Monte Carlo sample counts (Connolly–Higham–Mary,
          SIAM J. Sci. Comput., 2021). The √n·u improvement over Wilkinson's n·u
          bound is immeasurable against 1/√N for any n ≥ 1.

        Counter scheme (bit-exact with threefry_normal_nki for same seed/offset):
          c0 = lane index [0..127]  (static, from c0_ref)
          c1 = (start_batch + tile_idx) & 0xFFFFFF  (computed per tile)
          c2 = c3 = 0
          start_batch = counter_offset + launch × _PROGRAM_TILES

        Inputs:
          c0_ref:          (P, 1) int32 — lane indices 0..127
          start_batch_ref: (P, 1) int32 — base batch index for this launch
          k0_ref..k3_ref:  (P, 1) int32 — Threefry key words

        Output: (P, 4 × _PROGRAM_TILES) = (128, 128) float32 N(0,1) samples.
        """
        P = c0_ref.shape[0]

        # Fixed inputs: loaded once, shared across all _PROGRAM_TILES iterations.
        c0_b = _b_split(c0_ref)
        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)
        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]

        # c2=c3=0 (compile-time constants, same for all tiles).
        c2_b = _b_from_scalar(P, 0)
        c3_b = _b_from_scalar(P, 0)

        # Runtime start_batch loaded once from HBM; tile_idx offset added per tile.
        # Passing start_batch as a runtime arg lets the SAME NEFF serve every
        # stream_into call — the NEFF cache key depends only on shapes/dtypes.
        start_batch_u = nl.load(start_batch_ref, dtype=nl.uint32)

        # Box-Muller constants — hoisted outside the loop (compile-once into NEFF).
        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        scale256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        scale65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)
        clamp_eps = nl.full((P, 1), 1e-7, dtype=nl.float32)
        neg_two = nl.full((P, 1), -2.0, dtype=nl.float32)
        two_pi = nl.full((P, 1), TWO_PI, dtype=nl.float32)

        # Output: wider free dimension avoids large input-tensor slicing.
        # Shape (P, 4×_PROGRAM_TILES) = (128, 128) fits in a single HBM tile.
        out = nl.ndarray((P, 4 * _PROGRAM_TILES), dtype=nl.float32, buffer=nl.shared_hbm)

        for tile_idx in nl.static_range(_PROGRAM_TILES):
            # c1 = (start_batch + tile_idx) & 0xFFFFFF — unique counter per tile.
            # tile_idx is a compile-time Python int from nl.static_range;
            # nl.full materializes it as a constant tile for the add.
            c1_u = nl.bitwise_and(
                nl.add(
                    start_batch_u,
                    nl.full((P, 1), tile_idx, dtype=nl.uint32),
                    dtype=nl.uint32,
                ),
                0xFFFFFF,
                dtype=nl.uint32,
            )
            # Byte-split c1 from an SBUF tile (not a ref — no nl.load).
            # b3=0 guaranteed: c1_u ≤ 0xFFFFFF (24 bits after masking).
            c1_b = [
                nl.bitwise_and(c1_u, 0xFF, dtype=nl.uint32),
                nl.bitwise_and(nl.right_shift(c1_u, 8, dtype=nl.uint32), 0xFF, dtype=nl.uint32),
                nl.bitwise_and(nl.right_shift(c1_u, 16, dtype=nl.uint32), 0xFF, dtype=nl.uint32),
                nl.full((P, 1), 0, dtype=nl.uint32),
            ]

            # Threefry4×32-20: key injection then 20 rounds.
            _ki = _key_inject_b([c0_b, c1_b, c2_b, c3_b], ks_b, 0)
            x0_b = _ki[0]
            x1_b = _ki[1]
            x2_b = _ki[2]
            x3_b = _ki[3]

            for round_num in nl.static_range(THREEFRY_ROUNDS):
                pair_idx = round_num % 4
                if round_num % 2 == 0:
                    q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                    q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                    x0_b, x1_b = _mix_b(x0_b, x1_b, q0, r0)
                    x2_b, x3_b = _mix_b(x2_b, x3_b, q1, r1)
                else:
                    q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                    q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                    x0_b, x3_b = _mix_b(x0_b, x3_b, q0, r0)
                    x2_b, x1_b = _mix_b(x2_b, x1_b, q1, r1)
                if (round_num + 1) % 4 == 0:
                    _ki = _key_inject_b([x0_b, x1_b, x2_b, x3_b], ks_b, (round_num + 1) // 4)
                    x0_b = _ki[0]
                    x1_b = _ki[1]
                    x2_b = _ki[2]
                    x3_b = _ki[3]

            # Convert to 4 uniform floats (SBUF-resident, no HBM write yet).
            # Inlined x4 words — NKI compiler rejects list comprehensions inside @nki.jit.
            b = x0_b
            u0 = nl.maximum(
                nl.multiply(
                    nl.add(
                        nl.add(
                            nl.copy(b[0], dtype=nl.float32),
                            nl.multiply(
                                nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32
                            ),
                            dtype=nl.float32,
                        ),
                        nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    inv24,
                    dtype=nl.float32,
                ),
                clamp_eps,
            )
            b = x1_b
            u1 = nl.maximum(
                nl.multiply(
                    nl.add(
                        nl.add(
                            nl.copy(b[0], dtype=nl.float32),
                            nl.multiply(
                                nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32
                            ),
                            dtype=nl.float32,
                        ),
                        nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    inv24,
                    dtype=nl.float32,
                ),
                clamp_eps,
            )
            b = x2_b
            u2 = nl.maximum(
                nl.multiply(
                    nl.add(
                        nl.add(
                            nl.copy(b[0], dtype=nl.float32),
                            nl.multiply(
                                nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32
                            ),
                            dtype=nl.float32,
                        ),
                        nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    inv24,
                    dtype=nl.float32,
                ),
                clamp_eps,
            )
            b = x3_b
            u3 = nl.maximum(
                nl.multiply(
                    nl.add(
                        nl.add(
                            nl.copy(b[0], dtype=nl.float32),
                            nl.multiply(
                                nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32
                            ),
                            dtype=nl.float32,
                        ),
                        nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    inv24,
                    dtype=nl.float32,
                ),
                clamp_eps,
            )

            # Box-Muller pairs: (u0,u1) → (z0,z1), (u2,u3) → (z2,z3).
            bm_r0 = nl.sqrt(nl.multiply(nl.log(u0), neg_two))
            bm_theta0 = nl.multiply(u1, two_pi)
            z0 = nl.multiply(bm_r0, nl.cos(bm_theta0))
            z1 = nl.multiply(bm_r0, nl.sin(bm_theta0))
            bm_r1 = nl.sqrt(nl.multiply(nl.log(u2), neg_two))
            bm_theta1 = nl.multiply(u3, two_pi)
            z2 = nl.multiply(bm_r1, nl.cos(bm_theta1))
            z3 = nl.multiply(bm_r1, nl.sin(bm_theta1))

            col = tile_idx * 4
            out[:, col : col + 1] = z0
            out[:, col + 1 : col + 2] = z1
            out[:, col + 2 : col + 3] = z2
            out[:, col + 3 : col + 4] = z3

        return out

    @nki.jit
    def threefry_streaming_uniform_kernel(c0_ref, start_batch_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Streaming Threefry4×32-20: _PROGRAM_TILES tiles → float32 uniforms in [0,1).

        Same counter/key scheme and NEFF-caching strategy as
        threefry_streaming_normal_kernel but emits raw uniform samples instead of
        applying Box-Muller. Use when downstream needs U(0,1) values (importance-
        sampling weights, quantile transforms, Bernoulli draws, etc.).

        Engine orchestration:
          GpSimd:  Threefry byte-tile arithmetic
          Vector:  uniform float assembly (multiply/add)
          Tensor:  intentionally idle
        Stochastic rounding: NOT used (same rationale as streaming normal kernel).

        Output: (P, 4 × _PROGRAM_TILES) = (128, 128) float32 values in [0, 1).
        """
        P = c0_ref.shape[0]

        c0_b = _b_split(c0_ref)
        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)
        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]

        c2_b = _b_from_scalar(P, 0)
        c3_b = _b_from_scalar(P, 0)

        start_batch_u = nl.load(start_batch_ref, dtype=nl.uint32)

        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        scale256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        scale65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)

        out = nl.ndarray((P, 4 * _PROGRAM_TILES), dtype=nl.float32, buffer=nl.shared_hbm)

        for tile_idx in nl.static_range(_PROGRAM_TILES):
            c1_u = nl.bitwise_and(
                nl.add(
                    start_batch_u,
                    nl.full((P, 1), tile_idx, dtype=nl.uint32),
                    dtype=nl.uint32,
                ),
                0xFFFFFF,
                dtype=nl.uint32,
            )
            c1_b = [
                nl.bitwise_and(c1_u, 0xFF, dtype=nl.uint32),
                nl.bitwise_and(nl.right_shift(c1_u, 8, dtype=nl.uint32), 0xFF, dtype=nl.uint32),
                nl.bitwise_and(nl.right_shift(c1_u, 16, dtype=nl.uint32), 0xFF, dtype=nl.uint32),
                nl.full((P, 1), 0, dtype=nl.uint32),
            ]

            _ki = _key_inject_b([c0_b, c1_b, c2_b, c3_b], ks_b, 0)
            x0_b = _ki[0]
            x1_b = _ki[1]
            x2_b = _ki[2]
            x3_b = _ki[3]

            for round_num in nl.static_range(THREEFRY_ROUNDS):
                pair_idx = round_num % 4
                if round_num % 2 == 0:
                    q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                    q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                    x0_b, x1_b = _mix_b(x0_b, x1_b, q0, r0)
                    x2_b, x3_b = _mix_b(x2_b, x3_b, q1, r1)
                else:
                    q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                    q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                    x0_b, x3_b = _mix_b(x0_b, x3_b, q0, r0)
                    x2_b, x1_b = _mix_b(x2_b, x1_b, q1, r1)
                if (round_num + 1) % 4 == 0:
                    _ki = _key_inject_b([x0_b, x1_b, x2_b, x3_b], ks_b, (round_num + 1) // 4)
                    x0_b = _ki[0]
                    x1_b = _ki[1]
                    x2_b = _ki[2]
                    x3_b = _ki[3]

            # Uniform float assembly — inlined x4 words, direct to output slice.
            col = tile_idx * 4
            b = x0_b
            out[:, col : col + 1] = nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            )
            b = x1_b
            out[:, col + 1 : col + 2] = nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            )
            b = x2_b
            out[:, col + 2 : col + 3] = nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            )
            b = x3_b
            out[:, col + 3 : col + 4] = nl.multiply(
                nl.add(
                    nl.add(
                        nl.copy(b[0], dtype=nl.float32),
                        nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                        dtype=nl.float32,
                    ),
                    nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
                    dtype=nl.float32,
                ),
                inv24,
                dtype=nl.float32,
            )

        return out

    def threefry_stream_normal(
        n_elements: int,
        seed: int = 0,
        counter_offset: int = 0,
    ) -> torch.Tensor:
        """Host wrapper: streaming Threefry4×32-20 + Box-Muller → N(0,1) floats.

        Each call reuses the fixed-size NEFF compiled for `threefry_streaming_normal_kernel`,
        running _PROGRAM_TILES Threefry tiles in one XLA invocation — 16,384 samples per
        XLA call vs 512 per call in threefry_normal_nki. For 1M samples: 245 XLA calls vs
        7813 in the per-tile wrapper (~31× reduction in launch overhead).

        Counter-offset compatibility: outputs are bit-exact with threefry_normal_nki when
        both use the same seed and counter_offset. Launch L tile T uses
        c1 = (counter_offset + L*_PROGRAM_TILES + T) & 0xFFFFFF, matching the per-tile
        wrapper's c1 = (batch + counter_offset) & 0xFFFFFF at batch = L*_PROGRAM_TILES + T.
        """
        LANES = _THREEFRY_LANES
        NORMALS_PER_LAUNCH = _PROGRAM_TILES * LANES * 4  # 32 × 128 × 4 = 16,384

        n_launches = (n_elements + NORMALS_PER_LAUNCH - 1) // NORMALS_PER_LAUNCH

        k0_val = seed & 0xFFFFFF
        k1_val = (seed >> 24) & 0xFFFFFF

        c0_np = np.arange(LANES, dtype=np.int32).reshape(-1, 1)
        k0_np = np.full((LANES, 1), k0_val, dtype=np.int32)
        k1_np = np.full((LANES, 1), k1_val, dtype=np.int32)
        k2_np = np.zeros((LANES, 1), dtype=np.int32)
        k3_np = np.zeros((LANES, 1), dtype=np.int32)

        chunks = []
        for launch in range(n_launches):
            start_batch = (counter_offset + launch * _PROGRAM_TILES) & 0xFFFFFF
            start_batch_np = np.full((LANES, 1), start_batch, dtype=np.int32)

            if _use_simulator():
                out_np = nki.simulate(threefry_streaming_normal_kernel)(
                    c0_np, start_batch_np, k0_np, k1_np, k2_np, k3_np
                )
                out_tile = torch.from_numpy(np.asarray(out_np).reshape(-1))
            else:

                def _t(arr):
                    return torch.from_numpy(arr)

                (c0t, sb_t, k0t, k1t, k2t, k3t), orig = _to_xla(
                    _t(c0_np),
                    _t(start_batch_np),
                    _t(k0_np),
                    _t(k1_np),
                    _t(k2_np),
                    _t(k3_np),
                )
                out_tile = (
                    threefry_streaming_normal_kernel(c0t, sb_t, k0t, k1t, k2t, k3t)
                    .reshape(-1)
                    .to(orig)
                )
            chunks.append(out_tile)

        result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        return result.float()[:n_elements]

    def threefry_stream_uniform(
        n_elements: int,
        seed: int = 0,
        counter_offset: int = 0,
    ) -> torch.Tensor:
        """Host wrapper: streaming Threefry4×32-20 → float32 uniforms in [0, 1).

        Same launch/counter scheme as threefry_stream_normal; 16,384 uniforms per
        XLA call, ~31× fewer round-trips than the per-tile threefry_uniform_nki
        for large n.
        """
        LANES = _THREEFRY_LANES
        UNIFORMS_PER_LAUNCH = _PROGRAM_TILES * LANES * 4  # 32 × 128 × 4 = 16,384

        n_launches = (n_elements + UNIFORMS_PER_LAUNCH - 1) // UNIFORMS_PER_LAUNCH

        k0_val = seed & 0xFFFFFF
        k1_val = (seed >> 24) & 0xFFFFFF

        c0_np = np.arange(LANES, dtype=np.int32).reshape(-1, 1)
        k0_np = np.full((LANES, 1), k0_val, dtype=np.int32)
        k1_np = np.full((LANES, 1), k1_val, dtype=np.int32)
        k2_np = np.zeros((LANES, 1), dtype=np.int32)
        k3_np = np.zeros((LANES, 1), dtype=np.int32)

        chunks = []
        for launch in range(n_launches):
            start_batch = (counter_offset + launch * _PROGRAM_TILES) & 0xFFFFFF
            start_batch_np = np.full((LANES, 1), start_batch, dtype=np.int32)

            if _use_simulator():
                out_np = nki.simulate(threefry_streaming_uniform_kernel)(
                    c0_np, start_batch_np, k0_np, k1_np, k2_np, k3_np
                )
                out_tile = torch.from_numpy(np.asarray(out_np).reshape(-1))
            else:

                def _t(arr):
                    return torch.from_numpy(arr)

                (c0t, sb_t, k0t, k1t, k2t, k3t), orig = _to_xla(
                    _t(c0_np),
                    _t(start_batch_np),
                    _t(k0_np),
                    _t(k1_np),
                    _t(k2_np),
                    _t(k3_np),
                )
                out_tile = (
                    threefry_streaming_uniform_kernel(c0t, sb_t, k0t, k1t, k2t, k3t)
                    .reshape(-1)
                    .to(orig)
                )
            chunks.append(out_tile)

        result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        return result.float()[:n_elements]
