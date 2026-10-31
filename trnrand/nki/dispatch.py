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

import importlib.util
import math
import os
import warnings

import numpy as np
import torch

try:
    import nki
    import nki.isa as nisa
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


# ── Device dispatch path: native TorchNeuron vs XLA ──────────────────────────
#
# Two Neuron PyTorch lineages coexist:
#   native  torch-neuronx 2.14+ (TorchNeuron, Trn2/Trn3): Trainium is a
#           PyTorch `neuron` device; kernels are custom ops registered via
#           `torch_neuronx.nki_op` (see `_native.py`).
#   xla     torch-neuronx 2.9 lineage (PyTorch/XLA; Trn1 and Inf2 stay
#           here): kernels dispatch on the torch_xla device via `_to_xla()`.
# Anything else is "cpu": no device path, so a kernel launch outside the
# simulator raises (from `_to_xla`) exactly as it did before the native
# path existed. Detection runs on the first launch, not at import, and the
# native probe never imports torch_xla.

_NEURON_DEVICE = "neuron"
_dispatch_path: str | None = None
_native_ops: dict | None = None


def _native_available() -> bool:
    try:
        import torch_neuronx
    except Exception:
        return False
    # The XLA-lineage torch_neuronx has no `nki_op`.
    if not hasattr(torch_neuronx, "nki_op"):
        return False
    try:
        torch.device(_NEURON_DEVICE)  # torch_neuronx renames PrivateUse1 to "neuron"
        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def _detect_dispatch_path() -> str:
    # TRNSCI_NKI_DISPATCH=auto|native|xla is shared by every trnsci library, so
    # one variable pins the whole suite on hosts that have both lineages.
    forced = os.environ.get("TRNSCI_NKI_DISPATCH", "auto").strip().lower()
    if forced == "native":
        if not _native_available():
            raise RuntimeError(
                "TRNSCI_NKI_DISPATCH=native, but no TorchNeuron backend with a "
                "'neuron' device is available"
            )
        return "native"
    if forced == "xla":
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError("TRNSCI_NKI_DISPATCH=xla, but torch_xla is not installed")
        return "xla"
    if forced != "auto":
        raise ValueError(f"TRNSCI_NKI_DISPATCH must be auto, native, or xla; got {forced!r}")
    if _native_available():
        return "native"
    if importlib.util.find_spec("torch_xla") is not None:
        return "xla"
    return "cpu"


def _get_dispatch_path() -> str:
    """Return "native", "xla", or "cpu" (cached after the first call)."""
    global _dispatch_path
    if _dispatch_path is None:
        _dispatch_path = _detect_dispatch_path()
    return _dispatch_path


def _get_native_ops() -> dict:
    """Import `_native` (registering the trnrand:: custom ops) on first use."""
    global _native_ops
    if _native_ops is None:
        from . import _native

        _native_ops = _native.OPS
    return _native_ops


# ── Host results through the caching host allocator ─────────────────────────
#
# Same fix as trnblas#61/#64. On Trn2 (SDK 2.33, native TorchNeuron), a
# device→host copy into fresh pageable memory (`.cpu()`, `torch.empty` +
# `copy_`) runs at ~4 GB/s, because the host page-faults every page it
# touches. A copy into memory that is already faulted in runs at the link
# rate, 13-14 GB/s. `torch.empty(..., pin_memory=True)` comes from the
# accelerator's caching host allocator: each call still returns a fresh tensor
# (a block is recycled only after the tensor that owned it is freed, so results
# never alias), and recycled blocks come back already faulted in.
#
# Costs: blocks are rounded up to a power of two, and freed blocks stay cached
# until `torch.accelerator.empty_host_cache()`. TRNRAND_PINNED_OUTPUT=0 turns
# this off (plain pageable memory). Results under `_PINNED_OUTPUT_MIN_BYTES`
# always use pageable memory. The XLA path, the simulator and torch.compile
# tracing are unchanged. The values are the same bits either way.
_PINNED_OUTPUT_MIN_BYTES = 1 << 20


def _pinned_output_enabled() -> bool:
    value = os.environ.get("TRNRAND_PINNED_OUTPUT", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _use_host_cache(device, nbytes: int) -> bool:
    """True when a CPU result of `nbytes` should come from the caching host
    allocator: native eager path, CPU destination, large enough, not disabled."""
    return (
        torch.device(device).type == "cpu"
        and nbytes >= _PINNED_OUTPUT_MIN_BYTES
        and _pinned_output_enabled()
        and not _use_simulator()
        and _get_dispatch_path() == "native"
        and not torch.compiler.is_compiling()
    )


def _host_empty(shape, dtype: torch.dtype, device="cpu") -> torch.Tensor:
    """`torch.empty(shape, dtype=dtype, device=device)`; a large CPU tensor on the
    native path comes from the caching host allocator (fresh, already faulted in)."""
    shape = tuple(shape)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if _use_host_cache(device, nbytes):
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype, device=device)


def _to_host(t: torch.Tensor, device="cpu") -> torch.Tensor:
    """`t.to(device)`. A large device → CPU copy lands in a fresh tensor from
    the caching host allocator, about 3x faster than `.cpu()` (trnblas#61)."""
    if t.device.type == "cpu" or not _use_host_cache(device, t.numel() * t.element_size()):
        return t.to(device)
    return _host_empty(t.shape, t.dtype, device).copy_(t)


def _launch(kernel, op_name: str, *tensors: torch.Tensor) -> torch.Tensor:
    """Run one kernel launch on the Neuron device and return the host result.

    `op_name` is the `trnrand::<op_name>` custom op used on the native path;
    the XLA path calls `kernel` directly on torch_xla tensors.
    """
    if _get_dispatch_path() == "native":
        orig = tensors[0].device
        op = _get_native_ops()[op_name]
        return _to_host(op(*(t.to(_NEURON_DEVICE) for t in tensors)), orig)
    xs, orig = _to_xla(*tensors)
    return kernel(*xs).to(orig)


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

    # Same bit pattern reinterpreted as int32 (matches the kernel's .view(nl.int32)).
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

# 128 lanes per Threefry tile (Trainium partition-axis limit). Module level so the
# host wrappers below `if HAS_NKI:` work (and are testable) without neuronxcc.
_THREEFRY_LANES = 128


# Bulk kernels (#60): F Threefry batches per (P, F) chunk, C chunks per launch.
# F is capped at _BULK_COLS_MAX (SBUF per partition: the kernel keeps ~16 live
# (P, F) uint32 state tiles plus temporaries) and C at _BULK_CHUNKS_MAX (chunks
# are unrolled, so C bounds instruction count and compile time). Both are
# rounded up to powers of two so a handful of NEFFs serve every request size.
# TRNRAND_BULK_COLS overrides the column cap (must be a power of two <= 4096).
_BULK_COLS_MIN = 32
_BULK_COLS_MAX = int(os.environ.get("TRNRAND_BULK_COLS", "512"))
_BULK_CHUNKS_MAX = 16


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _chunk_bucket(chunks: int) -> int:
    """Smallest allowed chunk count >= `chunks`: powers of two and their 1.5x / 1.25x
    / 1.75x midpoints (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, ...), capped at
    _BULK_CHUNKS_MAX. At most ~4 chunk shapes per octave, so the NEFF count stays
    bounded, and a request never computes more than ~25% extra chunks (the
    pure power-of-two rounding cost up to 2x: 2^20 + 1 values ran 8 chunks)."""
    chunks = max(1, int(chunks))
    if chunks <= 4:
        return chunks
    p = _next_pow2(chunks) // 2  # chunks in (p, 2p]
    step = p // 4
    return min(_BULK_CHUNKS_MAX, -(-chunks // step) * step)


def _bulk_plan(n_elements: int) -> tuple[int, int, int]:
    """(cols F, chunks C, launches) for `n_elements` values on the bulk kernels.

    Launches split the chunks as evenly as possible, then C is rounded up to a
    `_chunk_bucket` size, so the total work tracks the request instead of
    jumping to the next power of two.
    """
    n_batches = max(1, -(-int(n_elements) // (_THREEFRY_LANES * 4)))
    if n_batches <= _BULK_COLS_MAX:
        return max(_BULK_COLS_MIN, _next_pow2(n_batches)), 1, 1
    F = _BULK_COLS_MAX
    chunks = -(-n_batches // F)
    launches = -(-chunks // _BULK_CHUNKS_MAX)
    C = _chunk_bucket(-(-chunks // launches))
    return F, C, -(-chunks // C)


def _bulk_final_order(op_name: str) -> bool:
    """True for the bulk kernels that write per-tile order themselves (the full-word ones)."""
    return "bulk32" in op_name


def _bulk_lane_tile(cols: int, final_order: bool) -> np.ndarray:
    """The (P, cols) int32 lane input `c0` of the bulk kernels.

    Old layout (8/16-bit kernels): lane p in every column. Final-order layout
    (full-word kernels): element (p, j) is pair t = p·cols + j, lane t & 127.
    """
    lanes = _THREEFRY_LANES
    if final_order:
        t = np.arange(lanes * cols, dtype=np.int32).reshape(lanes, cols)
        return np.ascontiguousarray(t & (lanes - 1))
    return np.ascontiguousarray(
        np.broadcast_to(np.arange(lanes, dtype=np.int32).reshape(-1, 1), (lanes, cols))
    )


def _bulk_tile_order(out: torch.Tensor, cols: int) -> torch.Tensor:
    """Flatten one bulk launch's (P, C × 4 × F) output in per-tile (batch, lane, word) order.

    The bulk kernels write chunk c, word w, column j of lane p at
    out[p, c·4F + w·F + j], and column j of chunk c is batch c·F + j of the
    launch. Module-level so it is testable without NKI.
    """
    lanes = out.shape[0]
    chunks = out.shape[1] // (4 * cols)
    return out.reshape(lanes, chunks, 4, cols).permute(1, 3, 0, 2).contiguous().reshape(-1)


def _stream_tile_order(out: torch.Tensor) -> torch.Tensor:
    """Flatten one streaming launch's (P, 4 × _PROGRAM_TILES) output in per-tile order.

    The streaming kernels write tile T's four words for lane p to
    out[p, 4T : 4T + 4], so a row-major flatten yields (P, tiles, 4) order. The
    per-tile wrappers (threefry_normal_nki / threefry_uniform_nki) emit
    (tiles, P, 4): all lanes of batch 0, then batch 1, and so on. Permuting on the
    host makes a stream element-for-element identical to the per-tile path for the
    same seed and counter_offset (#58). Module-level so it is testable without NKI.
    """
    lanes = out.shape[0]
    return out.reshape(lanes, _PROGRAM_TILES, 4).permute(1, 0, 2).contiguous().reshape(-1)


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

    (s, a, m_init) rows are from Joe-Kuo ``new-joe-kuo-6.21201``, the table
    ``torch.quasirandom.SobolEngine`` uses; ``tests/test_quasi.py`` checks the
    result against ``SobolEngine.sobolstate`` for every dimension (#59).
    """
    joe_kuo = [
        (1, 0, [1]),  # dim 2
        (2, 1, [1, 3]),  # dim 3
        (3, 1, [1, 3, 1]),  # dim 4
        (3, 2, [1, 1, 1]),  # dim 5
        (4, 1, [1, 1, 3, 3]),  # dim 6
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


def sobol_reference(n_points: int, n_dims: int, start_index: int = 0) -> torch.Tensor:
    """CPU reference for the NKI Sobol path (unscrambled), bit-exact.

    Mirrors ``sobol_gray_code_kernel``: point ``i`` is the XOR of
    ``_SOBOL_DIR_VECS[d][k]`` over the set bits ``k`` of ``gray(i)``, scaled by
    ``2^-24``. Equals the top 24 bits of ``SobolEngine(scramble=False)`` after
    ``fast_forward(start_index)``.

    Returns:
        Float32 tensor of shape (n_points, n_dims) with values in [0, 1).
    """
    assert 1 <= n_dims <= _SOBOL_MAX_DIMS
    idx = np.arange(start_index, start_index + n_points, dtype=np.int64)
    gray = idx ^ (idx >> 1)
    dirs = np.asarray(_SOBOL_DIR_VECS[:n_dims], dtype=np.int64)  # (n_dims, B)
    acc = np.zeros((n_points, n_dims), dtype=np.int64)
    for k in range(_SOBOL_BITS):
        bit = ((gray >> k) & 1).astype(bool)
        acc[bit] ^= dirs[:, k]
    return torch.from_numpy((acc * _SOBOL_SCALE).astype(np.float32))


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

    # Integer add/multiply go through `nisa.tensor_tensor` directly, not
    # `nl.add` / `nl.multiply`. `nl`'s binary ops lower any (P, 1) operand —
    # a Python scalar *or* a (P, 1) tile, which is every tile in these
    # kernels — to `nisa.tensor_scalar`, and arithmetic `tensor_scalar`
    # (`tensor_scalar_arith`) requires a float32 operand. Newer compilers
    # (neuronxcc 2.26, NKI 0.5+) reject a uint32 operand outright:
    # "'nisa.tensor_scalar_arith' op 'operand0' must be float32, got 'ui32'"
    # (#56). `tensor_tensor` accepts integer operands. Bitwise ops and
    # shifts lower to `tensor_scalar_bitvec`, which takes integer operands,
    # so they keep their scalar operands.

    def _nki_mul_u32(x, y):
        out = nl.ndarray(x.shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_tensor(out, x, y, op=nl.multiply)
        return out

    def _nki_add_u32(x, y):
        out = nl.ndarray(x.shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_tensor(out, x, y, op=nl.add)
        return out

    # int32 <-> uint32 crossings use `.view(dtype)`, a bit reinterpretation
    # (no data copied, like `numpy.ndarray.view`), never `nl.copy(x, dtype=...)`.
    # `nl.copy` and dtype-converting loads/stores are VALUE conversions: on
    # Trn2 hardware and in `nki.simulate` they saturate (uint32 >= 2^31 ->
    # int32 0x7FFFFFFF, negative int32 -> uint32 0), which corrupted Philox,
    # Threefry keys >= 2^31 and the streaming kernels. Kernel inputs and
    # outputs stay int32 (torch has no general uint32 device path); the
    # arithmetic is all uint32, so the bits are reinterpreted once at each
    # boundary.

    def _as_u32(t):
        """Reinterpret an int32 tile's bits as uint32 (no value conversion)."""
        return t.view(nl.uint32)

    def _as_i32(t):
        """Reinterpret a uint32 tile's bits as int32 (no value conversion)."""
        return t.view(nl.int32)

    def _nki_carry_step(acc, col):
        """Extract low byte and carry, add carry to next column."""
        byte = nl.bitwise_and(acc, 0xFF, dtype=nl.uint32)
        carry = nl.right_shift(acc, 8, dtype=nl.uint32)
        return byte, _nki_add_u32(carry, col)

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
        """32×32→64 multiply returning (hi32, lo32) uint32 tiles.

        8-bit byte decomposition:

            a = a3·2^24 + a2·2^16 + a1·2^8 + a0   (ai ∈ [0,255])
            b = b3·2^24 + b2·2^16 + b1·2^8 + b0

            p_ij = ai · bj                       ≤ 65025   (2^16)
            c_k  = Σ{i+j=k} p_ij                 ≤ 2^18    (k=0..6)

        Byte-wise carry propagation over c0..c6 yields eight output
        bytes; low four pack lo32, high four pack hi32. Every
        intermediate stays well below 2^24, avoiding the float32
        precision loss on the activation-engine multiply path.

        `a` is a uint32 tile of shape (P, 1). `b_l`, `b_h` are Python-int
        constants (low/high 16 bits of the full 32-bit multiplier).
        """
        # `a` is uint32 so the dst/src dtypes match in the MLIR
        # tensor_scalar_bitvec ops; the hardware verifier rejects
        # src=i32/dst=ui32 mismatches.
        a_u = a
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

        # Stay uint32: the Philox state is uint32 end to end and is only
        # reinterpreted as int32 at the output store.
        return hi32_u, lo32_u

    @nki.jit
    def philox4x32_kernel(counter_lo_ref, key_lo_ref, key_hi_ref):
        """Philox 4×32-10 NKI kernel on GpSimd.

        Each of the P partition-axis lanes runs one independent Philox
        stream. Input shape is `(P, 1)` per NKI convention (partition,
        free). Output is `(P, 4)` — one int32 per round-output word per
        lane. The wrapper flattens to `(P*4,)` interleaved.
        """
        P = counter_lo_ref.shape[0]
        # Inputs are int32 bit patterns of uint32 words: reinterpret, don't
        # convert (a value conversion saturates words >= 2^31).
        c0 = _as_u32(nl.load(counter_lo_ref))
        c1 = nl.zeros_like(c0)
        c2 = nl.zeros_like(c0)
        c3 = nl.zeros_like(c0)
        k0 = _as_u32(nl.load(key_lo_ref))
        k1 = _as_u32(nl.load(key_hi_ref))

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
            k0 = _nki_add_u32(k0, w0_vec)
            k1 = _nki_add_u32(k1, w1_vec)

        # int32 output holding the uint32 bit patterns (reinterpret, not
        # a saturating uint32 -> int32 value conversion).
        out = nl.ndarray((P, 4), dtype=nl.int32, buffer=nl.shared_hbm)
        out[:, 0:1] = _as_i32(c0)
        out[:, 1:2] = _as_i32(c1)
        out[:, 2:3] = _as_i32(c2)
        out[:, 3:4] = _as_i32(c3)
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

        On the XLA path, importing `torch_xla` inside `_to_xla` also populates
        `sys.modules['torch_neuronx']` before the first kernel invocation.
        On the native path the launch goes through `trnrand::philox4x32`.

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
                # Kernel returns (tile_len, 4); flatten to interleaved
                # [c0_0, c1_0, c2_0, c3_0, c0_1, c1_1, ...].
                out_tile = _launch(
                    philox4x32_kernel, "philox4x32", cl_tile, kl_tile, kh_tile
                ).reshape(-1)
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
                # Kernel returns (tile_len, 2); flatten to interleaved [z1, z2, z1, z2, ...].
                out_tile = _launch(box_muller_kernel, "box_muller", tile_slice).reshape(-1)
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

    def _b_split(v_ref):
        """Load a (P,1) int32 tile and split into 4 byte tiles, each in [0,255].

        Returns [b0, b1, b2, b3] where b0 is the least-significant byte.
        All returned tiles are uint32-typed with values ≤ 255.
        """
        v_u = _as_u32(nl.load(v_ref))
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
        s0 = _nki_add_u32(a_b[0], b_b[0])  # ≤ 510
        c0 = nl.right_shift(s0, 8, dtype=nl.uint32)
        r0 = nl.bitwise_and(s0, 0xFF, dtype=nl.uint32)
        s1 = _nki_add_u32(_nki_add_u32(a_b[1], b_b[1]), c0)
        c1 = nl.right_shift(s1, 8, dtype=nl.uint32)
        r1 = nl.bitwise_and(s1, 0xFF, dtype=nl.uint32)
        s2 = _nki_add_u32(_nki_add_u32(a_b[2], b_b[2]), c1)
        c2 = nl.right_shift(s2, 8, dtype=nl.uint32)
        r2 = nl.bitwise_and(s2, 0xFF, dtype=nl.uint32)
        s3 = _nki_add_u32(_nki_add_u32(a_b[3], b_b[3]), c2)
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
    # Direction vectors and the CPU reference live at module level (above
    # `if HAS_NKI:`) so CPU-only installs can check them against SobolEngine.

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
        g = _as_u32(nl.load(gray_ref))

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
                        _nki_mul_u32(bit_k, nl.full((P, 1), v_dk, dtype=nl.uint32)),
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
                out_tile = _launch(sobol_gray_code_kernel, "sobol_gray_code", gray_tile)
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
                out_tile = _launch(halton_kernel, "halton", tile)
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
        start_batch_u = _as_u32(nl.load(start_batch_ref))

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
                _nki_add_u32(start_batch_u, nl.full((P, 1), tile_idx, dtype=nl.uint32)),
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

        start_batch_u = _as_u32(nl.load(start_batch_ref))

        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        scale256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        scale65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)

        out = nl.ndarray((P, 4 * _PROGRAM_TILES), dtype=nl.float32, buffer=nl.shared_hbm)

        for tile_idx in nl.static_range(_PROGRAM_TILES):
            c1_u = nl.bitwise_and(
                _nki_add_u32(start_batch_u, nl.full((P, 1), tile_idx, dtype=nl.uint32)),
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

    # ── Bulk Threefry kernels: wide free dimension, one launch per request (#60) ──
    #
    # The streaming kernels above hold every byte of every word as a (P, 1)
    # tile, so each NKI instruction does 128 elements of work (one per
    # partition) and a 16,384-value launch is ~55k instructions. That is why
    # `normal()` was linear at ~0.23 ms per 1k values on Trn2: instruction
    # issue, not the engines, set the rate. The bulk kernels run the same
    # byte-tile arithmetic on (P, F) tiles, F columns = F Threefry batches at
    # once, so the instruction count per chunk is the same as one old tile
    # but each instruction covers P × F elements.
    #
    # Layout: column j of chunk c is batch (start[c] + j) & 0xFFFFFF, lane p
    # is c0 = p, exactly the per-tile counter mapping (#58). The chunk's four
    # output words land in one (P, 4F) SBUF tile, word w at columns
    # [w F, (w + 1) F), and leave in a single DMA. The host reorders
    # (chunks, words, cols) into per-tile order (`_bulk_tile_order`).
    #
    # Key words are added with `tensor_scalar` against float32 (P, 1) byte
    # operands (byte + byte <= 510, exact in float32), so the key schedule is
    # never broadcast to (P, F). Integer state + state adds stay on
    # `tensor_tensor` with matching uint32 dtypes.

    def _ts_add_u32(x, f):
        """x (P, F) uint32 + f (P, 1) float32 -> (P, F) uint32 (exact for sums < 2^24)."""
        out = nl.ndarray(x.shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_scalar(out, x, op0=nl.add, operand0=f)
        return out

    def _bytes_f32(b):
        """Four (P, 1) uint32 byte tiles -> four (P, 1) float32 byte tiles (values <= 255)."""
        return [
            nl.copy(b[0], dtype=nl.float32),
            nl.copy(b[1], dtype=nl.float32),
            nl.copy(b[2], dtype=nl.float32),
            nl.copy(b[3], dtype=nl.float32),
        ]

    def _add32_bs(a_b, k_f):
        """(a + k) mod 2^32 in byte tiles; a: 4 x (P, F) uint32, k: 4 x (P, 1) float32."""
        s0 = _ts_add_u32(a_b[0], k_f[0])
        c0 = nl.right_shift(s0, 8, dtype=nl.uint32)
        r0 = nl.bitwise_and(s0, 0xFF, dtype=nl.uint32)
        s1 = _nki_add_u32(_ts_add_u32(a_b[1], k_f[1]), c0)
        c1 = nl.right_shift(s1, 8, dtype=nl.uint32)
        r1 = nl.bitwise_and(s1, 0xFF, dtype=nl.uint32)
        s2 = _nki_add_u32(_ts_add_u32(a_b[2], k_f[2]), c1)
        c2 = nl.right_shift(s2, 8, dtype=nl.uint32)
        r2 = nl.bitwise_and(s2, 0xFF, dtype=nl.uint32)
        s3 = _nki_add_u32(_ts_add_u32(a_b[3], k_f[3]), c2)
        r3 = nl.bitwise_and(s3, 0xFF, dtype=nl.uint32)
        return [r0, r1, r2, r3]

    def _key_inject_bs(x_b_list, ks_f, ksp_f, step):
        """`_key_inject_b` with (P, 1) float32 key bytes; ksp_f[s] = ks[(s+3)%5] + s."""
        out0 = _add32_bs(x_b_list[0], ks_f[(step + 0) % 5])
        out1 = _add32_bs(x_b_list[1], ks_f[(step + 1) % 5])
        out2 = _add32_bs(x_b_list[2], ks_f[(step + 2) % 5])
        out3 = _add32_bs(x_b_list[3], ksp_f[step])
        return [out0, out1, out2, out3]

    def _u24_f32(b, scale256, scale65536):
        """Low 24 bits of a byte-tile word as float32 b0 + 256 b1 + 65536 b2 (exact)."""
        return nl.add(
            nl.add(
                nl.copy(b[0], dtype=nl.float32),
                nl.multiply(nl.copy(b[1], dtype=nl.float32), scale256, dtype=nl.float32),
                dtype=nl.float32,
            ),
            nl.multiply(nl.copy(b[2], dtype=nl.float32), scale65536, dtype=nl.float32),
            dtype=nl.float32,
        )

    def _threefry_bulk_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, normal):
        P = c0_ref.shape[0]
        F = c0_ref.shape[1]
        C = start_ref.shape[1]

        # Key schedule, (P, 1): bytes as uint32 for the xor, then float32 operands.
        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)
        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]
        ks_f = [
            _bytes_f32(k0_b),
            _bytes_f32(k1_b),
            _bytes_f32(k2_b),
            _bytes_f32(k3_b),
            _bytes_f32(k4_b),
        ]
        # x3 injection adds ks[(s+3)%5] + s; fold s into the key word once.
        ksp_f = [
            _bytes_f32(_add32_b(ks_b[3], _b_from_scalar(P, 0))),
            _bytes_f32(_add32_b(ks_b[4], _b_from_scalar(P, 1))),
            _bytes_f32(_add32_b(ks_b[0], _b_from_scalar(P, 2))),
            _bytes_f32(_add32_b(ks_b[1], _b_from_scalar(P, 3))),
            _bytes_f32(_add32_b(ks_b[2], _b_from_scalar(P, 4))),
            _bytes_f32(_add32_b(ks_b[3], _b_from_scalar(P, 5))),
        ]

        zero = nl.full((P, F), 0, dtype=nl.uint32)
        c0_b = _b_split(c0_ref)  # (P, F): lane index in every column

        # Column index j (0..F-1) in SBUF, as byte tiles; F <= 2^16.
        col = nl.ndarray((P, F), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(col, [[1, F]], offset=0, channel_multiplier=0)
        col_b = [
            nl.bitwise_and(col, 0xFF, dtype=nl.uint32),
            nl.right_shift(col, 8, dtype=nl.uint32),
            zero,
            zero,
        ]

        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        scale256 = nl.full((P, 1), 256.0, dtype=nl.float32)
        scale65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)
        clamp_eps = nl.full((P, 1), 1e-7, dtype=nl.float32)
        neg_two = nl.full((P, 1), -2.0, dtype=nl.float32)
        two_pi = nl.full((P, 1), TWO_PI, dtype=nl.float32)

        out = nl.ndarray((P, C * 4 * F), dtype=nl.float32, buffer=nl.shared_hbm)

        for c in nl.static_range(C):
            # c1 = (start[c] + j) & 0xFFFFFF; start[c] is already masked to 24 bits.
            start_b = _b_split(start_ref[:, c : c + 1])
            c1_full = _add32_bs(col_b, _bytes_f32(start_b))
            c1_b = [c1_full[0], c1_full[1], c1_full[2], zero]

            _ki = _key_inject_bs(
                [c0_b, c1_b, [zero, zero, zero, zero], [zero, zero, zero, zero]], ks_f, ksp_f, 0
            )
            x0_b = _ki[0]
            x1_b = _ki[1]
            x2_b = _ki[2]
            x3_b = _ki[3]

            for round_num in nl.static_range(THREEFRY_ROUNDS):
                pair_idx = round_num % 4
                q0, r0 = _THREEFRY_ROT_QR[pair_idx * 2 + 0]
                q1, r1 = _THREEFRY_ROT_QR[pair_idx * 2 + 1]
                if round_num % 2 == 0:
                    x0_b, x1_b = _mix_b(x0_b, x1_b, q0, r0)
                    x2_b, x3_b = _mix_b(x2_b, x3_b, q1, r1)
                else:
                    x0_b, x3_b = _mix_b(x0_b, x3_b, q0, r0)
                    x2_b, x1_b = _mix_b(x2_b, x1_b, q1, r1)
                if (round_num + 1) % 4 == 0:
                    _ki = _key_inject_bs(
                        [x0_b, x1_b, x2_b, x3_b], ks_f, ksp_f, (round_num + 1) // 4
                    )
                    x0_b = _ki[0]
                    x1_b = _ki[1]
                    x2_b = _ki[2]
                    x3_b = _ki[3]

            res = nl.ndarray((P, 4 * F), dtype=nl.float32, buffer=nl.sbuf)
            if normal:
                # Same op sequence as threefry_streaming_normal_kernel, on (P, F).
                u0 = nl.maximum(
                    nl.multiply(_u24_f32(x0_b, scale256, scale65536), inv24, dtype=nl.float32),
                    clamp_eps,
                )
                u1 = nl.maximum(
                    nl.multiply(_u24_f32(x1_b, scale256, scale65536), inv24, dtype=nl.float32),
                    clamp_eps,
                )
                u2 = nl.maximum(
                    nl.multiply(_u24_f32(x2_b, scale256, scale65536), inv24, dtype=nl.float32),
                    clamp_eps,
                )
                u3 = nl.maximum(
                    nl.multiply(_u24_f32(x3_b, scale256, scale65536), inv24, dtype=nl.float32),
                    clamp_eps,
                )
                bm_r0 = nl.sqrt(nl.multiply(nl.log(u0), neg_two))
                bm_theta0 = nl.multiply(u1, two_pi)
                bm_r1 = nl.sqrt(nl.multiply(nl.log(u2), neg_two))
                bm_theta1 = nl.multiply(u3, two_pi)
                nisa.tensor_tensor(res[:, 0:F], bm_r0, nl.cos(bm_theta0), op=nl.multiply)
                nisa.tensor_tensor(res[:, F : 2 * F], bm_r0, nl.sin(bm_theta0), op=nl.multiply)
                nisa.tensor_tensor(res[:, 2 * F : 3 * F], bm_r1, nl.cos(bm_theta1), op=nl.multiply)
                nisa.tensor_tensor(res[:, 3 * F : 4 * F], bm_r1, nl.sin(bm_theta1), op=nl.multiply)
            else:
                nisa.tensor_scalar(
                    res[:, 0:F],
                    _u24_f32(x0_b, scale256, scale65536),
                    op0=nl.multiply,
                    operand0=inv24,
                )
                nisa.tensor_scalar(
                    res[:, F : 2 * F],
                    _u24_f32(x1_b, scale256, scale65536),
                    op0=nl.multiply,
                    operand0=inv24,
                )
                nisa.tensor_scalar(
                    res[:, 2 * F : 3 * F],
                    _u24_f32(x2_b, scale256, scale65536),
                    op0=nl.multiply,
                    operand0=inv24,
                )
                nisa.tensor_scalar(
                    res[:, 3 * F : 4 * F],
                    _u24_f32(x3_b, scale256, scale65536),
                    op0=nl.multiply,
                    operand0=inv24,
                )

            base = c * 4 * F
            nl.store(out[:, base : base + 4 * F], res)

        return out

    @nki.jit
    def threefry_bulk_normal_kernel(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Threefry4×32-20 + Box-Muller over C chunks of (P, F) batches in one launch (#60).

        Inputs:
          c0_ref:    (P, F) int32 — lane index p in every column (sets F)
          start_ref: (P, C) int32 — chunk c's first batch index, masked to 24 bits (sets C)
          k0..k3:    (P, 1) int32 — Threefry key words

        Output: (P, C × 4 × F) float32; chunk c, word w, column j at
        c·4F + w·F + j is normal w of lane p, batch start[c] + j.
        """
        return _threefry_bulk_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, True)

    @nki.jit
    def threefry_bulk_uniform_kernel(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Uniform twin of `threefry_bulk_normal_kernel`: float32 in [0, 1), same layout."""
        return _threefry_bulk_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, False)

    # ── 16-bit limb variant of the bulk kernels (#60, TRNRAND_LIMB_BITS=16, diagnostic) ──
    #
    # Why limbs at all: aws-neuron-sdk#1308 / ed05ad3 -- integer arithmetic
    # (`nl.add`, `nl.multiply`, `tensor_scalar_arith`) runs through the
    # engines' float32 datapath, so a uint32 add is exact only while every
    # operand and result is < 2^24. A full-word (mod 2^32) add is therefore
    # not available; bitwise ops (and/or/xor/shifts, `tensor_scalar_bitvec`)
    # are integer-exact. Bytes (8-bit limbs) were chosen originally because
    # the Philox 32x32 multiply needed byte sub-products (<= 2^16) and the
    # Threefry port reused the same representation. Threefry has no multiply,
    # so 16-bit limbs suffice: limb + limb + carry <= 2^17 - 1, exact in fp32.
    # Every word is [lo, hi], each a (P, F) uint32 tile in [0, 2^16).
    # Half the limbs -> about half the instructions per add / xor / rotate.
    # Left shifts mask first, so no intermediate ever reaches 2^16 either.

    _THREEFRY_ROT_QR16 = tuple((rot // 16, rot % 16) for pair in THREEFRY_ROTATIONS for rot in pair)

    def _w16_from_bytes_f32(b):
        """Four (P, 1) uint32 byte tiles -> [lo, hi] (P, 1) float32 limbs (< 2^16)."""
        lo = nl.bitwise_or(b[0], nl.left_shift(b[1], 8, dtype=nl.uint32), dtype=nl.uint32)
        hi = nl.bitwise_or(b[2], nl.left_shift(b[3], 8, dtype=nl.uint32), dtype=nl.uint32)
        return [nl.copy(lo, dtype=nl.float32), nl.copy(hi, dtype=nl.float32)]

    def _add32_h(a_h, b_h):
        """(a + b) mod 2^32, both [lo, hi] (P, F) uint32 limbs; sums < 2^17."""
        s0 = _nki_add_u32(a_h[0], b_h[0])
        c0 = nl.right_shift(s0, 16, dtype=nl.uint32)
        r0 = nl.bitwise_and(s0, 0xFFFF, dtype=nl.uint32)
        s1 = _nki_add_u32(_nki_add_u32(a_h[1], b_h[1]), c0)
        r1 = nl.bitwise_and(s1, 0xFFFF, dtype=nl.uint32)
        return [r0, r1]

    def _add32_hs(a_h, k_f):
        """(a + k) mod 2^32; a: [lo, hi] (P, F) uint32, k: [lo, hi] (P, 1) float32."""
        s0 = _ts_add_u32(a_h[0], k_f[0])
        c0 = nl.right_shift(s0, 16, dtype=nl.uint32)
        r0 = nl.bitwise_and(s0, 0xFFFF, dtype=nl.uint32)
        s1 = _nki_add_u32(_ts_add_u32(a_h[1], k_f[1]), c0)
        r1 = nl.bitwise_and(s1, 0xFFFF, dtype=nl.uint32)
        return [r0, r1]

    def _rot_limb16(h, lo_src, r):
        """((h << r) | (lo_src >> (16 - r))) & 0xFFFF with h masked before the shift."""
        return nl.bitwise_or(
            nl.left_shift(
                nl.bitwise_and(h, (1 << (16 - r)) - 1, dtype=nl.uint32), r, dtype=nl.uint32
            ),
            nl.right_shift(lo_src, 16 - r, dtype=nl.uint32),
            dtype=nl.uint32,
        )

    def _rotl32_h(x_h, q, r):
        """Rotate-left by 16 q + r bits on [lo, hi] limbs."""
        if q == 0:
            a, b = x_h[0], x_h[1]
        else:
            a, b = x_h[1], x_h[0]
        # After the q swap the word is b:a; output lo = (a << r) | (b >> (16 - r)), hi likewise.
        if r == 0:
            return [a, b]
        return [_rot_limb16(a, b, r), _rot_limb16(b, a, r)]

    def _mix_h(a_h, b_h, q, r):
        a_new = _add32_h(a_h, b_h)
        b_rot = _rotl32_h(b_h, q, r)
        b_new = [
            nl.bitwise_xor(b_rot[0], a_new[0], dtype=nl.uint32),
            nl.bitwise_xor(b_rot[1], a_new[1], dtype=nl.uint32),
        ]
        return a_new, b_new

    def _key_inject_hs(x_h, ks_f, ksp_f, step):
        out0 = _add32_hs(x_h[0], ks_f[(step + 0) % 5])
        out1 = _add32_hs(x_h[1], ks_f[(step + 1) % 5])
        out2 = _add32_hs(x_h[2], ks_f[(step + 2) % 5])
        out3 = _add32_hs(x_h[3], ksp_f[step])
        return [out0, out1, out2, out3]

    def _u24_f32_h(w, scale65536):
        """Low 24 bits of a [lo, hi] word as float32 lo + 65536 (hi & 0xFF) (exact)."""
        return nl.add(
            nl.copy(w[0], dtype=nl.float32),
            nl.multiply(
                nl.copy(nl.bitwise_and(w[1], 0xFF, dtype=nl.uint32), dtype=nl.float32),
                scale65536,
                dtype=nl.float32,
            ),
            dtype=nl.float32,
        )

    def _threefry_bulk16_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, normal):
        P = c0_ref.shape[0]
        F = c0_ref.shape[1]
        C = start_ref.shape[1]

        # Key schedule on (P, 1) byte tiles (as in the 8-bit body), then 16-bit limbs.
        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)
        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]
        ks_f = [
            _w16_from_bytes_f32(k0_b),
            _w16_from_bytes_f32(k1_b),
            _w16_from_bytes_f32(k2_b),
            _w16_from_bytes_f32(k3_b),
            _w16_from_bytes_f32(k4_b),
        ]
        ksp_f = [
            _w16_from_bytes_f32(_add32_b(ks_b[3], _b_from_scalar(P, 0))),
            _w16_from_bytes_f32(_add32_b(ks_b[4], _b_from_scalar(P, 1))),
            _w16_from_bytes_f32(_add32_b(ks_b[0], _b_from_scalar(P, 2))),
            _w16_from_bytes_f32(_add32_b(ks_b[1], _b_from_scalar(P, 3))),
            _w16_from_bytes_f32(_add32_b(ks_b[2], _b_from_scalar(P, 4))),
            _w16_from_bytes_f32(_add32_b(ks_b[3], _b_from_scalar(P, 5))),
        ]

        zero = nl.full((P, F), 0, dtype=nl.uint32)
        c0_u = _as_u32(nl.load(c0_ref))  # lane index p < 128 in every column
        c0_h = [c0_u, zero]

        col = nl.ndarray((P, F), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(col, [[1, F]], offset=0, channel_multiplier=0)
        col_h = [col, zero]  # F <= 2^16

        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        scale65536 = nl.full((P, 1), 65536.0, dtype=nl.float32)
        clamp_eps = nl.full((P, 1), 1e-7, dtype=nl.float32)
        neg_two = nl.full((P, 1), -2.0, dtype=nl.float32)
        two_pi = nl.full((P, 1), TWO_PI, dtype=nl.float32)

        out = nl.ndarray((P, C * 4 * F), dtype=nl.float32, buffer=nl.shared_hbm)

        for c in nl.static_range(C):
            # c1 = (start[c] + j) & 0xFFFFFF; start[c] is already masked to 24 bits.
            start_b = _b_split(start_ref[:, c : c + 1])
            c1_full = _add32_hs(col_h, _w16_from_bytes_f32(start_b))
            c1_h = [c1_full[0], nl.bitwise_and(c1_full[1], 0xFF, dtype=nl.uint32)]

            _ki = _key_inject_hs([c0_h, c1_h, [zero, zero], [zero, zero]], ks_f, ksp_f, 0)
            x0 = _ki[0]
            x1 = _ki[1]
            x2 = _ki[2]
            x3 = _ki[3]

            for round_num in nl.static_range(THREEFRY_ROUNDS):
                pair_idx = round_num % 4
                q0, r0 = _THREEFRY_ROT_QR16[pair_idx * 2 + 0]
                q1, r1 = _THREEFRY_ROT_QR16[pair_idx * 2 + 1]
                if round_num % 2 == 0:
                    x0, x1 = _mix_h(x0, x1, q0, r0)
                    x2, x3 = _mix_h(x2, x3, q1, r1)
                else:
                    x0, x3 = _mix_h(x0, x3, q0, r0)
                    x2, x1 = _mix_h(x2, x1, q1, r1)
                if (round_num + 1) % 4 == 0:
                    _ki = _key_inject_hs([x0, x1, x2, x3], ks_f, ksp_f, (round_num + 1) // 4)
                    x0 = _ki[0]
                    x1 = _ki[1]
                    x2 = _ki[2]
                    x3 = _ki[3]

            res = nl.ndarray((P, 4 * F), dtype=nl.float32, buffer=nl.sbuf)
            if normal:
                u0 = nl.maximum(
                    nl.multiply(_u24_f32_h(x0, scale65536), inv24, dtype=nl.float32), clamp_eps
                )
                u1 = nl.maximum(
                    nl.multiply(_u24_f32_h(x1, scale65536), inv24, dtype=nl.float32), clamp_eps
                )
                u2 = nl.maximum(
                    nl.multiply(_u24_f32_h(x2, scale65536), inv24, dtype=nl.float32), clamp_eps
                )
                u3 = nl.maximum(
                    nl.multiply(_u24_f32_h(x3, scale65536), inv24, dtype=nl.float32), clamp_eps
                )
                bm_r0 = nl.sqrt(nl.multiply(nl.log(u0), neg_two))
                bm_theta0 = nl.multiply(u1, two_pi)
                bm_r1 = nl.sqrt(nl.multiply(nl.log(u2), neg_two))
                bm_theta1 = nl.multiply(u3, two_pi)
                nisa.tensor_tensor(res[:, 0:F], bm_r0, nl.cos(bm_theta0), op=nl.multiply)
                nisa.tensor_tensor(res[:, F : 2 * F], bm_r0, nl.sin(bm_theta0), op=nl.multiply)
                nisa.tensor_tensor(res[:, 2 * F : 3 * F], bm_r1, nl.cos(bm_theta1), op=nl.multiply)
                nisa.tensor_tensor(res[:, 3 * F : 4 * F], bm_r1, nl.sin(bm_theta1), op=nl.multiply)
            else:
                nisa.tensor_scalar(
                    res[:, 0:F], _u24_f32_h(x0, scale65536), op0=nl.multiply, operand0=inv24
                )
                nisa.tensor_scalar(
                    res[:, F : 2 * F], _u24_f32_h(x1, scale65536), op0=nl.multiply, operand0=inv24
                )
                nisa.tensor_scalar(
                    res[:, 2 * F : 3 * F],
                    _u24_f32_h(x2, scale65536),
                    op0=nl.multiply,
                    operand0=inv24,
                )
                nisa.tensor_scalar(
                    res[:, 3 * F : 4 * F],
                    _u24_f32_h(x3, scale65536),
                    op0=nl.multiply,
                    operand0=inv24,
                )

            base = c * 4 * F
            nl.store(out[:, base : base + 4 * F], res)

        return out

    @nki.jit
    def threefry_bulk16_normal_kernel(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """`threefry_bulk_normal_kernel` on 16-bit limbs; same inputs, layout and bits."""
        return _threefry_bulk16_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, True)

    @nki.jit
    def threefry_bulk16_uniform_kernel(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """`threefry_bulk_uniform_kernel` on 16-bit limbs; same inputs, layout and bits."""
        return _threefry_bulk16_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, False)

    # ── Full-word variant (#60, TRNRAND_LIMB_BITS=32, the native default) ──────
    #
    # NKI 0.7 documents that `nisa.tensor_tensor` with all-int32/uint32
    # operands runs on the GpSimd engine with native integer arithmetic
    # ("exact results for all 32-bit integer values"); earlier NKI routed it
    # through float32. The docs do not say the add WRAPS mod 2^32 and the
    # simulator cannot tell; the Trn2 run of ea3800e confirmed it (bit-identical
    # to the 8-bit kernels, tests/test_nki_native.py). Adds are pinned to
    # GpSimd; xor and rotate are bitvec ops (Vector engine, bit patterns).
    # Key words are broadcast to (P, F) once per launch, since
    # `tensor_scalar` arithmetic operands must be float32.

    def _add32_w(x, y):
        out = nl.ndarray(x.shape, dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_tensor(out, x, y, op=nl.add, engine=nisa.engine.gpsimd)
        return out

    def _rotl32_w(x, rot):
        return nl.bitwise_or(
            nl.left_shift(x, rot, dtype=nl.uint32),
            nl.right_shift(x, 32 - rot, dtype=nl.uint32),
            dtype=nl.uint32,
        )

    def _mix_w(a, b, rot):
        a_new = _add32_w(a, b)
        b_new = nl.bitwise_xor(_rotl32_w(b, rot), a_new, dtype=nl.uint32)
        return a_new, b_new

    def _pack_bytes_wide(b, zero):
        """Four (P, 1) uint32 byte tiles -> one (P, F) uint32 word (broadcast over F)."""
        w = nl.bitwise_or(
            nl.bitwise_or(b[0], nl.left_shift(b[1], 8, dtype=nl.uint32), dtype=nl.uint32),
            nl.bitwise_or(
                nl.left_shift(b[2], 16, dtype=nl.uint32),
                nl.left_shift(b[3], 24, dtype=nl.uint32),
                dtype=nl.uint32,
            ),
            dtype=nl.uint32,
        )
        return nl.bitwise_or(zero, w, dtype=nl.uint32)

    def _key_inject_w(x, ks_w, ksp_w, step):
        return [
            _add32_w(x[0], ks_w[(step + 0) % 5]),
            _add32_w(x[1], ks_w[(step + 1) % 5]),
            _add32_w(x[2], ks_w[(step + 2) % 5]),
            _add32_w(x[3], ksp_w[step]),
        ]

    # Layout and engine plan of the full-word kernels (#60 follow-up):
    #
    # * Output in final order. Partition q of chunk c holds the F consecutive
    #   (batch, lane) pairs t = q·F + f (lane = t & 127, batch = start[c] +
    #   (t >> 7)), and each pair's four words sit next to each other, so the
    #   (C·P, F, 4) output flattens straight to per-tile (batch, lane, word)
    #   order. The host does no reorder, and every partition leaves in one
    #   contiguous DMA. A chunk still covers batches start[c] .. start[c] + F - 1
    #   on all 128 lanes, so the values are the same as the 8-bit kernels.
    # * Both physical cores of a logical core (LNC=2, the Trn2 default). NKI
    #   traces the kernel once per core (`program_id`); core k computes chunks
    #   [k·C/2, (k+1)·C/2) when C is even, else columns [k·F/2, (k+1)·F/2) of
    #   every chunk. A gridless kernel that ignores `program_id` runs the whole
    #   kernel on both cores and writes it twice (what the 8/16-bit ones do).
    #   Per-core tiles are never wider than F, so SBUF use is as before.
    # * GpSimd integer adds are ~2/3 of the time and do not overlap the Vector
    #   bit ops: the two engines share the SBUF port, and a Vector instruction
    #   issued during a GpSimd add stretches to the add's length (Trn2 profile
    #   on #60). So the levers are fewer adds, fewer Vector instructions, and
    #   both cores; double-buffering the two engines buys nothing.
    #   Chunk-invariant work is hoisted: x0 = lane + ks0 is the same for every
    #   chunk, and x2 = 0 + ks2, x3 = 0 + ks3 are the key words themselves.
    # * Box-Muller: (x & 0xFFFFFF) feeds one arithmetic `tensor_scalar`
    #   (multiply by 2^-24, then max with 1e-7); its implicit uint32 -> float32
    #   cast is exact below 2^24, so it equals copy + multiply + maximum.

    def _threefry_bulk32_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, normal):
        P = c0_ref.shape[0]
        F = c0_ref.shape[1]
        C = start_ref.shape[1]
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        pid = nl.program_id(0) if n_prog > 1 else 0
        if C % n_prog == 0:  # whole chunks per core: full-width instructions
            W = F
            c_lo = pid * (C // n_prog)
            c_hi = c_lo + C // n_prog
        elif F % n_prog == 0:  # e.g. C = 1: split every chunk's columns
            W = F // n_prog
            c_lo = 0
            c_hi = C
        else:
            n_prog = 1
            pid = 0
            W = F
            c_lo = 0
            c_hi = C
        j0 = 0 if W == F else pid * W
        zero = nl.full((P, W), 0, dtype=nl.uint32)

        k0_b = _b_split(k0_ref)
        k1_b = _b_split(k1_ref)
        k2_b = _b_split(k2_ref)
        k3_b = _b_split(k3_ref)
        parity_b = _b_from_scalar(P, THREEFRY_SKEIN_KS_PARITY)
        k4_b = _xor32_b(_xor32_b(_xor32_b(_xor32_b(k0_b, k1_b), k2_b), k3_b), parity_b)
        ks_b = [k0_b, k1_b, k2_b, k3_b, k4_b]
        ks_w = [
            _pack_bytes_wide(k0_b, zero),
            _pack_bytes_wide(k1_b, zero),
            _pack_bytes_wide(k2_b, zero),
            _pack_bytes_wide(k3_b, zero),
            _pack_bytes_wide(k4_b, zero),
        ]
        ksp_w = [
            _pack_bytes_wide(_add32_b(ks_b[3], _b_from_scalar(P, 0)), zero),
            _pack_bytes_wide(_add32_b(ks_b[4], _b_from_scalar(P, 1)), zero),
            _pack_bytes_wide(_add32_b(ks_b[0], _b_from_scalar(P, 2)), zero),
            _pack_bytes_wide(_add32_b(ks_b[1], _b_from_scalar(P, 3)), zero),
            _pack_bytes_wide(_add32_b(ks_b[2], _b_from_scalar(P, 4)), zero),
            _pack_bytes_wide(_add32_b(ks_b[3], _b_from_scalar(P, 5)), zero),
        ]

        # Lane of pair t = q·F + f comes in as c0; its batch within the chunk is t >> 7.
        lane = _as_u32(nl.load(c0_ref[:, j0 : j0 + W]))
        t_idx = nl.ndarray((P, W), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(t_idx, [[1, W]], offset=j0, channel_multiplier=F)
        b_local = nl.right_shift(t_idx, 7, dtype=nl.uint32)
        x0_init = _add32_w(lane, ks_w[0])

        inv24 = nl.full((P, 1), 1.0 / 16777216.0, dtype=nl.float32)
        clamp_eps = nl.full((P, 1), 1e-7, dtype=nl.float32)
        neg_two = nl.full((P, 1), -2.0, dtype=nl.float32)
        two_pi = nl.full((P, 1), TWO_PI, dtype=nl.float32)

        out = nl.ndarray((C * P, F, 4), dtype=nl.float32, buffer=nl.shared_hbm)

        for c in nl.static_range(c_lo, c_hi):
            start_w = _pack_bytes_wide(_b_split(start_ref[:, c : c + 1]), zero)
            c1_w = nl.bitwise_and(_add32_w(b_local, start_w), 0xFFFFFF, dtype=nl.uint32)

            # Key injection 0 on counters (lane, c1, 0, 0).
            x0 = x0_init
            x1 = _add32_w(c1_w, ks_w[1])
            x2 = ks_w[2]
            x3 = ksp_w[0]

            for round_num in nl.static_range(THREEFRY_ROUNDS):
                rot0, rot1 = THREEFRY_ROTATIONS[round_num % 4]
                if round_num % 2 == 0:
                    x0, x1 = _mix_w(x0, x1, rot0)
                    x2, x3 = _mix_w(x2, x3, rot1)
                else:
                    x0, x3 = _mix_w(x0, x3, rot0)
                    x2, x1 = _mix_w(x2, x1, rot1)
                if (round_num + 1) % 4 == 0:
                    _ki = _key_inject_w([x0, x1, x2, x3], ks_w, ksp_w, (round_num + 1) // 4)
                    x0 = _ki[0]
                    x1 = _ki[1]
                    x2 = _ki[2]
                    x3 = _ki[3]

            m0 = nl.bitwise_and(x0, 0xFFFFFF, dtype=nl.uint32)
            m1 = nl.bitwise_and(x1, 0xFFFFFF, dtype=nl.uint32)
            m2 = nl.bitwise_and(x2, 0xFFFFFF, dtype=nl.uint32)
            m3 = nl.bitwise_and(x3, 0xFFFFFF, dtype=nl.uint32)
            res = nl.ndarray((P, W, 4), dtype=nl.float32, buffer=nl.sbuf)
            if normal:
                u0 = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                u1 = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                u2 = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                u3 = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    u0, m0, op0=nl.multiply, operand0=inv24, op1=nl.maximum, operand1=clamp_eps
                )
                nisa.tensor_scalar(
                    u1, m1, op0=nl.multiply, operand0=inv24, op1=nl.maximum, operand1=clamp_eps
                )
                nisa.tensor_scalar(
                    u2, m2, op0=nl.multiply, operand0=inv24, op1=nl.maximum, operand1=clamp_eps
                )
                nisa.tensor_scalar(
                    u3, m3, op0=nl.multiply, operand0=inv24, op1=nl.maximum, operand1=clamp_eps
                )
                bm_r0 = nl.sqrt(nl.multiply(nl.log(u0), neg_two))
                bm_theta0 = nl.multiply(u1, two_pi)
                bm_r1 = nl.sqrt(nl.multiply(nl.log(u2), neg_two))
                bm_theta1 = nl.multiply(u3, two_pi)
                nisa.tensor_tensor(res[:, :, 0], bm_r0, nl.cos(bm_theta0), op=nl.multiply)
                nisa.tensor_tensor(res[:, :, 1], bm_r0, nl.sin(bm_theta0), op=nl.multiply)
                nisa.tensor_tensor(res[:, :, 2], bm_r1, nl.cos(bm_theta1), op=nl.multiply)
                nisa.tensor_tensor(res[:, :, 3], bm_r1, nl.sin(bm_theta1), op=nl.multiply)
            else:
                nisa.tensor_scalar(res[:, :, 0], m0, op0=nl.multiply, operand0=inv24)
                nisa.tensor_scalar(res[:, :, 1], m1, op0=nl.multiply, operand0=inv24)
                nisa.tensor_scalar(res[:, :, 2], m2, op0=nl.multiply, operand0=inv24)
                nisa.tensor_scalar(res[:, :, 3], m3, op0=nl.multiply, operand0=inv24)

            nl.store(out[c * P : (c + 1) * P, j0 : j0 + W, :], res)

        return out

    @nki.jit
    def threefry_bulk32_normal_kernel(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """`threefry_bulk_normal_kernel` on full uint32 words (GpSimd integer adds).

        Inputs as `threefry_bulk_normal_kernel`, except c0 is the lane of pair
        t = p·F + j, i.e. (p·F + j) & 127 (`_bulk_lane_tile`). Output:
        (C × P, F, 4) float32 in per-tile (batch, lane, word) order (#60).
        """
        return _threefry_bulk32_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, True)

    @nki.jit
    def threefry_bulk32_uniform_kernel(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref):
        """Uniform twin of `threefry_bulk32_normal_kernel`: float32 in [0, 1), same layout."""
        return _threefry_bulk32_body(c0_ref, start_ref, k0_ref, k1_ref, k2_ref, k3_ref, False)


# ── Threefry host wrappers (module level) ────────────────────────────────────
#
# Defined outside `if HAS_NKI:` so launch counts and output order are testable
# on CPU-only hosts with a stand-in `_launch` (tests/test_launch_granularity.py).
# Kernels are looked up by name at call time; they exist only when HAS_NKI.


def _kernel(name: str):
    """Return the `@nki.jit` kernel `name`, or None when NKI is not installed."""
    return globals().get(name)


def _neuronx_cc_version():
    """(major, minor) of the installed neuronx-cc, or None if it is absent."""
    try:
        from importlib.metadata import version

        major, minor = version("neuronx-cc").split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return None


def _bulk_streaming() -> bool:
    """Whether bulk uniform/normal generation uses the multi-tile streaming kernels.

    The per-tile kernels emit 512 values per launch, so `normal(4M)` was 8,192
    launches at the ~0.2 ms native dispatch floor or worse (#60). The streaming
    kernels emit _PROGRAM_TILES × 512 = 16,384 values per launch, element-for-
    element equal to the per-tile path (#58), so they are the default.

    They do not compile on neuronx-cc 2.26 (SBUF OOM, #57), the compiler on the
    Trn1 / XLA DLAMI, so the XLA path keeps the per-tile kernels on neuronx-cc
    older than 2.28. TRNRAND_BULK_STREAMING=0/1 overrides the choice.
    """
    env = os.environ.get("TRNRAND_BULK_STREAMING", "").lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    if _use_simulator() or _get_dispatch_path() == "native":
        return True
    ver = _neuronx_cc_version()
    return ver is None or ver >= (2, 28)


def threefry_uniform_nki(
    n_elements: int,
    seed: int = 0,
    counter_offset: int = 0,
) -> torch.Tensor:
    """Threefry4×32-20 → float32 uniforms in [0, 1).

    Routes through `threefry_stream_uniform` (16,384 values per kernel launch)
    unless `_bulk_streaming()` says otherwise, else `_threefry_uniform_per_tile`
    (512 per launch). Both give the same values in the same order for a given
    seed and counter_offset (#58, #60); counter_offset is in 512-value tiles.
    """
    if _bulk_streaming():
        return threefry_stream_uniform(n_elements, seed=seed, counter_offset=counter_offset)
    return _threefry_uniform_per_tile(n_elements, seed=seed, counter_offset=counter_offset)


def threefry_normal_nki(
    n_elements: int,
    seed: int = 0,
    counter_offset: int = 0,
) -> torch.Tensor:
    """Threefry4×32-20 + Box-Muller → N(0,1) float32.

    Same routing as `threefry_uniform_nki`: `threefry_stream_normal` by default,
    `_threefry_normal_per_tile` on old compilers; identical output either way.
    """
    if _bulk_streaming():
        return threefry_stream_normal(n_elements, seed=seed, counter_offset=counter_offset)
    return _threefry_normal_per_tile(n_elements, seed=seed, counter_offset=counter_offset)


def _threefry_uniform_per_tile(
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
            out_np = nki.simulate(_kernel("threefry4x32_kernel"))(
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
            out_tile = _launch(
                _kernel("threefry4x32_kernel"),
                "threefry4x32",
                *(
                    torch.from_numpy(a)
                    for a in (c0_np, c1_np, c2_np, c3_np, k0_np, k1_np, k2_np, k3_np)
                ),
            ).reshape(-1)
        chunks.append(out_tile)

    result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
    return result.float()[:n_elements]


def _threefry_normal_per_tile(
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
            out_np = nki.simulate(_kernel("threefry_normal_kernel"))(
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
            out_tile = _launch(
                _kernel("threefry_normal_kernel"),
                "threefry_normal",
                *(
                    torch.from_numpy(a)
                    for a in (c0_np, c1_np, c2_np, c3_np, k0_np, k1_np, k2_np, k3_np)
                ),
            ).reshape(-1)
        chunks.append(out_tile)

    result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
    return result.float()[:n_elements]


def _h2d(t: torch.Tensor, device: str | None = None) -> torch.Tensor:
    """Host → Neuron device copy (one hook so tests can count copies)."""
    return t.to(device or _NEURON_DEVICE)


def _d2h(t: torch.Tensor) -> torch.Tensor:
    """Neuron device → host copy (one hook so tests can count copies); large
    results land in cached host memory (`_to_host`)."""
    return _to_host(t)


def _d2h_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Neuron device → preallocated host slice (the multi-core gather, one hook for tests)."""
    dst.copy_(src)


def _bulk_host_inputs(normal: bool, n_elements, seed, counter_offset):
    """Host (numpy) inputs of the bulk kernels for one request: plan, op and tiles."""
    LANES = _THREEFRY_LANES
    F, C, n_launches = _bulk_plan(n_elements)
    op_name = _bulk_op_name(normal)
    k0_val = seed & 0xFFFFFF
    k1_val = (seed >> 24) & 0xFFFFFF
    final = _bulk_final_order(op_name)
    c0_np = _bulk_lane_tile(F, final)
    k_np = [
        np.full((LANES, 1), k0_val, dtype=np.int32),
        np.full((LANES, 1), k1_val, dtype=np.int32),
        np.zeros((LANES, 1), dtype=np.int32),
        np.zeros((LANES, 1), dtype=np.int32),
    ]
    starts = (counter_offset + np.arange(n_launches * C, dtype=np.int64) * F) & 0xFFFFFF
    sb_np = np.ascontiguousarray(
        np.broadcast_to(starts.astype(np.int32).reshape(n_launches, 1, C), (n_launches, LANES, C))
    )
    return op_name, F, n_launches, final, c0_np, sb_np, k_np


def _bulk_launches(normal: bool, n_elements, seed, counter_offset):
    """Run the bulk Threefry kernel; return host values in per-tile order (#60).

    One launch covers up to _BULK_CHUNKS_MAX × _BULK_COLS_MAX batches
    (16 × 512 × 512 = 4,194,304 values at the defaults), so `normal(4M)` is
    one launch instead of 256. Chunk c of launch L starts at batch
    (counter_offset + (L·C + c)·F) & 0xFFFFFF, and column j of it is batch
    (start + j) & 0xFFFFFF: the same counter for every element as the
    per-tile and streaming paths, so the values are bit-identical.

    Native path (`_bulk_launches_native`): the (P, F) lane tile, key words and
    every launch's (P, C) start tile go to the device in one packed copy, and
    each launch's output is copied straight into its slice of one host tensor.
    The full-word kernels write per-tile order themselves, so their output is
    only flattened; the 8/16-bit kernels' output is reordered on the host
    (`_bulk_tile_order`). With more than one core requested (`set_cores` /
    TRNRAND_CORES) and a large enough request, the batch range is also split
    across logical cores; the values are the same.
    """
    if not _use_simulator():
        return _bulk_launches_native(
            normal, n_elements, seed, counter_offset, _plan_cores(n_elements)
        )
    op_name, F, n_launches, final, c0_np, sb_np, k_np = _bulk_host_inputs(
        normal, n_elements, seed, counter_offset
    )
    sim = nki.simulate(_kernel(op_name + "_kernel"))
    outs = [torch.from_numpy(np.asarray(sim(c0_np, sb_np[i], *k_np))) for i in range(n_launches)]
    return _bulk_result(outs, F, n_elements, final)


# ── Multi-core bulk draws (#60) ──────────────────────────────────────────────
#
# Threefry is counter-based, so a request's batch range splits across logical
# NeuronCores the same way `Generator(partition_rank=r, partition_size=P)`
# splits it across chips: core r draws the contiguous batches
# [counter_offset + b_r, counter_offset + b_{r+1}), and its output, in per-tile
# order, is exactly elements [512·b_r, 512·b_{r+1}) of the single-core stream.
# Every core's launches are queued before any result is read back (TorchNeuron
# launches are asynchronous), and the parts are copied into one preallocated
# host tensor in order. Values never depend on the core count.
#
# Off by default. TRNRAND_CORES=<n>|auto, or `set_cores(n)`, opts in; requests
# smaller than TRNRAND_CORES_MIN_ELEMENTS stay on one core. Measured on Trn2
# (#60): the kernels scale from 2M values (4M: 2.8 -> 1.6 ms on 2 cores), but
# the result goes to the host, where the device→host copy and page faults do not
# scale with cores; end to end, splitting only wins from about 8M values
# (8M: ~20 -> ~18 ms; 4M: 7 -> 8-10 ms, a loss).

_CORES_MIN_ELEMENTS_DEFAULT = 8 * 2**20
_cores_override: int | str | None = None


def _parse_cores(value) -> int | str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("auto", "all"):
            return "auto"
        value = int(v)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"cores must be a positive int or 'auto'; got {value!r}")
    return value


def set_cores(cores: int | str | None) -> None:
    """Spread large native-path `uniform`/`normal` draws over `cores` logical NeuronCores.

    `cores` is a positive int, ``"auto"`` (every visible core), or None to go
    back to the ``TRNRAND_CORES`` environment setting (default 1). The values
    are bit-identical for every core count; only the work is split. Requests
    under ``TRNRAND_CORES_MIN_ELEMENTS`` (default 8,388,608) stay on one core.
    The cores are the logical devices ``neuron:0..`` visible to the process
    (``NEURON_RT_VISIBLE_CORES``), starting at the current device.
    """
    global _cores_override
    _cores_override = None if cores is None else _parse_cores(cores)


def _visible_cores() -> int:
    """Logical NeuronCores visible to this process on the native path, else 1."""
    if _use_simulator() or _get_dispatch_path() != "native":
        return 1
    try:
        return max(1, int(torch.neuron.device_count()))
    except Exception:
        return 1


def get_cores() -> int:
    """Logical cores a large native bulk draw is spread over (see `set_cores`)."""
    req = _cores_override
    if req is None:
        env = os.environ.get("TRNRAND_CORES", "").strip()
        req = _parse_cores(env) if env else 1
    vis = _visible_cores()
    return vis if req == "auto" else min(req, vis)


def _cores_min_elements() -> int:
    env = os.environ.get("TRNRAND_CORES_MIN_ELEMENTS", "").strip()
    return int(env) if env else _CORES_MIN_ELEMENTS_DEFAULT


def _plan_cores(n_elements: int) -> int:
    """Logical cores for one bulk request: 1 unless opted in and over the threshold."""
    if _use_simulator() or _get_dispatch_path() != "native":
        return 1
    if n_elements < _cores_min_elements():
        return 1
    return get_cores()


def _core_split(n_elements: int, cores: int) -> list[tuple[int, int]]:
    """Split a request into per-core (first batch, n values) parts, in stream order.

    Parts are whole (P, F) chunks of _BULK_COLS_MAX batches when there are at
    least `cores` of them (so each core runs a full-width plan), else whole
    batches. Every part but the last is a whole number of batches, so the parts
    concatenate to the single-core stream. May return fewer than `cores` parts.
    """
    per_batch = _THREEFRY_LANES * 4
    n_batches = max(1, -(-int(n_elements) // per_batch))
    unit = _BULK_COLS_MAX if n_batches >= cores * _BULK_COLS_MAX else 1
    units = -(-n_batches // unit)
    per = -(-units // cores) * unit  # batches per core
    parts, b = [], 0
    while b < n_batches:
        nb = min(per, n_batches - b)
        parts.append((b, min(nb * per_batch, n_elements - b * per_batch)))
        b += nb
    return parts


def _core_devices(k: int) -> list[str]:
    """Device names of `k` logical cores, starting at the current device."""
    vis = _visible_cores()
    base = _current_core() or 0
    return [f"{_NEURON_DEVICE}:{(base + i) % vis}" for i in range(k)]


def _bulk_launches_native(normal: bool, n_elements, seed, counter_offset, cores: int = 1):
    """Native bulk draw on `cores` logical cores; host values in per-tile order (#60).

    Same values as the single-core path, element for element (see `_core_split`).
    Three phases, so the cores run together: every core's inputs go to its
    device (one packed copy each), then every launch is queued (TorchNeuron
    launches are asynchronous), then each launch's output is copied into its
    slice of one preallocated host tensor. The inputs are all copied before any
    launch because a host→device copy waits for work already queued.
    Copying each launch's output into place also replaces the old device-side
    stack and host-side concatenation of multi-launch draws (one full copy).
    """
    parts = _core_split(n_elements, cores)
    # Explicit device names, also for one core: "neuron" means the current
    # device, which can change between calls, and the lane-tile cache is per device.
    devices = _core_devices(len(parts))
    # Touching other devices can move TorchNeuron's current device; put it back.
    base = _current_core()
    try:
        return _bulk_parts(normal, n_elements, seed, counter_offset, parts, devices)
    finally:
        if base is not None and _current_core() != base:
            torch.neuron.set_device(base)


def _current_core() -> int | None:
    try:
        return int(torch.neuron.current_device())
    except Exception:
        return None


def _bulk_parts(normal, n_elements, seed, counter_offset, parts, devices):
    staged = []
    for (b, m), dev in zip(parts, devices, strict=True):
        op_name, F, n_launches, final, c0_np, sb_np, k_np = _bulk_host_inputs(
            normal, m, seed, counter_offset + b
        )
        c0_d, sb_d, k_d = _h2d_packed(
            torch.from_numpy(c0_np),
            torch.from_numpy(sb_np),
            [torch.from_numpy(a) for a in k_np],
            c0_key=(F, final, dev),
            device=dev,
        )
        staged.append((b * _THREEFRY_LANES * 4, m, op_name, F, final, n_launches, c0_d, sb_d, k_d))
    pending = []
    for a, m, op_name, F, final, n_launches, c0_d, sb_d, k_d in staged:
        op = _get_native_ops()[op_name]
        pending.append((a, m, F, final, [op(c0_d, sb_d[i], *k_d) for i in range(n_launches)]))
    # Already-faulted-in cached host memory: the copies below run at the link
    # rate instead of the host page-fault rate (`_host_empty`).
    out = _host_empty((n_elements,), torch.float32)
    for a, m, F, final, outs_d in pending:
        if not final:  # 8/16-bit kernels: reorder on the host
            stacked = outs_d[0].unsqueeze(0) if len(outs_d) == 1 else torch.stack(outs_d)
            out[a : a + m] = _bulk_result(list(_d2h(stacked).unbind(0)), F, m, False)
            continue
        off, end = a, a + m
        for o in outs_d:  # each launch is the next contiguous run of the part
            o = o.reshape(-1)
            take = min(o.numel(), end - off)
            _d2h_into(out[off : off + take], o[:take])
            off += take
    return out


def _bulk_op_name(normal: bool) -> str:
    """Bulk kernel op name for the active limb width, e.g. ``threefry_bulk32_normal``.

    Full-word (32) kernels only on the native TorchNeuron (Trn2+) path and the
    simulator; anything else keeps the 8-bit kernels (aws-neuron-sdk#1308).
    """
    native = _use_simulator() or _get_dispatch_path() == "native"
    bits = _limb_bits() if native else 8
    variant = {8: "bulk", 16: "bulk16", 32: "bulk32"}[bits]
    return f"threefry_{variant}_{'normal' if normal else 'uniform'}"


def _bulk_result(outs, cols: int, n_elements: int, final_order: bool = False) -> torch.Tensor:
    if final_order:  # already per-tile order: flatten, no host reorder
        flat = [o.reshape(-1) for o in outs]
        result = torch.cat(flat) if len(flat) > 1 else flat[0]
        return result.float()[:n_elements]
    chunks = [_bulk_tile_order(o, cols) for o in outs]
    result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
    return result.float()[:n_elements]


def _limb_bits() -> int:
    """Word width of the bulk Threefry kernels on the native path (#60).

    TRNRAND_LIMB_BITS=32 (default): full uint32 words, adds on GpSimd native
    integer arithmetic (wraps mod 2^32 on Trn2). TRNRAND_LIMB_BITS=8: the
    byte-limb kernels, the documented fallback. 16 is a diagnostic width.
    All three are bit-identical. The XLA/Trn1 path never uses the bulk
    kernels (it streams), so this never changes it.
    """
    env = os.environ.get("TRNRAND_LIMB_BITS", "").strip()
    if env in ("", "32"):
        return 32
    if env in ("8", "16"):
        return int(env)
    raise ValueError(f"TRNRAND_LIMB_BITS must be 32, 8 or 16, got {env!r}")


def _use_bulk() -> bool:
    """Bulk kernels on the native path and the simulator; TRNRAND_BULK_KERNEL=0 opts out."""
    if os.environ.get("TRNRAND_BULK_KERNEL", "").lower() in ("0", "false", "no"):
        return False
    return _use_simulator() or _get_dispatch_path() == "native"


def _stream_launches(kernel_name: str, op_name: str, n_elements, seed, counter_offset):
    """Run the streaming kernel `ceil(n / 16,384)` times; return host values in per-tile order.

    Launch L uses start_batch = (counter_offset + L × _PROGRAM_TILES) & 0xFFFFFF.

    Native path (#60): the constant (P, 1) inputs (lane index, key words) and
    the start_batch inputs of all launches, as one (L, P, 1) tensor indexed per
    launch, go to the device in one packed copy. Each launch's (P, 4 × tiles)
    output stays on the device; they are stacked there and copied back in one
    transfer. That is 1 host→device copy and 1 device→host copy per call.

    XLA path and simulator: one `_launch` / `nki.simulate` per launch, unchanged.
    """
    LANES = _THREEFRY_LANES
    per_launch = _PROGRAM_TILES * LANES * 4  # 32 × 128 × 4 = 16,384
    n_launches = max(1, (n_elements + per_launch - 1) // per_launch)

    k0_val = seed & 0xFFFFFF
    k1_val = (seed >> 24) & 0xFFFFFF
    c0_np = np.arange(LANES, dtype=np.int32).reshape(-1, 1)
    k_np = [
        np.full((LANES, 1), k0_val, dtype=np.int32),
        np.full((LANES, 1), k1_val, dtype=np.int32),
        np.zeros((LANES, 1), dtype=np.int32),
        np.zeros((LANES, 1), dtype=np.int32),
    ]
    starts = (counter_offset + np.arange(n_launches, dtype=np.int64) * _PROGRAM_TILES) & 0xFFFFFF
    sb_np = np.ascontiguousarray(
        np.broadcast_to(starts.astype(np.int32).reshape(-1, 1, 1), (n_launches, LANES, 1))
    )

    if _use_simulator():
        sim = nki.simulate(_kernel(kernel_name))
        outs = [
            torch.from_numpy(np.asarray(sim(c0_np, sb_np[i], *k_np))) for i in range(n_launches)
        ]
    elif _get_dispatch_path() == "native":
        outs = _native_launches(
            op_name,
            n_launches,
            torch.from_numpy(c0_np),
            torch.from_numpy(sb_np),
            [torch.from_numpy(a) for a in k_np],
        )
    else:
        kernel = _kernel(kernel_name)
        outs = [
            _launch(
                kernel,
                op_name,
                torch.from_numpy(c0_np),
                torch.from_numpy(sb_np[i]),
                *(torch.from_numpy(a) for a in k_np),
            )
            for i in range(n_launches)
        ]

    return _stream_result(outs, n_elements)


# Device copies of the (P, cols) lane tiles, keyed by (cols, final_order). The
# lane tile depends only on the plan, so the eager bulk path copies it once per
# process instead of once per call (~0.5 ms of a 4M draw on Trn2, #60).
_device_lane_tiles: dict = {}


def _h2d_packed(c0, sb, k, c0_key=None, device=None):
    """Copy the kernel inputs to the device in ONE transfer.

    Each host→device copy has a fixed cost on Trn2 (0.05-0.2 ms measured for
    tensors up to 256 KB, #60), so the four
    key words, the start tiles and (unless cached under `c0_key`) the lane tile
    go over as one flat int32 buffer, and the kernel inputs are contiguous views
    into it (#60 follow-up).
    """
    cached = _device_lane_tiles.get(c0_key) if c0_key is not None else None
    parts = [*k, sb] if cached is not None else [*k, sb, c0]
    flat = torch.cat([t.reshape(-1).to(torch.int32) for t in parts])
    flat_d = _h2d(flat) if device is None else _h2d(flat, device)
    views, off = [], 0
    for t in parts:
        views.append(flat_d[off : off + t.numel()].view(t.shape))
        off += t.numel()
    if cached is None:
        cached = views[5]
        if c0_key is not None:
            _device_lane_tiles[c0_key] = cached
    return cached, views[4], views[0:4]


def _native_launches(op_name, n_launches, c0, sb, k, c0_key=None):
    """Native path: copy the inputs to the device once, launch, copy back once (#60).

    `c0` is the (P, cols) lane tile, the four `k` words are (P, 1) int32, and
    `sb` is the (n_launches, P, starts) int32 start tile of every launch; all of
    them go over in one packed copy (`_h2d_packed`), and `c0_key` (eager bulk
    path only) reuses a cached device lane tile. Returns the per-launch host
    outputs. Shared by the eager bulk and streaming wrappers and the
    torch.compile path (#61), so all of them launch identically.
    """
    op = _get_native_ops()[op_name]
    c0_d, sb_d, k_d = _h2d_packed(c0, sb, k, c0_key)
    outs_d = [op(c0_d, sb_d[i], *k_d) for i in range(n_launches)]
    stacked = outs_d[0].unsqueeze(0) if n_launches == 1 else torch.stack(outs_d)
    return list(_d2h(stacked).unbind(0))


def _stream_result(outs, n_elements):
    chunks = [_stream_tile_order(o) for o in outs]
    result = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
    return result.float()[:n_elements]


def _kernel_inputs(
    key_words: torch.Tensor,
    offset: torch.Tensor,
    n_launches: int,
    starts_per_launch: int,
    stride: int,
    cols: int,
    final_order: bool = False,
):
    """Native kernel inputs built with torch ops from a tensor key and counter (#61).

    Tensor-valued twin of the numpy setup in `_bulk_launches` / `_stream_launches`,
    value for value: c0 = `_bulk_lane_tile(cols, final_order)`, (k0, k1) = `key_words`
    (int32, each < 2^24), k2 = k3 = 0, and start i of launch L is
    (offset + (L × starts_per_launch + i) × stride) & 0xFFFFFF, laid out as
    (n_launches, P, starts_per_launch). `offset` is a 0-d integer tensor. Only
    plain aten ops, so the neuron backend can lower them, and the key and offset
    stay graph inputs rather than baked-in constants. Arithmetic after the first
    mask is int32 (the offset is < 2^24, the added span is one request's batches).
    """
    LANES = _THREEFRY_LANES
    if final_order:
        c0 = torch.arange(LANES * cols, dtype=torch.int32).reshape(LANES, cols) & (LANES - 1)
    else:
        c0 = torch.arange(LANES, dtype=torch.int32).reshape(-1, 1)
        if cols > 1:
            c0 = c0.expand(LANES, cols).contiguous()
    kw = key_words.to(torch.int32).reshape(1, 2).expand(LANES, 2)
    zero = torch.zeros(LANES, 1, dtype=torch.int32)
    k = [kw[:, 0:1].contiguous(), kw[:, 1:2].contiguous(), zero, zero.clone()]
    base = (offset & 0xFFFFFF).to(torch.int32)
    idx = torch.arange(n_launches * starts_per_launch, dtype=torch.int32)
    starts = (base + idx * stride) & 0xFFFFFF
    sb = (
        starts.reshape(n_launches, 1, starts_per_launch)
        .expand(n_launches, LANES, starts_per_launch)
        .contiguous()
    )
    return c0, sb, k


def _stream_inputs(key_words: torch.Tensor, offset: torch.Tensor, n_launches: int):
    """Streaming-kernel inputs: (P, 1) lanes, start_batch[L] = offset + L × _PROGRAM_TILES."""
    return _kernel_inputs(key_words, offset, n_launches, 1, _PROGRAM_TILES, 1)


def _bulk_inputs(
    key_words: torch.Tensor,
    offset: torch.Tensor,
    cols: int,
    chunks: int,
    n_launches: int,
    final_order: bool = False,
):
    """Bulk-kernel inputs for a `_bulk_plan` (cols F, chunks C, launches):
    (P, F) lanes, chunk c of launch L starts at offset + (L × C + c) × F."""
    return _kernel_inputs(key_words, offset, n_launches, chunks, cols, cols, final_order)


def _key_offset_from_state(state: torch.Tensor):
    """(key_words, offset) tensors from an int64 (seed, counter_offset) tensor (key_mode="draw")."""
    state = state.to(torch.int64)
    seed = state[0]
    return torch.stack([seed & 0xFFFFFF, (seed >> 24) & 0xFFFFFF]), state[1]


def _stream_inputs_from_state(state: torch.Tensor, n_launches: int):
    """`_stream_inputs` from an int64 (seed, counter_offset) tensor (key_mode="draw")."""
    return _stream_inputs(*_key_offset_from_state(state), n_launches)


def _n_stream_launches(n_elements: int) -> int:
    per_launch = _PROGRAM_TILES * _THREEFRY_LANES * 4
    return max(1, (n_elements + per_launch - 1) // per_launch)


def threefry_stream_from_key(
    kind: str, n_elements: int, key_words: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Native draw from tensor key words and counter offset (#61).

    Pure aten ops plus the nki_op launches, so it lowers on
    `torch.compile(backend="neuron")`. Takes the same route as the eager
    `threefry_stream_{kind}` on the native path: the bulk kernels (`_bulk_plan`
    launches, `_bulk_op_name` limb variant) unless TRNRAND_BULK_KERNEL=0, else
    the streaming kernels. Same values as `threefry_stream_{kind}(n_elements,
    seed=k0 | k1 << 24, counter_offset=offset)`.
    """
    if _use_bulk():
        F, C, n_launches = _bulk_plan(n_elements)
        op_name = _bulk_op_name(kind == "normal")
        final = _bulk_final_order(op_name)
        c0, sb, k = _bulk_inputs(key_words, offset, F, C, n_launches, final)
        outs = _native_launches(op_name, n_launches, c0, sb, k)
        return _bulk_result(outs, F, n_elements, final)
    n_launches = _n_stream_launches(n_elements)
    c0, sb, k = _stream_inputs(key_words, offset, n_launches)
    outs = _native_launches(f"threefry_streaming_{kind}", n_launches, c0, sb, k)
    return _stream_result(outs, n_elements)


def threefry_stream_from_state(kind: str, n_elements: int, state: torch.Tensor) -> torch.Tensor:
    """`threefry_stream_from_key` from a (seed, offset) tensor (#61).

    `state` is the int64 tensor produced by the `trnrand::gen_nki_state` op
    (key_mode="draw"; CPU compile backends only, see `trnrand/_compile.py`).
    """
    return threefry_stream_from_key(kind, n_elements, *_key_offset_from_state(state))


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
    Each launch's output is flattened in per-tile (tiles, P, 4) order by
    _stream_tile_order, so the stream equals the per-tile output element-for-element
    (#58).
    """
    if _use_bulk():
        return _bulk_launches(True, n_elements, seed, counter_offset)
    return _stream_launches(
        "threefry_streaming_normal_kernel",
        "threefry_streaming_normal",
        n_elements,
        seed,
        counter_offset,
    )


def threefry_stream_uniform(
    n_elements: int,
    seed: int = 0,
    counter_offset: int = 0,
) -> torch.Tensor:
    """Host wrapper: streaming Threefry4×32-20 → float32 uniforms in [0, 1).

    Same launch/counter scheme as threefry_stream_normal; 16,384 uniforms per
    XLA call, ~31× fewer round-trips than the per-tile threefry_uniform_nki
    for large n.
    Output is in per-tile order, element-for-element equal to
    threefry_uniform_nki for the same seed and counter_offset (#58).
    """
    if _use_bulk():
        return _bulk_launches(False, n_elements, seed, counter_offset)
    return _stream_launches(
        "threefry_streaming_uniform_kernel",
        "threefry_streaming_uniform",
        n_elements,
        seed,
        counter_offset,
    )
