"""Simulator-backed kernel correctness tests (NKI 0.3.0 Stable).

Run with `TRNRAND_USE_SIMULATOR=1` on any x86_64 Linux host that has
`nki>=0.3.0` installed. Bypasses torch_xla + NEFF compile; routes
kernel dispatch through `nki.simulate(kernel)(np_args)`.

Intentionally curated to small shapes — the CPU simulator is slow at
thousands of lanes. Correctness parity with hardware at tile size is
what we're verifying here; hardware still owns the perf numbers.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator

# Upstream platform block. NKI ops on uint32 tiles route through the
# float32 activation engine; values > 2^24 lose precision at the
# NKI boundary (including nl.copy's internal cast). No kernel-level
# decomposition can work around this — Philox counters themselves
# exceed 2^24. Tracked in aws-neuron-sdk#1308. Mark the dependent
# tests as xfail until AWS ships a true integer multiply primitive.
_XFAIL_NKI_1308 = pytest.mark.xfail(
    reason="aws-neuron-sdk#1308 — NKI uint32 ops lose precision above 2^24",
    strict=False,
)


@pytest.fixture(autouse=True)
def _simulator_enabled():
    """Skip the whole module if TRNRAND_USE_SIMULATOR isn't set.

    The marker alone isn't sufficient — users may `pytest -m nki_simulator`
    on a host where `nki` isn't importable or the env var hasn't been set.
    Fail loudly vs silently falling back.
    """
    if os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNRAND_USE_SIMULATOR=1 required")

    from trnrand.nki.dispatch import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki>=0.3.0 not importable on this host")


# Late imports. On hosts without `nki` installed (e.g. macOS dev box),
# the kernel objects don't exist in trnrand.nki.dispatch (they live
# inside `if HAS_NKI:`). Catch and set to None so module import
# succeeds; the autouse fixture below skips each test before it runs.
from trnrand.nki.dispatch import (  # noqa: E402
    UINT32_MASK,
    box_muller_cpu,
    philox4x32_reference,
    philox_uniform_cpu,
    threefry4x32_reference,
    threefry_uniform_cpu,
)

try:
    from trnrand.nki.dispatch import box_muller_kernel, philox4x32_kernel  # noqa: E402
except ImportError:
    philox4x32_kernel = None
    box_muller_kernel = None

try:
    from trnrand.nki.dispatch import (  # noqa: E402
        threefry4x32_kernel,
        threefry_normal_kernel,
    )
except ImportError:
    threefry4x32_kernel = None
    threefry_normal_kernel = None

_PHILOX_LANES_PER_TILE = 128
_THREEFRY_LANES = 128


# ── Philox conformance via nki.simulate ────────────────────────────────────


@pytest.mark.parametrize(
    "counter,key,expected",
    [
        # Salmon et al. SC'11 published Philox4×32-10 test vectors.
        # Run through the NKI kernel via nki.simulate(kernel). The kernel
        # only reads counter[0] (counter_lo); other words start at zero,
        # so spec vectors with non-zero counter[1..3] are skipped.
        (
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0x00000000, 0x00000000),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        ),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
@_XFAIL_NKI_1308
def test_philox_spec_vectors_via_simulator(counter, key, expected):
    """Lane 0 of the 128-lane tile must emit the Salmon SC'11 vector."""
    if counter[1] != 0 or counter[2] != 0 or counter[3] != 0:
        pytest.skip("Kernel only loads counter_lo; skip vectors that use counter[1..3].")

    import nki

    lanes = _PHILOX_LANES_PER_TILE
    counter_lo = np.zeros((lanes, 1), dtype=np.int32)
    counter_lo[0, 0] = np.int32(counter[0])
    key_lo = np.full((lanes, 1), np.int32(key[0]), dtype=np.int32)
    key_hi = np.full((lanes, 1), np.int32(key[1]), dtype=np.int32)

    out = nki.simulate(philox4x32_kernel)(counter_lo, key_lo, key_hi)
    lane0 = np.asarray(out)[0].astype(np.uint32)
    got = tuple(int(x) for x in lane0)
    assert got == expected, (
        f"Philox simulator mismatch on lane 0: got "
        f"({got[0]:#010x}, {got[1]:#010x}, {got[2]:#010x}, {got[3]:#010x}), "
        f"expected ({expected[0]:#010x}, {expected[1]:#010x}, "
        f"{expected[2]:#010x}, {expected[3]:#010x})"
    )


@_XFAIL_NKI_1308
def test_philox_kernel_matches_reference():
    """Full 128-lane tile: simulator output must equal CPU reference."""
    import nki

    lanes = _PHILOX_LANES_PER_TILE
    counter_lo = np.arange(lanes, dtype=np.int32).reshape(-1, 1)
    key_lo = np.full((lanes, 1), 0x12345678 & 0x7FFFFFFF, dtype=np.int32)
    key_hi = np.full((lanes, 1), 0x9ABCDEF0 & 0x7FFFFFFF, dtype=np.int32)

    out = np.asarray(nki.simulate(philox4x32_kernel)(counter_lo, key_lo, key_hi))

    ctr_ref = torch.zeros(lanes, 4, dtype=torch.int64)
    ctr_ref[:, 0] = torch.from_numpy(counter_lo.reshape(-1)).to(torch.int64)
    key_ref = torch.zeros(lanes, 2, dtype=torch.int64)
    key_ref[:, 0] = int(key_lo[0, 0]) & UINT32_MASK
    key_ref[:, 1] = int(key_hi[0, 0]) & UINT32_MASK
    expected = philox4x32_reference(ctr_ref, key_ref).reshape(lanes, 4).numpy()

    np.testing.assert_array_equal(out.astype(np.int64) & UINT32_MASK, expected)


@_XFAIL_NKI_1308
def test_philox_kernel_distribution():
    """128-lane × 4 uint32 outputs converted to floats should be ~U[0,1)."""
    import nki

    lanes = _PHILOX_LANES_PER_TILE
    counter_lo = np.arange(lanes, dtype=np.int32).reshape(-1, 1)
    key_lo = np.zeros((lanes, 1), dtype=np.int32)
    key_hi = np.zeros((lanes, 1), dtype=np.int32)

    out = np.asarray(nki.simulate(philox4x32_kernel)(counter_lo, key_lo, key_hi))
    u = (out.astype(np.int64) & UINT32_MASK).astype(np.float64) / 2**32
    assert abs(u.mean() - 0.5) < 0.05
    assert abs(u.var() - 1 / 12) < 0.02


# ── _mul32_hi_lo simulator parity vs numpy port ────────────────────────────


@_XFAIL_NKI_1308
def test_mul32_simulator_matches_numpy():
    """NKI's `_mul32_hi_lo` (via `nki.simulate`) must bit-match the numpy port.

    The numpy port is ground-truth-validated by
    `test_mul32_numpy_matches_ground_truth`. If *this* test fails, the
    bug is in NKI's uint32 op semantics — specific ops (`multiply`,
    `bitwise_and`, `right_shift`, `left_shift`, `bitwise_or`, `add`,
    `copy`) don't produce true uint32 arithmetic.

    Not trivially runnable — NKI won't let us call a helper function
    that isn't `@nki.jit`-decorated. We wrap the helper in a thin
    `@nki.jit` kernel that just calls it and stores the output.
    """
    import nki
    import nki.language as nl

    from trnrand.nki.dispatch import _mul32_hi_lo, _mul32_hi_lo_numpy

    @nki.jit
    def _mul32_kernel(a_ref, out_hi_ref, out_lo_ref):
        a = nl.load(a_ref)
        hi, lo = _mul32_hi_lo(a, 0x1F53, 0xD251)
        # Allocate output buffers via the output refs
        P = a_ref.shape[0]
        out_hi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        out_lo = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        out_hi[:, 0:1] = hi
        out_lo[:, 0:1] = lo
        return out_hi, out_lo

    # Same test inputs as the numpy ground-truth test.
    test_inputs = [
        0x00000000,
        0x00000001,
        0x0000FFFF,
        0x00010000,
        0xFFFE0001,
        0x7FFFFFFF,
        0x80000000,
        0xD2511F53,
        0xFFFFFFFF,
    ]
    # Pad to 128 lanes (partition-axis requirement). Build as uint32 then
    # reinterpret as int32 via view() — values like 0xFFFE0001 exceed
    # INT32_MAX and would OverflowError through np.array(..., dtype=int32).
    padded = test_inputs + [0] * (_PHILOX_LANES_PER_TILE - len(test_inputs))
    a_u32 = np.array(padded, dtype=np.uint32).reshape(-1, 1)
    a_arr = a_u32.view(np.int32)

    # NKI simulator
    sim_hi, sim_lo = nki.simulate(_mul32_kernel)(
        a_arr,
        np.zeros_like(a_arr),  # placeholder; unused but kernel signature requires
        np.zeros_like(a_arr),
    )

    # numpy ground truth (bit-exact by test_mul32_numpy_matches_ground_truth)
    np_hi, np_lo = _mul32_hi_lo_numpy(a_u32, 0x1F53, 0xD251)

    np.testing.assert_array_equal(
        np.asarray(sim_hi).astype(np.int64) & UINT32_MASK,
        np_hi.astype(np.int64) & UINT32_MASK,
        err_msg="NKI simulator hi32 differs from numpy port — NKI uint32 semantics gap",
    )
    np.testing.assert_array_equal(
        np.asarray(sim_lo).astype(np.int64) & UINT32_MASK,
        np_lo.astype(np.int64) & UINT32_MASK,
        err_msg="NKI simulator lo32 differs from numpy port — NKI uint32 semantics gap",
    )


# ── Box-Muller correctness via nki.simulate ────────────────────────────────


def test_box_muller_kernel_matches_reference():
    """Simulator output must match the CPU Box-Muller reference within tolerance."""
    import nki

    lanes = _PHILOX_LANES_PER_TILE
    u_torch = philox_uniform_cpu(lanes * 2, seed=99).to(torch.float32)
    uniforms = u_torch.numpy().reshape(lanes, 2)

    out = np.asarray(nki.simulate(box_muller_kernel)(uniforms))
    got_flat = out.reshape(-1)
    expected = box_muller_cpu(u_torch).to(torch.float32).numpy()

    np.testing.assert_allclose(got_flat, expected, rtol=1e-4, atol=1e-4)


def test_box_muller_kernel_distribution():
    """Box-Muller samples should be ~N(0, 1) — mean/std on a small tile."""
    import nki

    lanes = _PHILOX_LANES_PER_TILE
    u_torch = philox_uniform_cpu(lanes * 2, seed=11).to(torch.float32)
    uniforms = u_torch.numpy().reshape(lanes, 2)

    out = np.asarray(nki.simulate(box_muller_kernel)(uniforms))
    assert abs(float(out.mean())) < 0.15
    assert abs(float(out.std()) - 1.0) < 0.15


# ── Threefry 4×32-20 correctness via nki.simulate ─────────────────────────────
#
# No _XFAIL_NKI_1308 marks: Threefry uses no integer multiply and works
# entirely in byte-tile arithmetic where every intermediate ≤ 511 < 2^24.
# These tests should PASS on the simulator without qualification.


def _make_threefry_inputs(n_lanes, seed=0, batch=0):
    """Build the 8 (n_lanes, 1) int32 input arrays for threefry4x32_kernel."""
    c0 = np.arange(n_lanes, dtype=np.int32).reshape(-1, 1)
    c1 = np.full((n_lanes, 1), batch & 0xFFFFFF, dtype=np.int32)
    c2 = np.zeros((n_lanes, 1), dtype=np.int32)
    c3 = np.zeros((n_lanes, 1), dtype=np.int32)
    k0 = np.full((n_lanes, 1), seed & 0xFFFFFF, dtype=np.int32)
    k1 = np.full((n_lanes, 1), (seed >> 24) & 0xFFFFFF, dtype=np.int32)
    k2 = np.zeros((n_lanes, 1), dtype=np.int32)
    k3 = np.zeros((n_lanes, 1), dtype=np.int32)
    return c0, c1, c2, c3, k0, k1, k2, k3


def test_threefry_kernel_matches_reference():
    """Full 128-lane tile: simulator uniform output must match CPU reference.

    The kernel emits float32 uniforms in [0, 1) from the 3 low bytes of
    each Threefry output word. The CPU reference produces the same value
    by the same formula: mantissa = b0 + b1*256 + b2*65536; u = m/2^24.
    """
    import nki

    lanes = _THREEFRY_LANES
    seed = 0x12345678
    inputs = _make_threefry_inputs(lanes, seed=seed)
    out_np = np.asarray(nki.simulate(threefry4x32_kernel)(*inputs))  # (lanes, 4) float32

    # Build expected: same 3-byte mantissa extraction from threefry4x32_reference.
    ctr = torch.zeros(lanes, 4, dtype=torch.int64)
    ctr[:, 0] = torch.from_numpy(inputs[0].reshape(-1)).to(torch.int64)
    ctr[:, 1] = torch.from_numpy(inputs[1].reshape(-1)).to(torch.int64)
    key = torch.zeros(lanes, 4, dtype=torch.int64)
    key[:, 0] = seed & 0xFFFFFF
    key[:, 1] = (seed >> 24) & 0xFFFFFF
    ref_u32 = threefry4x32_reference(ctr, key).numpy()  # (lanes, 4) int64
    expected = ((ref_u32 & 0xFFFFFF) / 16777216.0).astype(np.float32)

    np.testing.assert_allclose(
        out_np, expected, rtol=1e-5, atol=1e-5,
        err_msg="Threefry simulator output differs from CPU reference",
    )


def test_threefry_spec_vectors_via_simulator():
    """Lane 0 of a 128-lane tile must emit the Random123 KAT vectors.

    Each run uses counter=(i, 0, 0, 0), key=(0,0,0,0). We check the raw
    float output equals the 3-LSB mantissa of the known-answer uint32 words.
    """
    import nki

    lanes = _THREEFRY_LANES
    # Random123 KAT vector 1: ctr=(0,0,0,0), key=(0,0,0,0)
    # Expected uint32: (0x3425621E, 0x64AF086C, 0x4939F9F4, 0x02F34BF4)
    kat_u32 = [0x3425621E, 0x64AF086C, 0x4939F9F4, 0x02F34BF4]
    kat_expected = np.array([(x & 0xFFFFFF) / 16777216.0 for x in kat_u32], dtype=np.float32)

    c0 = np.zeros((lanes, 1), dtype=np.int32)   # lane 0 counter = 0
    zeros = np.zeros((lanes, 1), dtype=np.int32)
    out_np = np.asarray(nki.simulate(threefry4x32_kernel)(
        c0, zeros, zeros, zeros, zeros, zeros, zeros, zeros
    ))
    lane0 = out_np[0]  # shape (4,)

    np.testing.assert_allclose(
        lane0, kat_expected, rtol=1e-5, atol=1e-5,
        err_msg="Threefry simulator lane 0 KAT vector mismatch",
    )


def test_threefry_kernel_distribution():
    """128-lane × 4 float32 outputs should be ~U[0, 1) — mean/var check."""
    import nki

    lanes = _THREEFRY_LANES
    inputs = _make_threefry_inputs(lanes, seed=7)
    out_np = np.asarray(nki.simulate(threefry4x32_kernel)(*inputs))  # (lanes, 4)
    u = out_np.reshape(-1)
    assert 0.0 <= u.min() < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.1
    assert abs(float(u.var()) - 1 / 12) < 0.02


def test_threefry_normal_kernel_distribution():
    """Fused Threefry + Box-Muller kernel: output should be ~N(0, 1)."""
    import nki

    lanes = _THREEFRY_LANES
    inputs = _make_threefry_inputs(lanes, seed=42)
    out_np = np.asarray(nki.simulate(threefry_normal_kernel)(*inputs))  # (lanes, 4)
    z = out_np.reshape(-1)
    assert abs(float(z.mean())) < 0.2
    assert abs(float(z.std()) - 1.0) < 0.2
