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
)

try:
    from trnrand.nki.dispatch import box_muller_kernel, philox4x32_kernel  # noqa: E402
except ImportError:
    philox4x32_kernel = None
    box_muller_kernel = None

_PHILOX_LANES_PER_TILE = 128


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
        0x00000000, 0x00000001, 0x0000FFFF, 0x00010000,
        0xFFFE0001, 0x7FFFFFFF, 0x80000000, 0xD2511F53, 0xFFFFFFFF,
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
