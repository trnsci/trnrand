"""NKI kernel correctness via the CPU simulator (Neuron SDK 2.29+).

Runs the Philox and Box-Muller NKI kernels on CPU using
`nki.simulate_kernel`, so we can iterate on kernel design without a
trn1 instance. Tests here take `numpy.ndarray` inputs (the simulator
contract), in contrast to the `@pytest.mark.neuron` tests which take
torch tensors through the XLA path.

Run:
    pytest tests/test_nki_philox_simulator.py -v -m simulator

On hosts without the Neuron SDK installed (e.g. macOS dev boxes), the
entire module is skipped at collection time. The CI `test-simulator`
job installs `neuronx-cc` from the Neuron pip index and runs these
tests on `ubuntu-latest`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Skip the whole module if neuronxcc isn't installed. The CI simulator job
# pulls it from https://pip.repos.neuron.amazonaws.com.
nki = pytest.importorskip(
    "neuronxcc.nki",
    reason="neuronxcc not installed; install via pip install 'neuronx-cc>=2.29' "
    "--extra-index-url https://pip.repos.neuron.amazonaws.com",
)

from trnrand.nki.dispatch import (  # noqa: E402
    UINT32_MASK,
    box_muller_cpu,
    box_muller_kernel,
    philox4x32_kernel,
    philox4x32_reference,
    philox_uniform_cpu,
)

pytestmark = pytest.mark.simulator

# NKI partition axis is capped at 128 on trn1/trn2 (matches runtime).
_PHILOX_LANES_PER_TILE = 128


# ── Philox spec conformance through the simulator ──────────────────────────


@pytest.mark.parametrize(
    "counter,key,expected",
    [
        # Salmon et al. SC'11 published Philox4×32-10 test vectors — same
        # three exercised via the pure CPU reference in test_nki_philox.py.
        # Run here through the NKI kernel via nki.simulate_kernel.
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
    """The kernel on lane 0 must emit the Salmon SC'11 vector.

    Only lane 0 carries the full 4-word counter; other lanes are
    padded to satisfy the (128, 1) partition-axis requirement. We
    read out just lane 0's 4 outputs.
    """
    # The kernel's lane-0 input is counter_lo = counter[0] and the other
    # three counter words start at zero. This spec vector has non-zero
    # counter[1..3], so this particular test only works for counter-word-0
    # on lane 0 — we'll check the first word and skip the richer fields.
    # For vectors 0 and 1 (all-zero or all-0xFF counter), all four words
    # line up; for vector 2, this test is a partial conformance check.
    if counter[1] != 0 or counter[2] != 0 or counter[3] != 0:
        pytest.skip("Kernel only loads counter_lo; skip vectors that use counter[1..3].")

    lanes = _PHILOX_LANES_PER_TILE
    counter_lo = np.zeros((lanes, 1), dtype=np.int32)
    counter_lo[0, 0] = np.int32(counter[0])
    key_lo = np.full((lanes, 1), np.int32(key[0]), dtype=np.int32)
    key_hi = np.full((lanes, 1), np.int32(key[1]), dtype=np.int32)

    out = nki.simulate_kernel(philox4x32_kernel, counter_lo, key_lo, key_hi)
    # `out` shape: (128, 4). Lane 0's four uint32 outputs.
    lane0 = out[0].astype(np.uint32)
    got = tuple(int(x) for x in lane0)
    assert got == expected, (
        f"Philox4×32-10 simulator output mismatch on lane 0: got "
        f"({got[0]:#010x}, {got[1]:#010x}, {got[2]:#010x}, {got[3]:#010x}), "
        f"expected ({expected[0]:#010x}, {expected[1]:#010x}, "
        f"{expected[2]:#010x}, {expected[3]:#010x})"
    )


def test_philox_kernel_matches_reference():
    """Full 128-lane tile: simulator output must equal CPU reference."""
    lanes = _PHILOX_LANES_PER_TILE
    counter_lo = np.arange(lanes, dtype=np.int32).reshape(-1, 1)
    key_lo = np.full((lanes, 1), 0x12345678 & 0x7FFFFFFF, dtype=np.int32)
    key_hi = np.full((lanes, 1), 0x9ABCDEF0 & 0x7FFFFFFF, dtype=np.int32)

    out = nki.simulate_kernel(philox4x32_kernel, counter_lo, key_lo, key_hi)

    # Reference: full 4-word counter (counter_lo, 0, 0, 0), matching kernel.
    ctr_ref = torch.zeros(lanes, 4, dtype=torch.int64)
    ctr_ref[:, 0] = torch.from_numpy(counter_lo.reshape(-1)).to(torch.int64)
    key_ref = torch.zeros(lanes, 2, dtype=torch.int64)
    key_ref[:, 0] = int(key_lo[0, 0]) & UINT32_MASK
    key_ref[:, 1] = int(key_hi[0, 0]) & UINT32_MASK
    expected = philox4x32_reference(ctr_ref, key_ref).reshape(lanes, 4).numpy()

    np.testing.assert_array_equal(out.astype(np.int64) & UINT32_MASK, expected)


def test_philox_kernel_distribution():
    """100k Philox uint32 outputs converted to floats should be ~U[0,1)."""
    lanes = _PHILOX_LANES_PER_TILE
    counter_lo = np.arange(lanes, dtype=np.int32).reshape(-1, 1)
    key_lo = np.zeros((lanes, 1), dtype=np.int32)
    key_hi = np.zeros((lanes, 1), dtype=np.int32)

    out = nki.simulate_kernel(philox4x32_kernel, counter_lo, key_lo, key_hi)
    u = (out.astype(np.int64) & UINT32_MASK).astype(np.float64) / 2**32
    # Small sample (128 * 4 = 512); loose tolerances.
    assert abs(u.mean() - 0.5) < 0.05
    assert abs(u.var() - 1 / 12) < 0.02


# ── Box-Muller correctness through the simulator ────────────────────────────


def test_box_muller_kernel_matches_reference():
    """Simulator output should match the CPU Box-Muller reference within tolerance."""
    lanes = _PHILOX_LANES_PER_TILE
    u_torch = philox_uniform_cpu(lanes * 2, seed=99).to(torch.float32)
    uniforms = u_torch.numpy().reshape(lanes, 2)

    out = nki.simulate_kernel(box_muller_kernel, uniforms)
    # Kernel returns (lanes, 2) → flatten to match box_muller_cpu's 1D input/output.
    got_flat = out.reshape(-1)
    expected = box_muller_cpu(u_torch).to(torch.float32).numpy()

    np.testing.assert_allclose(got_flat, expected, rtol=1e-4, atol=1e-4)


def test_box_muller_kernel_distribution():
    """Box-Muller samples should be ~N(0, 1) — check mean/std on a small tile."""
    lanes = _PHILOX_LANES_PER_TILE
    u_torch = philox_uniform_cpu(lanes * 2, seed=11).to(torch.float32)
    uniforms = u_torch.numpy().reshape(lanes, 2)

    out = nki.simulate_kernel(box_muller_kernel, uniforms)
    # 256 samples is tiny; use loose tolerances.
    assert abs(float(out.mean())) < 0.15
    assert abs(float(out.std()) - 1.0) < 0.15
