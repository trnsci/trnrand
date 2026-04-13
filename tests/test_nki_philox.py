"""Philox 4×32 NKI kernel tests.

CPU-side tests run anywhere and validate the reference implementation
(determinism, distributional uniformity). NKI-marked tests require
Trainium hardware and validate the on-device kernel against the CPU
reference.

Run all tests:        pytest tests/test_nki_philox.py -v
Skip neuron tests:    pytest tests/test_nki_philox.py -v -m "not neuron"
Hardware only:        pytest tests/test_nki_philox.py -v -m neuron
"""

from __future__ import annotations

import pytest
import torch

from trnrand.nki import HAS_NKI
from trnrand.nki.dispatch import (
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_W0,
    PHILOX_W1,
    UINT32_MASK,
    box_muller_cpu,
    philox4x32_reference,
    philox_uniform_cpu,
)

# ── CPU reference: spec invariants ────────────────────────────────────────────


class TestPhiloxReference:
    def test_constants(self):
        assert PHILOX_M0 == 0xD2511F53
        assert PHILOX_M1 == 0xCD9E8D57
        assert PHILOX_W0 == 0x9E3779B9
        assert PHILOX_W1 == 0xBB67AE85

    @pytest.mark.parametrize(
        "counter,key,expected",
        [
            # Published Philox4×32-10 test vectors from Salmon et al. SC'11
            # (Random123 reference) — also match the cuRAND / JAX
            # implementations. Any drift from these means the kernel does not
            # produce a standards-conformant stream.
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
    def test_spec_vectors(self, counter, key, expected):
        ctr = torch.tensor([list(counter)], dtype=torch.int64)
        k = torch.tensor([list(key)], dtype=torch.int64)
        out = philox4x32_reference(ctr, k)[0].tolist()
        assert out == list(expected), (
            f"Philox4×32-10 output mismatch: got "
            f"({out[0]:#010x}, {out[1]:#010x}, {out[2]:#010x}, {out[3]:#010x}), "
            f"expected ({expected[0]:#010x}, {expected[1]:#010x}, "
            f"{expected[2]:#010x}, {expected[3]:#010x})"
        )

    def test_zero_input_deterministic(self):
        ctr = torch.zeros(1, 4, dtype=torch.int64)
        key = torch.zeros(1, 2, dtype=torch.int64)
        out_a = philox4x32_reference(ctr, key)
        out_b = philox4x32_reference(ctr, key)
        assert torch.equal(out_a, out_b)

    def test_output_range(self):
        ctr = torch.arange(8, dtype=torch.int64).unsqueeze(-1).repeat(1, 4)
        key = torch.zeros(8, 2, dtype=torch.int64)
        out = philox4x32_reference(ctr, key)
        assert out.min().item() >= 0
        assert out.max().item() <= UINT32_MASK

    def test_different_counters_differ(self):
        ctr1 = torch.zeros(1, 4, dtype=torch.int64)
        ctr2 = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)
        key = torch.zeros(1, 2, dtype=torch.int64)
        out1 = philox4x32_reference(ctr1, key)
        out2 = philox4x32_reference(ctr2, key)
        assert not torch.equal(out1, out2)

    def test_different_keys_differ(self):
        ctr = torch.zeros(1, 4, dtype=torch.int64)
        key1 = torch.zeros(1, 2, dtype=torch.int64)
        key2 = torch.tensor([[42, 0]], dtype=torch.int64)
        out1 = philox4x32_reference(ctr, key1)
        out2 = philox4x32_reference(ctr, key2)
        assert not torch.equal(out1, out2)

    def test_disjoint_counter_ranges_no_overlap(self):
        # Two non-overlapping counter ranges produce disjoint output blocks.
        key = torch.zeros(64, 2, dtype=torch.int64)
        ctr_a = torch.zeros(64, 4, dtype=torch.int64)
        ctr_a[:, 0] = torch.arange(64)
        ctr_b = torch.zeros(64, 4, dtype=torch.int64)
        ctr_b[:, 0] = torch.arange(64, 128)

        out_a = philox4x32_reference(ctr_a, key).reshape(-1)
        out_b = philox4x32_reference(ctr_b, key).reshape(-1)
        # Probability of any uint32 collision in 256 vs 256 randoms is ~256²/2³² ≈ 1.5e-5.
        # Allow up to 1 spurious collision.
        a_set = set(out_a.tolist())
        overlap = sum(1 for x in out_b.tolist() if x in a_set)
        assert overlap <= 1

    def test_uniform_cpu_range(self):
        u = philox_uniform_cpu(10_000, seed=42)
        assert u.min().item() >= 0.0
        assert u.max().item() < 1.0
        assert u.dtype == torch.float32

    def test_uniform_cpu_distribution(self):
        u = philox_uniform_cpu(100_000, seed=42)
        # Mean of U[0,1) should be ~0.5, var ~1/12.
        assert abs(u.mean().item() - 0.5) < 0.01
        assert abs(u.var().item() - 1 / 12) < 0.005

    def test_uniform_cpu_seed_reproducible(self):
        u1 = philox_uniform_cpu(1024, seed=42)
        u2 = philox_uniform_cpu(1024, seed=42)
        assert torch.equal(u1, u2)

    def test_uniform_cpu_offset_disjoint(self):
        # Generating with offset N should equal the tail of a 2N-element draw.
        full = philox_uniform_cpu(2048, seed=42, counter_offset=0)
        tail = philox_uniform_cpu(1024, seed=42, counter_offset=256)  # 256 blocks × 4 = 1024
        assert torch.equal(full[1024:], tail)


# ── Box-Muller CPU reference ──────────────────────────────────────────────────


class TestBoxMullerReference:
    def test_shape_preserved(self):
        u = torch.empty(1024).uniform_()
        z = box_muller_cpu(u)
        assert z.shape == u.shape

    def test_distribution(self):
        # Feed 100k uniforms; output should be ~N(0, 1).
        u = philox_uniform_cpu(100_000, seed=7)
        z = box_muller_cpu(u)
        assert abs(z.mean().item()) < 0.02
        assert abs(z.std().item() - 1.0) < 0.02

    def test_deterministic(self):
        u = philox_uniform_cpu(2048, seed=123)
        z1 = box_muller_cpu(u)
        z2 = box_muller_cpu(u)
        assert torch.equal(z1, z2)


# ── NKI hardware: kernel matches CPU reference ────────────────────────────────


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestPhiloxNKI:
    """Validates the NKI kernel against the CPU reference.

    Runs on trn1/trn2 only. Iterate on `philox4x32_kernel` until these
    tests pass — the kernel is currently scaffolded and may need NKI API
    adjustments (mulhi/lo helpers, bitwise primitives, store layout).
    """

    def test_kernel_matches_reference_zero_key(self):
        from trnrand.nki.dispatch import philox4x32_nki

        n_lanes = 128
        counter_lo = torch.arange(n_lanes, dtype=torch.int32)
        key_lo = torch.zeros(n_lanes, dtype=torch.int32)
        key_hi = torch.zeros(n_lanes, dtype=torch.int32)

        out = philox4x32_nki(counter_lo, key_lo, key_hi)

        # Reference: same lanes, same 4-word counter (counter_lo, 0, 0, 0).
        ctr_ref = torch.zeros(n_lanes, 4, dtype=torch.int64)
        ctr_ref[:, 0] = counter_lo.to(torch.int64)
        key_ref = torch.zeros(n_lanes, 2, dtype=torch.int64)
        expected = philox4x32_reference(ctr_ref, key_ref).to(torch.int32).reshape(-1)

        assert torch.equal(out, expected)

    def test_kernel_with_nonzero_key(self):
        from trnrand.nki.dispatch import philox4x32_nki

        n_lanes = 128
        counter_lo = torch.arange(n_lanes, dtype=torch.int32)
        key_lo = torch.full((n_lanes,), 0x12345678 & 0x7FFFFFFF, dtype=torch.int32)
        key_hi = torch.full((n_lanes,), 0x9ABCDEF0 & 0x7FFFFFFF, dtype=torch.int32)

        out = philox4x32_nki(counter_lo, key_lo, key_hi)

        ctr_ref = torch.zeros(n_lanes, 4, dtype=torch.int64)
        ctr_ref[:, 0] = counter_lo.to(torch.int64)
        key_ref = torch.zeros(n_lanes, 2, dtype=torch.int64)
        key_ref[:, 0] = key_lo.to(torch.int64) & UINT32_MASK
        key_ref[:, 1] = key_hi.to(torch.int64) & UINT32_MASK
        expected = philox4x32_reference(ctr_ref, key_ref).to(torch.int32).reshape(-1)

        assert torch.equal(out, expected)

    def test_box_muller_kernel_matches_reference(self):
        from trnrand.nki.dispatch import box_muller_nki

        u = philox_uniform_cpu(4096, seed=99).to(torch.float32)
        out = box_muller_nki(u)
        expected = box_muller_cpu(u).to(torch.float32)
        # Allow small tolerance for hardware cos/sin/log/sqrt vs CPU libm.
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_box_muller_kernel_distribution(self):
        from trnrand.nki.dispatch import box_muller_nki

        u = philox_uniform_cpu(100_000, seed=11).to(torch.float32)
        out = box_muller_nki(u)
        assert abs(out.mean().item()) < 0.02
        assert abs(out.std().item() - 1.0) < 0.02

    def test_kernel_distribution(self):
        # 100k uint32 outputs converted to floats should be ~U[0,1).
        from trnrand.nki.dispatch import philox4x32_nki

        n_lanes = 25_000  # 4 outputs per lane → 100k samples
        counter_lo = torch.arange(n_lanes, dtype=torch.int32)
        key_lo = torch.zeros(n_lanes, dtype=torch.int32)
        key_hi = torch.zeros(n_lanes, dtype=torch.int32)

        out = philox4x32_nki(counter_lo, key_lo, key_hi)

        u = (out.to(torch.int64) & UINT32_MASK).to(torch.float64) / 2**32
        assert abs(u.mean().item() - 0.5) < 0.005
        assert abs(u.var().item() - 1 / 12) < 0.002
