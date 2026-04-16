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

import numpy as np
import pytest
import torch

from trnrand.nki import HAS_NKI
from trnrand.nki.dispatch import (
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_W0,
    PHILOX_W1,
    THREEFRY_ROTATIONS,
    THREEFRY_SKEIN_KS_PARITY,
    UINT32_MASK,
    _add32_bytes_numpy,
    _rotl32_bytes_numpy,
    box_muller_cpu,
    philox4x32_reference,
    philox_uniform_cpu,
    threefry4x32_reference,
    threefry_uniform_cpu,
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

    def test_mul32_numpy_matches_ground_truth(self):
        """Pure-numpy 16-bit decomposition must match Python unbounded-int ground truth.

        If this test fails, the 16-bit algorithm itself has a logic bug —
        NKI isn't in the picture. Compare numpy output to Python
        `(uint64(a) * uint64(b)) >> 32` and `& 0xFFFFFFFF`.
        """
        import numpy as np

        from trnrand.nki.dispatch import _mul32_hi_lo_numpy

        # Test inputs: boundary values + a handful of randoms.
        test_inputs = [
            0x00000000,
            0x00000001,
            0x00000002,
            0x0000FFFF,
            0x00010000,
            0xFFFE0001,
            0x7FFFFFFF,
            0x80000000,
            0xD2511F53,  # = PHILOX_M0
            0xFFFFFFFF,
        ]
        rng = np.random.default_rng(42)
        test_inputs.extend(int(x) for x in rng.integers(0, 2**32, size=10, dtype=np.uint64))

        # Multiplier: PHILOX_M0 — the actual constant our kernel uses most.
        m0_l = 0xD2511F53 & 0xFFFF
        m0_h = (0xD2511F53 >> 16) & 0xFFFF

        a_arr = np.array(test_inputs, dtype=np.uint32).reshape(-1, 1)
        hi_got, lo_got = _mul32_hi_lo_numpy(a_arr, m0_l, m0_h)

        for i, a in enumerate(test_inputs):
            full = (a * 0xD2511F53) & ((1 << 64) - 1)  # Python unbounded
            hi_expected = (full >> 32) & 0xFFFFFFFF
            lo_expected = full & 0xFFFFFFFF

            # Compare as uint32 bit pattern (got values are int32, same bits).
            got_hi_u = int(hi_got[i, 0]) & 0xFFFFFFFF
            got_lo_u = int(lo_got[i, 0]) & 0xFFFFFFFF

            assert got_hi_u == hi_expected, (
                f"hi mismatch for a={a:#010x}: got {got_hi_u:#010x}, expected {hi_expected:#010x}"
            )
            assert got_lo_u == lo_expected, (
                f"lo mismatch for a={a:#010x}: got {got_lo_u:#010x}, expected {lo_expected:#010x}"
            )


# ── Threefry 4×32-20 CPU reference ───────────────────────────────────────────


class TestThreefryReference:
    def test_constants(self):
        assert THREEFRY_SKEIN_KS_PARITY == 0x1BD11BDA
        assert THREEFRY_ROTATIONS == [(10, 26), (11, 21), (13, 27), (23, 5)]

    @pytest.mark.parametrize(
        "counter,key,expected",
        [
            # Random123 library KAT vectors for Threefry4×32-20.
            # Source: DE Shaw Research random123 test suite.
            (
                (0x00000000, 0x00000000, 0x00000000, 0x00000000),
                (0x00000000, 0x00000000, 0x00000000, 0x00000000),
                (0x3425621E, 0x64AF086C, 0x4939F9F4, 0x02F34BF4),
            ),
            (
                (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
                (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
                (0xFF0E3F66, 0x66CE18C2, 0xEBF1FE02, 0xC14E9FCA),
            ),
            (
                (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
                (0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89),
                (0x55FF6421, 0x9A904ECF, 0x02EB6042, 0xA0FC59B8),
            ),
        ],
    )
    def test_spec_vectors(self, counter, key, expected):
        ctr = torch.tensor([list(counter)], dtype=torch.int64)
        k = torch.tensor([list(key)], dtype=torch.int64)
        out = threefry4x32_reference(ctr, k)[0].tolist()
        assert out == list(expected), (
            f"Threefry4×32-20 output mismatch: got "
            f"({out[0]:#010x}, {out[1]:#010x}, {out[2]:#010x}, {out[3]:#010x}), "
            f"expected ({expected[0]:#010x}, {expected[1]:#010x}, "
            f"{expected[2]:#010x}, {expected[3]:#010x})"
        )

    def test_zero_input_deterministic(self):
        ctr = torch.zeros(1, 4, dtype=torch.int64)
        key = torch.zeros(1, 4, dtype=torch.int64)
        out_a = threefry4x32_reference(ctr, key)
        out_b = threefry4x32_reference(ctr, key)
        assert torch.equal(out_a, out_b)

    def test_output_range(self):
        ctr = torch.arange(8, dtype=torch.int64).unsqueeze(-1).repeat(1, 4)
        key = torch.zeros(8, 4, dtype=torch.int64)
        out = threefry4x32_reference(ctr, key)
        assert out.min().item() >= 0
        assert out.max().item() <= UINT32_MASK

    def test_different_counters_differ(self):
        ctr1 = torch.zeros(1, 4, dtype=torch.int64)
        ctr2 = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)
        key = torch.zeros(1, 4, dtype=torch.int64)
        out1 = threefry4x32_reference(ctr1, key)
        out2 = threefry4x32_reference(ctr2, key)
        assert not torch.equal(out1, out2)

    def test_different_keys_differ(self):
        ctr = torch.zeros(1, 4, dtype=torch.int64)
        key1 = torch.zeros(1, 4, dtype=torch.int64)
        key2 = torch.tensor([[42, 0, 0, 0]], dtype=torch.int64)
        out1 = threefry4x32_reference(ctr, key1)
        out2 = threefry4x32_reference(ctr, key2)
        assert not torch.equal(out1, out2)

    def test_disjoint_counter_ranges_no_overlap(self):
        key = torch.zeros(64, 4, dtype=torch.int64)
        ctr_a = torch.zeros(64, 4, dtype=torch.int64)
        ctr_a[:, 0] = torch.arange(64)
        ctr_b = torch.zeros(64, 4, dtype=torch.int64)
        ctr_b[:, 0] = torch.arange(64, 128)
        out_a = threefry4x32_reference(ctr_a, key).reshape(-1)
        out_b = threefry4x32_reference(ctr_b, key).reshape(-1)
        a_set = set(out_a.tolist())
        overlap = sum(1 for x in out_b.tolist() if x in a_set)
        assert overlap <= 1

    def test_uniform_cpu_range(self):
        u = threefry_uniform_cpu(10_000, seed=42)
        assert u.min().item() >= 0.0
        assert u.max().item() < 1.0
        assert u.dtype == torch.float32

    def test_uniform_cpu_distribution(self):
        u = threefry_uniform_cpu(100_000, seed=42)
        assert abs(u.mean().item() - 0.5) < 0.01
        assert abs(u.var().item() - 1 / 12) < 0.005

    def test_uniform_cpu_seed_reproducible(self):
        u1 = threefry_uniform_cpu(1024, seed=42)
        u2 = threefry_uniform_cpu(1024, seed=42)
        assert torch.equal(u1, u2)

    def test_uniform_cpu_different_seeds_differ(self):
        u1 = threefry_uniform_cpu(1024, seed=42)
        u2 = threefry_uniform_cpu(1024, seed=99)
        assert not torch.equal(u1, u2)

    def test_add32_bytes_numpy(self):
        """Byte-decomposed add must match Python unbounded-int ground truth."""
        test_pairs = [
            (0x00000000, 0x00000000),
            (0x00000001, 0x00000001),
            (0xFFFFFFFF, 0x00000001),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
            (0x7FFFFFFF, 0x80000001),
            (0xD2511F53, 0x9E3779B9),
        ]
        for a, b in test_pairs:
            expected = (a + b) & 0xFFFFFFFF
            a_arr = np.array([a], dtype=np.uint32)
            b_arr = np.array([b], dtype=np.uint32)
            got = int(_add32_bytes_numpy(a_arr, b_arr)[0])
            assert got == expected, (
                f"add32_bytes({a:#010x}, {b:#010x}): got {got:#010x}, expected {expected:#010x}"
            )

    def test_rotl32_bytes_numpy(self):
        """Byte-decomposed rotate-left must match Python ground truth."""
        test_inputs = [0x00000001, 0x12345678, 0xFFFFFFFF, 0x80000000, 0xABCDEF01]
        # All rotation constants used by Threefry4×32-20.
        rotations = [10, 26, 11, 21, 13, 27, 23, 5]
        for a in test_inputs:
            for R in rotations:
                expected = ((a << R) | (a >> (32 - R))) & 0xFFFFFFFF
                a_arr = np.array([a], dtype=np.uint32)
                got = int(_rotl32_bytes_numpy(a_arr, R)[0])
                assert got == expected, (
                    f"rotl32({a:#010x}, {R}): got {got:#010x}, expected {expected:#010x}"
                )


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


# ── NKI hardware: Threefry kernel matches CPU reference ───────────────────────


@pytest.mark.neuron
@pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
class TestThreefryNKI:
    """Validates the Threefry NKI kernels against the CPU reference on trn1/trn2.

    Hardware validation status (trn1, 2026-04-16):
      PASS: test_uniform_kernel_matches_cpu_reference
      PASS: test_uniform_kernel_distribution
      PASS: test_uniform_kernel_seed_deterministic
      PASS: test_uniform_kernel_different_seeds_differ
      xfail: test_normal_kernel_distribution  (NCC_IBIR605, see trnrand#2)
      xfail: test_normal_kernel_matches_box_muller_cpu  (NCC_IBIR605, see trnrand#2)

    Threefry uniform kernel is hardware-validated. The two normal-kernel tests
    are blocked by NCC_IBIR605 (trn1 compiler rejects nl.log with non-immediate
    bias) — the same restriction that gates standalone box_muller_kernel on trn1.
    Threefry byte-tile arithmetic (all intermediates ≤ 511 < 2²⁴) is unaffected
    by aws-neuron-sdk#1308 — no xfail marks on the uniform tests.

    The NKI kernel emits float32 uniforms using 3 low bytes of each output
    word (mantissa = b0 + b1×256 + b2×65536, divided by 2²⁴). The expected
    values below use the same formula applied to the CPU reference output.
    """

    def test_uniform_kernel_matches_cpu_reference(self):
        """NKI uniform output must match CPU reference (same 3-byte mantissa)."""
        from trnrand.nki.dispatch import threefry_uniform_nki

        # One full tile: 128 lanes × 4 words = 512 elements, batch 0.
        # Counter layout: c0=lane (0..127), c1=c2=c3=0.
        # Key layout: k0=seed&0xFFFFFF, k1=(seed>>24)&0xFFFFFF, k2=k3=0.
        seed = 0xABCD1234
        n = 512
        LANES = 128

        out = threefry_uniform_nki(n, seed=seed).cpu()

        ctr = torch.zeros(LANES, 4, dtype=torch.int64)
        ctr[:, 0] = torch.arange(LANES, dtype=torch.int64)
        k0 = seed & 0xFFFFFF
        k1 = (seed >> 24) & 0xFFFFFF
        key = torch.tensor([k0, k1, 0, 0], dtype=torch.int64).expand(LANES, 4)
        ref_u32 = threefry4x32_reference(ctr, key)  # (128, 4) int64
        expected = ((ref_u32 & 0xFFFFFF) / 16777216.0).to(torch.float32).reshape(-1)

        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_uniform_kernel_distribution(self):
        """100k Threefry NKI uniforms should be ~U[0, 1)."""
        from trnrand.nki.dispatch import threefry_uniform_nki

        u = threefry_uniform_nki(100_000, seed=42).cpu().to(torch.float64)
        assert u.min().item() >= 0.0
        assert u.max().item() < 1.0
        assert abs(u.mean().item() - 0.5) < 0.01
        assert abs(u.var().item() - 1 / 12) < 0.005

    def test_uniform_kernel_seed_deterministic(self):
        """Same seed must produce identical output across two calls."""
        from trnrand.nki.dispatch import threefry_uniform_nki

        u1 = threefry_uniform_nki(1024, seed=99).cpu()
        u2 = threefry_uniform_nki(1024, seed=99).cpu()
        assert torch.equal(u1, u2)

    def test_uniform_kernel_different_seeds_differ(self):
        """Different seeds must produce different output."""
        from trnrand.nki.dispatch import threefry_uniform_nki

        u1 = threefry_uniform_nki(512, seed=1).cpu()
        u2 = threefry_uniform_nki(512, seed=2).cpu()
        assert not torch.equal(u1, u2)

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "NCC_IBIR605: trn1 compiler rejects nl.log activation with non-immediate "
            "bias parameter. Same restriction that blocks box_muller_kernel on trn1. "
            "Tracked in trnrand#2. Does not apply to trn2+ architectures."
        ),
    )
    def test_normal_kernel_distribution(self):
        """100k Threefry+Box-Muller NKI normals should be ~N(0, 1)."""
        from trnrand.nki.dispatch import threefry_normal_nki

        z = threefry_normal_nki(100_000, seed=7).cpu()
        assert abs(z.mean().item()) < 0.02
        assert abs(z.std().item() - 1.0) < 0.02

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "NCC_IBIR605: trn1 compiler rejects nl.log activation with non-immediate "
            "bias parameter. Same restriction that blocks box_muller_kernel on trn1. "
            "Tracked in trnrand#2. Does not apply to trn2+ architectures."
        ),
    )
    def test_normal_kernel_matches_box_muller_cpu(self):
        """Fused NKI normal output must match CPU Box-Muller applied to same uniforms."""
        from trnrand.nki.dispatch import threefry_normal_nki, threefry_uniform_nki

        seed = 0x1234ABCD
        n = 512  # one full tile, pairs for Box-Muller

        # NKI fused path.
        z_nki = threefry_normal_nki(n, seed=seed).cpu()

        # CPU reference: same Threefry uniforms → same Box-Muller.
        u_cpu = threefry_uniform_nki(n, seed=seed).cpu()
        z_cpu = box_muller_cpu(u_cpu)

        # Hardware transcendentals (cos/sin/log/sqrt) may differ slightly
        # from libm — allow small tolerance.
        torch.testing.assert_close(z_nki, z_cpu, rtol=1e-3, atol=1e-3)
