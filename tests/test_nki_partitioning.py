"""NKI partition equivalence and hardware scaling tests (Phase 4).

Test tiers
----------
- nki_simulator : partition equivalence via the actual NKI Threefry kernels
  (requires TRNRAND_USE_SIMULATOR=1 + nki>=0.3.0).
- neuron        : zero cross-chip coordination (profiler) and near-linear
  strong scaling (trn1.32xlarge — manual run only).

None of these tests run on a bare dev host without neuronxcc.
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch

import trnrand
from trnrand.generator import _BATCH_SIZE, Generator
from trnrand.nki.program import ProgramBuilder

try:
    from trnrand.nki.dispatch import (
        _PROGRAM_TILES,
        threefry_stream_normal,
        threefry_stream_uniform,
    )
    from trnrand.nki.dispatch import (
        HAS_NKI as _HAS_NKI,
    )
except ImportError:
    _HAS_NKI = False
    _PROGRAM_TILES = 32
    threefry_stream_normal = None
    threefry_stream_uniform = None

# ── Autouse fixture ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _require_simulator(request):
    if request.node.get_closest_marker("nki_simulator") is None:
        return
    if os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNRAND_USE_SIMULATOR=1 required")
    if not _HAS_NKI:
        pytest.skip("nki>=0.3.0 not importable on this host")


# ── Helpers ───────────────────────────────────────────────────────────────────

_LANES = 128
_NORMALS_PER_LAUNCH = _PROGRAM_TILES * _LANES * 4  # 16,384


def _n_batches_streaming(n: int) -> int:
    """Counter batches consumed by a streaming dispatch of n elements."""
    return math.ceil(n / _NORMALS_PER_LAUNCH) * _PROGRAM_TILES


# ── Partition equivalence — per-tile dispatch ─────────────────────────────────


@pytest.mark.nki_simulator
class TestPartitionEquivalencePerTile:
    """Partition equivalence for per-tile dispatch (uniform, normal, exponential).

    For a 1-chip run of N samples and a P-chip run of N/P samples each,
    concatenating the P-chip outputs in rank order must produce the same
    tensor as the 1-chip run — bit-exact, via torch.equal.

    Equivalence holds when N is a multiple of 512 × P (the NKI batch size
    times the chip count). Tests use N values chosen to satisfy this.
    """

    @pytest.mark.parametrize("P", [2, 4])
    def test_uniform_partition_equivalence(self, P):
        N = 512 * P * 16  # exact multiple; no batch-boundary rounding
        seed = 42
        trnrand.set_backend("nki")

        # 1-chip reference
        g_full = Generator(seed=seed)
        full = trnrand.uniform(N, generator=g_full)

        # P-chip draws concatenated in rank order
        n_per = N // P
        chunks = []
        for r in range(P):
            g = Generator(seed=seed, partition_rank=r, partition_size=P)
            chunks.append(trnrand.uniform(n_per, generator=g))

        assert torch.equal(full, torch.cat(chunks)), (
            f"uniform P={P}: concatenated P-chip output ≠ 1-chip output"
        )

    @pytest.mark.parametrize("P", [2, 4])
    def test_normal_partition_equivalence(self, P):
        N = 512 * P * 16
        seed = 7
        trnrand.set_backend("nki")

        g_full = Generator(seed=seed)
        full = trnrand.normal(N, generator=g_full)

        n_per = N // P
        chunks = []
        for r in range(P):
            g = Generator(seed=seed, partition_rank=r, partition_size=P)
            chunks.append(trnrand.normal(n_per, generator=g))

        assert torch.equal(full, torch.cat(chunks)), (
            f"normal P={P}: concatenated P-chip output ≠ 1-chip output"
        )

    @pytest.mark.parametrize("P", [2, 4])
    def test_exponential_partition_equivalence(self, P):
        N = 512 * P * 16
        seed = 99
        trnrand.set_backend("nki")

        g_full = Generator(seed=seed)
        full = trnrand.exponential(N, generator=g_full)

        n_per = N // P
        chunks = []
        for r in range(P):
            g = Generator(seed=seed, partition_rank=r, partition_size=P)
            chunks.append(trnrand.exponential(n_per, generator=g))

        assert torch.equal(full, torch.cat(chunks)), (
            f"exponential P={P}: concatenated P-chip output ≠ 1-chip output"
        )


# ── Partition equivalence — streaming dispatch ────────────────────────────────


@pytest.mark.nki_simulator
class TestPartitionEquivalenceStreaming:
    """Partition equivalence for streaming kernels (threefry_stream_normal/uniform).

    Uses the dispatch functions directly so counter_offset arithmetic is
    testable without going through GeneratorProgram.
    """

    @pytest.mark.parametrize("P", [2, 4])
    def test_stream_normal_partition_equivalence(self, P):
        N = _NORMALS_PER_LAUNCH * P * 2  # exact multiple of one streaming launch
        seed = 13

        full = threefry_stream_normal(N, seed=seed, counter_offset=0)

        n_per = N // P
        batches_per_chip = _n_batches_streaming(n_per)
        chunks = [
            threefry_stream_normal(n_per, seed=seed, counter_offset=r * batches_per_chip)
            for r in range(P)
        ]
        assert torch.equal(full, torch.cat(chunks)), (
            f"stream_normal P={P}: concatenated output ≠ 1-chip"
        )

    @pytest.mark.parametrize("P", [2, 4])
    def test_stream_uniform_partition_equivalence(self, P):
        N = _NORMALS_PER_LAUNCH * P * 2
        seed = 17

        full = threefry_stream_uniform(N, seed=seed, counter_offset=0)

        n_per = N // P
        batches_per_chip = _n_batches_streaming(n_per)
        chunks = [
            threefry_stream_uniform(n_per, seed=seed, counter_offset=r * batches_per_chip)
            for r in range(P)
        ]
        assert torch.equal(full, torch.cat(chunks)), (
            f"stream_uniform P={P}: concatenated output ≠ 1-chip"
        )


# ── Partition equivalence — GeneratorProgram / stream_into ───────────────────


@pytest.mark.nki_simulator
class TestPartitionEquivalenceProgram:
    """Validates that GeneratorProgram.stream_into applies partition offsets.

    chip_r_prog.stream_into produces the same output as the corresponding
    slice of the single-chip program when counter_start is 0 for all chips.
    """

    @pytest.mark.parametrize("P", [2, 4])
    def test_normal_program_partition_equivalence(self, P):
        N = _NORMALS_PER_LAUNCH * P * 2
        n_per = N // P
        seed = 5

        # 1-chip reference
        full_prog = ProgramBuilder(seed=seed).normal(N, out="z").build()
        z_full = torch.empty(N)
        full_prog.stream_into({"z": z_full})

        # P-chip draws
        chunks = []
        for r in range(P):
            gen = Generator(seed=seed, partition_rank=r, partition_size=P)
            prog = gen.new_program().normal(n_per, out="z").build()
            z = torch.empty(n_per)
            prog.stream_into({"z": z})
            chunks.append(z)

        assert torch.equal(z_full, torch.cat(chunks)), (
            f"GeneratorProgram normal P={P}: partition output ≠ 1-chip"
        )

    def test_multi_distribution_program_equivalence_P2(self):
        P = 2
        N = _NORMALS_PER_LAUNCH * P * 2
        n_per = N // P
        seed = 3

        # 1-chip reference
        full_prog = ProgramBuilder(seed=seed).normal(N, out="z").uniform(N, out="u").build()
        z_full = torch.empty(N)
        u_full = torch.empty(N)
        full_prog.stream_into({"z": z_full, "u": u_full})

        # 2-chip draws
        z_chunks, u_chunks = [], []
        for r in range(P):
            gen = Generator(seed=seed, partition_rank=r, partition_size=P)
            prog = gen.new_program().normal(n_per, out="z").uniform(n_per, out="u").build()
            z = torch.empty(n_per)
            u = torch.empty(n_per)
            prog.stream_into({"z": z, "u": u})
            z_chunks.append(z)
            u_chunks.append(u)

        assert torch.equal(z_full, torch.cat(z_chunks)), "normal step: partition ≠ 1-chip"
        assert torch.equal(u_full, torch.cat(u_chunks)), "uniform step: partition ≠ 1-chip"


# ── Advance / resume ──────────────────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestAdvanceResumeEquivalence:
    """advance() followed by continued generation must match an uninterrupted run."""

    def test_advance_then_uniform(self):
        seed = 11
        N_skip = 512 * 8  # must be a multiple of _BATCH_SIZE
        N_draw = 512 * 4

        # Uninterrupted 1-chip run; take the tail
        trnrand.set_backend("nki")
        g_ref = Generator(seed=seed)
        full = trnrand.uniform(N_skip + N_draw, generator=g_ref)
        tail_ref = full[N_skip:]

        # Skip N_skip, then draw N_draw
        g_skip = Generator(seed=seed)
        g_skip.advance(N_skip)
        tail_skip = trnrand.uniform(N_draw, generator=g_skip)

        assert torch.equal(tail_ref, tail_skip), (
            "advance then uniform: output does not match uninterrupted tail"
        )

    def test_advance_then_normal(self):
        seed = 22
        N_skip = 512 * 8
        N_draw = 512 * 4

        trnrand.set_backend("nki")
        g_ref = Generator(seed=seed)
        full = trnrand.normal(N_skip + N_draw, generator=g_ref)
        tail_ref = full[N_skip:]

        g_skip = Generator(seed=seed)
        g_skip.advance(N_skip)
        tail_skip = trnrand.normal(N_draw, generator=g_skip)

        assert torch.equal(tail_ref, tail_skip)

    def test_advance_to_resume(self):
        """advance_to(position) picks up the same stream as advance(n)."""
        seed = 33
        N_skip = 512 * 6
        N_draw = 512 * 2

        trnrand.set_backend("nki")
        g_advance = Generator(seed=seed)
        g_advance.advance(N_skip)
        out_advance = trnrand.uniform(N_draw, generator=g_advance)

        g_to = Generator(seed=seed)
        g_to.advance_to(N_skip)
        out_to = trnrand.uniform(N_draw, generator=g_to)

        assert torch.equal(out_advance, out_to)


# ── Hardware-only tests (trn1.32xlarge) ───────────────────────────────────────


@pytest.mark.neuron
class TestHardwarePartition:
    """Hardware profiler and scaling tests.

    These require physical trn1.32xlarge hardware. Run manually via:
        pytest tests/test_nki_partitioning.py -m neuron -v

    For profiler capture:
        NEURON_PPROF_SINK=/tmp/profile pytest ... -k test_zero_cross_chip
    """

    def test_zero_cross_chip_coordination(self):
        """Smoke test: kernel executes without raising.

        Visual proof of zero coordination: run with NEURON_PPROF_SINK and
        inspect the trace — no collective ops should appear during the kernel.
        """
        gen = Generator(seed=42, partition_rank=0, partition_size=32)
        trnrand.set_backend("nki")
        trnrand.normal(1_000_000, generator=gen)  # must not raise

    def test_near_linear_strong_scaling(self, benchmark):
        """Throughput guard: 1-chip baseline for the strong-scaling comparison.

        Run on trn1.32xlarge: compare this number against 32-chip throughput.
        ≥ 87% efficiency (≥ 28× 1-chip) is the Phase 4 target.
        """
        gen = Generator(seed=0, partition_rank=0, partition_size=1)
        trnrand.set_backend("nki")

        def _draw():
            trnrand.normal(1_000_000, generator=gen)

        benchmark.pedantic(_draw, iterations=1, rounds=50, warmup_rounds=5)

    def test_partition_equivalence_P32_hardware(self):
        """Bit-exact partition equivalence on 32-chip hardware.

        This is a single-chip structural test — it validates the counter
        arithmetic produces the correct subrange that would concatenate to
        the 1-chip output. Full 32-chip concatenation requires orchestrating
        32 NeuronCores externally (not possible in a single-process test).
        """
        # Validate that chip 0 of 32 produces the FIRST slice of the 1-chip output.
        N = 512 * 32 * 16
        n_per = N // 32
        seed = 77

        trnrand.set_backend("nki")
        g_full = Generator(seed=seed)
        full = trnrand.uniform(N, generator=g_full)

        g_chip0 = Generator(seed=seed, partition_rank=0, partition_size=32)
        chip0 = trnrand.uniform(n_per, generator=g_chip0)

        assert torch.equal(full[:n_per], chip0), (
            "chip 0 of 32 does not match first slice of 1-chip output"
        )

    def test_neff_cache_reuse_with_partition(self):
        """Second stream_into with partition args must still reuse NEFF cache."""
        gen = Generator(seed=0, partition_rank=0, partition_size=4)
        prog = gen.new_program().normal(1_000_000, out="z").build()
        z = torch.empty(1_000_000)
        prog.stream_into({"z": z})  # warm-up / compile

        t0 = time.perf_counter()
        prog.stream_into({"z": z})
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        assert elapsed_ms < 100, (
            f"Second stream_into with partition took {elapsed_ms:.1f}ms — NEFF cache miss?"
        )
