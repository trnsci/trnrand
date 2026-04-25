"""Tests for counter-partitioned multi-chip RNG (Phase 4).

Test tiers
----------
- No marker   : CPU-only; runs on dev hosts without neuronxcc.
  Covers the Generator partition API and CPU-reference counter math.
- nki_simulator: partition equivalence over the actual Threefry NKI kernels.
- neuron       : hardware profiler trace, near-linear scaling (trn1.32xlarge).
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from trnrand.generator import _BATCH_SIZE, Generator
from trnrand.nki.dispatch import threefry_uniform_cpu

try:
    from trnrand.nki.dispatch import HAS_NKI as _HAS_NKI
except ImportError:
    _HAS_NKI = False


# ── Autouse fixture ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _require_simulator(request):
    if request.node.get_closest_marker("nki_simulator") is None:
        return
    if os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNRAND_USE_SIMULATOR=1 required")
    if not _HAS_NKI:
        pytest.skip("nki>=0.3.0 not importable on this host")


# ── Generator partition API ────────────────────────────────────────────────────


class TestGeneratorPartitionAPI:
    def test_default_single_chip(self):
        gen = Generator(seed=42)
        assert gen._partition_rank == 0
        assert gen._partition_size == 1
        assert gen.position() == 0

    def test_partition_kwargs_stored(self):
        gen = Generator(seed=0, partition_rank=2, partition_size=8)
        assert gen._partition_rank == 2
        assert gen._partition_size == 8

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="partition_rank"):
            Generator(seed=0, partition_rank=4, partition_size=4)

    def test_rank_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="partition_rank"):
            Generator(seed=0, partition_rank=8, partition_size=8)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="partition_size"):
            Generator(seed=0, partition_rank=0, partition_size=0)

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="partition_size"):
            Generator(seed=0, partition_rank=0, partition_size=-1)

    def test_advance_full_batch(self):
        gen = Generator(seed=0)
        gen.advance(_BATCH_SIZE)
        assert gen.position() == _BATCH_SIZE

    def test_advance_partial_batch_rounds_up(self):
        gen = Generator(seed=0)
        gen.advance(1)
        assert gen.position() == _BATCH_SIZE  # ceil(1/512) = 1 batch

    def test_advance_zero_is_noop(self):
        gen = Generator(seed=0)
        gen.advance(0)
        assert gen.position() == 0

    def test_advance_multiple_batches(self):
        gen = Generator(seed=0)
        gen.advance(4 * _BATCH_SIZE)
        assert gen.position() == 4 * _BATCH_SIZE

    def test_advance_negative_raises(self):
        gen = Generator(seed=0)
        with pytest.raises(ValueError, match="non-negative"):
            gen.advance(-1)

    def test_advance_to_sets_position(self):
        gen = Generator(seed=0)
        gen.advance_to(1024)
        assert gen.position() == 1024

    def test_advance_to_rounds_down_to_batch(self):
        gen = Generator(seed=0)
        gen.advance_to(_BATCH_SIZE + 1)
        # floor(513/512) = 1 batch → position = 512
        assert gen.position() == _BATCH_SIZE

    def test_advance_to_zero(self):
        gen = Generator(seed=0)
        gen.advance(3 * _BATCH_SIZE)
        gen.advance_to(0)
        assert gen.position() == 0

    def test_advance_to_negative_raises(self):
        gen = Generator(seed=0)
        with pytest.raises(ValueError, match="non-negative"):
            gen.advance_to(-1)

    def test_manual_seed_resets_counter(self):
        gen = Generator(seed=0)
        gen.advance(10 * _BATCH_SIZE)
        assert gen.position() > 0
        gen.manual_seed(42)
        assert gen.position() == 0

    def test_chip_offset_rank0_at_zero(self):
        gen = Generator(seed=0, partition_rank=0, partition_size=4)
        # rank 0: offset = 0 * n_batches + 0 = 0
        assert gen._chip_counter_offset(1024) == 0

    def test_chip_offset_rank1_size2(self):
        gen = Generator(seed=0, partition_rank=1, partition_size=2)
        n = 1024  # = 2 batches exactly
        n_batches = math.ceil(n / _BATCH_SIZE)
        assert gen._chip_counter_offset(n) == 1 * n_batches

    def test_chip_offset_rank3_size4(self):
        gen = Generator(seed=0, partition_rank=3, partition_size=4)
        n = 2048  # = 4 batches
        n_batches = math.ceil(n / _BATCH_SIZE)
        assert gen._chip_counter_offset(n) == 3 * n_batches

    def test_chip_offset_accumulates_counter(self):
        gen = Generator(seed=0, partition_rank=1, partition_size=2)
        gen.advance(_BATCH_SIZE)  # _counter = 1
        n = 1024
        n_batches = math.ceil(n / _BATCH_SIZE)
        # rank=1 offset = 1*n_batches + _counter(=1)
        assert gen._chip_counter_offset(n) == 1 * n_batches + 1

    def test_advance_by_elements_increments_counter(self):
        gen = Generator(seed=0)
        gen._advance_by_elements(1024)
        assert gen._counter == math.ceil(1024 / _BATCH_SIZE)

    def test_advance_by_elements_partial(self):
        gen = Generator(seed=0)
        gen._advance_by_elements(1)
        assert gen._counter == 1  # ceil(1/512) = 1

    def test_position_returns_samples_not_batches(self):
        gen = Generator(seed=0)
        gen._counter = 3
        assert gen.position() == 3 * _BATCH_SIZE


# ── CPU-reference partition equivalence ───────────────────────────────────────


def _n_blocks(n: int) -> int:
    """Number of Threefry 4-sample output blocks for n samples (CPU-reference unit)."""
    return math.ceil(n / 4)


class TestPartitionEquivalenceCPU:
    """Validates counter partitioning using threefry_uniform_cpu directly.

    threefry_uniform_cpu uses Threefry 4-sample output blocks as its
    counter_offset unit (1 block = 4 samples).  The NKI kernel aggregates
    128 lanes into a 512-sample batch — a coarser unit.  These tests prove
    the fundamental counter arithmetic at the 4-sample granularity.

    A 1-chip draw of N samples equals the concatenation of P-chip draws of
    N/P samples with counter_offsets [0, n_blocks_per, 2*n_blocks_per, ...].
    """

    def test_uniform_P2(self):
        N, P = 4096, 2
        full = threefry_uniform_cpu(N, seed=42, counter_offset=0)
        n_per = N // P
        chunks = [
            threefry_uniform_cpu(n_per, seed=42, counter_offset=r * _n_blocks(n_per))
            for r in range(P)
        ]
        assert torch.equal(full, torch.cat(chunks)), "P=2 partition not equivalent to 1-chip"

    def test_uniform_P4(self):
        N, P = 4096, 4
        full = threefry_uniform_cpu(N, seed=7, counter_offset=0)
        n_per = N // P
        chunks = [
            threefry_uniform_cpu(n_per, seed=7, counter_offset=r * _n_blocks(n_per))
            for r in range(P)
        ]
        assert torch.equal(full, torch.cat(chunks))

    def test_uniform_P8(self):
        N, P = 8192, 8
        full = threefry_uniform_cpu(N, seed=99, counter_offset=0)
        n_per = N // P
        chunks = [
            threefry_uniform_cpu(n_per, seed=99, counter_offset=r * _n_blocks(n_per))
            for r in range(P)
        ]
        assert torch.equal(full, torch.cat(chunks))

    def test_uniform_P16(self):
        N, P = 8192, 16
        full = threefry_uniform_cpu(N, seed=123, counter_offset=0)
        n_per = N // P
        chunks = [
            threefry_uniform_cpu(n_per, seed=123, counter_offset=r * _n_blocks(n_per))
            for r in range(P)
        ]
        assert torch.equal(full, torch.cat(chunks))

    def test_different_seeds_differ(self):
        N = 1024
        a = threefry_uniform_cpu(N, seed=1, counter_offset=0)
        b = threefry_uniform_cpu(N, seed=2, counter_offset=0)
        assert not torch.equal(a, b)

    def test_counter_offset_produces_disjoint_ranges(self):
        # Non-overlapping counter ranges → statistically independent outputs.
        N = 1024
        a = threefry_uniform_cpu(N, seed=42, counter_offset=0)
        b = threefry_uniform_cpu(N, seed=42, counter_offset=_n_blocks(N))
        assert not torch.equal(a, b)

    def test_checkpoint_resume(self):
        """Resuming at counter N_first produces the same tail as the uninterrupted draw."""
        N_full = 2048
        N_first = 1024
        full = threefry_uniform_cpu(N_full, seed=5, counter_offset=0)
        tail = threefry_uniform_cpu(N_full - N_first, seed=5, counter_offset=_n_blocks(N_first))
        assert torch.equal(full[N_first:], tail)
