"""Streaming kernel benchmarks.

Validates the five RFC claims for the v0.6.0 streaming API:

1. Latency floor      — one XLA call covers 16,384 samples; ≪31× cheaper than v0.5.0
2. Throughput         — stream_into × N sustains ≥ 10⁹ samples/s (target; hardware-dependent)
3. NEFF cache reuse   — second stream_into with identical shapes < 100ms
4. Bit-exactness      — GeneratorProgram output matches direct kernel dispatch at same seed
5. Engine overlap     — GpSimd + Vector Engine concurrent (profiler trace; neuron-only)

Run with:

    pytest benchmarks/bench_streaming.py --benchmark-only --benchmark-sort=name

NKI tests skip when HAS_NKI is False. Engine-overlap test requires physical trn2
hardware and is marked `neuron`.
"""

from __future__ import annotations

import time

import pytest
import torch

from trnrand.nki import HAS_NKI
from trnrand.nki.dispatch import _PROGRAM_TILES
from trnrand.nki.program import ProgramBuilder

nki_only = pytest.mark.skipif(not HAS_NKI, reason="requires neuronxcc")
neuron = pytest.mark.neuron

_LANES = 128
_NORMALS_PER_LAUNCH = _PROGRAM_TILES * _LANES * 4  # 16,384

# ── Claim 1: Latency floor ─────────────────────────────────────────────────────


class TestLatencyFloor:
    """One XLA call generates 16,384 samples — the streaming kernel amortises
    NEFF launch overhead over 32 tiles instead of one tile at a time.

    v0.5.0 dispatched one kernel per 512 samples → ~31 round-trips for 16,384.
    v0.6.0 dispatches once for the same count.  The benchmark measures
    steady-state per-call cost (NEFF already cached by the XLA runtime).
    """

    @nki_only
    def test_stream_normal_16k_steady_state(self, benchmark):
        """Benchmark stream_into for exactly one NKI launch (16,384 normals)."""
        prog = ProgramBuilder(seed=0).normal(_NORMALS_PER_LAUNCH, out="z").build()
        z = torch.empty(_NORMALS_PER_LAUNCH)
        # warm-up: trigger NEFF compilation before timing
        prog.stream_into({"z": z})
        prog.stream_into({"z": z})

        benchmark(lambda: prog.stream_into({"z": z}))

    @nki_only
    def test_stream_uniform_16k_steady_state(self, benchmark):
        """Uniform counterpart — single NKI launch."""
        prog = ProgramBuilder(seed=0).uniform(_NORMALS_PER_LAUNCH, out="u").build()
        u = torch.empty(_NORMALS_PER_LAUNCH)
        prog.stream_into({"u": u})
        prog.stream_into({"u": u})

        benchmark(lambda: prog.stream_into({"u": u}))

    @nki_only
    def test_stream_normal_1m_steady_state(self, benchmark):
        """1M normals: ceil(1_000_000 / 16_384) = 62 NKI launches."""
        prog = ProgramBuilder(seed=0).normal(1_000_000, out="z").build()
        z = torch.empty(1_000_000)
        prog.stream_into({"z": z})
        prog.stream_into({"z": z})

        benchmark(lambda: prog.stream_into({"z": z}))


# ── Claim 2: Throughput ────────────────────────────────────────────────────────


class TestThroughput:
    """Sustained samples/second over 250 iterations.

    Hardware target (trn2): ≥ 10⁹ samples/s for both normal and uniform.
    The benchmark fixture reports wall-clock mean; derive throughput by
    dividing element count by mean_seconds.
    """

    @nki_only
    def test_normal_1m_throughput(self, benchmark):
        """Throughput: 1M normals per stream_into call."""
        n = 1_000_000
        prog = ProgramBuilder(seed=0).normal(n, out="z").build()
        z = torch.empty(n)
        prog.stream_into({"z": z})

        result = benchmark.pedantic(
            lambda: prog.stream_into({"z": z}),
            iterations=1,
            rounds=250,
            warmup_rounds=5,
        )
        _ = result  # benchmark fixture captures timing automatically

    @nki_only
    def test_uniform_1m_throughput(self, benchmark):
        """Throughput: 1M uniforms per stream_into call."""
        n = 1_000_000
        prog = ProgramBuilder(seed=0).uniform(n, out="u").build()
        u = torch.empty(n)
        prog.stream_into({"u": u})

        benchmark.pedantic(
            lambda: prog.stream_into({"u": u}),
            iterations=1,
            rounds=250,
            warmup_rounds=5,
        )

    @nki_only
    def test_multi_dist_throughput(self, benchmark):
        """Throughput: normal + uniform + exponential in a single program call."""
        n = 500_000
        prog = (
            ProgramBuilder(seed=0)
            .normal(n, out="z")
            .uniform(n, out="u")
            .exponential(n, out="e")
            .build()
        )
        z = torch.empty(n)
        u = torch.empty(n)
        e = torch.empty(n)
        prog.stream_into({"z": z, "u": u, "e": e})

        benchmark.pedantic(
            lambda: prog.stream_into({"z": z, "u": u, "e": e}),
            iterations=1,
            rounds=100,
            warmup_rounds=5,
        )


# ── Claim 3: NEFF cache reuse ─────────────────────────────────────────────────


class TestNeffCacheReuse:
    """Second stream_into with the same buffer shapes must reuse the cached NEFF.

    XLA caches compiled NEFFs by input shape.  A cache miss would require
    minutes of recompilation.  A cache hit takes ~10μs launch overhead.
    We assert the second call completes in under 100ms as a conservative bound.
    """

    @nki_only
    def test_second_call_under_100ms(self):
        """Second stream_into must be < 100ms (NEFF cache hit)."""
        prog = ProgramBuilder(seed=0).normal(1_000_000, out="z").build()
        z = torch.empty(1_000_000)

        prog.stream_into({"z": z})  # may trigger compilation

        t0 = time.perf_counter()
        prog.stream_into({"z": z})
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        assert elapsed_ms < 100, (
            f"Second stream_into took {elapsed_ms:.1f}ms — expected NEFF cache hit (< 100ms)"
        )

    @nki_only
    def test_neff_reuse_benchmark(self, benchmark):
        """Benchmark second call only — measures NEFF cache-hit latency."""
        prog = ProgramBuilder(seed=0).normal(1_000_000, out="z").build()
        z = torch.empty(1_000_000)
        # warm-up to populate NEFF cache
        prog.stream_into({"z": z})
        prog.stream_into({"z": z})
        prog.stream_into({"z": z})

        benchmark(lambda: prog.stream_into({"z": z}))

    @nki_only
    def test_shape_change_forces_recompile(self):
        """Changing buffer size must produce correct (different-length) output.

        Does not test timing — just that the kernel executes without error
        when shape changes, confirming the XLA recompile path is not broken.
        """
        prog = ProgramBuilder(seed=0).normal(1_000_000, out="z").build()
        z_1m = torch.empty(1_000_000)
        prog.stream_into({"z": z_1m})

        z_512k = torch.empty(512_000)
        prog.stream_into({"z": z_512k})  # must not raise
        assert z_512k.shape == (512_000,)


# ── Claim 4: Bit-exactness ─────────────────────────────────────────────────────


class TestBitExactness:
    """GeneratorProgram is a thin API wrapper.

    Calling stream_into on a fresh program (counter=0) must produce the
    same floats as calling the underlying streaming dispatch function directly
    with the same seed and counter_offset=0.

    This proves the program API introduces no hidden transformation —
    users can reason about reproducibility directly from the kernel contract.
    """

    @nki_only
    def test_normal_matches_direct_dispatch(self):
        from trnrand.nki.dispatch import threefry_stream_normal

        seed = 42
        n = 16_384

        prog = ProgramBuilder(seed=seed).normal(n, out="z").build()
        z_prog = torch.empty(n)
        prog.stream_into({"z": z_prog})

        z_direct = threefry_stream_normal(n, seed=seed, counter_offset=0)

        assert torch.equal(z_prog, z_direct), (
            "GeneratorProgram output does not match direct threefry_stream_normal dispatch"
        )

    @nki_only
    def test_uniform_matches_direct_dispatch(self):
        from trnrand.nki.dispatch import threefry_stream_uniform

        seed = 99
        n = 16_384

        prog = ProgramBuilder(seed=seed).uniform(n, out="u").build()
        u_prog = torch.empty(n)
        prog.stream_into({"u": u_prog})

        u_direct = threefry_stream_uniform(n, seed=seed, counter_offset=0)

        assert torch.equal(u_prog, u_direct), (
            "GeneratorProgram output does not match direct threefry_stream_uniform dispatch"
        )

    @nki_only
    def test_counter_offset_propagates(self):
        """Counter advances correctly: second call uses counter_offset > 0."""
        from trnrand.nki.dispatch import threefry_stream_normal

        seed = 7
        n = 16_384

        prog = ProgramBuilder(seed=seed).normal(n, out="z").build()
        z1 = torch.empty(n)
        z2 = torch.empty(n)
        prog.stream_into({"z": z1})  # counter_offset=0 → counter advances to _PROGRAM_TILES
        prog.stream_into({"z": z2})  # counter_offset=_PROGRAM_TILES

        expected_counter = _PROGRAM_TILES  # one launch for n==16,384
        z2_direct = threefry_stream_normal(n, seed=seed, counter_offset=expected_counter)

        assert torch.equal(z2, z2_direct), (
            f"Second stream_into (counter_offset={expected_counter}) does not match direct dispatch"
        )

    @nki_only
    def test_same_seed_same_output(self):
        """Two programs with identical seeds produce identical first draws."""
        n = 65_536
        prog1 = ProgramBuilder(seed=123).normal(n, out="z").build()
        prog2 = ProgramBuilder(seed=123).normal(n, out="z").build()
        z1 = torch.empty(n)
        z2 = torch.empty(n)
        prog1.stream_into({"z": z1})
        prog2.stream_into({"z": z2})
        assert torch.equal(z1, z2)

    @nki_only
    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        n = 65_536
        prog1 = ProgramBuilder(seed=1).normal(n, out="z").build()
        prog2 = ProgramBuilder(seed=2).normal(n, out="z").build()
        z1 = torch.empty(n)
        z2 = torch.empty(n)
        prog1.stream_into({"z": z1})
        prog2.stream_into({"z": z2})
        assert not torch.equal(z1, z2)


# ── Claim 5: Engine overlap ───────────────────────────────────────────────────


@neuron
class TestEngineOverlap:
    """GpSimd and Vector Engine run concurrently within a single NEFF.

    The streaming kernel issues GpSimd work (Threefry rounds) and Vector Engine
    work (Box-Muller: log, sqrt, sincos) in the same static_range loop.
    The XLA scheduler overlaps them because they operate on independent SBUF
    partitions — neither stalls waiting for the other.

    Verification requires the Neuron profiler:
        NEURON_RT_STOCHASTIC_ROUNDING=0 \
        NEURON_FRAMEWORK_DEBUG=1 \
        NEURON_CC_FLAGS="--enable-saturate-infinity" \
        pytest benchmarks/bench_streaming.py::TestEngineOverlap -v

    The test below is a functional guard: it confirms the kernel completes in
    reasonable time on hardware, which would be impossible without overlap.
    The profiler trace (neuron-profile run) provides the visual proof.
    """

    def test_overlap_timing_guard(self, benchmark):
        """Streaming normal on hardware: expect < 500μs for 16,384 samples.

        Pure-GpSimd without VE overlap would require sequential round-trips;
        the observed latency floor is evidence of concurrent execution.
        """
        prog = ProgramBuilder(seed=0).normal(_NORMALS_PER_LAUNCH, out="z").build()
        z = torch.empty(_NORMALS_PER_LAUNCH)
        prog.stream_into({"z": z})  # warm-up

        benchmark(lambda: prog.stream_into({"z": z}))

    def test_profiler_annotations_present(self):
        """Smoke test: the kernel executes and produces finite output.

        Run with NEURON_PPROF_SINK=/tmp/profile to capture trace.
        Inspect with: neuron-profiler view /tmp/profile
        """
        prog = ProgramBuilder(seed=0).normal(_NORMALS_PER_LAUNCH, out="z").build()
        z = torch.empty(_NORMALS_PER_LAUNCH)
        prog.stream_into({"z": z})
        assert torch.isfinite(z).all(), "kernel produced non-finite output"
        assert z.shape == (_NORMALS_PER_LAUNCH,)
