"""Tests for ProgramBuilder / GeneratorProgram API.

Test tiers
----------
- No marker   : CPU-only; runs on dev hosts without neuronxcc.
  ProgramBuilder and GeneratorProgram have a torch.Generator CPU fallback,
  so shape/dtype/API tests run everywhere.
- nki_simulator: requires TRNRAND_USE_SIMULATOR=1 + HAS_NKI.
- neuron       : requires physical trn2 hardware.
"""

import os
import time

import pytest
import torch

import trnrand
from trnrand.generator import Generator
from trnrand.nki.program import GeneratorProgram, ProgramBuilder

try:
    from trnrand.nki.dispatch import HAS_NKI as _HAS_NKI
except ImportError:
    _HAS_NKI = False


# ── Autouse fixture (mirrors test_nki_sim.py / test_nki_streaming.py) ─────────


@pytest.fixture(autouse=True)
def _require_simulator(request):
    if request.node.get_closest_marker("nki_simulator") is None:
        return
    if os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNRAND_USE_SIMULATOR=1 required")
    if not _HAS_NKI:
        pytest.skip("nki>=0.3.0 not importable on this host")


# ── Version tests ──────────────────────────────────────────────────────────────


class TestVersion:
    def test_version_not_stale(self):
        assert trnrand.__version__ != "0.2.0", (
            "__version__ is still the stale hardcoded value '0.2.0'"
        )

    def test_version_is_string(self):
        assert isinstance(trnrand.__version__, str)
        assert len(trnrand.__version__) > 0

    def test_version_not_unknown_when_installed(self):
        # If the package is installed (editable or otherwise), version should resolve.
        # "unknown" is the fallback for running from a bare source tree — not a failure,
        # but worth flagging if the package IS installed.
        if trnrand.__version__ == "unknown":
            pytest.skip("package not installed — version fallback is expected")
        # If installed, version should look like a semver string.
        parts = trnrand.__version__.split(".")
        assert len(parts) >= 2, f"unexpected version format: {trnrand.__version__!r}"


# ── Exports ────────────────────────────────────────────────────────────────────


class TestExports:
    def test_generator_program_exported(self):
        assert hasattr(trnrand, "GeneratorProgram")
        assert trnrand.GeneratorProgram is GeneratorProgram

    def test_generator_program_in_all(self):
        assert "GeneratorProgram" in trnrand.__all__

    def test_program_builder_internal(self):
        # ProgramBuilder is internal — not in __all__, not exported at package level.
        assert "ProgramBuilder" not in trnrand.__all__
        assert not hasattr(trnrand, "ProgramBuilder")


# ── ProgramBuilder fluent API (CPU) ───────────────────────────────────────────


class TestProgramBuilderAPI:
    def test_fluent_normal_uniform_build(self):
        prog = ProgramBuilder(seed=0).normal(1000, out="z").uniform(500, out="u").build()
        assert isinstance(prog, GeneratorProgram)

    def test_fluent_exponential(self):
        prog = ProgramBuilder(seed=1).exponential(200, rate=2.0, out="e").build()
        assert isinstance(prog, GeneratorProgram)

    def test_chained_three_distributions(self):
        prog = (
            ProgramBuilder(seed=42)
            .normal(100, out="z")
            .uniform(100, out="u")
            .exponential(100, out="e")
            .build()
        )
        assert isinstance(prog, GeneratorProgram)

    def test_build_empty_raises(self):
        with pytest.raises(ValueError, match="no distribution steps"):
            ProgramBuilder(seed=0).build()

    def test_default_out_names(self):
        # Default out names: normal→"z", uniform→"u", exponential→"e"
        prog = ProgramBuilder().normal(10).uniform(10).exponential(10).build()
        z = torch.empty(10)
        u = torch.empty(10)
        e = torch.empty(10)
        prog.stream_into({"z": z, "u": u, "e": e})  # should not raise

    def test_custom_out_names(self):
        prog = ProgramBuilder().normal(10, out="noise").uniform(10, out="mask").build()
        noise = torch.empty(10)
        mask = torch.empty(10)
        prog.stream_into({"noise": noise, "mask": mask})  # should not raise


# ── Generator.new_program() entry point ───────────────────────────────────────


class TestGeneratorNewProgram:
    def test_new_program_returns_builder(self):
        gen = Generator(seed=7)
        builder = gen.new_program()
        assert isinstance(builder, ProgramBuilder)

    def test_new_program_seed_propagates(self):
        gen = Generator(seed=99)
        builder = gen.new_program()
        assert builder._seed == 99

    def test_unseeded_generator_uses_zero(self):
        gen = Generator()  # seed=None
        builder = gen.new_program()
        assert builder._seed == 0

    def test_full_pipeline(self):
        gen = Generator(seed=42)
        prog = gen.new_program().normal(500, out="z").build()
        z = torch.empty(500)
        prog.stream_into({"z": z})
        assert z.shape == (500,)
        assert z.dtype == torch.float32


# ── stream_into shapes and dtypes (CPU PyTorch fallback) ──────────────────────


class TestStreamIntoShapes:
    @pytest.mark.parametrize("n", [1, 100, 1000, 16384, 16385])
    def test_normal_shape_preserved(self, n):
        prog = ProgramBuilder(seed=0).normal(n, out="z").build()
        z = torch.empty(n)
        prog.stream_into({"z": z})
        assert z.shape == (n,)

    def test_2d_shape_preserved(self):
        prog = ProgramBuilder(seed=0).normal(200, out="z").build()
        z = torch.empty(10, 20)
        prog.stream_into({"z": z})
        assert z.shape == (10, 20)

    def test_uniform_shape_preserved(self):
        prog = ProgramBuilder(seed=0).uniform(300, out="u").build()
        u = torch.empty(300)
        prog.stream_into({"u": u})
        assert u.shape == (300,)

    def test_exponential_shape_preserved(self):
        prog = ProgramBuilder(seed=0).exponential(150, out="e").build()
        e = torch.empty(150)
        prog.stream_into({"e": e})
        assert e.shape == (150,)

    def test_dtype_float32(self):
        prog = ProgramBuilder(seed=0).normal(100, out="z").build()
        z = torch.empty(100)
        prog.stream_into({"z": z})
        assert z.dtype == torch.float32

    def test_multi_distribution_shapes(self):
        prog = ProgramBuilder(seed=0).normal(200, out="z").uniform(100, out="u").build()
        z = torch.empty(200)
        u = torch.empty(100)
        prog.stream_into({"z": z, "u": u})
        assert z.shape == (200,)
        assert u.shape == (100,)


# ── Value properties (CPU fallback) ───────────────────────────────────────────


class TestStreamIntoValues:
    def test_uniform_in_range(self):
        prog = ProgramBuilder(seed=0).uniform(10_000, out="u").build()
        u = torch.empty(10_000)
        prog.stream_into({"u": u})
        assert (u >= 0.0).all()
        assert (u < 1.0).all()

    def test_uniform_custom_range(self):
        prog = ProgramBuilder(seed=0).uniform(10_000, low=2.0, high=5.0, out="u").build()
        u = torch.empty(10_000)
        prog.stream_into({"u": u})
        assert (u >= 2.0).all()
        assert (u < 5.0).all()

    def test_exponential_positive(self):
        prog = ProgramBuilder(seed=0).exponential(5_000, rate=1.0, out="e").build()
        e = torch.empty(5_000)
        prog.stream_into({"e": e})
        assert (e > 0.0).all()

    def test_normal_finite(self):
        prog = ProgramBuilder(seed=0).normal(5_000, out="z").build()
        z = torch.empty(5_000)
        prog.stream_into({"z": z})
        assert torch.isfinite(z).all()


# ── Determinism (CPU fallback) ─────────────────────────────────────────────────


class TestDeterminism:
    def test_same_seed_same_output(self):
        z1 = torch.empty(500)
        z2 = torch.empty(500)
        ProgramBuilder(seed=7).normal(500, out="z").build().stream_into({"z": z1})
        ProgramBuilder(seed=7).normal(500, out="z").build().stream_into({"z": z2})
        assert torch.equal(z1, z2), "same seed produced different output"

    def test_different_seeds_differ(self):
        z1 = torch.empty(500)
        z2 = torch.empty(500)
        ProgramBuilder(seed=1).normal(500, out="z").build().stream_into({"z": z1})
        ProgramBuilder(seed=2).normal(500, out="z").build().stream_into({"z": z2})
        assert not torch.equal(z1, z2)

    def test_consecutive_calls_differ(self):
        prog = ProgramBuilder(seed=0).normal(500, out="z").build()
        z1 = torch.empty(500)
        z2 = torch.empty(500)
        prog.stream_into({"z": z1})
        prog.stream_into({"z": z2})
        assert not torch.equal(z1, z2), "consecutive stream_into calls produced identical output"


# ── NKI simulator tests ────────────────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestSimulatorStreamInto:
    def test_normal_moments(self):
        prog = ProgramBuilder(seed=0).normal(500_000, out="z").build()
        z = torch.empty(500_000)
        prog.stream_into({"z": z})
        assert abs(z.mean().item()) < 0.01, f"mean {z.mean().item():.4f}"
        assert abs(z.std().item() - 1.0) < 0.01, f"std {z.std().item():.4f}"

    def test_normal_scaled(self):
        prog = ProgramBuilder(seed=0).normal(200_000, mean=5.0, std=2.0, out="z").build()
        z = torch.empty(200_000)
        prog.stream_into({"z": z})
        assert abs(z.mean().item() - 5.0) < 0.02
        assert abs(z.std().item() - 2.0) < 0.02

    def test_uniform_range(self):
        prog = ProgramBuilder(seed=0).uniform(100_000, out="u").build()
        u = torch.empty(100_000)
        prog.stream_into({"u": u})
        assert (u >= 0.0).all()
        assert (u < 1.0).all()

    def test_exponential_range(self):
        prog = ProgramBuilder(seed=0).exponential(100_000, rate=1.0, out="e").build()
        e = torch.empty(100_000)
        prog.stream_into({"e": e})
        assert (e > 0.0).all()
        assert torch.isfinite(e).all()

    def test_counter_advances_independently(self):
        prog = ProgramBuilder(seed=42).normal(16_384, out="z").build()
        z1 = torch.empty(16_384)
        z2 = torch.empty(16_384)
        prog.stream_into({"z": z1})
        prog.stream_into({"z": z2})
        assert not torch.equal(z1, z2), "consecutive calls produced identical output"

    def test_neff_cache_reuse_timing(self):
        """Second stream_into with same shapes should be faster than first."""
        if os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() in ("1", "true", "yes"):
            pytest.skip(
                "NEFF cache timing is not meaningful under the NKI simulator "
                "(simulator compiles fresh every call; no NEFF cache)"
            )
        prog = ProgramBuilder(seed=0).normal(16_384, out="z").build()
        z = torch.empty(16_384)

        t0 = time.perf_counter()
        prog.stream_into({"z": z})  # may trigger NEFF compilation
        _first_ms = (time.perf_counter() - t0) * 1000  # noqa: F841

        t0 = time.perf_counter()
        prog.stream_into({"z": z})  # should reuse cached NEFF
        second_ms = (time.perf_counter() - t0) * 1000

        # Second call must be < 500ms regardless of first (NEFF cache hit).
        assert second_ms < 500, f"second stream_into took {second_ms:.1f}ms — NEFF cache miss?"

    def test_multi_distribution_nki(self):
        prog = ProgramBuilder(seed=0).normal(50_000, out="z").uniform(50_000, out="u").build()
        z = torch.empty(50_000)
        u = torch.empty(50_000)
        prog.stream_into({"z": z, "u": u})
        # Normal moments
        assert abs(z.mean().item()) < 0.02
        assert abs(z.std().item() - 1.0) < 0.02
        # Uniform range
        assert (u >= 0.0).all()
        assert (u < 1.0).all()
        # Buffers independent (different counters used for each distribution)
        assert not torch.equal(z[:50_000].float(), u)
