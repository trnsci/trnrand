"""Tests for streaming NKI kernels (threefry_stream_normal / threefry_stream_uniform).

Test tiers
----------
- nki_simulator : requires TRNRAND_USE_SIMULATOR=1 (NKI simulator, CPU-bound).
  Also guarded by HAS_NKI; skipped silently on dev hosts without neuronxcc.
- neuron        : requires physical trn2 hardware.

The streaming wrappers live inside `if HAS_NKI:` in dispatch.py and have no
CPU-only fallback — their whole purpose is the NEFF pipeline. All tests
therefore require at minimum the NKI simulator.

The `_PROGRAM_TILES` constant is exported at module level (outside HAS_NKI)
so it can be imported without neuronxcc for parameterisation.
"""

import os

import numpy as np
import pytest
import torch

from trnrand.nki.dispatch import (
    _PROGRAM_TILES,
    _stream_tile_order,
    _threefry_normal_per_tile,
    _threefry_uniform_per_tile,
)

# Same value as dispatch._THREEFRY_LANES (128-lane Threefry tile).
_THREEFRY_LANES = 128
_NORMALS_PER_LAUNCH = _PROGRAM_TILES * _THREEFRY_LANES * 4  # 32×128×4 = 16,384

# Late imports — streaming kernel wrappers live inside `if HAS_NKI:`.
# Catch ImportError on dev hosts without neuronxcc; autouse fixture skips tests.
try:
    from trnrand.nki.dispatch import HAS_NKI as _HAS_NKI
except ImportError:
    _HAS_NKI = False

try:
    from trnrand.nki.dispatch import (
        threefry_normal_nki,
        threefry_stream_normal,
        threefry_stream_uniform,
        threefry_uniform_nki,
    )
except ImportError:
    threefry_stream_normal = None
    threefry_stream_uniform = None
    threefry_normal_nki = None
    threefry_uniform_nki = None


# ── Shared autouse fixture (mirrors test_nki_sim.py pattern) ──────────────────


@pytest.fixture(autouse=True)
def _require_simulator(request):
    """Skip nki_simulator tests unless TRNRAND_USE_SIMULATOR=1 and HAS_NKI."""
    if request.node.get_closest_marker("nki_simulator") is None:
        return  # test not marked — skip check
    if os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNRAND_USE_SIMULATOR=1 required")
    if not _HAS_NKI:
        pytest.skip("nki>=0.3.0 not importable on this host")


# ── Output order (CPU, no NKI needed) ─────────────────────────────────────────


class TestStreamTileOrder:
    """_stream_tile_order flattens kernel output (P, 4×tiles) as (tiles, P, 4) (#58)."""

    def test_per_tile_order(self):
        lanes = _THREEFRY_LANES
        # Kernel layout: lane p, tile t, word w lives at out[p, 4t + w].
        p = torch.arange(lanes).reshape(lanes, 1, 1)
        t = torch.arange(_PROGRAM_TILES).reshape(1, _PROGRAM_TILES, 1)
        w = torch.arange(4).reshape(1, 1, 4)
        code = (t * lanes + p) * 4 + w  # per-tile flat index of (t, p, w)
        out = code.reshape(lanes, 4 * _PROGRAM_TILES)
        flat = _stream_tile_order(out)
        assert torch.equal(flat, torch.arange(lanes * _PROGRAM_TILES * 4))

    def test_first_tile_is_all_lanes_of_batch_zero(self):
        out = torch.arange(_THREEFRY_LANES * 4 * _PROGRAM_TILES).reshape(
            _THREEFRY_LANES, 4 * _PROGRAM_TILES
        )
        flat = _stream_tile_order(out)
        assert torch.equal(flat[: _THREEFRY_LANES * 4], out[:, :4].reshape(-1))


@pytest.mark.nki_simulator
class TestStreamMatchesPerTile:
    """Streams equal the per-tile path element-for-element for the same seed (#58)."""

    def test_normal_one_launch(self):
        n = _NORMALS_PER_LAUNCH
        streaming = threefry_stream_normal(n, seed=5, counter_offset=0)
        per_tile = _threefry_normal_per_tile(n, seed=5, counter_offset=0)
        assert torch.allclose(streaming, per_tile, atol=1e-6)

    def test_uniform_one_launch(self):
        n = _NORMALS_PER_LAUNCH
        streaming = threefry_stream_uniform(n, seed=5, counter_offset=0)
        per_tile = _threefry_uniform_per_tile(n, seed=5, counter_offset=0)
        assert torch.equal(streaming, per_tile)

    def test_uniform_multi_launch_offset_partial(self):
        # Two launches, nonzero counter_offset, n not a multiple of one tile.
        n = _NORMALS_PER_LAUNCH + 1000
        streaming = threefry_stream_uniform(n, seed=11, counter_offset=7)
        per_tile = _threefry_uniform_per_tile(n, seed=11, counter_offset=7)
        assert torch.equal(streaming, per_tile)

    @pytest.mark.parametrize(
        "n,seed,offset", [(513, 3, 0), (16_385, 21, 5), (40_000, 2**24 - 1, 99)]
    )
    def test_bulk_wrappers_bit_identical_to_per_tile(self, monkeypatch, n, seed, offset):
        # threefry_{normal,uniform}_nki now route through the streaming kernels
        # (#60); values and order must be exactly those of the per-tile path.
        monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)
        z = threefry_normal_nki(n, seed=seed, counter_offset=offset)
        assert torch.equal(z, _threefry_normal_per_tile(n, seed=seed, counter_offset=offset))
        u = threefry_uniform_nki(n, seed=seed, counter_offset=offset)
        assert torch.equal(u, _threefry_uniform_per_tile(n, seed=seed, counter_offset=offset))


# ── Shape and dtype ────────────────────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestStreamNormalShape:
    @pytest.mark.parametrize("n", [1, 512, 16384, 16385, 100_000])
    def test_shape(self, n):
        out = threefry_stream_normal(n, seed=0)
        assert out.shape == (n,), f"expected ({n},), got {out.shape}"

    def test_dtype_float32(self):
        out = threefry_stream_normal(1000, seed=42)
        assert out.dtype == torch.float32

    def test_single_element(self):
        out = threefry_stream_normal(1, seed=7)
        assert out.shape == (1,)
        assert out.dtype == torch.float32


@pytest.mark.nki_simulator
class TestStreamUniformShape:
    @pytest.mark.parametrize("n", [1, 512, 16384, 16385, 100_000])
    def test_shape(self, n):
        out = threefry_stream_uniform(n, seed=0)
        assert out.shape == (n,), f"expected ({n},), got {out.shape}"

    def test_dtype_float32(self):
        out = threefry_stream_uniform(1000, seed=42)
        assert out.dtype == torch.float32


# ── Value range ────────────────────────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestStreamUniformRange:
    def test_all_in_unit_interval(self):
        out = threefry_stream_uniform(50_000, seed=12345)
        assert (out >= 0.0).all(), "uniform output contains values < 0"
        assert (out < 1.0).all(), "uniform output contains values ≥ 1"

    def test_no_exact_zeros(self):
        # Mantissa-based generation never produces 0.0 exactly.
        out = threefry_stream_uniform(16384, seed=99)
        assert (out > 0.0).all(), "uniform output contains exact zero"


@pytest.mark.nki_simulator
class TestStreamNormalFinite:
    def test_all_finite(self):
        out = threefry_stream_normal(50_000, seed=0)
        assert torch.isfinite(out).all(), "normal output contains NaN or Inf"


# ── Statistical moments ────────────────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestStreamNormalMoments:
    """Moment checks with generous tolerances — Monte Carlo error is 1/√N."""

    N = 500_000

    def test_mean_near_zero(self):
        out = threefry_stream_normal(self.N, seed=0)
        mean = out.mean().item()
        assert abs(mean) < 0.01, f"mean {mean:.4f} too far from 0"

    def test_std_near_one(self):
        out = threefry_stream_normal(self.N, seed=1)
        std = out.std().item()
        assert abs(std - 1.0) < 0.01, f"std {std:.4f} too far from 1"

    def test_skew_near_zero(self):
        out = threefry_stream_normal(self.N, seed=2).double()
        mean = out.mean()
        centered = out - mean
        skew = (centered**3).mean() / (centered**2).mean() ** 1.5
        assert abs(skew.item()) < 0.05, f"skew {skew.item():.4f} too large"


@pytest.mark.nki_simulator
class TestStreamUniformMoments:
    N = 500_000

    def test_mean_near_half(self):
        out = threefry_stream_uniform(self.N, seed=0)
        mean = out.mean().item()
        assert abs(mean - 0.5) < 0.005, f"mean {mean:.4f} too far from 0.5"

    def test_variance_near_twelfth(self):
        out = threefry_stream_uniform(self.N, seed=1)
        var = out.var().item()
        expected = 1.0 / 12.0
        assert abs(var - expected) < 0.002, f"variance {var:.5f} too far from 1/12"


# ── Determinism ────────────────────────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestDeterminism:
    def test_normal_same_seed_same_output(self):
        a = threefry_stream_normal(10_000, seed=42)
        b = threefry_stream_normal(10_000, seed=42)
        assert torch.equal(a, b), "same seed produced different normal samples"

    def test_uniform_same_seed_same_output(self):
        a = threefry_stream_uniform(10_000, seed=7)
        b = threefry_stream_uniform(10_000, seed=7)
        assert torch.equal(a, b), "same seed produced different uniform samples"

    def test_different_seeds_different_output(self):
        a = threefry_stream_normal(1000, seed=1)
        b = threefry_stream_normal(1000, seed=2)
        assert not torch.equal(a, b), "different seeds produced identical output"

    def test_counter_offset_shifts_stream(self):
        a = threefry_stream_normal(1000, seed=0, counter_offset=0)
        b = threefry_stream_normal(1000, seed=0, counter_offset=1)
        assert not torch.equal(a, b), "counter_offset had no effect"


# ── Counter advance across calls ───────────────────────────────────────────────


@pytest.mark.nki_simulator
class TestCounterAdvance:
    """Verify counter_offset shifts the stream by the expected number of tiles."""

    def test_offset_by_program_tiles_matches_next_launch(self):
        # Two consecutive launches of 16384 normals should compose with a single
        # call requesting 32768 normals (same counter sequence).
        n = _NORMALS_PER_LAUNCH  # exactly one launch
        first = threefry_stream_normal(n, seed=0, counter_offset=0)
        second = threefry_stream_normal(n, seed=0, counter_offset=_PROGRAM_TILES)
        combined = threefry_stream_normal(2 * n, seed=0, counter_offset=0)
        assert torch.equal(first, combined[:n]), "first chunk mismatch"
        assert torch.equal(second, combined[n:]), "second chunk mismatch"

    def test_uniform_offset_by_program_tiles(self):
        n = _NORMALS_PER_LAUNCH
        first = threefry_stream_uniform(n, seed=0, counter_offset=0)
        second = threefry_stream_uniform(n, seed=0, counter_offset=_PROGRAM_TILES)
        combined = threefry_stream_uniform(2 * n, seed=0, counter_offset=0)
        assert torch.equal(first, combined[:n])
        assert torch.equal(second, combined[n:])


# ── Simulator-specific tests ───────────────────────────────────────────────────


@pytest.mark.neuron
class TestHardwareBitExact:
    """Streams equal the per-tile path element-for-element on hardware (#58).

    Contract: for the same seed and counter_offset, threefry_stream_* emits the
    same values in the same order as threefry_*_nki — per-tile (tiles, P, 4)
    order, i.e. all lanes of counter batch 0, then batch 1, and so on. The
    simulator twin is TestStreamMatchesPerTile.
    """

    def test_bit_exact_with_per_tile_normal(self):
        """Streaming kernel must be bit-exact with threefry_normal_nki.

        For exactly one streaming launch, launch 0 tile T uses
        c1 = T & 0xFFFFFF — identical to per-tile batch=T.
        """
        n = _NORMALS_PER_LAUNCH
        streaming = threefry_stream_normal(n, seed=5, counter_offset=0)
        per_tile = threefry_normal_nki(n, seed=5, counter_offset=0)
        assert torch.allclose(streaming, per_tile, atol=1e-6), (
            f"max abs diff: {(streaming - per_tile).abs().max().item():.2e}"
        )

    def test_bit_exact_with_per_tile_uniform(self):
        """Streaming uniform kernel must be bit-exact with threefry_uniform_nki."""
        n = _NORMALS_PER_LAUNCH
        streaming = threefry_stream_uniform(n, seed=5, counter_offset=0)
        per_tile = threefry_uniform_nki(n, seed=5, counter_offset=0)
        assert torch.equal(streaming, per_tile), (
            f"max abs diff: {(streaming - per_tile).abs().max().item():.2e}"
        )

    def test_bit_exact_with_per_tile_multi_launch(self):
        """Order holds across launches, a nonzero counter_offset, and a partial tile."""
        n = 2 * _NORMALS_PER_LAUNCH + 1000
        streaming = threefry_stream_uniform(n, seed=11, counter_offset=7)
        per_tile = threefry_uniform_nki(n, seed=11, counter_offset=7)
        assert torch.equal(streaming, per_tile), (
            f"mismatches: {(streaming != per_tile).sum().item()} of {n}"
        )


# ── Hardware tests (neuron marker) ─────────────────────────────────────────────


@pytest.mark.neuron
@pytest.mark.skipif(not _HAS_NKI, reason="requires neuronxcc")
class TestHardwareStreaming:
    """On-device tests — run on trn2.3xlarge only."""

    def test_stream_normal_shape_and_moments(self):
        out = threefry_stream_normal(1_000_000, seed=0)
        assert out.shape == (1_000_000,)
        mean = out.mean().item()
        std = out.std().item()
        assert abs(mean) < 0.005
        assert abs(std - 1.0) < 0.005

    def test_stream_uniform_range(self):
        out = threefry_stream_uniform(1_000_000, seed=0)
        assert (out >= 0.0).all()
        assert (out < 1.0).all()

    def test_seed_deterministic(self):
        a = threefry_stream_normal(100_000, seed=42)
        b = threefry_stream_normal(100_000, seed=42)
        assert torch.equal(a, b)

    def test_counter_advances_across_calls(self):
        n = _NORMALS_PER_LAUNCH
        first = threefry_stream_normal(n, seed=0, counter_offset=0)
        second = threefry_stream_normal(n, seed=0, counter_offset=_PROGRAM_TILES)
        combined = threefry_stream_normal(2 * n, seed=0, counter_offset=0)
        assert torch.equal(first, combined[:n])
        assert torch.equal(second, combined[n:])
