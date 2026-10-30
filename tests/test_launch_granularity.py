"""
Launch granularity of the bulk Threefry wrappers (#60).

CPU-only: `dispatch._launch` is replaced by a stand-in that counts launches per
op and computes each kernel's output from its counter/key inputs with the CPU
Threefry reference, laid out exactly as the real kernels lay it out
(per-tile: (P, 4); streaming: (P, 4 × _PROGRAM_TILES), tile T in columns
4T..4T+3). That checks the host-side plumbing (counters, launch split, tile
order, truncation) of the streaming route against the per-tile route without
NKI or a device. The kernels' own bit-equality is covered on the simulator
(tests/test_nki_streaming.py::TestStreamMatchesPerTile).
"""

from collections import Counter

import pytest
import torch

import trnrand
from trnrand import distributions
from trnrand.nki import dispatch
from trnrand.nki.dispatch import _PROGRAM_TILES, threefry4x32_reference

LANES = 128
PER_TILE = LANES * 4  # 512 values per per-tile launch
PER_STREAM = _PROGRAM_TILES * PER_TILE  # 16,384 values per streaming launch


def _words(c0, c1, k0, k1):
    """(P, 4) Threefry output words for counters (c0, c1, 0, 0), key (k0, k1, 0, 0)."""
    P = c0.shape[0]
    ctr = torch.zeros(P, 4, dtype=torch.int64)
    ctr[:, 0] = c0.reshape(-1).to(torch.int64)
    ctr[:, 1] = c1.reshape(-1).to(torch.int64)
    key = torch.zeros(P, 4, dtype=torch.int64)
    key[:, 0] = k0.reshape(-1).to(torch.int64)
    key[:, 1] = k1.reshape(-1).to(torch.int64)
    return threefry4x32_reference(ctr, key)


def _uniform(w):
    return (w & 0xFFFFFF).to(torch.float32) / 16777216.0


def _normal(w):
    # Any deterministic, distinct transform will do: this test checks plumbing.
    return _uniform(w) * 2.0 - 1.0 + 10.0


class _FakeLauncher:
    def __init__(self):
        self.calls = Counter()

    def __call__(self, kernel, op_name, *t):
        self.calls[op_name] += 1
        if op_name in ("threefry4x32", "threefry_normal"):
            c0, c1, _c2, _c3, k0, k1, _k2, _k3 = t
            w = _words(c0, c1, k0, k1)
            return _uniform(w) if op_name == "threefry4x32" else _normal(w)
        if op_name in ("threefry_streaming_uniform", "threefry_streaming_normal"):
            c0, start, k0, k1, _k2, _k3 = t
            f = _uniform if op_name == "threefry_streaming_uniform" else _normal
            cols = [f(_words(c0, (start + T) & 0xFFFFFF, k0, k1)) for T in range(_PROGRAM_TILES)]
            return torch.cat(cols, dim=1)  # (P, 4 × tiles), tile T at 4T..4T+3
        raise AssertionError(f"unexpected op {op_name}")


@pytest.fixture
def fake(monkeypatch):
    launcher = _FakeLauncher()
    monkeypatch.setattr(dispatch, "_launch", launcher)
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)
    return launcher


def _ceil(a, b):
    return -(-a // b)


SIZES = [1, 511, 512, 513, 16_383, 16_384, 16_385, 50_000, 100_000]


class TestStreamingMatchesPerTile:
    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("seed,offset", [(0, 0), (42, 0), (12345, 7), (2**24 - 1, 1000)])
    def test_uniform_bit_identical(self, fake, n, seed, offset):
        old = dispatch._threefry_uniform_per_tile(n, seed=seed, counter_offset=offset)
        new = dispatch.threefry_stream_uniform(n, seed=seed, counter_offset=offset)
        assert new.shape == old.shape == (n,)
        assert torch.equal(new, old)

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("seed,offset", [(0, 0), (7, 3), (999, 2**24 - 5)])
    def test_normal_bit_identical(self, fake, n, seed, offset):
        old = dispatch._threefry_normal_per_tile(n, seed=seed, counter_offset=offset)
        new = dispatch.threefry_stream_normal(n, seed=seed, counter_offset=offset)
        assert torch.equal(new, old)


class TestLaunchCount:
    @pytest.mark.parametrize("n", [1, 512, 16_384, 16_385, 262_144, 1_048_576])
    def test_per_tile_baseline(self, fake, n):
        dispatch._threefry_normal_per_tile(n, seed=1)
        assert fake.calls == Counter({"threefry_normal": _ceil(n, PER_TILE)})

    @pytest.mark.parametrize("n", [1, 512, 16_384, 16_385, 262_144, 1_048_576])
    @pytest.mark.parametrize("fn,op", [("normal", "normal"), ("uniform", "uniform")])
    def test_bulk_routes_through_streaming(self, fake, n, fn, op):
        getattr(dispatch, f"threefry_{fn}_nki")(n, seed=1)
        assert fake.calls == Counter({f"threefry_streaming_{op}": _ceil(n, PER_STREAM)})

    def test_4m_normal_is_256_launches_not_8192(self, fake):
        n = 4_194_304
        dispatch.threefry_normal_nki(n, seed=3)
        assert fake.calls["threefry_streaming_normal"] == n // PER_STREAM == 256

    @pytest.mark.parametrize("fn", ["normal", "uniform"])
    def test_public_api_launch_count(self, fake, monkeypatch, fn):
        monkeypatch.setattr(distributions, "_nki_active", lambda: True)
        n = 100_000
        g = trnrand.Generator(seed=11)
        out = getattr(trnrand, fn)(n, generator=g)
        assert out.shape == (n,)
        assert sum(fake.calls.values()) == _ceil(n, PER_STREAM) == 7

    def test_public_api_values_unchanged(self, fake, monkeypatch):
        """trnrand.normal gives the same values through either route."""
        monkeypatch.setattr(distributions, "_nki_active", lambda: True)
        n = 40_000
        new = trnrand.normal(n, generator=trnrand.Generator(seed=5))
        monkeypatch.setenv("TRNRAND_BULK_STREAMING", "0")
        old = trnrand.normal(n, generator=trnrand.Generator(seed=5))
        assert fake.calls["threefry_normal"] == _ceil(n, PER_TILE)
        assert torch.equal(new, old)


class _FakeNative:
    """Stand-in for the native path: `trnrand::` ops plus counted host↔device copies."""

    def __init__(self):
        self.launcher = _FakeLauncher()
        self.h2d = 0
        self.d2h = 0

    def ops(self):
        def make(name):
            return lambda *t: self.launcher(None, name, *t)

        return {n: make(n) for n in ("threefry_streaming_normal", "threefry_streaming_uniform")}

    def to_device(self, t):
        self.h2d += 1
        return t.clone()

    def to_host(self, t):
        self.d2h += 1
        return t.clone()


@pytest.fixture
def fake_native(monkeypatch):
    nat = _FakeNative()
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(dispatch, "_get_native_ops", nat.ops)
    monkeypatch.setattr(dispatch, "_h2d", nat.to_device)
    monkeypatch.setattr(dispatch, "_d2h", nat.to_host)
    monkeypatch.setattr(dispatch, "_launch", _unexpected_launch)
    monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)
    # The native path now defaults to the bulk kernels (tests/test_bulk_kernel.py);
    # this class covers the streaming-kernel fallback behind TRNRAND_BULK_KERNEL=0.
    monkeypatch.setenv("TRNRAND_BULK_KERNEL", "0")
    return nat


def _unexpected_launch(*a, **k):
    raise AssertionError("native streaming must not go through per-launch _launch")


class TestNativeDeviceResident:
    """Native path: inputs copied once, outputs copied back once (#60 follow-up)."""

    @pytest.mark.parametrize("n", [1, 16_384, 16_385, 100_000, 1_048_576])
    @pytest.mark.parametrize("fn", ["normal", "uniform"])
    def test_copy_and_launch_counts(self, fake_native, n, fn):
        getattr(dispatch, f"threefry_{fn}_nki")(n, seed=4, counter_offset=3)
        assert fake_native.launcher.calls == Counter(
            {f"threefry_streaming_{fn}": _ceil(n, PER_STREAM)}
        )
        # c0, 4 key words and the (launches, P, 1) start_batch tensor go over as
        # one packed copy (`_h2d_packed`); one copy back.
        assert fake_native.h2d == 1
        assert fake_native.d2h == 1

    @pytest.mark.parametrize("n", [1, 513, 16_385, 50_000, 100_000])
    @pytest.mark.parametrize("seed,offset", [(0, 0), (12345, 7), (999, 2**24 - 5)])
    def test_bit_identical_to_per_tile(self, fake_native, monkeypatch, n, seed, offset):
        new_z = dispatch.threefry_normal_nki(n, seed=seed, counter_offset=offset)
        new_u = dispatch.threefry_uniform_nki(n, seed=seed, counter_offset=offset)
        # Per-tile reference through the XLA-style `_launch` stand-in.
        monkeypatch.setattr(dispatch, "_launch", _FakeLauncher())
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "xla")
        old_z = dispatch._threefry_normal_per_tile(n, seed=seed, counter_offset=offset)
        old_u = dispatch._threefry_uniform_per_tile(n, seed=seed, counter_offset=offset)
        assert torch.equal(new_z, old_z)
        assert torch.equal(new_u, old_u)


class TestBulkStreamingGate:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)

    def test_env_override(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
        monkeypatch.setenv("TRNRAND_BULK_STREAMING", "0")
        assert not dispatch._bulk_streaming()
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "xla")
        monkeypatch.setattr(dispatch, "_neuronx_cc_version", lambda: (2, 26))
        monkeypatch.setenv("TRNRAND_BULK_STREAMING", "1")
        assert dispatch._bulk_streaming()

    def test_native_path_streams(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
        monkeypatch.setattr(dispatch, "_neuronx_cc_version", lambda: (2, 26))
        assert dispatch._bulk_streaming()

    @pytest.mark.parametrize(
        "ver,expected", [((2, 26), False), ((2, 27), False), ((2, 28), True), ((3, 0), True)]
    )
    def test_xla_path_gated_on_compiler(self, monkeypatch, ver, expected):
        # neuronx-cc 2.26 hits SBUF OOM compiling the streaming kernels (#57).
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "xla")
        monkeypatch.setattr(dispatch, "_neuronx_cc_version", lambda: ver)
        assert dispatch._bulk_streaming() is expected
