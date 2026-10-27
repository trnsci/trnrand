"""
Bulk Threefry kernels: one launch per request, values bit-identical to caa9aec (#60).

CPU tier (no NKI): the native path is driven with stand-in `trnrand::threefry_bulk_*`
ops that compute each launch's output from its (P, F) lane / (P, C) start / key
inputs with an independent numpy Threefry4x32-20 + Box-Muller reference, laid
out exactly as the kernel lays it out (chunk c, word w, column j at
c·4F + w·F + j). The wrapper output is compared with the same reference math
evaluated directly in per-tile counter order (batch b = counter_offset + i,
lane p, word w), which is the mapping the per-tile and streaming kernels use.

Simulator tier (`TRNRAND_USE_SIMULATOR=1`, needs `nki`): the real bulk kernels
against the caa9aec streaming kernels, element for element.
"""

import os
from collections import Counter

import numpy as np
import pytest
import torch

import trnrand
from trnrand import distributions
from trnrand.nki import dispatch

LANES = 128
M32 = np.uint64(0xFFFFFFFF)
ROT = [(10, 26), (11, 21), (13, 27), (23, 5)]
PARITY = 0x1BD11BDA


def _threefry_np(c0, c1, k0, k1):
    """numpy Threefry4x32-20 for counters (c0, c1, 0, 0), key (k0, k1, 0, 0)."""
    c0 = np.asarray(c0, dtype=np.uint64)
    c1 = np.asarray(c1, dtype=np.uint64)
    shape = np.broadcast(c0, c1).shape
    x = [
        np.broadcast_to(c0, shape).copy(),
        np.broadcast_to(c1, shape).copy(),
        np.zeros(shape, np.uint64),
        np.zeros(shape, np.uint64),
    ]
    ks = [np.uint64(k0), np.uint64(k1), np.uint64(0), np.uint64(0)]
    ks.append(np.uint64(k0 ^ k1 ^ PARITY))

    def inject(s):
        for i in range(4):
            x[i] = (x[i] + ks[(s + i) % 5] + (np.uint64(s) if i == 3 else np.uint64(0))) & M32

    def rotl(v, r):
        return ((v << np.uint64(r)) | (v >> np.uint64(32 - r))) & M32

    inject(0)
    for r in range(20):
        a, b = ROT[r % 4]
        pairs = [(0, 1, a), (2, 3, b)] if r % 2 == 0 else [(0, 3, a), (2, 1, b)]
        for i, j, rot in pairs:
            x[i] = (x[i] + x[j]) & M32
            x[j] = rotl(x[j], rot) ^ x[i]
        if (r + 1) % 4 == 0:
            inject((r + 1) // 4)
    return x


def _u24(w):
    return ((w & np.uint64(0xFFFFFF)).astype(np.float32) * np.float32(1.0 / 16777216.0)).astype(
        np.float32
    )


def _box_muller(u0, u1):
    u0 = np.maximum(u0, np.float32(1e-7))
    r = np.sqrt(np.log(u0) * np.float32(-2.0))
    t = u1 * np.float32(dispatch.TWO_PI)
    return r * np.cos(t), r * np.sin(t)


def _words_to_values(x, normal):
    """Four word arrays -> four float32 value arrays (uniform or Box-Muller pairs)."""
    u = [_u24(w) for w in x]
    if not normal:
        return u
    z0, z1 = _box_muller(u[0], u[1])
    u2 = np.maximum(u[2], np.float32(1e-7))
    z2, z3 = _box_muller(u2, u[3])
    return [z0, z1, z2, z3]


def _reference(n, seed, offset, normal):
    """Reference stream in per-tile order: batch b, lane p, word w."""
    n_batches = -(-n // (LANES * 4))
    b = (offset + np.arange(n_batches, dtype=np.uint64)) & np.uint64(0xFFFFFF)
    p = np.arange(LANES, dtype=np.uint64)
    x = _threefry_np(p[None, :], b[:, None], seed & 0xFFFFFF, (seed >> 24) & 0xFFFFFF)
    vals = np.stack(_words_to_values(x, normal), axis=-1)  # (batches, P, 4)
    return torch.from_numpy(vals.reshape(-1)[:n].copy())


def _fake_bulk(normal):
    def op(c0, start, k0, k1, _k2, _k3):
        c0 = c0.numpy().astype(np.uint64)  # (P, F)
        F = c0.shape[1]
        cols = []
        for s in start.numpy()[0].astype(np.uint64):  # one start per chunk
            c1 = (s + np.arange(F, dtype=np.uint64)[None, :]) & np.uint64(0xFFFFFF)
            x = _threefry_np(c0, c1, int(k0[0, 0]), int(k1[0, 0]))
            cols.extend(_words_to_values(x, normal))  # word w at columns [wF, (w+1)F)
        return torch.from_numpy(np.concatenate(cols, axis=1))

    return op


def _nki_seeds(seed, k):
    """The first k 24-bit kernel seeds `distributions._nki_seed` draws for Generator(seed)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return [int(torch.randint(0, 2**24, (1,), generator=g).item()) for _ in range(k)]


class _FakeNative:
    def __init__(self):
        self.calls = Counter()
        self.h2d = 0
        self.d2h = 0

    def ops(self):
        def counted(name, fn):
            def run(*t):
                self.calls[name] += 1
                return fn(*t)

            return run

        return {
            "threefry_bulk_normal": counted("threefry_bulk_normal", _fake_bulk(True)),
            "threefry_bulk_uniform": counted("threefry_bulk_uniform", _fake_bulk(False)),
        }

    def to_device(self, t):
        self.h2d += 1
        return t.clone()

    def to_host(self, t):
        self.d2h += 1
        return t.clone()


@pytest.fixture
def native(monkeypatch):
    nat = _FakeNative()
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(dispatch, "_get_native_ops", nat.ops)
    monkeypatch.setattr(dispatch, "_h2d", nat.to_device)
    monkeypatch.setattr(dispatch, "_d2h", nat.to_host)
    monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)
    monkeypatch.delenv("TRNRAND_BULK_KERNEL", raising=False)
    return nat


SIZES = [1, 511, 513, 16_384, 16_385, 100_003, 262_145]


class TestBulkPlan:
    @pytest.mark.parametrize(
        "n,plan",
        [
            (1, (32, 1, 1)),
            (16_384, (32, 1, 1)),
            (16_385, (64, 1, 1)),
            (262_144, (512, 1, 1)),
            (262_145, (512, 2, 1)),
            (1_048_576, (512, 4, 1)),
            (4_194_304, (512, 16, 1)),
            (4_194_305, (512, 16, 2)),
        ],
    )
    def test_plan(self, n, plan):
        assert dispatch._bulk_plan(n) == plan

    @pytest.mark.parametrize("n", [1, 777, 16_385, 300_000, 5_000_000])
    def test_plan_covers_request(self, n):
        F, C, L = dispatch._bulk_plan(n)
        assert n <= F * C * L * LANES * 4

    def test_tile_order(self):
        F, C = 3, 2
        out = torch.arange(LANES * C * 4 * F).reshape(LANES, C * 4 * F)
        flat = dispatch._bulk_tile_order(out, F)
        # Element (batch c·F + j, lane p, word w) comes from out[p, c·4F + w·F + j].
        for c in range(C):
            for j in range(F):
                for p in (0, 5, 127):
                    for w in range(4):
                        idx = ((c * F + j) * LANES + p) * 4 + w
                        assert flat[idx] == out[p, c * 4 * F + w * F + j]


class TestBulkMatchesReference:
    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("seed,offset", [(0, 0), (12345, 7), (2**40 + 3, 2**24 - 5)])
    @pytest.mark.parametrize("normal", [True, False])
    def test_values(self, native, n, seed, offset, normal):
        fn = dispatch.threefry_normal_nki if normal else dispatch.threefry_uniform_nki
        got = fn(n, seed=seed, counter_offset=offset)
        assert torch.equal(got, _reference(n, seed, offset, normal))

    @pytest.mark.parametrize("n", [100_003, 262_145])
    def test_multi_launch_values(self, native, monkeypatch, n):
        # Force several launches of several chunks each.
        monkeypatch.setattr(dispatch, "_BULK_COLS_MAX", 32)
        monkeypatch.setattr(dispatch, "_BULK_CHUNKS_MAX", 4)
        F, C, L = dispatch._bulk_plan(n)
        assert L > 1 and C == 4
        got = dispatch.threefry_normal_nki(n, seed=99, counter_offset=2**24 - 40)
        assert torch.equal(got, _reference(n, 99, 2**24 - 40, True))
        assert native.calls["threefry_bulk_normal"] == L
        assert native.h2d == 6 and native.d2h == 1

    @pytest.mark.parametrize("n", [1, 16_384, 1_048_576, 4_194_304])
    def test_one_launch_up_to_4m(self, monkeypatch, n):
        nat = _FakeNative()
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
        monkeypatch.setattr(dispatch, "_h2d", nat.to_device)
        monkeypatch.setattr(dispatch, "_d2h", nat.to_host)
        F, C, _ = dispatch._bulk_plan(n)
        shapes = []

        def op(c0, start, *k):
            shapes.append((tuple(c0.shape), tuple(start.shape)))
            return torch.zeros(LANES, start.shape[1] * 4 * c0.shape[1])

        monkeypatch.setattr(dispatch, "_get_native_ops", lambda: {"threefry_bulk_normal": op})
        assert dispatch.threefry_normal_nki(n, seed=1).shape == (n,)
        assert shapes == [((LANES, F), (LANES, C))]

    @pytest.mark.parametrize("fn", ["normal", "uniform"])
    @pytest.mark.parametrize("parts", [2, 4])
    def test_partitioned_generators(self, native, monkeypatch, fn, parts):
        monkeypatch.setattr(distributions, "_nki_active", lambda: True)
        n = 512 * parts * 37  # multiple of 512 × P: ranks concatenate to one stream
        (k,) = _nki_seeds(31, 1)
        full = getattr(trnrand, fn)(n, generator=trnrand.Generator(seed=31))
        assert torch.equal(full, _reference(n, k, 0, fn == "normal"))
        chunks = []
        for r in range(parts):
            g = trnrand.Generator(seed=31, partition_rank=r, partition_size=parts)
            off = g._chip_counter_offset(n // parts)
            out = getattr(trnrand, fn)(n // parts, generator=g)
            assert torch.equal(out, _reference(n // parts, k, off, fn == "normal"))
            chunks.append(out)
        assert torch.equal(torch.cat(chunks), full)

    def test_generator_advances_like_before(self, native, monkeypatch):
        monkeypatch.setattr(distributions, "_nki_active", lambda: True)
        g = trnrand.Generator(seed=8)
        a = trnrand.normal(1000, generator=g)
        b = trnrand.normal(70_000, generator=g)
        ka, kb = _nki_seeds(8, 2)
        assert torch.equal(a, _reference(1000, ka, 0, True))
        assert torch.equal(b, _reference(70_000, kb, 2, True))  # 1000 values = 2 batches

    def test_opt_out_uses_streaming_kernels(self, native, monkeypatch):
        monkeypatch.setenv("TRNRAND_BULK_KERNEL", "0")
        assert not dispatch._use_bulk()


# ── Simulator: real bulk kernels vs the caa9aec streaming kernels ─────────────


def _sim():
    return dispatch.HAS_NKI and os.environ.get("TRNRAND_USE_SIMULATOR", "").lower() in (
        "1",
        "true",
        "yes",
    )


@pytest.mark.nki_simulator
@pytest.mark.skipif(not _sim(), reason="TRNRAND_USE_SIMULATOR=1 and nki required")
class TestBulkKernelSimulator:
    def _old(self, normal, n, seed, offset):
        name = "normal" if normal else "uniform"
        return dispatch._stream_launches(
            f"threefry_streaming_{name}_kernel", f"threefry_streaming_{name}", n, seed, offset
        )

    @pytest.mark.parametrize("normal", [True, False])
    @pytest.mark.parametrize("n,seed,offset", [(1, 0, 0), (513, 7, 3), (16_385, 12345, 2**24 - 5)])
    def test_matches_streaming_kernel(self, normal, n, seed, offset):
        new = dispatch._bulk_launches(normal, n, seed, offset)
        assert torch.equal(new, self._old(normal, n, seed, offset))
        if not normal:
            assert torch.equal(new, _reference(n, seed, offset, False))

    def test_multi_chunk_multi_launch(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_BULK_COLS_MAX", 32)
        monkeypatch.setattr(dispatch, "_BULK_CHUNKS_MAX", 2)
        n = 3 * 32 * 512 + 100  # 4 chunks of 32 columns -> 2 launches of C = 2
        assert dispatch._bulk_plan(n) == (32, 2, 2)
        for normal in (True, False):
            new = dispatch._bulk_launches(normal, n, 77, 2**24 - 50)
            assert torch.equal(new, self._old(normal, n, 77, 2**24 - 50))
