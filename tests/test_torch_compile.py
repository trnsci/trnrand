"""
torch.compile coverage for normal() / uniform() (#61).

Dynamo cannot proxy a Generator argument, so the generator-state step runs in
opaque `trnrand::gen_*` custom ops (trnrand/_compile.py). These CPU tests check
that `fullgraph=True` compiles with zero graph breaks and that compiled output
is bit-identical to eager for fixed seeds, on:

- the PyTorch backend (`backend="eager"` stands in for `backend="neuron"`);
- the native TorchNeuron NKI path, with the `trnrand::threefry_bulk*_*`
  kernels (#60; every limb variant and multi-launch plans), the streaming
  kernels (`TRNRAND_BULK_KERNEL=0`) and host/device copies replaced by CPU
  stand-ins, so the traced tensor-state setup is compared against the eager
  numpy setup;
- the non-native NKI path (XLA / per-tile), where the whole draw is one op.

On-hardware confirmation (`backend="neuron"` on Trn2) is the `neuron`-marked
test at the bottom.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest
import torch

import trnrand
from trnrand import _compile, distributions
from trnrand.nki import dispatch

pytestmark = pytest.mark.skipif(not _compile.AVAILABLE, reason="needs torch.library.custom_op")

SEEDS = [0, 1, 42, 2**31 - 1]
FNS = {
    "normal": lambda gen, *s: trnrand.normal(*s, generator=gen),
    "uniform": lambda gen, *s: trnrand.uniform(*s, generator=gen),
    "normal_ms": lambda gen, *s: trnrand.normal(*s, mean=2.0, std=0.5, generator=gen),
    "uniform_lh": lambda gen, *s: trnrand.uniform(*s, low=-3.0, high=5.0, generator=gen),
    "standard_normal": lambda gen, *s: trnrand.standard_normal(*s, generator=gen),
}


@pytest.fixture(autouse=True)
def _reset():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    trnrand.set_backend("auto")


def _compiled(fn):
    return torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)


def _breaks(fn):
    torch._dynamo.reset()
    e = torch._dynamo.explain(fn)()
    torch._dynamo.reset()
    return e.graph_count, e.graph_break_count, [str(b.reason) for b in e.break_reasons]


def _assert_compiles_like_eager(make_gen, fn, calls=3):
    """Eager vs compiled on identically seeded generators, `calls` draws in a row."""
    g_e, g_c = make_gen(), make_gen()
    cf = _compiled(lambda: fn(g_c))
    for _ in range(calls):
        ref, got = fn(g_e), cf()
        assert torch.equal(got, ref)
    assert g_c.position() == g_e.position()
    # Back in eager, the compiled generator continues the same stream.
    assert torch.equal(fn(g_c), fn(g_e))


class TestPytorchBackend:
    @pytest.fixture(autouse=True)
    def _pytorch(self):
        trnrand.set_backend("pytorch")

    @pytest.mark.parametrize("name", list(FNS))
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("size", [(64,), (3, 5)])
    def test_matches_eager(self, name, seed, size):
        fn = FNS[name]
        _assert_compiles_like_eager(lambda: trnrand.Generator(seed=seed), lambda g: fn(g, *size))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_generator(self, seed):
        # Mirrors trnsci#35 B6: reseed the default generator, compile a no-arg lambda.
        trnrand.manual_seed(seed)
        ref = (trnrand.normal(256), trnrand.uniform(256))
        trnrand.manual_seed(seed)
        got = _compiled(lambda: (trnrand.normal(256), trnrand.uniform(256)))()
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])

    def test_tuple_size_and_dtype(self):
        def fn(g):
            return trnrand.normal((4, 6), dtype=torch.float64, generator=g)

        _assert_compiles_like_eager(lambda: trnrand.Generator(seed=7), fn)

    def test_several_draws_in_one_graph(self):
        def fn(g):
            a = trnrand.normal(32, generator=g)
            b = trnrand.uniform(32, generator=g)
            c = trnrand.normal(32, generator=g)  # same args as `a`: must not be CSE'd
            return torch.stack([a, b, c])

        _assert_compiles_like_eager(lambda: trnrand.Generator(seed=3), fn)

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    def test_zero_graph_breaks(self, name):
        trnrand.manual_seed(0)
        graphs, breaks, reasons = _breaks(lambda: FNS[name](None, 1024))
        assert (graphs, breaks) == (1, 0), reasons


def _unused_draw(g):
    trnrand.normal(32, generator=g)  # discarded, but must still advance the stream
    return trnrand.uniform(8, generator=g)


@pytest.mark.parametrize("be", ["aot_eager", "inductor"])
def test_unused_draw_still_advances(be):
    """AOT/Inductor must not DCE or reorder a draw (the ops mutate a token)."""
    trnrand.set_backend("pytorch")
    g_e, g_c = trnrand.Generator(seed=5), trnrand.Generator(seed=5)
    cf = torch.compile(lambda: _unused_draw(g_c), backend=be, fullgraph=True)
    for _ in range(2):
        assert torch.equal(cf(), _unused_draw(g_e))


# ── NKI paths with CPU stand-ins ─────────────────────────────────────────────

LANES = dispatch._THREEFRY_LANES
TILES = dispatch._PROGRAM_TILES


def _fake_stream(c0, start, k0, k1, k2, k3, bias):
    # (P, 4 × tiles); any injective-enough function of every input catches misplumbing.
    t = torch.arange(TILES, dtype=torch.int64).repeat_interleave(4).reshape(1, -1)
    j = torch.arange(4, dtype=torch.int64).repeat(TILES).reshape(1, -1)
    v = (
        c0.to(torch.int64) * 1009
        + ((start.to(torch.int64) + t) & 0xFFFFFF) * 7919
        + k0.to(torch.int64) * 104729
        + k1.to(torch.int64) * 3
        + (k2.to(torch.int64) + k3.to(torch.int64)) * 5
        + j
    ) % 1_000_003
    return v.to(torch.float32) / 1_000_003.0 + bias


@torch.library.custom_op("trnrand_test::stream_uniform", mutates_args=())
def _stream_uniform(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _fake_stream(c0, start, k0, k1, k2, k3, 0.0)


@torch.library.custom_op("trnrand_test::stream_normal", mutates_args=())
def _stream_normal(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _fake_stream(c0, start, k0, k1, k2, k3, -10.0)


for _op in (_stream_uniform, _stream_normal):
    _op.register_fake(lambda c0, *a: c0.new_empty((c0.shape[0], 4 * TILES), dtype=torch.float32))


def _fake_bulk(c0, start, k0, k1, k2, k3, bias):
    # (P, C × 4 × F): chunk c, word w, column j at c·4F + w·F + j, like the kernels.
    P, F = c0.shape
    C = start.shape[1]
    j = torch.arange(F, dtype=torch.int64).reshape(1, 1, 1, F)
    w = torch.arange(4, dtype=torch.int64).reshape(1, 1, 4, 1)
    v = (
        c0.to(torch.int64).reshape(P, 1, 1, F) * 1009
        + ((start.to(torch.int64).reshape(P, C, 1, 1) + j) & 0xFFFFFF) * 7919
        + k0.to(torch.int64).reshape(P, 1, 1, 1) * 104729
        + k1.to(torch.int64).reshape(P, 1, 1, 1) * 3
        + (k2.to(torch.int64) + k3.to(torch.int64)).reshape(P, 1, 1, 1) * 5
        + w
    ) % 1_000_003
    return (v.to(torch.float32) / 1_000_003.0 + bias).reshape(P, C * 4 * F)


def _make_fake_bulk(name, bias):
    def impl(
        c0: torch.Tensor,
        start: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        k3: torch.Tensor,
    ) -> torch.Tensor:
        return _fake_bulk(c0, start, k0, k1, k2, k3, bias)

    op = torch.library.custom_op(f"trnrand_test::{name}", impl, mutates_args=())
    op.register_fake(
        lambda c0, start, *a: c0.new_empty(
            (c0.shape[0], start.shape[1] * 4 * c0.shape[1]), dtype=torch.float32
        )
    )
    return op


# Distinct biases per limb variant: a wrong variant in the graph shows up as a mismatch.
_BULK_VARIANTS = {"bulk": 8, "bulk16": 16, "bulk32": 32}
_FAKE_OPS = {
    "threefry_streaming_uniform": _stream_uniform,
    "threefry_streaming_normal": _stream_normal,
}
for _variant, _bits in _BULK_VARIANTS.items():
    for _kind, _bias in (("uniform", 0.0), ("normal", -10.0)):
        _FAKE_OPS[f"threefry_{_variant}_{_kind}"] = _make_fake_bulk(
            f"{_variant}_{_kind}", _bias + _bits * 100.0
        )


def _partitioned(seed, rank=0, size=1, advance=0, key_mode="draw"):
    def make():
        g = trnrand.Generator(
            seed=seed, partition_rank=rank, partition_size=size, key_mode=key_mode
        )
        g.advance(advance)
        return g

    return make


GENS = [
    _partitioned(0),
    _partitioned(42),
    _partitioned(12345, rank=2, size=4),
    _partitioned(7, advance=(2**24 - 3) * 512),  # start_batch wraps at 2^24
]
COUNTER_GENS = [
    _partitioned(0, key_mode="counter"),
    _partitioned(42, key_mode="counter"),
    _partitioned(2**40 + 12345, rank=2, size=4, key_mode="counter"),  # k1 != 0
    _partitioned(7, advance=(2**24 - 3) * 512, key_mode="counter"),
]


@pytest.fixture
def nki_native(monkeypatch):
    monkeypatch.setattr(distributions, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "native")
    monkeypatch.setattr(dispatch, "_get_native_ops", lambda: _FAKE_OPS)
    monkeypatch.setattr(dispatch, "_h2d", lambda t: t.clone())
    monkeypatch.setattr(dispatch, "_d2h", lambda t: t.clone())
    monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)
    monkeypatch.delenv("TRNRAND_BULK_KERNEL", raising=False)
    monkeypatch.delenv("TRNRAND_LIMB_BITS", raising=False)


# Native-path kernel routes: (env, op-name fragment). Bulk limb variants (#60) and
# the streaming opt-out. Values differ per route (distinct stand-ins).
ROUTES = {
    "bulk32": ({}, "bulk32"),
    "bulk32-explicit": ({"TRNRAND_LIMB_BITS": "32"}, "bulk32"),
    "bulk16": ({"TRNRAND_LIMB_BITS": "16"}, "bulk16"),
    "bulk8": ({"TRNRAND_LIMB_BITS": "8"}, "bulk_"),
    "streaming": ({"TRNRAND_BULK_KERNEL": "0"}, "stream_"),
}


@pytest.fixture
def small_plan(monkeypatch):
    """Tiny bulk caps so modest requests need several chunks and launches."""
    monkeypatch.setattr(dispatch, "_BULK_COLS_MAX", 32)
    monkeypatch.setattr(dispatch, "_BULK_CHUNKS_MAX", 4)


def _launch_targets(fn):
    seen = []

    def backend(gm, example_inputs):
        seen.extend(
            str(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method")
        )
        return gm

    torch.compile(fn, backend=backend, fullgraph=True)()
    return seen


class TestNativeNkiPath:
    @pytest.mark.parametrize("seed", [0, 5, 2**24 - 1, 2**24, 2**31 + 17])
    @pytest.mark.parametrize("offset", [0, 7, 2**24 - 5, 2**40 + 3])
    @pytest.mark.parametrize("n_launches", [1, 2, 7])
    def test_inputs_match_numpy_setup(self, seed, offset, n_launches):
        c0, sb, k = dispatch._stream_inputs_from_state(
            torch.tensor([seed, offset], dtype=torch.int64), n_launches
        )
        k_np = [(seed & 0xFFFFFF), ((seed >> 24) & 0xFFFFFF), 0, 0]
        starts = (offset + np.arange(n_launches, dtype=np.int64) * TILES) & 0xFFFFFF
        assert torch.equal(c0, torch.arange(LANES, dtype=torch.int32).reshape(-1, 1))
        for kw, v in zip(k, k_np, strict=True):
            assert kw.shape == (LANES, 1) and kw.dtype == torch.int32
            assert torch.equal(kw, torch.full((LANES, 1), v, dtype=torch.int32))
        want = np.broadcast_to(starts.astype(np.int32).reshape(-1, 1, 1), (n_launches, LANES, 1))
        assert sb.dtype == torch.int32 and torch.equal(sb, torch.from_numpy(want.copy()))

    @pytest.mark.parametrize("seed", [0, 5, 2**24 - 1, 2**24, 2**31 + 17])
    @pytest.mark.parametrize("offset", [0, 7, 2**24 - 5, 2**40 + 3])
    @pytest.mark.parametrize("n", [1, 513, 300_000, 5_000_000, 40_000_000])
    def test_bulk_inputs_match_numpy_setup(self, seed, offset, n):
        """`_bulk_inputs` equals the numpy setup in `_bulk_launches` for the request's plan."""
        F, C, n_launches = dispatch._bulk_plan(n)
        c0, sb, k = dispatch._bulk_inputs(
            *dispatch._key_offset_from_state(torch.tensor([seed, offset], dtype=torch.int64)),
            F,
            C,
            n_launches,
        )
        want_c0 = np.broadcast_to(np.arange(LANES, dtype=np.int32).reshape(-1, 1), (LANES, F))
        assert c0.dtype == torch.int32 and torch.equal(c0, torch.from_numpy(want_c0.copy()))
        k_np = [(seed & 0xFFFFFF), ((seed >> 24) & 0xFFFFFF), 0, 0]
        for kw, v in zip(k, k_np, strict=True):
            assert kw.shape == (LANES, 1) and kw.dtype == torch.int32
            assert torch.equal(kw, torch.full((LANES, 1), v, dtype=torch.int32))
        starts = (offset + np.arange(n_launches * C, dtype=np.int64) * F) & 0xFFFFFF
        want = np.broadcast_to(
            starts.astype(np.int32).reshape(n_launches, 1, C), (n_launches, LANES, C)
        )
        assert sb.dtype == torch.int32 and torch.equal(sb, torch.from_numpy(want.copy()))

    @pytest.mark.parametrize("name", list(FNS))
    @pytest.mark.parametrize("make_gen", GENS)
    @pytest.mark.parametrize("n", [1, 513, 16_385, 50_000])
    def test_matches_eager(self, nki_native, name, make_gen, n):
        fn = FNS[name]
        _assert_compiles_like_eager(make_gen, lambda g: fn(g, n))

    @pytest.mark.parametrize("route", list(ROUTES))
    @pytest.mark.parametrize("name", ["normal", "uniform"])
    @pytest.mark.parametrize("make_gen", [GENS[2], GENS[3]])
    def test_routes_match_eager(self, nki_native, monkeypatch, route, name, make_gen):
        """Every native route (limb variants, streaming opt-out) compiles like eager."""
        env, frag = ROUTES[route]
        for key, val in env.items():
            monkeypatch.setenv(key, val)
        _assert_compiles_like_eager(make_gen, lambda g: FNS[name](g, 50_000))
        g = make_gen()
        targets = _launch_targets(lambda: FNS[name](g, 50_000))
        assert any(frag in t and name in t for t in targets), targets

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    @pytest.mark.parametrize("make_gen", GENS)
    @pytest.mark.parametrize("n", [1, 16_385, 70_000, 300_000])
    def test_multi_launch_plan_matches_eager(self, nki_native, small_plan, name, make_gen, n):
        """Plan buckets: multi-chunk, multi-launch bulk plans trace like eager."""
        _assert_compiles_like_eager(make_gen, lambda g: FNS[name](g, n))

    def test_eager_path_unchanged(self, nki_native):
        # The eager route still goes through the numpy host setup.
        g = trnrand.Generator(seed=11)
        seed = distributions._nki_seed(trnrand.Generator(seed=11))
        want = dispatch.threefry_stream_normal(1000, seed=seed, counter_offset=0)
        assert torch.equal(trnrand.normal(1000, generator=g), want)

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    def test_zero_graph_breaks(self, nki_native, name):
        trnrand.manual_seed(0)
        graphs, breaks, reasons = _breaks(lambda: FNS[name](None, 20_000))
        assert (graphs, breaks) == (1, 0), reasons

    def test_kernel_launch_stays_in_graph(self, nki_native):
        g = trnrand.Generator(seed=1)
        seen = _launch_targets(lambda: trnrand.normal(40_000, generator=g))
        assert any("gen_nki_state" in t for t in seen)
        # 40,000 values = 79 batches: one bulk launch, F = 128, C = 1 (#60).
        assert dispatch._bulk_plan(40_000) == (128, 1, 1)
        assert sum("trnrand_test.bulk32_normal" in t for t in seen) == 1
        assert not any("stream_" in t for t in seen)
        assert not any("gen_nki_draw" in t for t in seen)

    def test_multi_launch_stays_in_graph(self, nki_native, small_plan):
        # 300,000 values = 586 batches; caps F = 32, C <= 4 -> 19 chunks in 5 launches.
        F, C, n_launches = dispatch._bulk_plan(300_000)
        assert (F, n_launches) == (32, 5) and C * n_launches >= 19
        g = trnrand.Generator(seed=1)
        seen = _launch_targets(lambda: trnrand.normal(300_000, generator=g))
        assert sum("trnrand_test.bulk32_normal" in t for t in seen) == n_launches


class TestNonNativeNkiPath:
    """XLA lineage / per-tile kernels: the whole draw is the opaque gen_nki_draw op."""

    @pytest.fixture(autouse=True)
    def _xla(self, monkeypatch):
        def fake_launch(kernel, op_name, *t):
            assert op_name in ("threefry4x32", "threefry_normal")
            c0, c1, _c2, _c3, k0, k1, _k2, _k3 = t
            bias = 0.0 if op_name == "threefry4x32" else -10.0
            words = torch.cat([c0, c1, k0, k1], dim=1).to(torch.float32)
            return words / 2**24 + bias

        monkeypatch.setattr(distributions, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_get_dispatch_path", lambda: "xla")
        monkeypatch.setattr(dispatch, "_neuronx_cc_version", lambda: (2, 26))
        monkeypatch.setattr(dispatch, "_launch", fake_launch)
        monkeypatch.delenv("TRNRAND_BULK_STREAMING", raising=False)

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    @pytest.mark.parametrize("make_gen", GENS[:3])
    def test_matches_eager(self, name, make_gen):
        _assert_compiles_like_eager(make_gen, lambda g: FNS[name](g, 1500))

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    def test_zero_graph_breaks(self, name):
        trnrand.manual_seed(0)
        graphs, breaks, reasons = _breaks(lambda: FNS[name](None, 1500))
        assert (graphs, breaks) == (1, 0), reasons


class TestCounterKeyMode:
    """key_mode="counter": the draw lowerable on backend="neuron" (#61 round 2)."""

    def test_default_is_draw(self, monkeypatch):
        monkeypatch.delenv("TRNRAND_KEY_MODE", raising=False)
        assert trnrand.Generator(seed=1).key_mode == "draw"
        monkeypatch.setenv("TRNRAND_KEY_MODE", "counter")
        assert trnrand.Generator(seed=1).key_mode == "counter"
        with pytest.raises(ValueError):
            trnrand.Generator(seed=1, key_mode="bogus")

    @pytest.mark.parametrize("seed", [0, 7, 2**24 + 5, 2**40 + 12345, 2**63 + 9])
    def test_key_words(self, seed):
        g = trnrand.Generator(seed=seed, key_mode="counter")
        k = (seed & (2**48 - 1)) ^ (seed >> 48)
        assert g._key48 == k
        assert g._key_words.tolist() == [k & 0xFFFFFF, (k >> 24) & 0xFFFFFF]
        g.manual_seed(3)
        assert g._key_words.tolist() == [3, 0]

    def test_eager_stream(self, nki_native):
        g = trnrand.Generator(seed=2**30 + 99, key_mode="counter")
        torch_state = g.get_state().clone()
        a, b = trnrand.normal(1000, generator=g), trnrand.normal(1000, generator=g)
        # Fixed key, counter advances by ceil(1000 / 512) = 2 batches per call.
        assert torch.equal(a, dispatch.threefry_stream_normal(1000, g._key48, 0))
        assert torch.equal(b, dispatch.threefry_stream_normal(1000, g._key48, 2))
        assert torch.equal(g.get_state(), torch_state)  # host RNG untouched
        assert g.position() == 4 * 512

    @pytest.mark.parametrize("name", list(FNS))
    @pytest.mark.parametrize("make_gen", COUNTER_GENS)
    @pytest.mark.parametrize("n", [1, 513, 16_385, 50_000])
    def test_matches_eager(self, nki_native, name, make_gen, n):
        _assert_compiles_like_eager(make_gen, lambda g: FNS[name](g, n))

    @pytest.mark.parametrize("route", list(ROUTES))
    @pytest.mark.parametrize("name", ["normal", "uniform"])
    @pytest.mark.parametrize("make_gen", [COUNTER_GENS[2], COUNTER_GENS[3]])
    def test_routes_match_eager(self, nki_native, monkeypatch, route, name, make_gen):
        env, frag = ROUTES[route]
        for key, val in env.items():
            monkeypatch.setenv(key, val)
        _assert_compiles_like_eager(make_gen, lambda g: FNS[name](g, 50_000))
        g = make_gen()
        targets = _launch_targets(lambda: FNS[name](g, 50_000))
        assert any(frag in t and name in t for t in targets), targets
        assert not any("gen_nki" in t for t in targets)

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    @pytest.mark.parametrize("make_gen", COUNTER_GENS)
    @pytest.mark.parametrize("n", [1, 70_000, 300_000])
    def test_multi_launch_plan_matches_eager(self, nki_native, small_plan, name, make_gen, n):
        _assert_compiles_like_eager(make_gen, lambda g: FNS[name](g, n))

    def test_counter_advance_stays_in_graph(self, nki_native):
        g = trnrand.Generator(seed=3, key_mode="counter")
        targets = _launch_targets(lambda: trnrand.normal(40_000, generator=g))
        assert "add_" in targets, targets  # the in-place counter advance
        assert g.position() == 79 * 512

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    def test_zero_graph_breaks(self, nki_native, name):
        g = trnrand.Generator(seed=0, key_mode="counter")
        graphs, breaks, reasons = _breaks(lambda: FNS[name](g, 20_000))
        assert (graphs, breaks) == (1, 0), reasons

    def test_aten_graph_has_only_aten_and_kernel_ops(self, nki_native):
        """What backend="neuron" lowers: no trnrand custom op except the nki_op kernels."""
        from torch._dynamo.backends.common import aot_autograd

        targets = []

        def fw(gm, example_inputs):
            targets.extend(n.target for n in gm.graph.nodes if n.op == "call_function")
            return gm

        g = trnrand.Generator(seed=5, key_mode="counter")

        def fn():
            return trnrand.normal(40_000, generator=g) + trnrand.uniform(40_000, generator=g)

        ref_g = trnrand.Generator(seed=5, key_mode="counter")
        ref = trnrand.normal(40_000, generator=ref_g) + trnrand.uniform(40_000, generator=ref_g)
        got = torch.compile(fn, backend=aot_autograd(fw_compiler=fw), fullgraph=True)()
        assert torch.equal(got, ref) and g.position() == ref_g.position()

        kernels = {"bulk32_normal", "bulk32_uniform"}
        bad = []
        for t in targets:
            ns = getattr(t, "namespace", None)
            if t is operator.getitem or ns == "aten":
                continue
            if ns == "trnrand_test" and t._opname in kernels:
                continue
            bad.append(str(t))
        assert not bad, bad
        # One bulk launch per 40,000-value draw (#60; was 3 streaming launches each).
        assert sum(getattr(t, "namespace", None) == "trnrand_test" for t in targets) == 2

    @pytest.mark.parametrize("be", ["aot_eager", "inductor"])
    def test_unused_draw_still_advances(self, nki_native, be):
        g_e = trnrand.Generator(seed=5, key_mode="counter")
        g_c = trnrand.Generator(seed=5, key_mode="counter")
        cf = torch.compile(lambda: _unused_draw(g_c), backend=be, fullgraph=True)
        for _ in range(2):
            assert torch.equal(cf(), _unused_draw(g_e))
        assert g_c.position() == g_e.position()


@pytest.mark.neuron
@pytest.mark.parametrize("name", ["normal", "uniform"])
def test_neuron_backend_fullgraph(name):
    """Trn2: torch.compile(backend="neuron", fullgraph=True) on the native NKI path.

    Needs key_mode="counter": the default "draw" mode's per-call key comes from
    the host Mersenne Twister and cannot lower to StableHLO (#61).
    """
    pytest.importorskip("torch_neuronx")
    assert dispatch._get_dispatch_path() == "native"
    n = 512 * 512
    g_e = trnrand.Generator(seed=0, key_mode="counter")
    g_c = trnrand.Generator(seed=0, key_mode="counter")
    cf = torch.compile(lambda: FNS[name](g_c, n), backend="neuron", fullgraph=True, dynamic=False)
    for _ in range(2):  # second call: counter advanced in-graph, no recompile
        assert torch.equal(cf().cpu(), FNS[name](g_e, n).cpu())
    assert g_c.position() == g_e.position()
