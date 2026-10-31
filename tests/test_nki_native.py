"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. The existing kernel-correctness
suites (test_nki_philox, test_nki_streaming, ...) also exercise the native
path automatically on a TorchNeuron host; this file covers what is specific
to it — custom-op registration, no torch_xla, torch.compile, and the
streaming program path.
"""

import sys

import pytest
import torch

from trnrand.nki import HAS_NKI
from trnrand.nki.dispatch import (
    UINT32_MASK,
    _get_dispatch_path,
    _use_simulator,
    philox4x32_reference,
    threefry4x32_reference,
)


def _native() -> bool:
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

LANES = 128


def _threefry_expected(seed: int) -> torch.Tensor:
    ctr = torch.zeros(LANES, 4, dtype=torch.int64)
    ctr[:, 0] = torch.arange(LANES, dtype=torch.int64)
    key = torch.tensor([seed & 0xFFFFFF, (seed >> 24) & 0xFFFFFF, 0, 0], dtype=torch.int64)
    ref_u32 = threefry4x32_reference(ctr, key.expand(LANES, 4))
    return ((ref_u32 & 0xFFFFFF) / 16777216.0).to(torch.float32)


def _threefry_inputs(seed: int) -> list[torch.Tensor]:
    c0 = torch.arange(LANES, dtype=torch.int32).reshape(-1, 1)
    zeros = torch.zeros(LANES, 1, dtype=torch.int32)
    k0 = torch.full((LANES, 1), seed & 0xFFFFFF, dtype=torch.int32)
    k1 = torch.full((LANES, 1), (seed >> 24) & 0xFFFFFF, dtype=torch.int32)
    return [t.to("neuron") for t in (c0, zeros, zeros, zeros, k0, k1, zeros, zeros)]


class TestNativeDispatch:
    def test_native_path_never_imports_torch_xla(self):
        # In-process: a subprocess can't acquire NeuronCores while this pytest
        # process holds them. Lazy registration itself is covered on CPU in
        # tests/test_native_dispatch.py.
        from trnrand.nki.dispatch import _get_dispatch_path, threefry_uniform_nki

        threefry_uniform_nki(512, seed=1)
        assert _get_dispatch_path() == "native"
        assert "torch_xla" not in sys.modules

    def test_ops_registered_in_torch_ops(self):
        from trnrand.nki.dispatch import _get_native_ops

        _get_native_ops()
        for name in ("philox4x32", "threefry4x32", "threefry_streaming_normal"):
            assert hasattr(torch.ops.trnrand, name)

    def test_threefry_op_eager_matches_reference(self):
        seed = 0xABCD1234
        out = torch.ops.trnrand.threefry4x32(*_threefry_inputs(seed))
        assert out.device.type == "neuron"
        torch.testing.assert_close(out.cpu(), _threefry_expected(seed), rtol=1e-5, atol=1e-5)

    def test_threefry_op_inside_torch_compile(self):
        from trnrand.nki.dispatch import _get_native_ops

        threefry4x32 = _get_native_ops()["threefry4x32"]

        @torch.compile(backend="neuron", fullgraph=True)
        def scaled(*args):
            return threefry4x32(*args) * 2.0

        seed = 0x1234
        out = scaled(*_threefry_inputs(seed)).cpu()
        torch.testing.assert_close(out, _threefry_expected(seed) * 2.0, rtol=1e-5, atol=1e-5)

    @pytest.mark.xfail(
        strict=False,
        reason="Philox integer precision on hardware: aws-neuron-sdk#1308, "
        "gated on Spike C (trnsci/trnsci#60)",
    )
    def test_philox_op_matches_reference(self):
        counter_lo = torch.arange(LANES, dtype=torch.int32).reshape(-1, 1)
        zeros = torch.zeros(LANES, 1, dtype=torch.int32)
        out = torch.ops.trnrand.philox4x32(
            counter_lo.to("neuron"), zeros.to("neuron"), zeros.to("neuron")
        ).cpu()

        ctr = torch.zeros(LANES, 4, dtype=torch.int64)
        ctr[:, 0] = counter_lo.reshape(-1).to(torch.int64)
        key = torch.zeros(LANES, 2, dtype=torch.int64)
        expected = (philox4x32_reference(ctr, key) & UINT32_MASK).to(torch.int32)
        assert torch.equal(out, expected)

    def test_generator_program_streams_on_native_path(self):
        import trnrand
        from trnrand.nki.dispatch import _threefry_normal_per_tile

        n = 16_384
        prog = trnrand.Generator(seed=5).new_program().normal(n, out="z").build()
        z = torch.empty(n)
        prog.stream_into({"z": z})
        # Streams emit per-tile order: element-for-element equal to the per-tile
        # wrapper for the same seed/offset (#58).
        torch.testing.assert_close(z, _threefry_normal_per_tile(n, seed=5), rtol=0, atol=0)

    @pytest.mark.parametrize("n,seed,offset", [(16_385, 3, 0), (3 * 16_384 + 100, 21, 7)])
    def test_bulk_multi_launch_matches_per_tile(self, n, seed, offset):
        # Bulk wrappers stream with device-resident inputs indexed per launch and
        # one stacked copy back (#60); values/order must equal the per-tile path.
        from trnrand.nki.dispatch import (
            _threefry_normal_per_tile,
            _threefry_uniform_per_tile,
            threefry_normal_nki,
            threefry_uniform_nki,
        )

        z = threefry_normal_nki(n, seed=seed, counter_offset=offset)
        torch.testing.assert_close(
            z, _threefry_normal_per_tile(n, seed=seed, counter_offset=offset), rtol=0, atol=0
        )
        u = threefry_uniform_nki(n, seed=seed, counter_offset=offset)
        assert torch.equal(u, _threefry_uniform_per_tile(n, seed=seed, counter_offset=offset))

    @pytest.mark.parametrize("bits", ["16", "32"])
    @pytest.mark.parametrize(
        "n,seed,offset",
        [
            (1, 0, 0),
            (16_385, 12345, 2**24 - 5),
            (1_048_577, 2**40 + 3, 7),
            (4_194_305, 9, 2**24 - 40),
        ],
    )
    def test_limb_variants_bit_identical(self, monkeypatch, bits, n, seed, offset):
        # TRNRAND_LIMB_BITS=16|32 (#60) must reproduce the 8-bit bulk kernels exactly.
        from trnrand.nki.dispatch import threefry_normal_nki, threefry_uniform_nki

        for fn in (threefry_normal_nki, threefry_uniform_nki):
            monkeypatch.setenv("TRNRAND_LIMB_BITS", "8")
            want = fn(n, seed=seed, counter_offset=offset)
            monkeypatch.setenv("TRNRAND_LIMB_BITS", bits)
            assert torch.equal(fn(n, seed=seed, counter_offset=offset), want)

    def test_partitioned_program_concatenates_to_single_chip(self):
        import trnrand

        n, P = 32_768, 2
        full = torch.empty(n)
        trnrand.Generator(seed=9).new_program().uniform(n, out="u").build().stream_into({"u": full})
        parts = []
        for r in range(P):
            gen = trnrand.Generator(seed=9, partition_rank=r, partition_size=P)
            buf = torch.empty(n // P)
            gen.new_program().uniform(n // P, out="u").build().stream_into({"u": buf})
            parts.append(buf)
        assert torch.equal(torch.cat(parts), full)


class TestMultiCore:
    """Bulk draws split over logical NeuronCores (#60): bit-identical for every core count.

    Needs at least two visible logical cores (e.g. NEURON_RT_VISIBLE_CORES=0-3)."""

    @pytest.fixture(autouse=True)
    def _cores(self, monkeypatch):
        from trnrand.nki import dispatch

        if dispatch._visible_cores() < 2:
            pytest.skip("needs >= 2 visible logical NeuronCores")
        monkeypatch.setenv("TRNRAND_CORES_MIN_ELEMENTS", "0")
        yield
        dispatch.set_cores(None)

    @pytest.mark.parametrize("bits", ["32", "8"])
    @pytest.mark.parametrize(
        "n,seed,offset",
        [
            (1, 0, 0),
            (1_048_577, 2**40 + 3, 7),
            (4_194_305, 9, 2**24 - 40),
            (16_777_216, 12345, 2**24 - 3000),
        ],
    )
    def test_bit_identical_across_core_counts(self, monkeypatch, bits, n, seed, offset):
        from trnrand.nki import dispatch

        if bits == "8" and n > 4_194_305:
            pytest.skip("8-bit kernels are slow; 4M covers them")
        monkeypatch.setenv("TRNRAND_LIMB_BITS", bits)
        vis = dispatch._visible_cores()
        for fn in (dispatch.threefry_normal_nki, dispatch.threefry_uniform_nki):
            dispatch.set_cores(1)
            want = fn(n, seed=seed, counter_offset=offset)
            for k in sorted({2, 3, vis}):
                if k > vis:
                    continue
                dispatch.set_cores(k)
                assert torch.equal(fn(n, seed=seed, counter_offset=offset), want), k

    def test_public_api(self):
        import trnrand

        n = 4_194_304 + 513
        trnrand.set_cores(1)
        a = trnrand.normal(n, mean=1.0, std=3.0, generator=trnrand.Generator(seed=5))
        trnrand.set_cores("auto")
        b = trnrand.normal(n, mean=1.0, std=3.0, generator=trnrand.Generator(seed=5))
        assert torch.equal(a, b)
