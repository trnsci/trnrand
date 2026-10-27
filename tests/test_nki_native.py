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
        from trnrand.nki.dispatch import threefry_normal_nki

        n = 16_384
        prog = trnrand.Generator(seed=5).new_program().normal(n, out="z").build()
        z = torch.empty(n)
        prog.stream_into({"z": z})
        # Streams emit per-tile order: element-for-element equal to the per-tile
        # wrapper for the same seed/offset (#58).
        torch.testing.assert_close(z, threefry_normal_nki(n, seed=5), rtol=0, atol=0)

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
