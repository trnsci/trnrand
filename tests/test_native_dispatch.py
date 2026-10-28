"""
Dispatch-path selection (native TorchNeuron vs XLA vs CPU) and lazy
registration of the `trnrand::` custom ops.

CPU-only: `torch_neuronx` and `torch_xla` are replaced with stand-ins via
`sys.modules` / `find_spec` monkeypatching, so these run in the default
`pytest -m "not neuron"` job on any host.
"""

import importlib.util
import subprocess
import sys
import types

import pytest
import torch

import trnrand.nki
from trnrand.nki import dispatch

_REAL_FIND_SPEC = importlib.util.find_spec

EXPECTED_OPS = {
    "philox4x32",
    "box_muller",
    "threefry4x32",
    "threefry_normal",
    "sobol_gray_code",
    "halton",
    "threefry_streaming_normal",
    "threefry_streaming_uniform",
    "threefry_bulk_normal",
    "threefry_bulk_uniform",
    "threefry_bulk16_normal",
    "threefry_bulk16_uniform",
    "threefry_bulk32_normal",
    "threefry_bulk32_uniform",
}


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Each test sees an undetected path and an unregistered op table."""
    monkeypatch.setattr(dispatch, "_dispatch_path", None)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.delenv("TRNSCI_NKI_DISPATCH", raising=False)
    # Drop any `_native` left by an earlier test so registration re-runs.
    monkeypatch.delitem(sys.modules, "trnrand.nki._native", raising=False)
    monkeypatch.delattr(trnrand.nki, "_native", raising=False)
    yield


def _fake_native_torch_neuronx(device_count=1, recorded=None):
    """Stand-in for the TorchNeuron `torch_neuronx` (exports `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            if recorded is not None:
                recorded[name] = (f, mutates_args, fake_fn)
            return f

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: device_count
    return mod


def _fake_xla_torch_neuronx():
    """Stand-in for the XLA-lineage `torch_neuronx` (no `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    mod.trace = lambda *a, **k: None
    return mod


def _set_torch_xla_present(monkeypatch, present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "torch_xla":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)


def _accept_neuron_device(monkeypatch):
    # Off-hardware, "neuron" is not a registered torch device type.
    real_device = torch.device

    def device(spec, *args, **kwargs):
        if spec == dispatch._NEURON_DEVICE:
            return real_device("cpu")
        return real_device(spec, *args, **kwargs)

    monkeypatch.setattr(dispatch.torch, "device", device)


class TestDispatchPathSelection:
    def test_native_when_nki_op_and_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "native"

    def test_native_probe_does_not_touch_torch_xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        # Any import of torch_xla would raise ImportError; any spec lookup
        # would fail the test.
        monkeypatch.setitem(sys.modules, "torch_xla", None)

        def find_spec(name, *args, **kwargs):
            assert name != "torch_xla", "native probe must not look up torch_xla"
            return _REAL_FIND_SPEC(name, *args, **kwargs)

        monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)
        assert dispatch._get_dispatch_path() == "native"

    def test_xla_when_torch_neuronx_has_no_nki_op(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_xla_when_native_has_no_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(device_count=0)
        )
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_not_native_when_neuron_device_type_unregistered(self, monkeypatch):
        # torch_neuronx imported but "neuron" is not a torch device type
        # (e.g. another backend owns PrivateUse1).
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_not_native_when_device_count_raises(self, monkeypatch):
        mod = _fake_native_torch_neuronx()

        def boom():
            raise RuntimeError("no runtime")

        mod.device_count = boom
        monkeypatch.setitem(sys.modules, "torch_neuronx", mod)
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_cpu_when_nothing_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)  # import raises
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_detection_is_cached(self, monkeypatch):
        calls = []

        def detect():
            calls.append(1)
            return "xla"

        monkeypatch.setattr(dispatch, "_detect_dispatch_path", detect)
        assert dispatch._get_dispatch_path() == "xla"
        assert dispatch._get_dispatch_path() == "xla"
        assert len(calls) == 1


class TestDispatchOverride:
    """TRNSCI_NKI_DISPATCH=auto|native|xla (suite-wide) pins the path."""

    def _both_lineages(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)

    def test_auto_prefers_native(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "auto")
        assert dispatch._get_dispatch_path() == "native"

    def test_force_xla_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "XLA")
        assert dispatch._get_dispatch_path() == "xla"

    def test_force_native_without_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "native")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=native"):
            dispatch._get_dispatch_path()

    def test_force_xla_without_torch_xla_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "xla")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=xla"):
            dispatch._get_dispatch_path()

    def test_unknown_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cuda")
        with pytest.raises(ValueError, match="auto, native, or xla"):
            dispatch._get_dispatch_path()


class TestLaunchRouting:
    def test_native_launch_uses_custom_op_not_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        seen = {}

        def op(a, b):
            seen["args"] = (a, b)
            return a + b

        monkeypatch.setattr(dispatch, "_native_ops", {"fake": op})

        def no_xla(*t):
            raise AssertionError("_to_xla must not run on the native path")

        def no_kernel(*t):
            raise AssertionError("raw kernel must not be called on the native path")

        monkeypatch.setattr(dispatch, "_to_xla", no_xla)
        out = dispatch._launch(no_kernel, "fake", torch.ones(2, 1), torch.ones(2, 1))
        assert torch.equal(out, torch.full((2, 1), 2.0))
        assert out.device.type == "cpu"

    def test_xla_launch_calls_kernel_via_to_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        moved = []

        def fake_to_xla(*tensors):
            moved.append(len(tensors))
            return list(tensors), tensors[0].device

        monkeypatch.setattr(dispatch, "_to_xla", fake_to_xla)

        def no_ops():
            raise AssertionError("native ops must not load on the XLA path")

        monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
        out = dispatch._launch(lambda x: x * 3, "unused", torch.ones(4, 1))
        assert moved == [1]
        assert torch.equal(out, torch.full((4, 1), 3.0))

    def test_cpu_launch_keeps_pre_native_behaviour(self, monkeypatch):
        # No device path: the launch goes through `_to_xla`, which raises
        # when torch_xla is absent — the same failure as before.
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        with pytest.raises(ImportError):
            dispatch._launch(lambda x: x, "unused", torch.ones(1, 1))


class TestLazyRegistration:
    def test_import_does_not_register_ops(self):
        # Fresh interpreter: importing trnrand (and the dispatch module)
        # must not load `_native` or import torch_neuronx / torch_xla.
        code = (
            "import sys, trnrand, trnrand.nki.dispatch, trnrand.nki.program; "
            "bad = [m for m in ('trnrand.nki._native', 'torch_neuronx', 'torch_xla') "
            "if m in sys.modules]; "
            "assert not bad, bad"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ops_registered_on_first_native_use(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        assert "trnrand.nki._native" not in sys.modules
        assert recorded == {}

        ops = dispatch._get_native_ops()

        assert set(ops) == EXPECTED_OPS
        assert set(recorded) == {f"trnrand::{name}" for name in EXPECTED_OPS}
        assert all(mutates == {} for _, mutates, _ in recorded.values())
        assert all(fake is not None for _, _, fake in recorded.values())

        # Second call reuses the table; no re-registration.
        recorded.clear()
        assert dispatch._get_native_ops() is ops
        assert recorded == {}

    def test_op_signatures_build_a_schema(self, monkeypatch):
        # nki_op derives the schema with torch.library.infer_schema; check
        # every wrapper is fully annotated (Tensor args, Tensor return).
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        dispatch._get_native_ops()
        assert len(recorded) == len(EXPECTED_OPS)
        for name, (fn, mutates, _) in recorded.items():
            schema = torch.library.infer_schema(fn, mutates_args=mutates)
            assert schema.endswith("-> Tensor"), (name, schema)
            assert "Tensor" in schema.split("->")[0], (name, schema)

    def test_fake_fns_give_kernel_output_shapes(self, monkeypatch):
        recorded = {}
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(recorded=recorded)
        )
        # These constants live inside `if HAS_NKI:`; supply them off-NKI.
        monkeypatch.setattr(dispatch, "_SOBOL_MAX_DIMS", 10, raising=False)
        monkeypatch.setattr(dispatch, "_HALTON_MAX_DIMS", 10, raising=False)
        dispatch._get_native_ops()

        i32 = torch.zeros(128, 1, dtype=torch.int32, device="meta")
        f32 = torch.zeros(128, 2, dtype=torch.float32, device="meta")

        def fake(name):
            return recorded[f"trnrand::{name}"][2]

        assert fake("philox4x32")(i32, i32, i32).shape == (128, 4)
        assert fake("philox4x32")(i32, i32, i32).dtype == torch.int32
        assert fake("box_muller")(f32).shape == (128, 2)
        assert fake("threefry4x32")(*[i32] * 8).shape == (128, 4)
        assert fake("threefry_normal")(*[i32] * 8).dtype == torch.float32
        assert fake("sobol_gray_code")(i32).shape == (128, 10)
        assert fake("halton")(i32).shape == (128, 10)
        streaming = (128, 4 * dispatch._PROGRAM_TILES)
        assert fake("threefry_streaming_normal")(*[i32] * 6).shape == streaming
        assert fake("threefry_streaming_uniform")(*[i32] * 6).shape == streaming
        lanes = torch.zeros(128, 512, dtype=torch.int32, device="meta")
        starts = torch.zeros(128, 16, dtype=torch.int32, device="meta")
        for name in ("threefry_bulk_normal", "threefry_bulk_uniform"):
            out = fake(name)(lanes, starts, *[i32] * 4)
            assert out.shape == (128, 16 * 4 * 512) and out.dtype == torch.float32
