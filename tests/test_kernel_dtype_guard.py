"""One dtype guard at the NKI kernel boundary (trnsci/trnsci#35).

Every trnrand kernel takes int32 words except Box-Muller (float32 uniforms).
These tests run the real host wrappers on CPU against stand-in kernels that
fail when an operand of any other dtype reaches them:

- a fake ``nki`` package (``nki.jit`` is the identity, ``nki.simulate``
  returns a checking stub) and a fake TorchNeuron ``torch_neuronx`` whose
  ``nki_op`` returns a checking stub, under which ``trnrand.nki.dispatch``
  and ``trnrand.nki._native`` are imported fresh (``sys.modules`` is
  restored afterwards, so other tests see the real modules);
- each launch route: native (``nki_op`` table), XLA (kernel called directly)
  and the simulator (``nki.simulate``).

Every public API that can reach a kernel is then driven with float64,
bfloat16, float16, int32, int64, bool and complex64 requests under backends
``nki`` and ``auto``, and must give the same result dtype (or the same
exception type) as the pytorch backend.

The float64 policy (trnrand#105) is covered in all three modes: backend
``auto`` draws float64 in torch (no kernel launch, the pytorch backend's
values), backend ``nki`` runs the float32 kernels and warns once per op, and
``TRNRAND_REQUIRE_NKI`` raises ``NotImplementedError``. A float64 buffer on
the neuron device (faked: a CPU tensor the neuron predicate reports as
device-resident) is filled on the host under ``auto`` with one warning per
op, unless ``TRNRAND_FP64_ON_DEVICE=1`` keeps the on-device kernels.
"""

import importlib.util
import pathlib
import sys
import types
import warnings

import numpy as np
import pytest
import torch

import trnrand
import trnrand.nki
from trnrand import distributions, quasi
from trnrand.nki import program

_PKG = pathlib.Path(trnrand.nki.__file__).parent

# The float64 cast warning under backend "nki" (trnrand#105) is asserted by the
# float64 policy tests below; elsewhere it is expected noise.
pytestmark = pytest.mark.filterwarnings("ignore:trnrand .*float64 on the NKI kernels:UserWarning")

# Independent of trnrand's own table: what the kernel signatures take.
_I32 = (torch.int32,)
ALLOWED = {
    "philox4x32": _I32,
    "box_muller": (torch.float32,),
    "threefry4x32": _I32,
    "threefry_normal": _I32,
    "sobol_gray_code": _I32,
    "halton": _I32,
    "threefry_streaming_normal": _I32,
    "threefry_streaming_uniform": _I32,
    "threefry_bulk_normal": _I32,
    "threefry_bulk_uniform": _I32,
    "threefry_bulk16_normal": _I32,
    "threefry_bulk16_uniform": _I32,
    "threefry_bulk32_normal": _I32,
    "threefry_bulk32_uniform": _I32,
}
_NP = {np.dtype(np.int32): torch.int32, np.dtype(np.float32): torch.float32}

DTYPES = [
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
]


class Kernels:
    """Stand-in kernels: record each launch, refuse operands outside ALLOWED."""

    def __init__(self):
        self.calls = []
        self.registered = {}
        self.fake_fns = {}

    def check(self, op, dtypes):
        bad = [d for d in dtypes if d not in ALLOWED[op]]
        assert not bad, f"kernel {op!r} reached with operand dtypes {dtypes}"
        self.calls.append(op)

    def out(self, op, tensors):
        return self.fake_fns[op](*tensors).zero_()

    # torch_neuronx.nki_op stand-in
    def nki_op(self, name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        op = name.split("::", 1)[1]

        def dec(f):
            self.registered[op] = f
            self.fake_fns[op] = fake_fn

            def stub(*args):
                self.check(op, [a.dtype for a in args])
                return self.out(op, args)

            return stub

        return dec if fn is None else dec(fn)

    # nki.simulate stand-in
    def simulate(self, kernel):
        op = kernel.__name__.removesuffix("_kernel")

        def run(*arrays):
            self.check(op, [_NP.get(a.dtype, f"numpy.{a.dtype}") for a in arrays])
            return self.out(op, [torch.from_numpy(np.asarray(a)) for a in arrays]).numpy()

        return run

    # XLA path: the raw kernel is called on (here: CPU) tensors
    def xla_kernel(self, op):
        def run(*args):
            self.check(op, [a.dtype for a in args])
            return self.out(op, args)

        run.__name__ = f"{op}_kernel"
        return run


def _fake_package(kernels):
    nki = types.ModuleType("nki")
    nki.__path__ = []

    def jit(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)

    nki.jit = jit
    nki.simulate = kernels.simulate
    lang = types.ModuleType("nki.language")
    isa = types.ModuleType("nki.isa")
    lang.__getattr__ = isa.__getattr__ = lambda name: object()
    nki.language, nki.isa = lang, isa
    tnx = types.ModuleType("torch_neuronx")
    tnx.nki_op = kernels.nki_op
    tnx.device_count = lambda: 1
    return {"nki": nki, "nki.language": lang, "nki.isa": isa, "torch_neuronx": tnx}


def _exec_fresh(monkeypatch, name, filename):
    spec = importlib.util.spec_from_file_location(name, _PKG / filename)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(monkeypatch):
    """Fresh dispatch + _native under fake nki / torch_neuronx; returns a namespace."""
    for var in (
        "TRNRAND_BULK_STREAMING",
        "TRNRAND_BULK_KERNEL",
        "TRNRAND_LIMB_BITS",
        "TRNRAND_CORES",
        "TRNSCI_NKI_DISPATCH",
    ):
        monkeypatch.delenv(var, raising=False)
    kernels = Kernels()
    for name, mod in _fake_package(kernels).items():
        monkeypatch.setitem(sys.modules, name, mod)
    d = _exec_fresh(monkeypatch, "trnrand.nki.dispatch", "dispatch.py")
    assert d.HAS_NKI
    monkeypatch.setattr(trnrand.nki, "dispatch", d)
    nat = _exec_fresh(monkeypatch, "trnrand.nki._native", "_native.py")
    monkeypatch.setattr(trnrand.nki, "_native", nat, raising=False)
    d._NEURON_DEVICE = "cpu"
    d._REQUIRE_NKI = False
    d._USE_SIMULATOR = False
    d._to_xla = lambda *t: (list(t), t[0].device)
    for op in nat.OPS:
        setattr(d, f"{op}_kernel", kernels.xla_kernel(op))
    for m in (distributions, quasi):
        monkeypatch.setattr(m, "HAS_NKI", True)
        monkeypatch.setattr(m, "get_backend", d.get_backend)
    monkeypatch.setattr(program, "HAS_NKI", True)

    ns = types.SimpleNamespace(d=d, nat=nat, kernels=kernels)

    def use_path(path):
        d._dispatch_path = "native" if path == "sim" else path
        d._native_ops = None
        d._USE_SIMULATOR = path == "sim"

    ns.use_path = use_path
    use_path("native")
    return ns


# ── Public API drivers: fn(dtype) -> tensor ─────────────────────────────────


def _gen(**kw):
    return trnrand.Generator(seed=7, **kw)


def _into(fn, **kw):
    def run(dt):
        buf = torch.empty(40, dtype=dt)
        fn(buf, generator=_gen(), **kw)
        return buf

    return run


def _program(kind):
    def run(dt):
        prog = getattr(_gen().new_program(), kind)(40, out="x").build()
        buf = torch.empty(40, dtype=dt)
        if distributions.get_backend() == "pytorch":
            prog._stream_into_pytorch({"x": buf})  # the CPU path of stream_into
        else:
            prog.stream_into({"x": buf})
        return buf

    return run


APIS = {
    "uniform": lambda dt: trnrand.uniform(40, dtype=dt, generator=_gen()),
    "uniform_ab": lambda dt: trnrand.uniform(40, low=-1.0, high=3.0, dtype=dt, generator=_gen()),
    "uniform_counter": lambda dt: trnrand.uniform(40, dtype=dt, generator=_gen(key_mode="counter")),
    "normal": lambda dt: trnrand.normal(40, mean=1.0, std=2.0, dtype=dt, generator=_gen()),
    "normal_part": lambda dt: trnrand.normal(
        40, dtype=dt, generator=_gen(partition_rank=1, partition_size=2)
    ),
    "standard_normal": lambda dt: trnrand.standard_normal(4, 10, dtype=dt, generator=_gen()),
    "exponential": lambda dt: trnrand.exponential(40, rate=2.0, dtype=dt, generator=_gen()),
    "bernoulli": lambda dt: trnrand.bernoulli(40, p=0.3, dtype=dt, generator=_gen()),
    "randint": lambda dt: trnrand.randint(40, low=0, high=5, dtype=dt, generator=_gen()),
    "randperm": lambda dt: trnrand.randperm(40, dtype=dt, generator=_gen()),
    "truncated_normal": lambda dt: trnrand.truncated_normal(40, dtype=dt, generator=_gen()),
    "gamma": lambda dt: trnrand.gamma(40, shape=2.0, dtype=dt, generator=_gen()),
    "gamma_small": lambda dt: trnrand.gamma(40, shape=0.5, dtype=dt, generator=_gen()),
    "chi_squared": lambda dt: trnrand.chi_squared(40, df=3.0, dtype=dt, generator=_gen()),
    "beta": lambda dt: trnrand.beta(40, alpha=2.0, beta=3.0, dtype=dt, generator=_gen()),
    "poisson": lambda dt: trnrand.poisson(40, lam=30.0, dtype=dt, generator=_gen()),
    "poisson_small": lambda dt: trnrand.poisson(40, lam=2.0, dtype=dt, generator=_gen()),
    "normal_into": _into(trnrand.normal_into, mean=1.0, std=2.0),
    "uniform_into": _into(trnrand.uniform_into, low=-1.0, high=3.0),
    "exponential_into": _into(trnrand.exponential_into, rate=2.0),
    "program_normal": _program("normal"),
    "program_uniform": _program("uniform"),
    "program_exponential": _program("exponential"),
    "sobol": lambda dt: trnrand.sobol(40, 3, seed=5, dtype=dt),
    "sobol_noscramble": lambda dt: trnrand.sobol(40, 3, seed=5, scramble=False, dtype=dt),
    "halton": lambda dt: trnrand.halton(40, 3, dtype=dt),
    "latin_hypercube": lambda dt: trnrand.latin_hypercube(40, 3, dtype=dt),
}


def _outcome(fn, dt):
    try:
        return fn(dt)
    except Exception as exc:  # noqa: BLE001 - the type is the outcome
        return exc


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("api", sorted(APIS))
def test_public_api_dtypes(env, path, backend, api):
    env.use_path(path)
    fn = APIS[api]
    for dt in DTYPES:
        env.d.set_backend("pytorch")
        ref = _outcome(fn, dt)
        env.d.set_backend(backend)
        got = _outcome(fn, dt)
        env.d.set_backend("auto")
        where = f"{api}({dt}) backend={backend} path={path}"
        if isinstance(got, AssertionError):
            raise got  # a stand-in kernel saw an unsupported dtype
        if api.startswith("program_") and not dt.is_floating_point:
            # GeneratorProgram has no torch route on the NKI path: a clear refusal.
            assert isinstance(got, NotImplementedError), f"{where}: got {got!r}"
            continue
        if isinstance(ref, Exception):
            assert type(got) is type(ref), f"{where}: pytorch raised {ref!r}, got {got!r}"
            continue
        assert not isinstance(got, Exception), f"{where}: raised {got!r}"
        assert got.dtype == ref.dtype, f"{where}: dtype {got.dtype}, pytorch {ref.dtype}"
        assert got.shape == ref.shape, where


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
def test_float_requests_reach_the_kernels(env, path):
    """The real floating requests (float64 included) still run on the kernels."""
    env.use_path(path)
    env.d.set_backend("nki")
    try:
        for dt in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            env.kernels.calls.clear()
            for api in ("uniform", "normal", "truncated_normal", "uniform_into", "halton"):
                APIS[api](dt)
            assert env.kernels.calls, dt
    finally:
        env.d.set_backend("auto")


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
def test_reduced_precision_uniform_into_stays_below_high(env, path):
    env.use_path(path)
    env.d.set_backend("nki")
    # Every stand-in uniform is 0, so u * (high - low) + low == low; draw with
    # low just under high, where the float32 values round up to `high`.
    try:
        for dt in (torch.float16, torch.bfloat16):
            buf = torch.empty(40, dtype=dt)
            trnrand.uniform_into(buf, low=1.0 - 2**-20, high=1.0, generator=_gen())
            assert bool((buf < 1.0).all()), dt
    finally:
        env.d.set_backend("auto")


# ── Direct wrappers that take a caller's tensor ─────────────────────────────


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("dt", [torch.int32, torch.int64, torch.int16, torch.uint8])
def test_philox_integer_words(env, path, dt):
    env.use_path(path)
    words = torch.arange(130).to(dt)
    out = env.d.philox4x32_nki(words, words, words)
    assert out.dtype == torch.int32 and out.shape == (130 * 4,)


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.bool, torch.complex64])
def test_philox_refuses_non_integer_words(env, path, dt):
    env.use_path(path)
    words = torch.ones(8, dtype=dt)
    with pytest.raises(TypeError, match="philox4x32_nki"):
        env.d.philox4x32_nki(words, words, words)


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("dt", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_box_muller_float_inputs(env, path, dt):
    env.use_path(path)
    u = torch.rand(300).to(dt)
    out = env.d.box_muller_nki(u)
    assert out.dtype == env.d.box_muller_cpu(u).dtype == dt
    assert out.shape == (300,)


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("dt", [torch.int32, torch.int64, torch.bool, torch.complex64])
def test_box_muller_refuses_non_real_float(env, path, dt):
    env.use_path(path)
    with pytest.raises(TypeError, match="box_muller_nki"):
        env.d.box_muller_nki(torch.ones(8, dtype=dt))


# ── TRNRAND_REQUIRE_NKI ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
def test_require_nki_raises_instead_of_routing(env, path):
    env.use_path(path)
    env.d._REQUIRE_NKI = True
    env.d.set_backend("nki")
    try:
        for dt in (torch.int32, torch.int64, torch.bool, torch.complex64):
            for api in (
                "uniform",
                "normal",
                "standard_normal",
                "truncated_normal",
                "normal_into",
                "uniform_into",
                "exponential_into",
            ):
                with pytest.raises(NotImplementedError, match="TRNRAND_REQUIRE_NKI"):
                    APIS[api](dt)
        with pytest.raises(NotImplementedError, match="TRNRAND_REQUIRE_NKI"):
            APIS["exponential"](torch.complex64)
        with pytest.raises(NotImplementedError, match="TRNRAND_REQUIRE_NKI"):
            env.d.box_muller_nki(torch.rand(8, dtype=torch.float64))
        # Exact casts still run; float64 raises (trnrand#105, tested below).
        assert env.d.box_muller_nki(torch.rand(8).bfloat16()).dtype == torch.bfloat16
        assert APIS["uniform"](torch.float32).dtype == torch.float32
        assert APIS["normal"](torch.float16).dtype == torch.float16
    finally:
        env.d.set_backend("auto")


@pytest.mark.parametrize("dt", [torch.int32, torch.bool, torch.complex64])
def test_program_refuses_non_real_float_buffers(env, dt):
    env.d.set_backend("nki")
    try:
        prog = _gen().new_program().normal(8, out="z").uniform(8, out="u").build()
        before = prog._counter
        with pytest.raises(NotImplementedError, match="'u'"):
            prog.stream_into({"z": torch.empty(8), "u": torch.empty(8, dtype=dt)})
        assert prog._counter == before  # nothing drawn
    finally:
        env.d.set_backend("auto")


# ── Structural: the launch routes enforce the table themselves ─────────────


def test_every_op_has_a_table_entry(env):
    env.d._get_native_ops()
    assert set(env.kernels.registered) == set(env.nat.OPS) == set(ALLOWED)
    assert set(env.d._KERNEL_DTYPES) == set(ALLOWED)
    assert {k: tuple(v) for k, v in env.d._KERNEL_DTYPES.items()} == ALLOWED


def test_op_without_table_entry_cannot_load(env, monkeypatch):
    monkeypatch.delitem(env.d._KERNEL_DTYPES, "halton")
    env.d._native_ops = None
    with pytest.raises(TypeError, match="halton"):
        env.d._get_native_ops()


@pytest.mark.parametrize("path", ["native", "xla"])
@pytest.mark.parametrize("dt", [torch.float64, torch.float32, torch.int64, torch.bool])
def test_launch_refuses_bad_operands(env, path, dt):
    env.use_path(path)
    t = torch.zeros(128, 1, dtype=dt)
    with pytest.raises(TypeError, match="'threefry4x32'"):
        env.d._launch(env.d.threefry4x32_kernel, "threefry4x32", *([t] * 8))
    assert env.kernels.calls == []


def test_launch_refuses_unknown_op(env):
    with pytest.raises(TypeError, match="no _KERNEL_DTYPES entry"):
        env.d._launch(lambda *a: a[0], "not_a_kernel", torch.zeros(1, dtype=torch.int32))


@pytest.mark.parametrize("dt", [torch.float64, torch.float32, torch.int64])
def test_native_op_table_refuses_bad_operands(env, dt):
    op = env.d._get_native_ops()["threefry_bulk32_normal"]
    t = torch.zeros(128, 1, dtype=dt)
    with pytest.raises(TypeError, match="'threefry_bulk32_normal'"):
        op(*([t] * 6))
    assert env.kernels.calls == []


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int64, np.uint32])
def test_simulator_refuses_bad_operands(env, dt):
    sim = env.d._simulator("halton", env.d.halton_kernel)
    with pytest.raises(TypeError, match="'halton'"):
        sim(np.zeros((128, 1), dtype=dt))
    assert env.kernels.calls == []


def test_box_muller_refuses_int_operand_at_launch(env):
    # The float32 kernel is guarded too, not only the integer-word ones.
    with pytest.raises(TypeError, match="'box_muller'"):
        env.d._launch(env.d.box_muller_kernel, "box_muller", torch.zeros(4, 2, dtype=torch.int32))


def test_every_op_is_driven(env, monkeypatch):
    """Across the launch routes and limb widths, the drivers above reach every op."""
    seen = set()

    def drive():
        env.d.set_backend("nki")
        try:
            for api in ("uniform", "normal", "sobol", "halton", "program_normal"):
                APIS[api](torch.float64)
            w = torch.arange(8)
            env.d.philox4x32_nki(w, w, w)
            env.d.box_muller_nki(torch.rand(8, dtype=torch.float64))
        finally:
            env.d.set_backend("auto")
        seen.update(env.kernels.calls)

    for limb in ("32", "16", "8"):
        monkeypatch.setenv("TRNRAND_LIMB_BITS", limb)
        env.use_path("native")
        drive()
    monkeypatch.delenv("TRNRAND_LIMB_BITS")
    monkeypatch.setenv("TRNRAND_BULK_KERNEL", "0")
    drive()  # native streaming kernels
    monkeypatch.setenv("TRNRAND_BULK_STREAMING", "0")
    env.use_path("xla")
    drive()  # per-tile kernels
    assert seen == set(ALLOWED)


# ── float64 policy (trnrand#105) ───────────────────────────────────────────

# APIs whose float64 request takes torch under "auto" -> warning op name.
FP64_ROUTED = {
    "uniform": "uniform",
    "uniform_ab": "uniform",
    "uniform_counter": "uniform",
    "normal": "normal",
    "normal_part": "normal",
    "standard_normal": "normal",
    "exponential": "exponential",
    "truncated_normal": "truncated_normal",
    "gamma": "gamma",
    "gamma_small": "gamma",
    "chi_squared": "chi_squared",
    "beta": "beta",
    "poisson": "poisson",
    "normal_into": "normal_into",
    "uniform_into": "uniform_into",
    "exponential_into": "exponential_into",
    "halton": "halton",
    "program_normal": "GeneratorProgram.stream_into",
    "program_uniform": "GeneratorProgram.stream_into",
    "program_exponential": "GeneratorProgram.stream_into",
}
# The pytorch backend's float64 Sobol is a float32 SobolEngine draw cast, so
# float64 Sobol keeps the kernel on every backend (no warning, no raise).
FP64_KERNEL = ("sobol", "sobol_noscramble")


def _fp64_warnings(record):
    return [w for w in record if "float64 on the NKI kernels" in str(w.message)]


def test_fp64_tables_cover_every_kernel_api():
    no_kernel = {"bernoulli", "randint", "randperm", "latin_hypercube", "poisson_small"}
    assert set(FP64_ROUTED) | set(FP64_KERNEL) | no_kernel == set(APIS)


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("api", sorted(FP64_ROUTED))
def test_fp64_auto_draws_in_torch(env, path, api):
    """backend "auto": no kernel launch, no warning, float64-precision values."""
    env.use_path(path)
    fn = APIS[api]
    env.d.set_backend("pytorch")
    ref = fn(torch.float64)
    env.d.set_backend("auto")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(torch.float64)
    assert env.kernels.calls == [], f"{api}: kernel launched under auto"
    assert not _fp64_warnings(rec)
    assert got.dtype == torch.float64 and got.shape == ref.shape
    if api.startswith("program_"):
        # The no-NKI program path seeds differently; check the values instead.
        assert bool(torch.isfinite(got).all())
    else:
        assert torch.equal(got, ref), f"{api}: not the pytorch backend's float64 draw"
    if api in ("uniform", "uniform_into", "program_uniform", "halton"):
        # Not on the float32 kernels' 24-bit grid (k * 2^-24).
        grid = (got.double() * 2**24) % 1
        assert bool((grid != 0).any()), f"{api}: float32-resolution values"


@pytest.mark.parametrize("kind", ["normal", "uniform", "exponential"])
def test_fp64_program_step_keeps_other_streams(env, kind):
    """A float64 step drawn in torch takes the same counter slots as its kernel step."""

    def run(dt):
        env.kernels.calls.clear()
        prog = getattr(_gen().new_program().normal(40, out="a"), kind)(40, out="x")
        prog = prog.normal(40, out="b").build()
        bufs = {"a": torch.empty(40), "x": torch.empty(40, dtype=dt), "b": torch.empty(40)}
        prog.stream_into(bufs)
        return prog._counter, len(env.kernels.calls)

    c32, k32 = run(torch.float32)
    c64, k64 = run(torch.float64)
    assert c64 == c32
    assert 0 < k64 < k32  # only the float64 step left the kernels


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("api", sorted(FP64_ROUTED))
def test_fp64_nki_casts_and_warns_once(env, path, api):
    """backend "nki": the float32 kernels run (the stand-ins check the int32 /
    float32 operands) and the op warns exactly once across repeated calls."""
    env.use_path(path)
    env.d.set_backend("nki")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            outs = [APIS[api](torch.float64) for _ in range(3)]
    finally:
        env.d.set_backend("auto")
    assert env.kernels.calls, f"{api}: no kernel launched under nki"
    assert all(o.dtype == torch.float64 for o in outs)
    got = _fp64_warnings(rec)
    assert len(got) == 1, [str(w.message) for w in got]
    assert issubclass(got[0].category, UserWarning)
    assert f"trnrand {FP64_ROUTED[api]}:" in str(got[0].message)
    assert got[0].filename == __file__, got[0].filename  # points at the caller


def test_fp64_nki_warns_once_per_op(env):
    env.d.set_backend("nki")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in range(2):
                for api in ("uniform", "uniform_ab", "normal", "standard_normal", "halton"):
                    APIS[api](torch.float64)
                APIS["uniform"](torch.float32)
    finally:
        env.d.set_backend("auto")
    msgs = [str(w.message).split(":")[0] for w in _fp64_warnings(rec)]
    assert sorted(msgs) == ["trnrand halton", "trnrand normal", "trnrand uniform"]


@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("api", FP64_KERNEL)
def test_fp64_sobol_keeps_the_kernel(env, backend, api):
    env.d.set_backend(backend)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = APIS[api](torch.float64)
        env.d._REQUIRE_NKI = True
        assert APIS[api](torch.float64).dtype == torch.float64  # no raise either
    finally:
        env.d.set_backend("auto")
    assert out.dtype == torch.float64
    assert "sobol_gray_code" in env.kernels.calls
    assert not _fp64_warnings(rec)


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
@pytest.mark.parametrize("backend", ["nki", "auto"])
@pytest.mark.parametrize("api", sorted(FP64_ROUTED))
def test_fp64_require_nki_raises(env, path, backend, api):
    env.use_path(path)
    env.d._REQUIRE_NKI = True
    env.d.set_backend(backend)
    try:
        with pytest.raises(NotImplementedError, match="TRNRAND_REQUIRE_NKI"):
            APIS[api](torch.float64)
        assert APIS[api](torch.float32).dtype == torch.float32  # float32 still runs
    finally:
        env.d.set_backend("auto")


def test_fp64_require_nki_program_draws_nothing(env):
    env.d._REQUIRE_NKI = True
    prog = _gen().new_program().normal(8, out="z").uniform(8, out="u").build()
    before = prog._counter
    with pytest.raises(NotImplementedError, match="TRNRAND_REQUIRE_NKI"):
        prog.stream_into({"z": torch.empty(8), "u": torch.empty(8, dtype=torch.float64)})
    assert prog._counter == before
    assert env.kernels.calls == []


def test_fp64_poisson_small_lam_stays_in_torch(env):
    # lam < 20 never used a kernel; REQUIRE_NKI does not change that.
    env.d._REQUIRE_NKI = True
    env.d.set_backend("nki")
    try:
        assert APIS["poisson_small"](torch.float64).dtype == torch.float64
    finally:
        env.d.set_backend("auto")
    assert env.kernels.calls == []


@pytest.mark.parametrize("path", ["native", "xla", "sim"])
def test_box_muller_nki_fp64_warns_once(env, path):
    env.use_path(path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            assert env.d.box_muller_nki(torch.rand(8, dtype=torch.float64)).dtype == torch.float64
    got = _fp64_warnings(rec)
    assert len(got) == 1 and "box_muller_nki" in str(got[0].message)


# ── float64 buffers on the neuron device: filled on the host ───────────────
#
# Device arithmetic on Trainium is float32 (measured on Trn2), so under
# backend "auto" a float64 `*_into` / `GeneratorProgram` buffer on the neuron
# device is filled on the host at float64 precision and copied back into the
# device buffer, with one UserWarning per op. TRNRAND_FP64_ON_DEVICE=1 keeps
# the on-device float32 kernels. Here a "neuron" buffer is a CPU tensor that
# `dispatch._on_neuron_device` reports as neuron-resident; every other trnrand
# result is a host tensor, so the buffers are the only device-resident case.

_HOST_MSG = (
    "float64 on the neuron device is computed on the host for float64 accuracy "
    "(device arithmetic is fp32); pass float32 to stay on device"
)
DEVICE_BUFFER_APIS = {
    "normal_into": (trnrand.normal_into, {"mean": 1.0, "std": 2.0}),
    "uniform_into": (trnrand.uniform_into, {"low": -1.0, "high": 3.0}),
    "exponential_into": (trnrand.exponential_into, {"rate": 2.0}),
    "program_normal": ("normal", {}),
    "program_uniform": ("uniform", {}),
    "program_exponential": ("exponential", {}),
}


def _host_warnings(record):
    return [w for w in record if _HOST_MSG in str(w.message)]


@pytest.fixture
def neuron_bufs(env, monkeypatch):
    """Marks buffers as neuron-resident and records copies into them."""
    marked: set[int] = set()
    copies: list[tuple[torch.device, torch.dtype]] = []
    real_copy = torch.Tensor.copy_

    def copy_spy(self, src, *a, **k):
        if id(self) in marked:
            copies.append((src.device, src.dtype))
        return real_copy(self, src, *a, **k)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_spy)
    monkeypatch.setattr(
        env.d, "_on_neuron_device", lambda t: isinstance(t, torch.Tensor) and id(t) in marked
    )

    def make(dt, n=40):
        buf = torch.full((n,), float("nan"), dtype=dt)
        marked.add(id(buf))
        return buf

    return types.SimpleNamespace(make=make, copies=copies)


def _fill(api, buf):
    fn, kw = DEVICE_BUFFER_APIS[api]
    if api.startswith("program_"):
        prog = getattr(_gen().new_program(), fn)(buf.numel(), out="x", **kw).build()
        prog.stream_into({"x": buf})
    else:
        fn(buf, generator=_gen(), **kw)
    return buf


@pytest.fixture(params=["nki", "no_nki"])
def kernels_present(request, env, monkeypatch):
    """With the NKI kernels (native path), or without them (the torch path)."""
    if request.param == "no_nki":
        monkeypatch.setattr(distributions, "HAS_NKI", False)
        monkeypatch.setattr(program, "HAS_NKI", False)
    return request.param == "nki"


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp64_device_buffer_auto_fills_on_host(env, neuron_bufs, kernels_present, api):
    """auto: no kernel, the host's float64 values copied into the same device
    buffer (float64), one warning per op over repeated calls."""
    ref = _fill(api, torch.empty(40, dtype=torch.float64))  # host buffer, same seed
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        outs = []
        for _ in range(3):
            buf = neuron_bufs.make(torch.float64)
            assert _fill(api, buf) is buf
            outs.append(buf)
    assert env.kernels.calls == [], f"{api}: kernel launched"
    assert neuron_bufs.copies == [(torch.device("cpu"), torch.float64)] * 3
    for buf in outs:
        assert buf.dtype == torch.float64
        assert torch.allclose(buf, ref, rtol=1e-12, atol=1e-12), api
    if api in ("uniform_into", "program_uniform"):
        assert bool(((outs[0] * 2**24) % 1 != 0).any()), "float32-resolution values"
    got = _host_warnings(rec)
    assert len(got) == 1, [str(w.message) for w in got]
    assert issubclass(got[0].category, UserWarning)
    op = "GeneratorProgram.stream_into" if api.startswith("program_") else api
    assert str(got[0].message).startswith(f"trnrand {op}: ")
    assert "TRNRAND_FP64_ON_DEVICE=1" in str(got[0].message)
    assert got[0].filename == __file__, got[0].filename
    assert not _fp64_warnings(rec)


@pytest.mark.parametrize("api", ["normal_into", "uniform_into", "exponential_into"])
def test_fp64_device_buffer_matches_pytorch_backend(env, neuron_bufs, api):
    env.d.set_backend("pytorch")
    try:
        ref = _fill(api, torch.empty(40, dtype=torch.float64))
    finally:
        env.d.set_backend("auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = _fill(api, neuron_bufs.make(torch.float64))
    assert torch.equal(got, ref)


def test_fp64_device_buffer_warns_once_per_op(env, neuron_bufs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(2):
            for api in DEVICE_BUFFER_APIS:
                _fill(api, neuron_bufs.make(torch.float64))
    ops = sorted(str(w.message).split(":")[0] for w in _host_warnings(rec))
    assert ops == [
        "trnrand GeneratorProgram.stream_into",
        "trnrand exponential_into",
        "trnrand normal_into",
        "trnrand uniform_into",
    ]


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp64_on_device_override_keeps_the_kernels(env, neuron_bufs, api):
    """TRNRAND_FP64_ON_DEVICE=1: the on-device float32 kernels, no host fill,
    no warning; the buffer stays float64."""
    env.d._FP64_ON_DEVICE = True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        buf = _fill(api, neuron_bufs.make(torch.float64))
    assert env.kernels.calls, f"{api}: no kernel launched under the override"
    assert (torch.device("cpu"), torch.float64) not in neuron_bufs.copies  # no host fill
    assert buf.dtype == torch.float64 and not bool(buf.isnan().any())
    assert not _host_warnings(rec) and not _fp64_warnings(rec)


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp64_on_device_override_without_kernels(env, neuron_bufs, monkeypatch, api):
    """Without the NKI kernels there is no device route: host fill, no warning."""
    monkeypatch.setattr(distributions, "HAS_NKI", False)
    monkeypatch.setattr(program, "HAS_NKI", False)
    env.d._FP64_ON_DEVICE = True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        buf = _fill(api, neuron_bufs.make(torch.float64))
    assert neuron_bufs.copies == [(torch.device("cpu"), torch.float64)]
    assert buf.dtype == torch.float64
    assert not _host_warnings(rec)


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp64_device_buffer_require_nki_raises(env, neuron_bufs, api):
    env.d._REQUIRE_NKI = True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match="TRNRAND_REQUIRE_NKI"):
            _fill(api, neuron_bufs.make(torch.float64))
    assert env.kernels.calls == [] and neuron_bufs.copies == []
    assert not _host_warnings(rec)


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp64_device_buffer_nki_backend_unchanged(env, neuron_bufs, api):
    """Explicit nki: float32 kernels with the cast warning, no host fill."""
    env.d.set_backend("nki")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            buf = _fill(api, neuron_bufs.make(torch.float64))
    finally:
        env.d.set_backend("auto")
    assert env.kernels.calls and buf.dtype == torch.float64
    assert len(_fp64_warnings(rec)) == 1 and not _host_warnings(rec)


@pytest.mark.parametrize("api", ["normal_into", "uniform_into", "exponential_into"])
def test_fp64_device_buffer_pytorch_backend_no_warning(env, neuron_bufs, api):
    env.d.set_backend("pytorch")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            buf = _fill(api, neuron_bufs.make(torch.float64))
    finally:
        env.d.set_backend("auto")
    assert env.kernels.calls == [] and buf.dtype == torch.float64
    assert neuron_bufs.copies == [(torch.device("cpu"), torch.float64)]
    assert not _host_warnings(rec)


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp64_host_buffer_unchanged(env, neuron_bufs, kernels_present, api):
    """A host float64 buffer is filled in place in host torch, without warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        buf = _fill(api, torch.full((40,), float("nan"), dtype=torch.float64))
    assert env.kernels.calls == [] and not bool(buf.isnan().any())
    assert not _host_warnings(rec)


@pytest.mark.parametrize("api", sorted(DEVICE_BUFFER_APIS))
def test_fp32_device_buffer_stays_on_device(env, neuron_bufs, api):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _fill(api, neuron_bufs.make(torch.float32))
    assert env.kernels.calls
    assert not _host_warnings(rec)


@pytest.mark.parametrize("api", sorted(FP64_ROUTED))
def test_fp64_host_results_do_not_warn(env, api):
    """Every other trnrand result is a host tensor: no host-fill warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert APIS[api](torch.float64).dtype == torch.float64
    assert not _host_warnings(rec)
