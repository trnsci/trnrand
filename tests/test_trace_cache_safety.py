"""Guard against the NKI trace-cache closure bug (aws-neuron/aws-neuron-sdk#1454).

``import torch_neuronx`` turns on the NKI trace cache
(``NKI_ENABLE_TRACE_CACHE=1``), and the cache persists on disk across
processes. Its key covers a kernel's source and arguments, but not the
values a kernel body reads from an enclosing function (closure cells) or
from module globals. So an ``@nki.jit`` kernel that is specialised through
such a value -- the classic case being a kernel factory,
``def make(c): @nki.jit def k(x): ... c ...`` -- silently reuses the first
specialisation's trace. A module global that differs between processes
(say, one derived from an environment variable) goes stale the same way
through the on-disk cache.

These CPU tests make sure no kernel in this package depends on such a value:

* ``test_kernels_read_only_shared_constants`` parses every module, finds
  each ``@nki.jit`` kernel, and checks every free name its body reads
  (following module-level helper functions, which are traced inline). A
  name passes if it is a builtin, an import, a function or class, or a
  module global bound only to a constant expression and never rebound with
  ``global``. Anything else must be passed to the kernel as an argument
  (a static ``nki_op`` argument on the native path) or be listed in
  ``ALLOWED`` with the reason it is the same in every process.
* ``test_kernel_closure_cells_are_modules`` imports the kernel modules
  against a stub ``nki`` and checks each kernel function's
  ``__code__.co_freevars`` / ``__closure__`` at runtime: the only cells
  allowed are imported modules.

See trnsci/trnsci#35 for the TorchNeuron migration this belongs to.
"""

from __future__ import annotations

import ast
import builtins
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

PACKAGE = "trnrand"
PKG_DIR = Path(__file__).resolve().parents[1] / PACKAGE

# (module path relative to the repo root, kernel or "kernel->helper", name): reason.
# Every entry must still match a finding (see test_allowlist_is_current).
ALLOWED: dict[tuple[str, str, str], str] = {
    # Threefry-2x32 rotation constants (Salmon et al. 2011): a fixed list
    # literal, never mutated, identical in every process.
    **{
        ("trnrand/nki/dispatch.py", k, "THREEFRY_ROTATIONS"): "fixed Threefry rotation table"
        for k in (
            "threefry_bulk32_normal_kernel->_threefry_bulk32_body",
            "threefry_bulk32_uniform_kernel->_threefry_bulk32_body",
        )
    },
    # (quotient, remainder) splits of THREEFRY_ROTATIONS, built once at import
    # with tuple(<genexpr>): derived only from the fixed table above.
    **{
        ("trnrand/nki/dispatch.py", k, "_THREEFRY_ROT_QR"): "derived from the fixed rotation table"
        for k in (
            "threefry4x32_kernel",
            "threefry_normal_kernel",
            "threefry_streaming_normal_kernel",
            "threefry_streaming_uniform_kernel",
            "threefry_bulk_normal_kernel->_threefry_bulk_body",
            "threefry_bulk_uniform_kernel->_threefry_bulk_body",
        )
    },
    **{
        (
            "trnrand/nki/dispatch.py",
            k,
            "_THREEFRY_ROT_QR16",
        ): "derived from the fixed rotation table"
        for k in (
            "threefry_bulk16_normal_kernel->_threefry_bulk16_body",
            "threefry_bulk16_uniform_kernel->_threefry_bulk16_body",
        )
    },
    # Joe-Kuo Sobol direction numbers: a tuple of tuples computed at import by
    # _init_sobol_directions() from a literal table (checked against
    # torch.quasirandom.SobolEngine in tests/test_quasi.py).
    (
        "trnrand/nki/dispatch.py",
        "sobol_gray_code_kernel",
        "_SOBOL_DIR_VECS",
    ): "fixed Joe-Kuo direction table",
}

# Fewest kernels the scan must find, so a broken detector cannot pass vacuously.
MIN_KERNELS = 14

# --------------------------------------------------------------------------
# AST lint
# --------------------------------------------------------------------------

_BUILTINS = frozenset(dir(builtins))
# Calls that build an immutable value from constant arguments.
_PURE_CALLS = frozenset(
    {
        "int",
        "float",
        "bool",
        "str",
        "complex",
        "tuple",
        "frozenset",
        "len",
        "min",
        "max",
        "abs",
        "round",
    }
)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kernel: str
    name: str
    problem: str

    @property
    def key(self):
        return (self.path, self.kernel, self.name)

    def __str__(self):
        return (
            f"{self.path}:{self.line}: kernel {self.kernel!r} reads {self.name!r}: {self.problem}"
        )


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_jit(node):
    """``nki.jit`` / ``jit`` / ``nki.jit(...)`` (decorator or call target)."""
    if isinstance(node, ast.Call):
        node = node.func
    name = _dotted(node)
    return name is not None and (name == "jit" or name.endswith("nki.jit"))


def _body_nodes(fn):
    body = fn.body if isinstance(fn.body, list) else [fn.body]
    return ast.walk(ast.Module(body=body, type_ignores=[]))


def _scope_bindings(scope):
    """name -> set of binding kinds ("import" or "other") made directly in a scope."""
    out: dict[str, set[str]] = {}

    def add(name, kind):
        out.setdefault(name, set()).add(kind)

    if isinstance(scope, _FUNCS):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                add(arg.arg, "other")
        stack = list(scope.body) if isinstance(scope.body, list) else [scope.body]
    else:
        stack = list(scope.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(n.name, "other")
            stack.extend(n.decorator_list)
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            add(n.id, "other")
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                add((al.asname or al.name).split(".")[0], "import")
        elif isinstance(n, ast.ExceptHandler) and n.name:
            add(n.name, "other")
        stack.extend(ast.iter_child_nodes(n))
    return out


def _kernel_locals(fn):
    names = set(_scope_bindings(fn))
    for n in _body_nodes(fn):
        if isinstance(n, _FUNCS):
            names |= set(_scope_bindings(n))
        elif isinstance(n, ast.comprehension):
            names |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
        elif isinstance(n, ast.NamedExpr):
            names.add(n.target.id)
    return names


class _Module:
    def __init__(self, path: Path, rel: str, source: str | None = None):
        self.rel = rel
        self.tree = ast.parse(path.read_text() if source is None else source, filename=rel)
        self.parents = {c: n for n in ast.walk(self.tree) for c in ast.iter_child_nodes(n)}
        self.bindings: dict[str, list[tuple[str, ast.AST | None]]] = {}
        self.functions: dict[str, ast.AST] = {}
        self._collect(self.tree.body)
        self.rebound = {
            name for n in ast.walk(self.tree) if isinstance(n, ast.Global) for name in n.names
        }

    # Module-scope bindings, including those under module-level if/try/with.
    def _collect(self, body):
        for st in body:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                for al in st.names:
                    self._bind((al.asname or al.name).split(".")[0], "import", None)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._bind(st.name, "function", st)
                self.functions.setdefault(st.name, st)
            elif isinstance(st, ast.ClassDef):
                self._bind(st.name, "class", st)
            elif isinstance(st, ast.Assign):
                for t in st.targets:
                    self._bind_target(t, st.value)
            elif isinstance(st, ast.AnnAssign):
                self._bind_target(st.target, st.value)
            elif isinstance(st, ast.AugAssign):
                self._bind_target(st.target, None)
            elif isinstance(st, ast.If):
                self._collect(st.body)
                self._collect(st.orelse)
            elif isinstance(st, ast.Try):
                self._collect(st.body)
                for h in st.handlers:
                    self._collect(h.body)
                self._collect(st.orelse)
                self._collect(st.finalbody)
            elif isinstance(st, (ast.For, ast.While, ast.With)):
                for n in ast.walk(st):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        self._bind(n.id, "assign", None)

    def _bind(self, name, kind, value):
        self.bindings.setdefault(name, []).append((kind, value))

    def _bind_target(self, target, value):
        if isinstance(target, ast.Name):
            self._bind(target.id, "assign", value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            elts = getattr(value, "elts", None)
            same = elts is not None and len(elts) == len(target.elts)
            for i, t in enumerate(target.elts):
                self._bind_target(t, elts[i] if same else None)

    def is_const_name(self, name, seen=frozenset()):
        if name in self.rebound or name in seen or name not in self.bindings:
            return False
        for kind, value in self.bindings[name]:
            if kind == "assign" and (value is None or not self.is_const(value, seen | {name})):
                return False
        return True

    def is_const(self, e, seen=frozenset()):
        """True if ``e`` is an immutable value computed only from constants."""
        sub = []
        if isinstance(e, ast.Constant):
            return True
        if isinstance(e, ast.Name):
            return self.is_const_name(e.id, seen) if e.id in self.bindings else e.id in _BUILTINS
        if isinstance(e, ast.Tuple):
            sub = e.elts
        elif isinstance(e, ast.UnaryOp):
            sub = [e.operand]
        elif isinstance(e, ast.BinOp):
            sub = [e.left, e.right]
        elif isinstance(e, ast.BoolOp):
            sub = e.values
        elif isinstance(e, ast.Compare):
            sub = [e.left, *e.comparators]
        elif isinstance(e, ast.IfExp):
            sub = [e.test, e.body, e.orelse]
        elif isinstance(e, ast.Attribute):
            sub = [e.value]  # e.g. nl.float32: attribute of an import
        elif isinstance(e, ast.Subscript):
            sub = [e.value, e.slice]
        elif isinstance(e, ast.Slice):
            sub = [x for x in (e.lower, e.upper, e.step) if x is not None]
        elif isinstance(e, ast.Call):
            fn = _dotted(e.func) or ""
            pure = (fn in _PURE_CALLS and fn not in self.bindings) or fn.startswith("math.")
            if not pure:
                return False
            sub = [*e.args, *(k.value for k in e.keywords)]
        else:
            return False
        return all(self.is_const(x, seen) for x in sub)

    def enclosing_functions(self, node):
        out, p = [], self.parents.get(node)
        while p is not None:
            if isinstance(p, _FUNCS):
                out.append(p)
            p = self.parents.get(p)
        return out

    def kernels(self):
        found = {}
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                map(_is_jit, n.decorator_list)
            ):
                found[id(n)] = n
            elif (
                isinstance(n, ast.Call)
                and _is_jit(n.func)
                and n.args
                and isinstance(n.args[0], ast.Name)
            ):
                # nki.jit(fn) call form: mark the function it wraps.
                encl = self.enclosing_functions(n)
                scope = encl[0] if encl else self.tree
                for m in ast.walk(scope):
                    if (
                        isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and m.name == n.args[0].id
                    ):
                        found[id(m)] = m
        return list(found.values())


def _check(mod: _Module, fn, label, findings, visited):
    if id(fn) in visited:
        return
    visited.add(id(fn))
    declared_global = {x for n in _body_nodes(fn) if isinstance(n, ast.Global) for x in n.names}
    local = _kernel_locals(fn) - declared_global
    for n in _body_nodes(fn):
        if isinstance(n, ast.Nonlocal):
            for name in n.names:
                findings.append(
                    Finding(mod.rel, n.lineno, label, name, "nonlocal closure variable")
                )
    # Defaults are evaluated once per `def`: a per-instance default is a closure in disguise.
    a = fn.args
    for d in [*a.defaults, *(x for x in a.kw_defaults if x is not None)]:
        if not mod.is_const(d):
            findings.append(
                Finding(mod.rel, d.lineno, label, ast.unparse(d), "non-constant default argument")
            )
    enclosing = mod.enclosing_functions(fn)
    seen = set()
    for n in _body_nodes(fn):
        if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
            continue
        name = n.id
        if name in local or name in seen:
            continue
        seen.add(name)
        if name not in declared_global:
            owner = next((f for f in enclosing if name in _scope_bindings(f)), None)
            if owner is not None:
                if _scope_bindings(owner)[name] != {"import"}:  # a module object is shared
                    findings.append(
                        Finding(
                            mod.rel,
                            n.lineno,
                            label,
                            name,
                            f"closure variable of {getattr(owner, 'name', '<lambda>')!r}",
                        )
                    )
                continue
        if name not in mod.bindings:
            if name not in _BUILTINS:
                findings.append(Finding(mod.rel, n.lineno, label, name, "unresolved name"))
            continue
        kinds = {k for k, _ in mod.bindings[name]}
        if name in mod.rebound:
            findings.append(
                Finding(mod.rel, n.lineno, label, name, "module global rebound via `global`")
            )
        elif kinds <= {"import", "class"}:
            pass
        elif kinds <= {"import", "class", "function"}:
            # Helpers a kernel calls are traced inline: they must obey the same rule.
            _check(mod, mod.functions[name], f"{label}->{name}", findings, visited)
        elif not mod.is_const_name(name):
            findings.append(
                Finding(
                    mod.rel,
                    n.lineno,
                    label,
                    name,
                    "module global that is not a constant expression",
                )
            )


def scan_source(rel: str, source: str):
    mod = _Module(Path(rel), rel, source)
    findings = []
    kernels = []
    for k in mod.kernels():
        _check(mod, k, k.name, findings, set())
        encl = mod.enclosing_functions(k)
        # A zero-argument module-level factory can be called to build the kernel.
        outer = encl[-1] if encl else None
        if outer is not None and (mod.enclosing_functions(outer) or _has_params(outer)):
            outer = None
        kernels.append((k, getattr(outer, "name", None)))
    return kernels, findings


def _has_params(fn):
    a = fn.args
    return bool(a.posonlyargs or a.args or a.kwonlyargs or a.vararg or a.kwarg)


def scan_package(pkg_dir: Path = PKG_DIR):
    root = pkg_dir.parent
    kernels, findings = [], []
    for path in sorted(pkg_dir.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        ks, fs = scan_source(rel, path.read_text())
        kernels += [(rel, k, factory) for k, factory in ks]
        findings += fs
    return kernels, findings


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_kernels_found():
    kernels, _ = scan_package()
    assert len(kernels) >= MIN_KERNELS, [f"{r}:{k.name}" for r, k, _ in kernels]


def test_kernels_read_only_shared_constants():
    _, findings = scan_package()
    bad = [str(f) for f in findings if f.key not in ALLOWED]
    assert not bad, (
        "@nki.jit kernels read values the NKI trace cache does not key on "
        "(aws-neuron/aws-neuron-sdk#1454). Pass them as kernel arguments, or "
        "add them to ALLOWED with a reason:\n" + "\n".join(bad)
    )


def test_allowlist_is_current():
    _, findings = scan_package()
    keys = {f.key for f in findings}
    assert not [k for k in ALLOWED if k not in keys], "stale ALLOWED entries"


_FACTORY = """
import nki
import nki.language as nl
P = 128
SCALE = (2.0, 3.0)
_TABLE = [1, 2]
_FLAG = __import__("os").environ.get("X", "1") == "1"

def make_kernel(i):
    import nki.isa as nisa
    c = float(i + 1)
    @nki.jit
    def kernel(x):
        nisa.tensor_scalar(x, x, op0=nl.multiply, operand0=c)
        return x
    return kernel

@nki.jit
def clean(x, n=P):
    return helper(x) * SCALE[0] * P

def helper(x):
    return x + _TABLE[0]

@nki.jit
def env_flag(x):
    return x if _FLAG else -x

def make_default(c):
    @nki.jit
    def kernel_d(x, c=c):
        return x * c
    return kernel

def make_wrapped(c):
    def body(x):
        return x * c
    return nki.jit(body)
"""


def test_lint_catches_closure_specialisation():
    """The lint flags the #1454 pattern and its relatives, and passes clean kernels."""
    kernels, findings = scan_source("m.py", _FACTORY)
    assert {k.name for k, _ in kernels} == {"kernel", "clean", "env_flag", "kernel_d", "body"}
    got = {(f.kernel, f.name) for f in findings}
    assert got == {
        ("kernel", "c"),  # the #1454 factory: closure constant
        ("clean->helper", "_TABLE"),  # mutable global, read through a helper
        ("env_flag", "_FLAG"),  # differs between processes
        ("kernel_d", "c"),  # per-instance default argument
        ("body", "c"),  # nki.jit(fn) call form
    }, findings
    # The import captured by `make_kernel` (nisa) is a shared module: not flagged.
    assert ("kernel", "nisa") not in got


_RUNTIME_PROBE = r"""
import importlib, importlib.abc, importlib.machinery, json, sys, types

class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return _Any()
    def __getitem__(self, k): return _Any()
    def __iter__(self): return iter(())
    def __int__(self): return 128
    __index__ = __int__
    def __float__(self): return 128.0
    def __bool__(self): return True
    def __eq__(self, o): return False
    def __hash__(self): return 0
_ARITH = ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "and", "or")
for _op in _ARITH + ("lshift", "rshift"):
    setattr(_Any, f"__{_op}__", lambda self, o: _Any())
    setattr(_Any, f"__r{_op}__", lambda self, o: _Any())
for _op in ("lt", "le", "gt", "ge"):
    setattr(_Any, f"__{_op}__", lambda self, o: False)

JITTED = []

def jit(fn=None, **kw):
    if fn is None:
        return jit
    JITTED.append(fn)
    return fn

class _StubModule(types.ModuleType):
    __path__ = []
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return _Any()

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == "nki" or name.startswith("nki."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = _StubModule(spec.name)
        if spec.name == "nki":
            m.jit = jit
        return m
    def exec_module(self, module): pass

sys.meta_path.insert(0, _Finder())
mods, factories = json.loads(sys.argv[1])
for name in mods:
    importlib.import_module(name)
for mod_name, fn_name in factories:
    getattr(importlib.import_module(mod_name), fn_name)()
out = []
for fn in JITTED:
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
    bad = sorted(n for n, c in cells.items() if not isinstance(c.cell_contents, types.ModuleType))
    out.append([fn.__module__, fn.__qualname__, sorted(cells), bad])
print(json.dumps(out))
"""


def test_kernel_closure_cells_are_modules():
    """Runtime check: every kernel's closure holds only imported modules."""
    kernels, _ = scan_package()
    root = PKG_DIR.parent
    mods = sorted({rel[:-3].replace("/", ".") for rel, _, _ in kernels})
    factories = sorted({(rel[:-3].replace("/", "."), f) for rel, _, f in kernels if f})
    res = subprocess.run(
        [sys.executable, "-c", _RUNTIME_PROBE, json.dumps([mods, factories])],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    seen = json.loads(res.stdout.strip().splitlines()[-1])
    assert len(seen) >= MIN_KERNELS, seen
    bad = [f"{m}.{q}: closure cells {b}" for m, q, _, b in seen if b]
    assert not bad, "kernel closures capture non-module values (#1454):\n" + "\n".join(bad)
