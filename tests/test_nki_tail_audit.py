"""Tail-tile audit (trnsci/trnsci#35, after trnfft#100): awkward sizes on hardware.

Every NKI path is checked against an independent CPU oracle, not against another
device path: the Threefry stream is rebuilt from `threefry4x32_reference` with the
kernels' counter layout (lane, (batch + counter_offset) & 0xFFFFFF, 0, 0), the
Sobol points from `sobol_reference`, Halton from the CPU `quasi` path and the
Box-Muller / Philox kernels from their CPU twins. Sizes are deliberately not tile
multiples (1, 3, 127, 129, 200, 1000, 4097, 128·512 + 1, ...), offsets are odd,
and draws are chained so a dropped, duplicated or zero-filled tail shows up.

`CASES` is shared with the standalone hardware probe, which prints one line per
case (OK / SILENT / LOUD) instead of failing fast.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from trnrand.nki import HAS_NKI
from trnrand.nki.dispatch import (
    UINT32_MASK,
    _get_dispatch_path,
    _use_simulator,
    box_muller_cpu,
    philox4x32_reference,
    sobol_reference,
    threefry4x32_reference,
)


def _native() -> bool:
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

LANES = 128
BATCH = LANES * 4  # values per Threefry batch
ODD = [1, 3, 127, 129, 200, 1000, 4097, 128 * 512 + 1]
BIG = [262_145, 4_194_305]  # bulk C > 1, and more than one bulk launch


# ── Oracles ──────────────────────────────────────────────────────────────────


def _words_ref(n: int, seed: int, offset: int) -> torch.Tensor:
    """(n,) int64 Threefry words in the kernels' per-tile (batch, lane, word) order."""
    n_b = max(1, -(-n // BATCH))
    ctr = torch.zeros(n_b, LANES, 4, dtype=torch.int64)
    ctr[:, :, 0] = torch.arange(LANES, dtype=torch.int64)
    ctr[:, :, 1] = ((torch.arange(n_b, dtype=torch.int64) + offset) & 0xFFFFFF).reshape(-1, 1)
    key = torch.tensor([seed & 0xFFFFFF, (seed >> 24) & 0xFFFFFF, 0, 0], dtype=torch.int64)
    out = threefry4x32_reference(ctr.reshape(-1, 4), key.expand(n_b * LANES, 4))
    return out.reshape(-1)[:n]


def uniform_ref(n: int, seed: int, offset: int = 0) -> torch.Tensor:
    return ((_words_ref(n, seed, offset) & 0xFFFFFF).to(torch.float64) / 2**24).to(torch.float32)


def normal_ref(n: int, seed: int, offset: int = 0) -> torch.Tensor:
    m = -(-n // 4) * 4
    u = (_words_ref(m, seed, offset) & 0xFFFFFF).to(torch.float64) / 2**24
    u = torch.clamp(u, min=1e-7).reshape(-1, 2)
    r = torch.sqrt(-2.0 * torch.log(u[:, 0]))
    th = 2.0 * math.pi * u[:, 1]
    z = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1).reshape(-1)
    return z[:n].to(torch.float32)


def halton_ref(n: int, d: int, start: int = 0) -> np.ndarray:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:d]
    out = np.zeros((n, d))
    for j, p in enumerate(primes):
        idx = np.arange(start + 1, start + n + 1, dtype=np.int64)
        f = 1.0
        while idx.any():
            f /= p
            out[:, j] += f * (idx % p)
            idx //= p
    return out


# ── Comparison ───────────────────────────────────────────────────────────────


def compare(got: torch.Tensor, want: torch.Tensor, atol: float = 0.0) -> str | None:
    """None if equal (within atol), else a short description of what differs."""
    got = torch.as_tensor(got).detach().cpu()
    want = torch.as_tensor(want).detach().cpu()
    if got.shape != want.shape:
        return f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    g = got.to(torch.float64).reshape(-1)
    w = want.to(torch.float64).reshape(-1)
    bad = ~torch.isclose(g, w, rtol=atol, atol=atol, equal_nan=True) if atol else g != w
    if not bool(bad.any()):
        return None
    idx = torch.nonzero(bad).reshape(-1)
    err = (g - w).abs()[idx]
    zero = int((g[idx] == 0).sum())
    return (
        f"{idx.numel()}/{g.numel()} wrong, first {int(idx[0])} last {int(idx[-1])}, "
        f"max err {float(err.max()):.3g}, {zero} zeros"
    )


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _nki_backend():
    import trnrand

    prev = trnrand.get_backend()
    trnrand.set_backend("nki")
    try:
        yield
    finally:
        trnrand.set_backend(prev)


# ── Cases: (kernel, shape) -> thunk returning (got, want, atol) ──────────────

ATOL_NORMAL = 1e-3  # ScalarE log / sqrt / sin / cos against float64


def _draw(kind, n, seed, offset, env):
    from trnrand.nki import dispatch

    fn = dispatch.threefry_uniform_nki if kind == "uniform" else dispatch.threefry_normal_nki
    ref = uniform_ref if kind == "uniform" else normal_ref
    with _env(**env):
        got = fn(n, seed=seed, counter_offset=offset)
    return got, ref(n, seed, offset), (0.0 if kind == "uniform" else ATOL_NORMAL)


_ROUTES = {
    "bulk32": {},
    "bulk16": {"TRNRAND_LIMB_BITS": "16"},
    "bulk8": {"TRNRAND_LIMB_BITS": "8"},
    "streaming": {"TRNRAND_BULK_KERNEL": "0"},
    "per_tile": {"TRNRAND_BULK_STREAMING": "0", "TRNRAND_BULK_KERNEL": "0"},
}


def _public(kind, n, seed):
    import trnrand

    with _nki_backend():
        gen = trnrand.Generator(seed=seed, key_mode="counter")
        key = gen._key48
        got = (trnrand.uniform if kind == "uniform" else trnrand.normal)(n, generator=gen)
    ref = uniform_ref if kind == "uniform" else normal_ref
    return got, ref(n, key, 0), (0.0 if kind == "uniform" else ATOL_NORMAL)


def _chained(kind, sizes, seed):
    """Successive draws on one counter-mode Generator: each starts at the next whole batch."""
    import trnrand

    with _nki_backend():
        gen = trnrand.Generator(seed=seed, key_mode="counter")
        key = gen._key48
        fn = trnrand.uniform if kind == "uniform" else trnrand.normal
        got = torch.cat([fn(m, generator=gen) for m in sizes])
    ref = uniform_ref if kind == "uniform" else normal_ref
    want, off = [], 0
    for m in sizes:
        want.append(ref(m, key, off))
        off += -(-m // BATCH)
    return got, torch.cat(want), (0.0 if kind == "uniform" else ATOL_NORMAL)


def _into(kind, n, seed, chunks=2):
    """`*_into` twice on one Generator; the twin generator replays the key draws."""
    import trnrand
    from trnrand.nki.dispatch import _PROGRAM_TILES

    with _nki_backend():
        gen = trnrand.Generator(seed=seed)
        twin = trnrand.Generator(seed=seed)
        fn = {
            "uniform": trnrand.uniform_into,
            "normal": trnrand.normal_into,
            "exponential": trnrand.exponential_into,
        }[kind]
        got = []
        for _ in range(chunks):
            buf = torch.full((n,), float("nan"))
            fn(buf, generator=gen)
            got.append(buf)
    want, off = [], 0
    for _ in range(chunks):
        key = int(torch.randint(0, 2**24, (1,), generator=twin.torch_generator).item())
        if kind == "normal":
            want.append(normal_ref(n, key, off))
        elif kind == "uniform":
            want.append(uniform_ref(n, key, off))
        else:
            want.append(-torch.log(uniform_ref(n, key, off).to(torch.float64)).float())
        off += -(-n // (_PROGRAM_TILES * BATCH)) * _PROGRAM_TILES
    return torch.cat(got), torch.cat(want), (0.0 if kind == "uniform" else ATOL_NORMAL)


def _program(n_z, n_u, seed):
    import trnrand

    with _nki_backend():
        prog = trnrand.Generator(seed=seed).new_program().normal(n_z, out="z")
        prog = prog.uniform(n_u, out="u").build()
        z, u = torch.empty(n_z), torch.empty(n_u)
        prog.stream_into({"z": z, "u": u})
    from trnrand.nki.dispatch import _PROGRAM_TILES

    off_u = -(-n_z // (_PROGRAM_TILES * BATCH)) * _PROGRAM_TILES
    want_z, want_u = normal_ref(n_z, seed, 0), uniform_ref(n_u, seed, off_u)
    return torch.cat([z, u]), torch.cat([want_z, want_u]), ATOL_NORMAL


def _partitioned(kind, n, P, calls, key_mode):
    """P ranks x `calls` draws of n each, in (call, rank) order, against one chip drawing
    P·n per call: the ranks must tile the single-chip stream on every call, not
    only the first (each call advances the counter past all P slices)."""
    import trnrand

    fn = trnrand.uniform if kind == "uniform" else trnrand.normal
    with _nki_backend():
        gens = [
            trnrand.Generator(seed=21, partition_rank=r, partition_size=P, key_mode=key_mode)
            for r in range(P)
        ]
        per_rank = [[fn(n, generator=g) for _ in range(calls)] for g in gens]
        one = trnrand.Generator(seed=21, key_mode=key_mode)
        want = torch.cat([fn(P * n, generator=one) for _ in range(calls)])
    got = torch.cat([per_rank[r][c] for c in range(calls) for r in range(P)])
    return got, want, 0.0


def _cores(kind, n, seed, offset, k):
    from trnrand.nki import dispatch

    if dispatch._visible_cores() < k:
        raise pytest.skip.Exception(f"needs {k} visible logical cores")
    dispatch.set_cores(k)
    try:
        return _draw(kind, n, seed, offset, {"TRNRAND_CORES_MIN_ELEMENTS": "0"})
    finally:
        dispatch.set_cores(None)


def _sobol(n, d, start):
    from trnrand.nki.dispatch import sobol_nki

    return sobol_nki(n, d, seed=0, start_index=start), sobol_reference(n, d, start), 0.0


def _sobol_public(n, d):
    import trnrand

    with _nki_backend():
        got = trnrand.sobol(n, d, seed=0)
    return got, sobol_reference(n, d, 0), 0.0


def _halton(n, d, start):
    from trnrand.nki.dispatch import halton_nki

    got = halton_nki(n, d, start_index=start)
    return got, torch.from_numpy(halton_ref(n, d, start)).float(), 1e-6


def _halton_chunks(sizes, d):
    from trnrand.nki.dispatch import halton_nki

    got, start = [], 0
    for m in sizes:
        got.append(halton_nki(m, d, start_index=start))
        start += m
    return torch.cat(got), torch.from_numpy(halton_ref(start, d, 0)).float(), 1e-6


def _halton_public(n, d):
    import trnrand

    with _nki_backend():
        got = trnrand.halton(n, d)
    return got, torch.from_numpy(halton_ref(n, d, 0)).float(), 1e-6


def _philox(lanes, base):
    from trnrand.nki.dispatch import philox4x32_nki

    cl = torch.arange(base, base + lanes, dtype=torch.int32)
    kl = torch.full((lanes,), 0x1234, dtype=torch.int32)
    kh = torch.full((lanes,), 0x56, dtype=torch.int32)
    got = philox4x32_nki(cl, kl, kh)
    ctr = torch.zeros(lanes, 4, dtype=torch.int64)
    ctr[:, 0] = cl.to(torch.int64)
    key = torch.tensor([0x1234, 0x56], dtype=torch.int64).expand(lanes, 2)
    want = (philox4x32_reference(ctr, key) & UINT32_MASK).reshape(-1)
    return got.to(torch.int64) & UINT32_MASK, want, 0.0


def _box_muller(pairs):
    from trnrand.nki.dispatch import box_muller_nki

    g = torch.Generator().manual_seed(pairs)
    u = torch.rand(2 * pairs, generator=g, dtype=torch.float32).clamp(min=1e-6)
    return box_muller_nki(u), box_muller_cpu(u.to(torch.float64)).float(), ATOL_NORMAL


def _build_cases():
    cases = {}
    for kind in ("uniform", "normal"):
        for route, env in _ROUTES.items():
            sizes = ODD + (BIG if route in ("bulk32", "streaming") else [])
            if route == "per_tile":
                sizes = [1, 3, 127, 129, 200, 1000, 4097]
            for n in sizes:
                for off in (0, 7):
                    cases[f"{kind}_{route}[n={n},off={off}]"] = (
                        lambda kind=kind, n=n, off=off, env=env: _draw(kind, n, 99, off, env)
                    )
        cases[f"{kind}_bulk32[n=4097,off=2^24-3]"] = lambda kind=kind: _draw(
            kind, 4097, 5, 2**24 - 3, {}
        )
        for n in ODD + [0]:
            cases[f"{kind}_public[n={n}]"] = lambda kind=kind, n=n: _public(kind, n, 42)
        cases[f"{kind}_chained[1,3,129,4097,511,513]"] = lambda kind=kind: _chained(
            kind, [1, 3, 129, 4097, 511, 513], 42
        )
        for k in (2, 4):
            for n in (4097, 1_048_577, 4_194_305 + 2048):
                cases[f"{kind}_cores{k}[n={n},off=3]"] = lambda kind=kind, n=n, k=k: _cores(
                    kind, n, 7, 3, k
                )
        for mode in ("counter", "draw"):
            cases[f"{kind}_partitioned[P=2,n=1536,calls=2,{mode}]"] = lambda kind=kind, mode=mode: (
                _partitioned(kind, 1536, 2, 2, mode)
            )
    for kind in ("uniform", "normal", "exponential"):
        for n in (1, 129, 1000, 16_385):
            cases[f"{kind}_into[n={n}x2]"] = lambda kind=kind, n=n: _into(kind, n, 11)
    cases["program[z=1000,u=16385]"] = lambda: _program(1000, 16_385, 3)
    for n in (1, 127, 129, 200, 1000, 4097):
        for d in (1, 3, 10):
            cases[f"sobol[n={n},d={d},start=0]"] = lambda n=n, d=d: _sobol(n, d, 0)
            cases[f"halton[n={n},d={d},start=0]"] = lambda n=n, d=d: _halton(n, d, 0)
        cases[f"sobol[n={n},d=7,start=129]"] = lambda n=n: _sobol(n, 7, 129)
        cases[f"halton[n={n},d=7,start=129]"] = lambda n=n: _halton(n, 7, 129)
    cases["sobol_public[n=1000,d=5]"] = lambda: _sobol_public(1000, 5)
    cases["halton_public[n=1000,d=5]"] = lambda: _halton_public(1000, 5)
    cases["halton_chunks[128,1,129,200]"] = lambda: _halton_chunks([128, 1, 129, 200], 4)
    for lanes in (1, 127, 129, 200):
        cases[f"philox[lanes={lanes},base=5]"] = lambda lanes=lanes: _philox(lanes, 5)
    for pairs in (1, 127, 129, 200):
        cases[f"box_muller[pairs={pairs}]"] = lambda pairs=pairs: _box_muller(pairs)
    return cases


CASES = _build_cases()

# Known, tracked, not a tail bug: Philox integer multiply on hardware
# (aws-neuron-sdk#1308; same xfail as tests/test_nki_native.py).
XFAIL = {k for k in CASES if k.startswith("philox")}


@pytest.mark.parametrize("case", sorted(CASES))
def test_tail(case):
    if case in XFAIL:
        pytest.xfail("Philox integer precision on hardware: aws-neuron-sdk#1308")
    got, want, atol = CASES[case]()
    msg = compare(got, want, atol)
    assert msg is None, f"{case}: {msg}"


def test_sobol_index_limit_is_loud():
    from trnrand.nki.dispatch import sobol_nki

    with pytest.raises(AssertionError):
        sobol_nki(2, 2, start_index=2**24 - 1)
