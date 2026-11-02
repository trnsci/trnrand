"""Checkpoint / resume with advance, advance_to and get_state (#76).

`advance_to(position())` is a resume only with key_mode="counter": the key is
fixed by the seed, so the counter is the whole stream state. With the default
key_mode="draw" every NKI draw takes a fresh key from the host torch.Generator;
that key sequence depends on the number of draws, which `position()` does not
record, so advance / advance_to cannot resume it and warn (`AdvanceWarning`).
`get_state()` / `set_state()` resume in both modes and on both paths.

Each NKI test runs on three tiers, chosen by the `nki_path` fixture:

- ``standin`` (CPU, always): `threefry_{uniform,normal}_nki` are replaced by a
  Threefry reference with the kernels' counter layout, so the host-side
  key / counter bookkeeping is exercised without nki;
- ``simulator`` (`TRNRAND_USE_SIMULATOR=1`, nki installed): the real kernels
  under `nki.simulate`;
- ``device`` (`-m neuron`, native TorchNeuron dispatch): the real kernels on
  Trainium.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import trnrand
from trnrand import AdvanceWarning, Generator, distributions
from trnrand.nki import HAS_NKI, dispatch

DRAWS = {"uniform": trnrand.uniform, "normal": trnrand.normal}
SIZES = [512, 1000, 4096]  # the sizes of the Trn2 report in #76
BATCH = 512
LANES = 128


def _ref_words(n: int, seed: int, offset: int) -> torch.Tensor:
    """Threefry words in the kernels' (batch, lane, word) order, batch = offset + b."""
    n_b = max(1, -(-n // BATCH))
    ctr = torch.zeros(n_b, LANES, 4, dtype=torch.int64)
    ctr[:, :, 0] = torch.arange(LANES, dtype=torch.int64)
    ctr[:, :, 1] = ((torch.arange(n_b, dtype=torch.int64) + offset) & 0xFFFFFF).reshape(-1, 1)
    key = torch.tensor([seed & 0xFFFFFF, (seed >> 24) & 0xFFFFFF, 0, 0], dtype=torch.int64)
    out = dispatch.threefry4x32_reference(ctr.reshape(-1, 4), key.expand(n_b * LANES, 4))
    return out.reshape(-1)[:n]


def _standin_uniform(n_elements, seed=0, counter_offset=0):
    w = _ref_words(n_elements, seed, counter_offset)
    return ((w & 0xFFFFFF).to(torch.float64) / 2**24).to(torch.float32)


def _standin_normal(n_elements, seed=0, counter_offset=0):
    # Distinct from the uniform stand-in; only the key / counter layout matters here.
    return _standin_uniform(n_elements, seed, counter_offset) * 2.0 - 1.0


def _native() -> bool:
    return HAS_NKI and not dispatch._use_simulator() and dispatch._get_dispatch_path() == "native"


def _simulator() -> bool:
    return HAS_NKI and dispatch._use_simulator()


@pytest.fixture(
    params=[
        "standin",
        pytest.param("simulator", marks=pytest.mark.nki_simulator),
        pytest.param("device", marks=pytest.mark.neuron),
    ]
)
def nki_path(request, monkeypatch):
    """Route trnrand draws through the NKI host path on the given tier."""
    if request.param == "standin":
        monkeypatch.setattr(distributions, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "threefry_uniform_nki", _standin_uniform)
        monkeypatch.setattr(dispatch, "threefry_normal_nki", _standin_normal)
    elif request.param == "simulator" and not _simulator():
        pytest.skip("TRNRAND_USE_SIMULATOR=1 and nki required")
    elif request.param == "device" and not _native():
        pytest.skip("requires TorchNeuron (native) dispatch path")
    if request.param == "standin":  # backend "auto" + HAS_NKI routes to NKI
        yield request.param
        return
    prev = trnrand.get_backend()
    trnrand.set_backend("nki")
    yield request.param
    trnrand.set_backend(prev)


# ── key_mode="counter": advance / advance_to resume ──────────────────────────


@pytest.mark.parametrize("kind", list(DRAWS))
@pytest.mark.parametrize("n", SIZES)
def test_counter_mode_advance_to_position_resumes(nki_path, kind, n):
    draw = DRAWS[kind]
    g = Generator(seed=11, key_mode="counter")
    draw(n, generator=g)
    draw(n, generator=g)
    pos = g.position()
    ref = draw(n, generator=g)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AdvanceWarning)
        h = Generator(seed=11, key_mode="counter")
        h.advance_to(pos)
        k = Generator(seed=11, key_mode="counter")
        k.advance(pos)
    assert torch.equal(draw(n, generator=h), ref)
    assert torch.equal(draw(n, generator=k), ref)


@pytest.mark.parametrize("kind", list(DRAWS))
def test_counter_mode_resume_across_partition_size(nki_path, kind):
    """position() is the global counter: resume a 1-chip checkpoint on 2 ranks."""
    draw = DRAWS[kind]
    n = 2 * BATCH
    g = Generator(seed=5, key_mode="counter")
    draw(1000, generator=g)
    draw(n, generator=g)
    pos = g.position()
    ref = draw(2 * n, generator=g)

    parts = []
    for r in range(2):
        h = Generator(seed=5, partition_rank=r, partition_size=2, key_mode="counter")
        h.advance_to(pos)
        parts.append(draw(n, generator=h))
    assert torch.equal(torch.cat(parts), ref)


# ── key_mode="draw": warn, and get_state / set_state is the recipe ───────────


def test_draw_mode_advance_warns():
    g = Generator(seed=1)
    assert g.key_mode == "draw"
    with pytest.warns(AdvanceWarning, match="get_state"):
        g.advance_to(1024)
    with pytest.warns(AdvanceWarning, match="key_mode"):
        g.advance(512)
    assert g.position() == 3 * BATCH  # the counter still moves exactly as before


def test_counter_mode_advance_does_not_warn():
    g = Generator(seed=1, key_mode="counter")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.advance_to(1024)
        g.advance(512)
    assert g.position() == 3 * BATCH


def test_advance_warning_is_filterable():
    g = Generator(seed=1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", AdvanceWarning)
        g.advance_to(512)
    assert not rec


@pytest.mark.parametrize("kind", list(DRAWS))
def test_draw_mode_advance_to_does_not_resume(nki_path, kind):
    """Why draw mode warns: the resumed generator draws with the first host key."""
    draw = DRAWS[kind]
    n = 1000
    g = Generator(seed=11)
    draw(n, generator=g)
    draw(n, generator=g)
    pos = g.position()
    ref = draw(n, generator=g)

    h = Generator(seed=11)
    with pytest.warns(AdvanceWarning):
        h.advance_to(pos)
    assert not torch.equal(draw(n, generator=h), ref)


@pytest.mark.parametrize("key_mode", ["draw", "counter"])
@pytest.mark.parametrize("kind", list(DRAWS))
@pytest.mark.parametrize("n", SIZES)
def test_get_state_recipe_resumes(nki_path, key_mode, kind, n):
    """The documented checkpoint recipe (position() docstring), both key modes."""
    draw = DRAWS[kind]
    g = Generator(seed=11, key_mode=key_mode)
    draw(n, generator=g)
    draw(n, generator=g)
    state = g.get_state().clone()
    ref = torch.cat([draw(n, generator=g), draw(n, generator=g)])

    new = Generator(seed=11, key_mode=key_mode)
    new.set_state(state)
    assert torch.equal(torch.cat([draw(n, generator=new), draw(n, generator=new)]), ref)


@pytest.mark.parametrize("key_mode", ["draw", "counter"])
@pytest.mark.parametrize("kind", list(DRAWS))
def test_get_state_recipe_resumes_pytorch_path(key_mode, kind):
    """On the PyTorch path advance has no effect; get_state / set_state resumes."""
    draw = DRAWS[kind]
    prev = trnrand.get_backend()
    trnrand.set_backend("pytorch")
    try:
        g = Generator(seed=11, key_mode=key_mode)
        draw(1000, generator=g)
        state = g.get_state().clone()
        ref = draw(1000, generator=g)
        new = Generator(seed=11, key_mode=key_mode)
        new.set_state(state)
        assert torch.equal(draw(1000, generator=new), ref)
    finally:
        trnrand.set_backend(prev)


# ── Streams without advance are unchanged ────────────────────────────────────


@pytest.mark.parametrize("key_mode", ["draw", "counter"])
@pytest.mark.parametrize("kind", list(DRAWS))
def test_stream_without_advance_unchanged(nki_path, key_mode, kind):
    """No advance call: keys and counters are exactly what 0.12.1 used."""
    draw = DRAWS[kind]
    fn = dispatch.threefry_uniform_nki if kind == "uniform" else dispatch.threefry_normal_nki
    g = Generator(seed=2**30 + 7, key_mode=key_mode)
    twin = torch.Generator().manual_seed(2**30 + 7)
    off = 0
    for n in (1000, 512, 4096):
        got = draw(n, generator=g)
        if key_mode == "counter":
            key = g._key48
        else:
            key = int(torch.randint(0, 2**24, (1,), generator=twin).item())
        assert torch.equal(got, fn(n, seed=key, counter_offset=off))
        off += -(-n // BATCH)
    assert g.position() == off * BATCH
