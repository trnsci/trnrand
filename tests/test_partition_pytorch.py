"""partition_rank / partition_size on the PyTorch path (#99).

Through 0.13.0 the PyTorch path ignored the partition: every rank of
``Generator(seed=s, partition_rank=r, partition_size=P)`` drew identical
samples. It now matches the NKI path's contract (trnsci/trnrand#72): rank r
draws a contiguous slice of the single-generator stream, concatenating the
ranks in order reproduces it, and every draw moves past all P slices.
Single-rank (P = 1) streams are bit-identical to 0.13.0.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

import trnrand as tr


@pytest.fixture(autouse=True)
def _pytorch_backend():
    prev = tr.get_backend()
    tr.set_backend("pytorch")
    yield
    tr.set_backend(prev)


def _into(name):
    def draw(g, *size):
        b = torch.full(size, float("nan"))
        getattr(tr, name)(b, generator=g)
        return b

    return draw


DRAWS = {
    "uniform": lambda g, *s: tr.uniform(*s, low=-1.0, high=3.0, generator=g),
    "normal": lambda g, *s: tr.normal(*s, mean=2.0, std=0.5, generator=g),
    "exponential": lambda g, *s: tr.exponential(*s, rate=2.0, generator=g),
    "poisson": lambda g, *s: tr.poisson(*s, lam=3.0, generator=g),
    "gamma": lambda g, *s: tr.gamma(*s, shape=0.7, generator=g),
    "beta": lambda g, *s: tr.beta(*s, alpha=2.0, beta=3.0, generator=g),
    "chi_squared": lambda g, *s: tr.chi_squared(*s, df=3.0, generator=g),
    "truncated_normal": lambda g, *s: tr.truncated_normal(*s, generator=g),
    "bernoulli": lambda g, *s: tr.bernoulli(*s, p=0.3, generator=g),
    "randint": lambda g, *s: tr.randint(*s, high=100, generator=g),
    "normal_into": _into("normal_into"),
    "uniform_into": _into("uniform_into"),
    "exponential_into": _into("exponential_into"),
}


@pytest.mark.parametrize("P", [2, 4])
@pytest.mark.parametrize("dn", list(DRAWS))
def test_ranks_concatenate_to_single_generator(dn, P):
    d = DRAWS[dn]
    n = 300
    one = tr.Generator(seed=7)
    ranks = [tr.Generator(seed=7, partition_rank=r, partition_size=P) for r in range(P)]
    for call in range(3):
        ref = d(one, P * n)
        parts = [d(g, n) for g in ranks]
        assert not torch.equal(parts[0], parts[1]), f"call {call}: ranks identical"
        assert torch.equal(torch.cat(parts), ref), f"call {call}"


@pytest.mark.parametrize("dn", ["uniform", "normal", "bernoulli", "randint", "normal_into"])
def test_multidim_shape_is_flat_slice(dn):
    d = DRAWS[dn]
    P = 3
    ref = d(tr.Generator(seed=11), P * 4 * 5).reshape(-1)
    for r in range(P):
        part = d(tr.Generator(seed=11, partition_rank=r, partition_size=P), 4, 5)
        assert part.shape == (4, 5)
        assert torch.equal(part.reshape(-1), ref[r * 20 : (r + 1) * 20])


def test_randperm_rank_takes_its_permutation():
    P, n = 3, 50
    one = tr.Generator(seed=5)
    ranks = [tr.Generator(seed=5, partition_rank=r, partition_size=P) for r in range(P)]
    for _ in range(2):
        ref = [tr.randperm(n, generator=one) for _ in range(P)]
        for r, g in enumerate(ranks):
            assert torch.equal(tr.randperm(n, generator=g), ref[r])


def test_program_cpu_path_partitions(monkeypatch):
    from trnrand.nki import program

    monkeypatch.setattr(program, "HAS_NKI", False)
    P, n = 2, 1000

    def run(g, m, calls=2):
        prog = g.new_program().normal(m, out="z").uniform(m, out="u").build()
        out = []
        for _ in range(calls):
            b = {"z": torch.empty(m), "u": torch.empty(m)}
            prog.stream_into(b)
            out.append(b)
        return out

    ref = run(tr.Generator(seed=3), P * n)
    parts = [run(tr.Generator(seed=3, partition_rank=r, partition_size=P), n) for r in range(P)]
    for c in range(2):
        for k in ("z", "u"):
            assert torch.equal(torch.cat([parts[r][c][k] for r in range(P)]), ref[c][k])


# sha256[:16] of two consecutive 1000-sample draws from Generator(seed=2026),
# recorded on v0.13.0: single-rank streams must not change.
_PINNED = {
    "uniform": "2c98204dd5ea7aa1",
    "normal": "78a7b6e72bd0fb67",
    "exponential": "f84201c5d43d2b0c",
    "poisson": "04d8fedd2b969572",
    "gamma": "0f133b4be05d8a8c",
    "beta": "dd296e40cc9256a9",
    "chi_squared": "95adfdf1e56b82dc",
    "truncated_normal": "80b9967a2e08061b",
    "bernoulli": "0d5c760489091670",
    "randint": "821a5baf7cc04822",
    "randperm": "40ad18ed5773039d",
    "normal_into": "78a7b6e72bd0fb67",
    "uniform_into": "2c98204dd5ea7aa1",
    "exponential_into": "7a5972d4a997d171",
}
_KW = {
    "poisson": {"lam": 3.0},
    "gamma": {"shape": 0.7},
    "beta": {"alpha": 2.0, "beta": 3.0},
    "chi_squared": {"df": 3.0},
    "randint": {"high": 100},
}


@pytest.mark.parametrize("name", list(_PINNED))
def test_single_rank_streams_unchanged(name):
    g = tr.Generator(seed=2026)

    def draw():
        if name.endswith("_into"):
            b = torch.empty(1000)
            getattr(tr, name)(b, generator=g)
            return b
        return getattr(tr, name)(1000, generator=g, **_KW.get(name, {}))

    t = torch.cat([draw().double().reshape(-1) for _ in range(2)])
    assert hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16] == _PINNED[name]
