"""exponential_into / GeneratorProgram.exponential on the NKI path must stay
finite when the Threefry uniform is exactly 0 (#82).

CPU-only: the streaming uniform kernel is replaced by a stand-in that returns
zeros, which the real kernel emits with probability 2^-24 per sample.
"""

import torch

import trnrand
from trnrand import distributions
from trnrand.nki import dispatch, program


def _zeros(n_elements, seed=0, counter_offset=0):
    return torch.zeros(n_elements)


def test_exponential_into_finite_on_zero_uniform(monkeypatch):
    monkeypatch.setattr(distributions, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "threefry_stream_uniform", _zeros, raising=False)
    buf = torch.empty(64)
    trnrand.exponential_into(buf, rate=2.0, generator=trnrand.Generator(seed=1))
    assert torch.isfinite(buf).all()
    # Same value exponential() gives for u clamped to 1e-10.
    assert torch.allclose(buf, torch.full((64,), -torch.log(torch.tensor(1e-10)).item() / 2.0))


def test_program_exponential_finite_on_zero_uniform(monkeypatch):
    monkeypatch.setattr(program, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "threefry_stream_uniform", _zeros, raising=False)
    prog = trnrand.Generator(seed=1).new_program().exponential(64, rate=4.0, out="e").build()
    buf = torch.empty(64)
    prog.stream_into({"e": buf})
    assert torch.isfinite(buf).all()


def test_nonzero_uniform_unchanged(monkeypatch):
    u = torch.linspace(1e-6, 1 - 1e-6, 64)
    monkeypatch.setattr(distributions, "HAS_NKI", True)
    monkeypatch.setattr(
        dispatch,
        "threefry_stream_uniform",
        lambda n, seed=0, counter_offset=0: u.clone(),
        raising=False,
    )
    buf = torch.empty(64)
    trnrand.exponential_into(buf, rate=3.0, generator=trnrand.Generator(seed=1))
    assert torch.equal(buf, -torch.log(u) / 3.0)
