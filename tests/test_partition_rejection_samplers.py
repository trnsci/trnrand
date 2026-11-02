"""Partition ranks must not draw identical gamma / chi_squared / beta /
truncated_normal samples on the NKI path (#92).

CPU-only: the NKI wrappers are replaced by stand-ins that derive their output
from the key they get, as the real Threefry samplers do.
"""

import pytest
import torch

import trnrand
from trnrand import distributions
from trnrand.nki import dispatch

WRAPPERS = ("gamma_nki", "chi_squared_nki", "beta_nki", "truncated_normal_nki")

DRAWS = {
    "gamma": lambda g: trnrand.gamma(256, shape=2.0, generator=g),
    "chi_squared": lambda g: trnrand.chi_squared(256, df=3.0, generator=g),
    "beta": lambda g: trnrand.beta(256, alpha=2.0, beta=3.0, generator=g),
    "truncated_normal": lambda g: trnrand.truncated_normal(256, generator=g),
}


@pytest.fixture
def keys(monkeypatch):
    got = []

    def fake(n_elements, seed=0, counter_offset=0, **kw):
        got.append(seed)
        return torch.rand(n_elements, generator=torch.Generator().manual_seed(seed))

    monkeypatch.setattr(distributions, "HAS_NKI", True)
    for name in WRAPPERS:
        monkeypatch.setattr(dispatch, name, fake, raising=False)
    return got


@pytest.mark.parametrize("dist", list(DRAWS))
def test_ranks_draw_different_samples(keys, dist):
    P = 4
    outs = [
        DRAWS[dist](trnrand.Generator(seed=5, partition_rank=r, partition_size=P)) for r in range(P)
    ]
    for r in range(1, P):
        assert not torch.equal(outs[0], outs[r]), f"rank {r} duplicates rank 0"
    # Each call uses keys seed..seed+7 at most; ranks are 8 apart, never shared.
    assert len(set(keys)) == P
    assert all(
        (b - a) % (1 << 24) >= 8 for a, b in zip(sorted(keys), sorted(keys)[1:], strict=False)
    )


@pytest.mark.parametrize("dist", list(DRAWS))
def test_rank0_and_unpartitioned_unchanged(keys, dist):
    a = DRAWS[dist](trnrand.Generator(seed=5))
    b = DRAWS[dist](trnrand.Generator(seed=5, partition_rank=0, partition_size=4))
    assert torch.equal(a, b)
    assert keys[0] == keys[1] == distributions._nki_seed(trnrand.Generator(seed=5))
