"""uniform(dtype=float16/bfloat16) must stay in [low, high) on the NKI path (#94).

CPU-only: the Threefry uniform is replaced by a stand-in returning values at
and near its maximum, 1 - 2^-24, which round up to 1.0 in reduced precision.
"""

import pytest
import torch

import trnrand
from trnrand import distributions
from trnrand.nki import dispatch

U = torch.tensor([1 - 2**-24, 1 - 2**-10, 0.5, 0.0, 0.25])


@pytest.fixture
def nki_uniform(monkeypatch):
    monkeypatch.setattr(distributions, "HAS_NKI", True)
    monkeypatch.setattr(
        dispatch,
        "threefry_uniform_nki",
        lambda n, seed=0, counter_offset=0: U[:n].clone(),
        raising=False,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-3.0, 5.0), (10.0, 1000.0)])
def test_half_stays_below_high(nki_uniform, dtype, low, high):
    x = trnrand.uniform(5, low=low, high=high, dtype=dtype, generator=trnrand.Generator(seed=1))
    assert x.dtype == dtype
    assert (x < high).all() and (x >= low).all()
    # values not near high are the plain cast
    ref = (U * (high - low) + low).to(dtype)
    assert torch.equal(x[2:], ref[2:])


def test_float32_unchanged(nki_uniform):
    x = trnrand.uniform(5, dtype=torch.float32, generator=trnrand.Generator(seed=1))
    assert torch.equal(x, U)
