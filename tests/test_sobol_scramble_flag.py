"""sobol(scramble=False) must not be shifted on the NKI path (#90).

CPU-only: `sobol_nki` is replaced by a stand-in that records the seed it gets
(`seed != 0` means "apply the random shift" in the real wrapper).
"""

import pytest
import torch

import trnrand
from trnrand import quasi
from trnrand.nki import dispatch


@pytest.fixture
def seeds(monkeypatch):
    got = []

    def fake_sobol_nki(n_points, n_dims, seed=0, start_index=0):
        got.append(seed)
        return torch.zeros(n_points, n_dims)

    monkeypatch.setattr(quasi, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "sobol_nki", fake_sobol_nki, raising=False)
    return got


@pytest.mark.parametrize("seed", [None, 0, 7, 12345])
def test_scramble_false_never_shifts(seeds, seed):
    trnrand.sobol(8, 2, scramble=False, seed=seed)
    assert seeds == [0]


@pytest.mark.parametrize("seed", [7, 12345])
def test_scramble_true_keeps_seed(seeds, seed):
    trnrand.sobol(8, 2, scramble=True, seed=seed)
    assert seeds == [seed]
