"""NKI-path routing of `sobol(scramble=...)` (#90, #100).

CPU-only: `sobol_nki` / `sobol_owen_nki` are replaced by stand-ins that record
the call (`sobol_nki` shifts whenever `seed != 0` in the real wrapper).
"""

import pytest
import torch

import trnrand
from trnrand import quasi
from trnrand.nki import dispatch


@pytest.fixture
def calls(monkeypatch):
    got = []

    def fake_sobol_nki(n_points, n_dims, seed=0, start_index=0):
        got.append(("shift", seed))
        return torch.zeros(n_points, n_dims)

    def fake_sobol_owen_nki(n_points, n_dims, seed, start_index=0):
        got.append(("owen", seed))
        return torch.zeros(n_points, n_dims)

    monkeypatch.setattr(quasi, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "sobol_nki", fake_sobol_nki, raising=False)
    monkeypatch.setattr(dispatch, "sobol_owen_nki", fake_sobol_owen_nki, raising=False)
    return got


@pytest.mark.parametrize("seed", [None, 0, 7, 12345])
def test_scramble_false_never_shifts(calls, seed):
    trnrand.sobol(8, 2, scramble=False, seed=seed)
    assert calls == [("shift", 0)]


@pytest.mark.parametrize("seed", [0, 7, 12345])
@pytest.mark.parametrize("scramble", [True, "owen"])
def test_scramble_true_is_owen_with_seed(calls, seed, scramble):
    trnrand.sobol(8, 2, scramble=scramble, seed=seed)
    assert calls == [("owen", seed)]


def test_scramble_true_seed_none_draws_from_global_rng(calls):
    torch.manual_seed(3)
    trnrand.sobol(8, 2)
    torch.manual_seed(3)
    trnrand.sobol(8, 2)
    trnrand.sobol(8, 2)
    (m0, a), (m1, b), (m2, c) = calls
    assert m0 == m1 == m2 == "owen"
    assert isinstance(a, int) and a == b != c


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_scramble_shift_keeps_old_nki_call(calls, seed):
    trnrand.sobol(8, 2, scramble="shift", seed=seed)
    assert calls == [("shift", seed or 0)]


def test_bad_scramble_raises():
    with pytest.raises(ValueError, match="scramble"):
        trnrand.sobol(8, 2, scramble="lms")
