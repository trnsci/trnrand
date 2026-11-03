"""sobol() outside the NKI kernel's limits falls back to the CPU path (#89).

CPU-only: `sobol_nki` is replaced by a stand-in with the real wrapper's limit
asserts (n_dims <= 10, 1 <= n_points <= 2^24) that otherwise returns the
host mirror of the kernel. References are scipy.stats.qmc.Sobol.
"""

import warnings

import pytest
import torch
from scipy.stats import qmc

import trnrand
from trnrand import quasi
from trnrand.nki import dispatch


@pytest.fixture
def kernel_calls(monkeypatch):
    calls = []

    def fake_sobol_nki(n_points, n_dims, seed=0, start_index=0):
        assert 1 <= n_dims <= dispatch._SOBOL_MAX_DIMS
        assert n_points > 0
        assert start_index + n_points <= 1 << dispatch._SOBOL_BITS
        calls.append((n_points, n_dims))
        return dispatch.sobol_reference(n_points, n_dims, start_index)

    monkeypatch.setattr(quasi, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "sobol_nki", fake_sobol_nki, raising=False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    return calls


def _scipy_sobol(n, d):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The balance properties")
        return torch.from_numpy(qmc.Sobol(d, scramble=False).random(n))


@pytest.mark.parametrize("backend", ["auto", "nki"])
def test_high_dim_falls_back(kernel_calls, monkeypatch, backend):
    monkeypatch.setattr(dispatch, "_backend", backend)
    with pytest.warns(dispatch.NkiFallbackWarning, match="n_dims=11"):
        x = trnrand.sobol(64, 11, scramble=False, dtype=torch.float64)
    torch.testing.assert_close(x, _scipy_sobol(64, 11), atol=1e-7, rtol=0)
    assert kernel_calls == []


def test_more_than_2_pow_24_points_falls_back(kernel_calls):
    n = (1 << 24) + 1
    with pytest.warns(dispatch.NkiFallbackWarning, match="2\\^24"):
        x = trnrand.sobol(n, 1, scramble=False, dtype=torch.float64)
    assert tuple(x.shape) == (n, 1)
    ref = _scipy_sobol(n, 1)
    torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)
    assert kernel_calls == []


def test_shift_scrambled_high_dim_falls_back_to_sobol_engine(kernel_calls):
    with pytest.warns(dispatch.NkiFallbackWarning):
        x = trnrand.sobol(64, 11, seed=3, scramble="shift")
    ref = torch.quasirandom.SobolEngine(11, scramble=True, seed=3).draw(64)
    assert torch.equal(x, ref)


def test_owen_scrambled_high_dim_falls_back_to_cpu_owen(kernel_calls, monkeypatch):
    with pytest.warns(dispatch.NkiFallbackWarning):
        x = trnrand.sobol(64, 11, seed=3)
    monkeypatch.setattr(quasi, "HAS_NKI", False)
    assert torch.equal(x, trnrand.sobol(64, 11, seed=3))


@pytest.mark.parametrize("fake_nki", [False, True])
def test_zero_points_is_empty(kernel_calls, monkeypatch, fake_nki):
    monkeypatch.setattr(quasi, "HAS_NKI", fake_nki)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = trnrand.sobol(0, 3, dtype=torch.float64)
    assert tuple(x.shape) == (0, 3) and x.dtype == torch.float64
    assert kernel_calls == []


def test_require_nki_raises_value_error(kernel_calls, monkeypatch):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(ValueError, match="TRNRAND_REQUIRE_NKI"):
        trnrand.sobol(64, 11)
    with pytest.raises(ValueError, match="2\\^24"):
        trnrand.sobol((1 << 24) + 1, 1)


@pytest.mark.parametrize("n", [1, 129, 1 << 24])
def test_in_range_still_uses_kernel(kernel_calls, n):
    d = 10 if n < 1000 else 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = trnrand.sobol(n, d, scramble=False, dtype=torch.float64)
    assert kernel_calls == [(n, d)]
    assert tuple(x.shape) == (n, d)
