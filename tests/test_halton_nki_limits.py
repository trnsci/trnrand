"""halton() outside the NKI kernel's limits falls back to the CPU path (#104).

CPU-only: `halton_nki` is replaced by a stand-in with the real wrapper's limit
asserts (1 <= n_dims <= 10, n_points > 0, last index <= 2^22). References are
scipy.stats.qmc.Halton.
"""

import inspect
import re
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

    def fake_halton_nki(n_points, n_dims, start_index=0):
        assert 1 <= n_dims <= 10
        assert n_points > 0
        assert start_index + 1 + n_points <= 1 << 22
        calls.append((n_points, n_dims))
        return torch.full((n_points, n_dims), 0.5)

    monkeypatch.setattr(quasi, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "halton_nki", fake_halton_nki, raising=False)
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", False)
    return calls


def _scipy_halton(n, d):
    return torch.from_numpy(qmc.Halton(d, scramble=False).random(n + 1)[1:])


def test_limits_match_the_kernel_wrapper():
    # dispatch defines these inside `if HAS_NKI:`; read them from the source.
    src = inspect.getsource(dispatch)
    assert re.search(r"^\s+_HALTON_MAX_DIMS = 10$", src, re.M)
    assert re.search(r"^\s+_HALTON_MAX_POINTS = 1 << 22\b", src, re.M)
    assert quasi._HALTON_MAX_DIMS == 10 and quasi._HALTON_MAX_POINTS == 1 << 22


@pytest.mark.parametrize("backend", ["auto", "nki"])
@pytest.mark.parametrize("d", [11, 21])
def test_high_dim_falls_back(kernel_calls, monkeypatch, backend, d):
    monkeypatch.setattr(dispatch, "_backend", backend)
    # float32: a float64 request never considers the kernel under "auto" and
    # takes the cast under "nki" (the float64 policy, #105); see below.
    with pytest.warns(dispatch.NkiFallbackWarning, match=f"n_dims={d}"):
        x = trnrand.halton(64, d, dtype=torch.float32)
    torch.testing.assert_close(x, _scipy_halton(64, d).float(), atol=1e-6, rtol=0)
    assert kernel_calls == []


@pytest.mark.parametrize("d", [4, 11])
def test_float64_auto_takes_torch_before_the_limits(kernel_calls, monkeypatch, d):
    """Under "auto" float64 goes to the float64 torch path first (#105), so
    there is no kernel call and no limits fallback warning."""
    monkeypatch.setattr(dispatch, "_backend", "auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = trnrand.halton(64, d, dtype=torch.float64)
    torch.testing.assert_close(x, _scipy_halton(64, d), atol=1e-12, rtol=0)
    assert kernel_calls == []


@pytest.mark.parametrize("d", [1, 4, 10])
def test_zero_points_is_empty(kernel_calls, d):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = trnrand.halton(0, d, dtype=torch.float64)
    assert tuple(x.shape) == (0, d) and x.dtype == torch.float64
    assert kernel_calls == []


def test_too_many_points_falls_back(kernel_calls, monkeypatch):
    # The CPU path is a Python loop, too slow for 2^22 points in CI; stub it
    # and check only the routing here (its values are covered by test_quasi).
    monkeypatch.setattr(
        quasi, "_halton_sequence", lambda n, b, dt: torch.full((n,), 0.25, dtype=dt)
    )
    with pytest.warns(dispatch.NkiFallbackWarning, match="2\\^22"):
        x = trnrand.halton(1 << 22, 1)
    assert tuple(x.shape) == (1 << 22, 1) and bool((x == 0.25).all())
    assert kernel_calls == []


def test_require_nki_raises_value_error(kernel_calls, monkeypatch):
    monkeypatch.setattr(dispatch, "_REQUIRE_NKI", True)
    with pytest.raises(ValueError, match="TRNRAND_REQUIRE_NKI"):
        trnrand.halton(64, 11)
    with pytest.raises(ValueError, match="2\\^22"):
        trnrand.halton(1 << 22, 1)


@pytest.mark.parametrize("n, d", [(1, 1), (129, 10), ((1 << 22) - 1, 1)])
def test_in_range_still_uses_kernel(kernel_calls, n, d):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = trnrand.halton(n, d)
    assert kernel_calls == [(n, d)]
    assert tuple(x.shape) == (n, d)
