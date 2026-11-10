"""CPU checks for the device-known xfail rules and the device summary
(trnsci/trnsci#35): the rules pick the intended conformance cases, and the
conftest summary sorts xfail reasons by category."""

from types import SimpleNamespace

import pytest
from conftest import _expect_category, _expect_summary
from test_conformance import (
    _POISSON_NORMAL_THRESHOLD,
    DEVICE_KNOWN,
    _device_known_category,
)

from trnrand.nki import dispatch


def _node(originalname, **params):
    return SimpleNamespace(
        originalname=originalname, name=originalname, callspec=SimpleNamespace(params=params)
    )


@pytest.mark.parametrize(
    "test,params,cat",
    [
        ("test_parameters_match_scipy", {"name": "poisson_20"}, "poisson-normal-approx"),
        ("test_parameters_match_scipy", {"name": "poisson_1e4"}, "poisson-normal-approx"),
        ("test_parameters_match_scipy", {"name": "poisson_19.9"}, None),
        ("test_parameters_match_scipy", {"name": "gamma_shape1"}, None),
        ("test_distribution_matches_scipy", {"name": "poisson_large"}, "poisson-normal-approx"),
        ("test_distribution_matches_scipy", {"name": "poisson_small"}, None),
        ("test_same_seed_same_stream", {"dn": "poisson_large"}, None),
        ("test_output_shape", {"name": "poisson_large"}, None),
    ],
)
def test_device_known_rules(test, params, cat):
    assert _device_known_category(_node(test, **params)) == cat
    if cat is not None:
        assert cat in DEVICE_KNOWN


def test_device_known_unparametrized():
    assert _device_known_category(SimpleNamespace(originalname="test_x", name="test_x")) is None


def test_poisson_threshold_matches_library():
    assert dispatch._POISSON_NORMAL_THRESHOLD == _POISSON_NORMAL_THRESHOLD


def test_summary_categories():
    assert _expect_category("[device-known:poisson-normal-approx] trnsci/trnrand#88: x") == (
        "poisson-normal-approx"
    )
    assert _expect_category("trnsci/trnrand#94 open bug: y") == "other xfail"
    assert _expect_category("") == "other xfail"

    def rep(nodeid, wasxfail=None):
        r = SimpleNamespace(nodeid=nodeid, when="call")
        if wasxfail is not None:
            r.wasxfail = wasxfail
        return r

    s = _expect_summary(
        {
            "passed": [rep("a"), rep("b")],
            "xfailed": [rep("c", "[device-known:k1] i: w"), rep("d", "other")],
            "xpassed": [rep("e", "[device-known:k2] i: w")],
            "failed": [rep("f")],
        }
    )
    assert s["passed"] == 2
    assert s["xfailed"] == {"k1": ["c"], "other xfail": ["d"]}
    assert s["xpassed"] == {"k2": ["e"]}
    assert s["new_failures"] == ["f"]
