"""Reference-conformance suite (trnsci/trnsci#105).

Every public trnrand API is checked against an *independent* reference, not
against another trnrand backend: scipy.stats distributions (KS / chi-square
goodness of fit, closed-form moments), scipy.stats.qmc Sobol and Halton
points, the Latin-hypercube stratification property, support bounds, dtype /
shape contracts, and the seeding / state contracts in the docstrings. Audit
rounds 1-3 (trnsci/trnsci#35) found bugs that every backend shared; a
backend-vs-backend test cannot see those.

Sizes cover 0, 1, 127, 129, 1000 and the 512 / 16384 Threefry block
boundaries; dtypes float16, bfloat16, float32, float64; non-contiguous
``*_into`` buffers; non-default keyword values. Seeds are fixed and the
statistical thresholds sit near p = 1e-6, so the suite does not flake.

CPU (CI default, pytorch backend)::

    pytest tests/test_conformance.py

On a Neuron host, the same cases on the NKI path, with ``*_into`` /
``GeneratorProgram`` buffers on the device::

    pytest tests/test_conformance.py --backend nki --conformance-device neuron
    pytest tests/test_conformance.py --backend auto --conformance-device neuron

Known issues are ``xfail`` with the issue number. Needs-decision items are
``strict=True``, so the test fails (XPASS) once the behaviour changes and the
mark must be removed. Bugs with an open fix PR are ``strict=False``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.stats import qmc

import trnrand as tr
from trnrand.nki import dispatch

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
REDUCED = (torch.float16, torch.bfloat16)
SIZES = [1, 127, 129, 1000]

# scipy warns that Sobol balance needs n = 2^k; awkward n is the point here.
pytestmark = pytest.mark.filterwarnings("ignore:The balance properties of Sobol:UserWarning")
BLOCK_SIZES = [511, 512, 513, 16385]


# ── backend / device plumbing ──────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def _conformance_backend(request):
    prev = tr.get_backend()
    tr.set_backend(request.config.getoption("--backend"))
    yield
    tr.set_backend(prev)


@pytest.fixture
def backend(request) -> str:
    return request.config.getoption("--backend")


@pytest.fixture
def nki_like(backend) -> bool:
    """True when distributions take the NKI path in this run."""
    return backend != "pytorch" and tr.HAS_NKI


@pytest.fixture
def dev(request) -> torch.device:
    return torch.device(request.config.getoption("--conformance-device"))


def known(request, issue: str, why: str, *, strict: bool, when: bool = True):
    """Mark the running test xfail for a filed issue (only if `when`)."""
    if when:
        kind = "needs decision" if strict else "open bug"
        request.applymarker(
            pytest.mark.xfail(strict=strict, reason=f"trnsci/trnrand#{issue} {kind}: {why}")
        )


def host(t):
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def G(seed=1234, **kw):
    return tr.Generator(seed=seed, **kw)


def nan_buf(dev, *shape, dtype=torch.float32):
    return torch.full(shape, float("nan"), dtype=dtype, device=dev)


# ── statistical checks against scipy ───────────────────────────────────────


def ks_crit(n: int) -> float:
    # Kolmogorov critical value near p = 1e-6.
    return 1.95 / math.sqrt(n) * 1.6


def assert_ks(x: torch.Tensor, ref, slack: float = 0.0):
    xs = host(x).double().reshape(-1).numpy()
    d = stats.kstest(xs, ref.cdf).statistic
    assert d <= ks_crit(xs.size) + slack, f"KS D={d:.4f} > {ks_crit(xs.size) + slack:.4f}"


def assert_chi2_discrete(x: torch.Tensor, ref):
    xs = host(x).double().reshape(-1).numpy()
    n = xs.size
    vals, cnt = np.unique(xs, return_counts=True)
    lo, hi = int(ref.ppf(1e-4)), int(ref.ppf(1 - 1e-4))
    ks = np.arange(lo, hi + 1)
    obs = np.array([cnt[vals == k].sum() for k in ks], dtype=float)
    exp = ref.pmf(ks) * n
    keep = exp > 20
    chi = ((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum()
    p = stats.chi2(max(int(keep.sum()) - 1, 1)).sf(chi)
    assert p > 1e-6, f"chi-square GOF p={p:.2e}"


def assert_moments(x: torch.Tensor, ref, dtype, heavy: bool = False):
    x = host(x).double().reshape(-1)
    n = x.numel()
    m, v = float(ref.mean()), float(ref.var())
    eps = {torch.float16: 2**-10, torch.bfloat16: 2**-7, torch.float32: 1e-6, torch.float64: 1e-12}[
        dtype
    ]
    q = eps * (abs(m) + math.sqrt(v)) * 2
    tol_m = 6 * math.sqrt(v / n) + q
    assert abs(x.mean().item() - m) <= tol_m, f"mean {x.mean().item():.5f} vs {m:.5f}"
    tol_v = 10 * v * math.sqrt(2.0 / n) * (3 if heavy else 1) + 4 * q * math.sqrt(v)
    assert abs(x.var().item() - v) <= tol_v, f"var {x.var().item():.5f} vs {v:.5f}"


# name -> (draw(n, dtype, gen), scipy reference, support kind, heavy-tailed)
DISTS = {
    "uniform": (
        lambda n, dt, g: tr.uniform(n, dtype=dt, generator=g),
        stats.uniform(0, 1),
        ("half_open", 0.0, 1.0),
        False,
    ),
    "uniform_shift": (
        lambda n, dt, g: tr.uniform(n, low=-3.0, high=5.0, dtype=dt, generator=g),
        stats.uniform(-3.0, 8.0),
        ("half_open", -3.0, 5.0),
        False,
    ),
    "normal": (
        lambda n, dt, g: tr.normal(n, mean=2.0, std=3.0, dtype=dt, generator=g),
        stats.norm(2.0, 3.0),
        ("real",),
        False,
    ),
    "standard_normal": (
        lambda n, dt, g: tr.standard_normal(n, dtype=dt, generator=g),
        stats.norm(),
        ("real",),
        False,
    ),
    "exponential": (
        lambda n, dt, g: tr.exponential(n, rate=2.5, dtype=dt, generator=g),
        stats.expon(scale=1 / 2.5),
        ("closed", 0.0, math.inf),
        False,
    ),
    "truncated_normal": (
        lambda n, dt, g: tr.truncated_normal(
            n, mean=1.0, std=2.0, low=-1.0, high=1.5, dtype=dt, generator=g
        ),
        stats.truncnorm(-1.0, 1.5, loc=1.0, scale=2.0),
        ("closed", 1.0 - 2.0 - 1e-2, 1.0 + 3.0 + 1e-2),
        False,
    ),
    "gamma_lt1": (
        lambda n, dt, g: tr.gamma(n, shape=0.4, scale=2.0, dtype=dt, generator=g),
        stats.gamma(0.4, scale=2.0),
        ("closed", 0.0, math.inf),
        True,
    ),
    "gamma_gt1": (
        lambda n, dt, g: tr.gamma(n, shape=3.5, scale=0.5, dtype=dt, generator=g),
        stats.gamma(3.5, scale=0.5),
        ("closed", 0.0, math.inf),
        False,
    ),
    "chi_squared": (
        lambda n, dt, g: tr.chi_squared(n, df=5.0, dtype=dt, generator=g),
        stats.chi2(5.0),
        ("closed", 0.0, math.inf),
        False,
    ),
    "beta": (
        lambda n, dt, g: tr.beta(n, alpha=0.7, beta=2.5, dtype=dt, generator=g),
        stats.beta(0.7, 2.5),
        ("closed", 0.0, 1.0),
        True,
    ),
    "poisson_small": (
        lambda n, dt, g: tr.poisson(n, lam=3.0, dtype=dt, generator=g),
        stats.poisson(3.0),
        ("count",),
        False,
    ),
    "poisson_large": (
        lambda n, dt, g: tr.poisson(n, lam=40.0, dtype=dt, generator=g),
        stats.poisson(40.0),
        ("count",),
        False,
    ),
    "bernoulli": (
        lambda n, dt, g: tr.bernoulli(n, p=0.3, dtype=dt, generator=g),
        stats.bernoulli(0.3),
        ("count",),
        False,
    ),
}


def assert_support(x, support):
    x = host(x).double()
    kind = support[0]
    if kind == "half_open":
        lo, hi = support[1:]
        assert bool((x >= lo).all() and (x < hi).all()), (
            f"outside [{lo}, {hi}): {x.min()}..{x.max()}"
        )
    elif kind == "closed":
        lo, hi = support[1:]
        assert bool((x >= lo).all() and (x <= hi).all()), (
            f"outside [{lo}, {hi}]: {x.min()}..{x.max()}"
        )
    elif kind == "count":
        assert bool((x >= 0).all() and (x == torch.round(x)).all()), (
            "negative or non-integer counts"
        )


# ── distributions: dtype x reference ───────────────────────────────────────


@pytest.mark.parametrize("dt", FLOAT_DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize("name", list(DISTS))
def test_distribution_matches_scipy(request, nki_like, name, dt):
    known(
        request,
        "94",
        "NKI uniform in float16/bfloat16 returns high (fix in #95)",
        strict=False,
        when=nki_like and name.startswith("uniform") and dt in REDUCED,
    )
    draw, ref, support, heavy = DISTS[name]
    n = 100_000
    x = host(draw(n, dt, G(7)))
    assert x.dtype == dt
    assert tuple(x.shape) == (n,)
    assert bool(torch.isfinite(x).all()), f"{int((~torch.isfinite(x)).sum())} non-finite"
    assert_support(x, support)
    assert_moments(x, ref, dt, heavy=heavy)
    if support[0] == "count":
        assert_chi2_discrete(x[:50_000], ref)
    else:
        assert_ks(x[:50_000], ref, slack={torch.float16: 2e-3, torch.bfloat16: 1.2e-2}.get(dt, 0.0))


# Non-default keyword values against scipy.
PARAM_GRID = {
    "uniform_narrow": (
        lambda g, n: tr.uniform(n, low=-1e3, high=-999.0, generator=g),
        stats.uniform(-1e3, 1.0),
    ),
    "normal_tiny_std_fp64": (
        lambda g, n: tr.normal(n, mean=5.0, std=1e-3, dtype=torch.float64, generator=g),
        stats.norm(5.0, 1e-3),
    ),
    "exponential_rate_small": (
        lambda g, n: tr.exponential(n, rate=1e-3, generator=g),
        stats.expon(scale=1e3),
    ),
    "exponential_rate_big": (
        lambda g, n: tr.exponential(n, rate=1e3, generator=g),
        stats.expon(scale=1e-3),
    ),
    "trunc_one_sided": (
        lambda g, n: tr.truncated_normal(n, low=0.0, high=10.0, generator=g),
        stats.truncnorm(0.0, 10.0),
    ),
    "trunc_tail": (
        lambda g, n: tr.truncated_normal(n, low=1.5, high=3.0, mean=-2, std=0.5, generator=g),
        stats.truncnorm(1.5, 3.0, loc=-2, scale=0.5),
    ),
    "trunc_default": (lambda g, n: tr.truncated_normal(n, generator=g), stats.truncnorm(-2.0, 2.0)),
    "gamma_tiny_shape_fp64": (
        lambda g, n: tr.gamma(n, shape=0.05, generator=g, dtype=torch.float64),
        stats.gamma(0.05),
    ),
    "gamma_shape1": (
        lambda g, n: tr.gamma(n, shape=1.0, scale=3.0, generator=g),
        stats.gamma(1.0, scale=3.0),
    ),
    "gamma_big_shape": (
        lambda g, n: tr.gamma(n, shape=150.0, scale=0.1, generator=g),
        stats.gamma(150.0, scale=0.1),
    ),
    "chi2_df1": (lambda g, n: tr.chi_squared(n, df=1.0, generator=g), stats.chi2(1.0)),
    "chi2_df_frac": (lambda g, n: tr.chi_squared(n, df=0.6, generator=g), stats.chi2(0.6)),
    "beta_sym": (lambda g, n: tr.beta(n, alpha=5.0, beta=5.0, generator=g), stats.beta(5.0, 5.0)),
    "beta_u_fp64": (
        lambda g, n: tr.beta(n, alpha=0.3, beta=0.3, generator=g, dtype=torch.float64),
        stats.beta(0.3, 0.3),
    ),
    "poisson_19.9": (lambda g, n: tr.poisson(n, lam=19.9, generator=g), stats.poisson(19.9)),
    "poisson_20": (lambda g, n: tr.poisson(n, lam=20.0, generator=g), stats.poisson(20.0)),
    "poisson_1e4": (lambda g, n: tr.poisson(n, lam=1e4, generator=g), stats.poisson(1e4)),
    "bernoulli_p0.9": (lambda g, n: tr.bernoulli(n, p=0.9, generator=g), stats.bernoulli(0.9)),
}


@pytest.mark.parametrize("name", list(PARAM_GRID))
def test_parameters_match_scipy(name):
    f, ref = PARAM_GRID[name]
    x = host(f(G(17), 100_000))
    assert bool(torch.isfinite(x).all())
    if isinstance(ref.dist, stats.rv_discrete):
        assert_chi2_discrete(x, ref)
    else:
        assert_ks(x, ref, slack=1e-3)


@pytest.mark.parametrize(
    "f, value",
    [
        (lambda g: tr.normal(1000, mean=3.0, std=0.0, generator=g), 3.0),
        (lambda g: tr.uniform(1000, low=2.0, high=2.0, generator=g), 2.0),
        (lambda g: tr.bernoulli(1000, p=0.0, generator=g), 0.0),
        (lambda g: tr.bernoulli(1000, p=1.0, generator=g), 1.0),
        (lambda g: tr.poisson(1000, lam=0.0, generator=g), 0.0),
    ],
    ids=["normal_std0", "uniform_low_eq_high", "bernoulli_p0", "bernoulli_p1", "poisson_lam0"],
)
def test_degenerate_parameters(f, value):
    x = host(f(G(3)))
    assert bool((x == value).all())


@pytest.mark.parametrize("dt", [torch.int64, torch.bool, torch.float16], ids=lambda d: str(d)[6:])
def test_bernoulli_dtypes(dt):
    x = host(tr.bernoulli(50_000, p=0.3, dtype=dt, generator=G(1)))
    assert x.dtype == dt
    assert_chi2_discrete(x.double(), stats.bernoulli(0.3))


@pytest.mark.parametrize("dt", [torch.int64, torch.float32], ids=lambda d: str(d)[6:])
def test_poisson_dtypes(dt):
    x = host(tr.poisson(50_000, lam=3.0, dtype=dt, generator=G(1)))
    assert x.dtype == dt
    assert_chi2_discrete(x.double(), stats.poisson(3.0))


# ── randint / randperm ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dt",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float32],
    ids=lambda d: str(d)[6:],
)
def test_randint_dtype_range_uniformity(dt):
    lo, hi = (0, 7) if dt == torch.uint8 else (-5, 7)
    x = host(tr.randint(50_000, low=lo, high=hi, dtype=dt, generator=G(3)))
    assert x.dtype == dt
    assert int(x.min()) == lo and int(x.max()) == hi - 1
    cnt = torch.bincount(x.long() - lo)
    exp = 50_000 / (hi - lo)
    chi = (((cnt.double() - exp) ** 2) / exp).sum().item()
    assert stats.chi2(hi - lo - 1).sf(chi) > 1e-6


def test_randint_bounds():
    assert bool((host(tr.randint(10_000, low=3, high=4, generator=G(1))) == 3).all()), (
        "high not exclusive"
    )
    assert bool((host(tr.randint(10_000, generator=G(1))) == 0).all()), "default [0, 1)"
    x = host(tr.randint(100_000, low=-(2**40), high=2**40, generator=G(3)))
    assert x.dtype == torch.int64
    assert bool((x >= -(2**40)).all() and (x < 2**40).all())
    assert stats.kstest((x.double().numpy() + 2**40) / 2**41, "uniform").statistic < 0.01


@pytest.mark.parametrize("n", [0, 1, 2, 127, 129, 1000])
@pytest.mark.parametrize("dt", [torch.int16, torch.int32, torch.int64], ids=lambda d: str(d)[6:])
def test_randperm_is_permutation(n, dt):
    p = host(tr.randperm(n, dtype=dt, generator=G(5)))
    assert p.dtype == dt
    assert torch.equal(torch.sort(p.long()).values, torch.arange(n))


def test_randperm_uniform_over_positions():
    # Position of element 0 over many permutations of 8 is uniform.
    g = G(11)
    pos = torch.tensor([int((tr.randperm(8, generator=g) == 0).nonzero()) for _ in range(4000)])
    cnt = torch.bincount(pos, minlength=8).double()
    chi = (((cnt - 500) ** 2) / 500).sum().item()
    assert stats.chi2(7).sf(chi) > 1e-6


# ── shapes ─────────────────────────────────────────────────────────────────

SIZED = {
    "uniform": lambda *s: tr.uniform(*s, generator=G(1)),
    "normal": lambda *s: tr.normal(*s, generator=G(1)),
    "standard_normal": lambda *s: tr.standard_normal(*s, generator=G(1)),
    "exponential": lambda *s: tr.exponential(*s, generator=G(1)),
    "bernoulli": lambda *s: tr.bernoulli(*s, generator=G(1)),
    "randint": lambda *s: tr.randint(*s, high=5, generator=G(1)),
    "truncated_normal": lambda *s: tr.truncated_normal(*s, generator=G(1)),
    "gamma": lambda *s: tr.gamma(*s, shape=2.0, generator=G(1)),
    "chi_squared": lambda *s: tr.chi_squared(*s, df=3.0, generator=G(1)),
    "beta": lambda *s: tr.beta(*s, alpha=2.0, beta=3.0, generator=G(1)),
    "poisson": lambda *s: tr.poisson(*s, lam=3.0, generator=G(1)),
    "poisson_large": lambda *s: tr.poisson(*s, lam=50.0, generator=G(1)),
}

SHAPES = {
    "0": (0,),
    "3x0": (3, 0),
    "1": (1,),
    "1x1x1": (1, 1, 1),
    "127": (127,),
    "129": (129,),
    "1000": (1000,),
    "4x129": (4, 129),
    "long_thin": (1, 70_001),
}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("name", list(SIZED))
def test_output_shape(name, shape):
    s = SHAPES[shape]
    x = host(SIZED[name](*s))
    assert tuple(x.shape) == s
    if x.numel():
        assert bool(torch.isfinite(x.float()).all())


@pytest.mark.parametrize("n", BLOCK_SIZES)
@pytest.mark.parametrize("name", ["uniform", "normal", "standard_normal", "exponential"])
def test_block_boundaries_no_repeated_tail(name, n):
    x = host(SIZED[name](n))
    assert x.numel() == n and bool(torch.isfinite(x).all())
    if n > 1024:
        assert not torch.equal(x[-256:], x[:256]), "tail repeats head"
    # tail is distributed like the head (a dropped/zeroed tail tile would not be)
    tail = x[-min(n, 2000) :].double().numpy()
    assert len(np.unique(tail)) > 0.9 * tail.size


@pytest.mark.parametrize("name", list(SIZED))
def test_outputs_are_not_differentiable(name):
    with torch.enable_grad():
        x = SIZED[name](16)
    assert not x.requires_grad and x.grad_fn is None


# ── *_into: allocation-free variants ───────────────────────────────────────

INTO = {
    "normal_into": (tr.normal_into, {"mean": 1.0, "std": 2.0}, stats.norm(1.0, 2.0), tr.normal),
    "uniform_into": (
        tr.uniform_into,
        {"low": -1.0, "high": 3.0},
        stats.uniform(-1.0, 4.0),
        tr.uniform,
    ),
    "exponential_into": (
        tr.exponential_into,
        {"rate": 2.0},
        stats.expon(scale=0.5),
        tr.exponential,
    ),
}

VIEWS = {
    "transposed": ((300, 200), lambda b: b.t()),
    "step_sliced": ((600, 400), lambda b: b[::2, ::3]),
    "as_strided": ((200_000,), lambda b: b.as_strided((300, 150), (600, 3), 7)),
    "slice_T": ((300, 400), lambda b: b[10:290, 5:395:2].transpose(0, 1)),
}


@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize("iname", list(INTO))
def test_into_noncontiguous_writes_only_the_view(dev, iname, view):
    fn, kw, ref, _ = INTO[iname]
    shape, vf = VIEWS[view]
    base = nan_buf(dev, *shape)
    v = vf(base)
    idx = vf(torch.arange(math.prod(shape)).reshape(shape))
    inside = torch.zeros(math.prod(shape), dtype=torch.bool)
    inside[idx.reshape(-1)] = True
    ptr = v.data_ptr()
    assert fn(v, generator=G(11), **kw) is None
    assert v.data_ptr() == ptr, "buffer storage replaced"
    outside = host(base).reshape(-1)[~inside]
    assert bool(torch.isnan(outside).all()), (
        f"{int((~torch.isnan(outside)).sum())} elements outside the view written"
    )
    vals = host(v).reshape(-1)
    assert bool(torch.isfinite(vals).all()), "view elements left unwritten"
    assert_ks(vals, ref)


@pytest.mark.parametrize("iname", list(INTO))
def test_into_channels_last(dev, iname):
    fn, kw, ref, _ = INTO[iname]
    v = nan_buf(dev, 4, 8, 30, 30).contiguous(memory_format=torch.channels_last)
    fn(v, generator=G(11), **kw)
    vals = host(v).reshape(-1)
    assert bool(torch.isfinite(vals).all())
    assert_ks(vals, ref)


@pytest.mark.parametrize(
    "dt", [torch.float16, torch.bfloat16, torch.float64], ids=lambda d: str(d)[6:]
)
@pytest.mark.parametrize("iname", list(INTO))
def test_into_buffer_dtypes(dev, iname, dt):
    fn, kw, ref, _ = INTO[iname]
    b = nan_buf(dev, 50_000, dtype=dt)
    fn(b, generator=G(2), **kw)
    hb = host(b)
    assert hb.dtype == dt
    assert bool(torch.isfinite(hb).all())
    assert stats.kstest(hb.double().numpy(), ref.cdf).statistic < 0.02


@pytest.mark.parametrize("iname", list(INTO))
def test_into_matches_allocating_distribution(dev, iname):
    # Same distribution as the allocating variant (two-sample KS).
    fn, kw, _, alloc = INTO[iname]
    b = nan_buf(dev, 50_000)
    fn(b, generator=G(21), **kw)
    a = alloc(50_000, generator=G(22), **kw)
    d = stats.ks_2samp(host(b).double().numpy(), host(a).double().numpy()).statistic
    assert d < 1.95 * math.sqrt(2 / 50_000) * 1.6


@pytest.mark.parametrize("n", [0, 1, 127, 129, 1000, *BLOCK_SIZES])
@pytest.mark.parametrize("iname", list(INTO))
def test_into_sizes_fill_everything(dev, iname, n):
    fn, kw, _, _ = INTO[iname]
    b = nan_buf(dev, n)
    fn(b, generator=G(1), **kw)
    assert bool(torch.isfinite(host(b)).all()), "unwritten tail"


@pytest.mark.parametrize("iname", list(INTO))
def test_into_zero_dim_buffer(dev, iname):
    fn, kw, _, _ = INTO[iname]
    z = nan_buf(dev, 1).reshape(())
    fn(z, generator=G(1), **kw)
    assert bool(torch.isfinite(host(z)))


def test_into_rejects_overlapping_and_grad_leaf(dev):
    with pytest.raises(RuntimeError):
        tr.normal_into(nan_buf(dev, 1, 50).expand(40, 50), generator=G(1))
    with pytest.raises(RuntimeError):
        tr.normal_into(torch.zeros(10, requires_grad=True), generator=G(1))


def test_into_does_not_touch_other_tensors(dev):
    other = torch.randn(100)
    keep = other.clone()
    tr.uniform_into(nan_buf(dev, 1000), generator=G(3))
    assert torch.equal(other, keep)


def test_exponential_into_finite_at_scale(request, dev, nki_like):
    known(
        request,
        "82",
        "NKI exponential_into returns inf when u == 0 (fix in #86)",
        strict=False,
        when=nki_like,
    )
    n = 1 << 22 if nki_like else 1 << 20
    b = nan_buf(dev, n)
    tr.exponential_into(b, generator=G(100))
    assert int((~torch.isfinite(host(b))).sum()) == 0


# ── GeneratorProgram ───────────────────────────────────────────────────────


def _program(seed):
    return (
        G(seed)
        .new_program()
        .normal(5000, mean=1.0, std=2.0, out="z")
        .uniform(3000, low=-1, high=1, out="u")
        .exponential(4000, rate=3.0, out="e")
        .build()
    )


def test_program_stream_into_matches_scipy(dev):
    prog = _program(21)
    bufs = {"z": nan_buf(dev, 5000), "u": nan_buf(dev, 50, 60), "e": nan_buf(dev, 4000)}
    prog.stream_into(bufs)
    first = {k: host(v).clone() for k, v in bufs.items()}
    assert tuple(first["u"].shape) == (50, 60)
    for v in first.values():
        assert bool(torch.isfinite(v).all())
    assert_ks(first["z"], stats.norm(1, 2))
    assert_ks(first["u"], stats.uniform(-1, 2))
    assert bool((first["u"] >= -1).all() and (first["u"] < 1).all())
    assert_ks(first["e"], stats.expon(scale=1 / 3))
    prog.stream_into(bufs)
    assert not torch.equal(first["z"], host(bufs["z"])), "second stream_into repeated the first"
    b2 = {"z": nan_buf(dev, 5000), "u": nan_buf(dev, 50, 60), "e": nan_buf(dev, 4000)}
    _program(21).stream_into(b2)
    assert torch.equal(first["z"], host(b2["z"])), "same-seed programs differ"
    b3 = {"z": nan_buf(dev, 5000), "u": nan_buf(dev, 50, 60), "e": nan_buf(dev, 4000)}
    _program(22).stream_into(b3)
    assert not torch.equal(first["z"], host(b3["z"])), "different seeds give the same stream"


def test_program_builder_requires_steps():
    from trnrand.nki.program import ProgramBuilder

    with pytest.raises(ValueError):
        ProgramBuilder(seed=0).build()


def test_program_exponential_finite_at_scale(request, dev):
    known(
        request,
        "82",
        "NKI GeneratorProgram.exponential returns inf when u == 0 (fix in #86)",
        strict=False,
        when=tr.HAS_NKI,
    )
    n = 1 << 22 if tr.HAS_NKI else 1 << 18
    b = {"e": nan_buf(dev, n)}
    G(5).new_program().exponential(n, out="e").build().stream_into(b)
    assert int((~torch.isfinite(host(b["e"]))).sum()) == 0


# ── determinism, seeding and state ─────────────────────────────────────────


def _into_draw(fn):
    def draw(g, n):
        b = torch.full((n,), float("nan"))
        fn(b, generator=g)
        return b

    return draw


DRAWS = {
    "uniform": lambda g, n: tr.uniform(n, generator=g),
    "normal": lambda g, n: tr.normal(n, generator=g),
    "exponential": lambda g, n: tr.exponential(n, generator=g),
    "poisson_large": lambda g, n: tr.poisson(n, lam=30.0, generator=g),
    "gamma": lambda g, n: tr.gamma(n, shape=2.0, generator=g),
    "beta": lambda g, n: tr.beta(n, alpha=2.0, beta=3.0, generator=g),
    "chi_squared": lambda g, n: tr.chi_squared(n, df=3.0, generator=g),
    "truncated_normal": lambda g, n: tr.truncated_normal(n, generator=g),
    "bernoulli": lambda g, n: tr.bernoulli(n, generator=g),
    "randint": lambda g, n: tr.randint(n, high=100, generator=g),
    "randperm": lambda g, n: tr.randperm(n, generator=g),
    "normal_into": _into_draw(tr.normal_into),
    "uniform_into": _into_draw(tr.uniform_into),
    "exponential_into": _into_draw(tr.exponential_into),
}
KEY_MODES = ["draw", "counter"]


@pytest.mark.parametrize("km", KEY_MODES)
@pytest.mark.parametrize("dn", list(DRAWS))
def test_same_seed_same_stream(dn, km):
    d = DRAWS[dn]
    a = host(d(G(77, key_mode=km), 3000))
    assert torch.equal(a, host(d(G(77, key_mode=km), 3000)))
    assert not torch.equal(a, host(d(G(78, key_mode=km), 3000)))


@pytest.mark.parametrize("km", KEY_MODES)
@pytest.mark.parametrize("dn", list(DRAWS))
def test_state_roundtrip_replays(dn, km):
    d = DRAWS[dn]
    g = G(5, key_mode=km)
    d(g, 1000)
    s = g.get_state()
    a, a2 = host(d(g, 2000)), host(d(g, 700))
    h = G(999, key_mode=km)  # a different seed: the state must replace it fully
    h.set_state(s)
    assert torch.equal(a, host(d(h, 2000)))
    assert torch.equal(a2, host(d(h, 700)))


@pytest.mark.parametrize("km", KEY_MODES)
@pytest.mark.parametrize("dn", list(DRAWS))
def test_manual_seed_restarts_stream(dn, km):
    d = DRAWS[dn]
    g = G(1, key_mode=km)
    d(g, 1500)
    assert g.manual_seed(42) is g
    assert torch.equal(host(d(g, 1000)), host(d(G(42, key_mode=km), 1000)))


@pytest.mark.parametrize("km", KEY_MODES)
@pytest.mark.parametrize("dn", list(DRAWS))
def test_successive_draws_independent(dn, km):
    d = DRAWS[dn]
    g = G(3, key_mode=km)
    a, b = host(d(g, 4096)), host(d(g, 4096))
    assert not torch.equal(a, b)
    if dn not in ("bernoulli", "randint", "randperm"):
        r = np.corrcoef(a.double().numpy(), b.double().numpy())[0, 1]
        assert abs(r) < 0.08, f"successive draws correlated r={r:.3f}"


def test_state_tensors_do_not_alias():
    g = G(9)
    s = g.get_state()
    s.zero_()
    assert torch.equal(host(tr.uniform(1000, generator=g)), host(tr.uniform(1000, generator=G(9))))
    s = G(9).get_state()
    h = G(1)
    h.set_state(s)
    s.zero_()
    assert torch.equal(host(tr.uniform(1000, generator=h)), host(tr.uniform(1000, generator=G(9))))


def test_draws_return_fresh_tensors():
    g = G(3)
    a = tr.uniform(1000, generator=g)
    a0 = a.clone()
    b = tr.uniform(1000, generator=g)
    assert a.data_ptr() != b.data_ptr()
    assert torch.equal(host(a), host(a0))


def test_module_manual_seed_and_default_generator():
    g = tr.manual_seed(123)
    assert tr.get_default_generator() is g
    assert g.seed == 123
    a = host(tr.uniform(1000))
    assert torch.equal(a, host(tr.uniform(1000, generator=G(123))))
    tr.manual_seed(123)
    assert torch.equal(host(tr.normal(1000)), host(tr.normal(1000, generator=G(123))))


def test_generator_rejects_bad_arguments():
    with pytest.raises(ValueError):
        tr.Generator(seed=1, key_mode="bogus")
    with pytest.raises(ValueError):
        tr.Generator(seed=1, partition_size=0)
    with pytest.raises(ValueError):
        tr.Generator(seed=1, partition_rank=2, partition_size=2)
    with pytest.raises(ValueError):
        G(1).advance(-1)
    with pytest.raises(ValueError):
        G(1).advance_to(-1)


def test_unseeded_generators_are_independent(request):
    known(request, "98", "Generator(seed=None) always draws the same stream", strict=True)
    a = host(tr.uniform(100, generator=tr.Generator()))
    b = host(tr.uniform(100, generator=tr.Generator()))
    assert not torch.equal(a, b)


_NKI_PARTITIONED = {
    "uniform",
    "normal",
    "exponential",
    "poisson_large",
    "normal_into",
    "uniform_into",
    "exponential_into",
}
_NKI_92 = {"gamma", "beta", "chi_squared", "truncated_normal"}


@pytest.mark.parametrize("dn", list(DRAWS))
def test_partition_ranks_draw_different_samples(request, nki_like, dn):
    # Docs: each rank draws a disjoint subrange of the logical stream.
    if not nki_like or dn not in (_NKI_PARTITIONED | _NKI_92):
        known(request, "99", "partition_rank is ignored on the PyTorch path", strict=True)
    elif dn in _NKI_92:
        known(
            request,
            "92",
            "ranks identical on NKI for rejection samplers (fix in #93)",
            strict=False,
        )
    d = DRAWS[dn]
    a = host(d(G(55, partition_rank=0, partition_size=2), 2048))
    b = host(d(G(55, partition_rank=1, partition_size=2), 2048))
    assert not torch.equal(a, b)


@pytest.mark.parametrize("km", KEY_MODES)
@pytest.mark.parametrize("dn", ["uniform", "normal", "exponential"])
def test_partition_concat_equals_single_chip(request, nki_like, dn, km):
    # Generator docstring: concatenating ranks in order reproduces one chip.
    known(
        request,
        "99",
        "partition_rank is ignored on the PyTorch path",
        strict=True,
        when=not nki_like,
    )
    P, N = 4, 4 * 2048
    d = DRAWS[dn]
    g1 = G(31, key_mode=km)
    ref = [host(d(g1, N)) for _ in range(2)]
    gs = [G(31, key_mode=km, partition_rank=r, partition_size=P) for r in range(P)]
    for call in range(2):
        parts = [host(d(gs[r], N // P)) for r in range(P)]
        assert torch.equal(torch.cat(parts), ref[call]), f"call {call}"


def test_advance_in_draw_mode_warns():
    g = G(8)
    with pytest.warns(tr.AdvanceWarning):
        g.advance(1024)
    with pytest.warns(tr.AdvanceWarning):
        g.advance_to(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        G(8, key_mode="counter").advance(1024)
        G(8).advance(0)  # no move, no warning


def test_advance_counter_mode_skips_ahead(nki_like):
    if not nki_like:
        pytest.skip("advance only affects the NKI stream (documented in Generator.position)")
    full = host(tr.uniform(4096, generator=G(8, key_mode="counter")))
    h = G(8, key_mode="counter")
    h.advance(2048)
    assert h.position() == 2048
    assert torch.equal(host(tr.uniform(2048, generator=h)), full[2048:])
    h = G(8, key_mode="counter")
    h.advance_to(1024)
    assert torch.equal(host(tr.uniform(3072, generator=h)), full[1024:])
    g = G(1)
    tr.uniform(1000, generator=g)
    assert g.position() == 1024


@pytest.mark.parametrize("dn", ["uniform", "normal"])
@pytest.mark.parametrize("parts", [(512, 512), (16384, 16384), (1024, 512, 512)], ids=str)
def test_counter_mode_split_invariance(nki_like, dn, parts):
    if not nki_like:
        pytest.skip("counter streams are an NKI-path contract")
    d = DRAWS[dn]
    whole = host(d(G(8, key_mode="counter"), sum(parts)))
    g = G(8, key_mode="counter")
    assert torch.equal(whole, torch.cat([host(d(g, p)) for p in parts]))


def test_nki_draws_equal_threefry_reference(nki_like):
    if not nki_like:
        pytest.skip("NKI only")
    for n in (1000, 16384, 70_001):
        g = G(2**30 + 99, key_mode="counter")
        assert torch.equal(
            host(tr.normal(n, generator=g)), host(dispatch.threefry_stream_normal(n, g._key48, 0))
        )
        g = G(5, key_mode="counter")
        assert torch.equal(
            host(tr.uniform(n, generator=g)), host(dispatch.threefry_stream_uniform(n, g._key48, 0))
        )


# ── quasi-random sequences ─────────────────────────────────────────────────


@pytest.mark.parametrize("d", [1, 3, 10])
@pytest.mark.parametrize("n", SIZES)
def test_sobol_unscrambled_matches_scipy(n, d):
    x = host(tr.sobol(n, d, scramble=False, dtype=torch.float64))
    ref = torch.from_numpy(qmc.Sobol(d, scramble=False).random(n))
    assert tuple(x.shape) == (n, d)
    torch.testing.assert_close(x, ref, atol=1e-7, rtol=0)


@pytest.mark.parametrize("d", [1, 3, 10])
@pytest.mark.parametrize("n", SIZES)
def test_sobol_nki_reference_matches_scipy(n, d):
    # The host reference the NKI Sobol kernel is validated against.
    x = dispatch.sobol_reference(n, d).double()
    ref = torch.from_numpy(qmc.Sobol(d, scramble=False).random(n))
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


def test_sobol_scramble_false_ignores_seed(request, nki_like):
    known(
        request,
        "90",
        "NKI sobol(scramble=False, seed=s) is shifted (fix in #91)",
        strict=False,
        when=nki_like,
    )
    assert torch.equal(
        host(tr.sobol(256, 3, scramble=False)), host(tr.sobol(256, 3, scramble=False, seed=5))
    )


@pytest.mark.parametrize("seed", [None, 0, 5])
def test_sobol_scramble_true_is_scrambled(request, nki_like, seed):
    known(
        request,
        "100",
        "NKI default sobol(scramble=True) is unscrambled for seed None/0",
        strict=True,
        when=nki_like and not seed,
    )
    plain = host(tr.sobol(256, 3, scramble=False))
    assert not torch.equal(plain, host(tr.sobol(256, 3, scramble=True, seed=seed)))


def test_sobol_scrambled_is_still_a_net(request, nki_like):
    # Owen / LMS scrambling keeps one point per 1/256 stratum in each dim.
    known(request, "100", "NKI scramble is a random shift, not Owen", strict=True, when=nki_like)
    x = host(tr.sobol(256, 3, scramble=True, seed=5)).double()
    for d in range(3):
        cells = torch.floor(x[:, d] * 256).long()
        assert torch.equal(torch.sort(cells).values, torch.arange(256)), f"dim {d}"


def test_sobol_seed_determinism():
    a = host(tr.sobol(128, 4, seed=9))
    assert torch.equal(a, host(tr.sobol(128, 4, seed=9)))
    assert not torch.equal(a, host(tr.sobol(128, 4, seed=10)))


@pytest.mark.parametrize("d", [1, 4, 10])
@pytest.mark.parametrize("n", [0, *SIZES])
def test_halton_matches_scipy(n, d):
    x = host(tr.halton(n, d, dtype=torch.float64))
    ref = torch.from_numpy(qmc.Halton(d, scramble=False).random(n + 1)[1:])
    assert tuple(x.shape) == (n, d)
    torch.testing.assert_close(x, ref, atol=1e-6, rtol=0)


@pytest.mark.parametrize("dt", FLOAT_DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize(
    "fn",
    [
        lambda n, dt: tr.sobol(n, 4, seed=3, dtype=dt),
        lambda n, dt: tr.sobol(n, 4, scramble=False, dtype=dt),
        lambda n, dt: tr.halton(n, 4, dtype=dt),
        lambda n, dt: tr.latin_hypercube(
            n, 4, dtype=dt, generator=torch.Generator().manual_seed(1)
        ),
    ],
    ids=["sobol", "sobol_unscrambled", "halton", "latin_hypercube"],
)
def test_quasi_dtype_and_unit_interval(request, fn, dt):
    # sobol: [0, 1); halton: (0, 1); latin_hypercube: [0, 1) by construction.
    lhs = "latin" in request.node.callspec.id
    n = 250 if lhs else 4096  # bf16 cannot hold 4096 LHS strata (raises by design, #96)
    x = host(fn(n, dt))
    assert x.dtype == dt
    assert tuple(x.shape) == (n, 4)
    assert bool((x >= 0).all() and (x < 1).all()), f"{int((x >= 1).sum())} points >= 1"


@pytest.mark.parametrize("dt", FLOAT_DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize("n", [1, 7, 127, 129, 250, 1000])
def test_latin_hypercube_one_point_per_stratum(n, dt):
    if dt in REDUCED and n > 250:
        pytest.skip("reduced precision cannot represent n strata (raises by design, #96)")
    x = host(tr.latin_hypercube(n, 3, dtype=dt, generator=torch.Generator().manual_seed(1)))
    assert x.dtype == dt and tuple(x.shape) == (n, 3)
    assert bool((x >= 0).all() and (x < 1).all())
    for d in range(3):
        strata = torch.floor(x[:, d].double() * n).long()
        assert torch.equal(torch.sort(strata).values, torch.arange(n)), f"dim {d}"


def test_latin_hypercube_marginals_uniform():
    x = host(
        tr.latin_hypercube(2000, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    )
    for d in range(2):
        # Within each stratum the offset is U(0, 1).
        off = (x[:, d] * 2000) - torch.floor(x[:, d] * 2000)
        assert_ks(off, stats.uniform(0, 1))


@pytest.mark.parametrize("n, d", [(1, 1), (5, 1), (3, 10), (0, 2)])
@pytest.mark.parametrize("name", ["sobol", "halton", "latin_hypercube"])
def test_quasi_edge_shapes(name, n, d):
    if name == "sobol" and n == 0:
        pytest.skip("torch.quasirandom.SobolEngine rejects n=0 (loud)")
    x = host(getattr(tr, name)(n, d))
    assert tuple(x.shape) == (n, d)


@pytest.mark.parametrize("name", ["sobol", "halton"])
def test_quasi_high_dim(request, nki_like, name):
    known(
        request,
        "89",
        "NKI sobol raises AssertionError for d > 10 instead of falling back",
        strict=False,
        when=nki_like and name == "sobol",
    )
    x = host(getattr(tr, name)(64, 21))
    assert tuple(x.shape) == (64, 21)
    assert bool((x >= 0).all() and (x < 1).all())
    if name == "halton":
        ref = torch.from_numpy(qmc.Halton(21, scramble=False).random(65)[1:])
        torch.testing.assert_close(x.double(), ref, atol=1e-6, rtol=0)


# ── backend controls ───────────────────────────────────────────────────────


def test_backend_controls(backend):
    assert isinstance(tr.HAS_NKI, bool)
    assert tr.get_backend() == backend
    with pytest.raises((AssertionError, ValueError)):
        tr.set_backend("bogus")
    assert tr.get_backend() == backend
    if not tr.HAS_NKI:
        with pytest.raises(RuntimeError):
            tr.set_backend("nki")
        assert tr.get_backend() == backend


def test_cores_controls():
    try:
        tr.set_cores(2)
        assert 1 <= tr.get_cores() <= 2
        tr.set_cores("auto")
        assert tr.get_cores() >= 1
        with pytest.raises(ValueError):
            tr.set_cores(0)
    finally:
        tr.set_cores(None)
    assert tr.get_cores() >= 1
