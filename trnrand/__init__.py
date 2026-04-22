"""
trnrand — Random number generation for AWS Trainium via NKI.

Pseudo-random and quasi-random sequences for scientific computing.
Reproducible seeding, standard distributions, and low-discrepancy
sequences for quasi-Monte Carlo integration.
Part of the trnsci scientific computing suite.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("trnrand")
except PackageNotFoundError:
    __version__ = "unknown"

# Generator
# Distributions
from .distributions import (
    bernoulli,
    beta,
    chi_squared,
    exponential,
    gamma,
    normal,
    poisson,
    randint,
    randperm,
    standard_normal,
    truncated_normal,
    uniform,
)
from .generator import Generator, get_default_generator, manual_seed

# Backend control
from .nki import HAS_NKI, get_backend, set_backend

# Streaming generator program
from .nki.program import GeneratorProgram

# Quasi-random sequences
from .quasi import halton, latin_hypercube, sobol

__all__ = [
    # Generator
    "Generator",
    "manual_seed",
    "get_default_generator",
    # Distributions
    "uniform",
    "normal",
    "standard_normal",
    "exponential",
    "bernoulli",
    "randint",
    "randperm",
    "truncated_normal",
    "gamma",
    "chi_squared",
    "beta",
    "poisson",
    # Quasi-random
    "sobol",
    "halton",
    "latin_hypercube",
    # Streaming program API
    "GeneratorProgram",
    # Backend
    "HAS_NKI",
    "set_backend",
    "get_backend",
]
