"""
trnrand — Random number generation for AWS Trainium via NKI.

Pseudo-random and quasi-random sequences for scientific computing.
Reproducible seeding, standard distributions, and low-discrepancy
sequences for quasi-Monte Carlo integration.
Part of the trnsci scientific computing suite.
"""

__version__ = "0.2.0"

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
    # Backend
    "HAS_NKI",
    "set_backend",
    "get_backend",
]
