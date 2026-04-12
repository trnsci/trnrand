"""
trnrand — Random number generation for AWS Trainium via NKI.

Pseudo-random and quasi-random sequences for scientific computing.
Reproducible seeding, standard distributions, and low-discrepancy
sequences for quasi-Monte Carlo integration.
Part of the trn-* scientific computing suite by Playground Logic.
"""

__version__ = "0.1.0"

# Generator
from .generator import Generator, manual_seed, get_default_generator

# Distributions
from .distributions import (
    uniform, normal, standard_normal, exponential,
    bernoulli, randint, randperm, truncated_normal,
)

# Quasi-random sequences
from .quasi import sobol, halton, latin_hypercube

# Backend control
from .nki import HAS_NKI, set_backend, get_backend

__all__ = [
    # Generator
    "Generator", "manual_seed", "get_default_generator",
    # Distributions
    "uniform", "normal", "standard_normal", "exponential",
    "bernoulli", "randint", "randperm", "truncated_normal",
    # Quasi-random
    "sobol", "halton", "latin_hypercube",
    # Backend
    "HAS_NKI", "set_backend", "get_backend",
]
