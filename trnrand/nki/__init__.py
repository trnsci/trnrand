"""NKI kernel dispatch for Trainium RNG acceleration."""

from .dispatch import HAS_NKI, get_backend, get_cores, set_backend, set_cores

__all__ = ["HAS_NKI", "set_backend", "get_backend", "set_cores", "get_cores"]
