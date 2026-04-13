"""NKI kernel dispatch for Trainium RNG acceleration."""

from .dispatch import HAS_NKI, get_backend, set_backend

__all__ = ["HAS_NKI", "set_backend", "get_backend"]
