"""NKI kernel dispatch for Trainium RNG acceleration."""

from .dispatch import HAS_NKI, set_backend, get_backend

__all__ = ["HAS_NKI", "set_backend", "get_backend"]
