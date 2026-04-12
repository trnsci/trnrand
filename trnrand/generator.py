"""
Random number generator state management for Trainium.

Provides a Generator class that wraps torch.Generator for CPU/CUDA
and will wrap NKI-based on-device RNG for Trainium. Reproducible
seeding is critical for scientific computing — same seed, same stream,
regardless of backend.

The Philox counter-based RNG is the natural NKI target because it's
stateless (counter + key → output), embarrassingly parallel across
tiles, and used by both PyTorch and JAX as their default engine.
"""

from __future__ import annotations

import torch
from typing import Optional


class Generator:
    """Seeded random number generator with backend dispatch.

    Wraps torch.Generator for CPU/CUDA. On NKI, would use a Philox
    counter-based RNG running on the GpSimd engine.
    """

    def __init__(self, seed: Optional[int] = None, device: str = "cpu"):
        self._device = device
        self._torch_gen = torch.Generator(device=device)
        if seed is not None:
            self._torch_gen.manual_seed(seed)
        self._seed = seed

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def device(self) -> str:
        return self._device

    def manual_seed(self, seed: int) -> Generator:
        self._seed = seed
        self._torch_gen.manual_seed(seed)
        return self

    def get_state(self) -> torch.Tensor:
        return self._torch_gen.get_state()

    def set_state(self, state: torch.Tensor):
        self._torch_gen.set_state(state)

    @property
    def torch_generator(self) -> torch.Generator:
        """Access underlying torch.Generator for use with torch ops."""
        return self._torch_gen


# Module-level default generator
_default_generator = Generator(seed=None)


def manual_seed(seed: int) -> Generator:
    """Seed the default generator."""
    global _default_generator
    _default_generator = Generator(seed=seed)
    return _default_generator


def get_default_generator() -> Generator:
    return _default_generator
