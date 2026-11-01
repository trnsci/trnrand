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

import itertools
import math
import os
import weakref
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .nki.program import ProgramBuilder

# Threefry batch: _THREEFRY_LANES × 4 output words = 128 × 4 = 512 samples.
# Drives partition counter arithmetic and advance() / position() conversions.
_BATCH_SIZE: int = 128 * 4

# Live generators by integer handle. Under torch.compile the state step runs in
# an opaque custom op (trnrand/_compile.py), which cannot take a Generator or
# torch.Generator argument (Dynamo cannot proxy either, #61); it takes the
# handle and looks the generator up here at run time.
_HANDLES = itertools.count()

_KEY_MODES = ("draw", "counter")


def _key48(seed: int) -> int:
    """Fold a 64-bit seed to the 48 key bits the Threefry kernels take (k0, k1 < 2^24)."""
    seed &= 0xFFFFFFFFFFFFFFFF
    return (seed & 0xFFFFFFFFFFFF) ^ (seed >> 48)


_REGISTRY: weakref.WeakValueDictionary[int, Generator] = weakref.WeakValueDictionary()


def _lookup(handle: int) -> Generator:
    gen = _REGISTRY.get(handle)
    if gen is None:
        raise RuntimeError(f"trnrand Generator handle {handle} is no longer alive")
    return gen


class Generator:
    """Seeded random number generator with backend dispatch.

    Wraps torch.Generator for CPU/CUDA. On NKI, uses Threefry counter-based
    RNG running on the GpSimd engine.

    Multi-chip usage (Phase 4)::

        gen = Generator(
            seed=42,
            partition_rank=xm.get_ordinal(),
            partition_size=xm.xrt_world_size(),
        )
        # Each chip draws from a disjoint subrange of the same logical stream.
        # Concatenating in rank order reproduces the single-chip output exactly.

    On the native TorchNeuron path (no torch_xla), take the rank and world
    size from ``torch.distributed.get_rank()`` / ``get_world_size()``
    instead. The partition arithmetic is host-side and identical on both
    dispatch paths.

    Single-chip defaults (``partition_rank=0, partition_size=1``) preserve
    all existing behaviour identically.
    """

    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        partition_rank: int = 0,
        partition_size: int = 1,
        key_mode: str | None = None,
    ):
        if key_mode is None:
            key_mode = os.environ.get("TRNRAND_KEY_MODE", "draw").strip().lower()
        if key_mode not in _KEY_MODES:
            raise ValueError(f"key_mode must be one of {_KEY_MODES}; got {key_mode!r}")
        if partition_size < 1:
            raise ValueError(f"partition_size must be >= 1; got {partition_size}")
        if not (0 <= partition_rank < partition_size):
            raise ValueError(
                f"partition_rank must be in [0, partition_size); "
                f"got rank={partition_rank}, size={partition_size}"
            )
        self._partition_rank = partition_rank
        self._partition_size = partition_size
        self._key_mode = key_mode
        # Current position in Threefry batch units. A 0-d tensor, so a compiled
        # graph can advance it with a plain aten add_ (#61); `_counter` is the
        # int view eager code uses.
        self._ctr = torch.zeros((), dtype=torch.int64)
        self._device = device
        self._torch_gen = torch.Generator(device=device)
        if seed is not None:
            self._torch_gen.manual_seed(seed)
        self._seed = seed
        # key_mode="counter": Threefry key words (k0, k1), int32, fixed per seed.
        self._key_words = torch.zeros(2, dtype=torch.int32)
        self._set_key()
        self._handle: int = next(_HANDLES)
        _REGISTRY[self._handle] = self
        # Bumped by the torch.compile state-step ops, which declare it mutated so
        # graph passes keep every draw, in order, even when its output is unused.
        self._compile_token = torch.zeros((), dtype=torch.int64)

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def key_mode(self) -> str:
        """How NKI draws pick their Threefry key.

        ``"draw"`` (default): each call draws a fresh 24-bit key from the
        underlying ``torch.Generator``. ``"counter"``: the key is fixed by the
        seed and only the counter advances between calls, so no host RNG state
        is touched and the draw is traceable by ``torch.compile`` on the
        ``neuron`` backend (#61). The two modes give different streams.
        Default from ``TRNRAND_KEY_MODE``.
        """
        return self._key_mode

    @property
    def _counter(self) -> int:
        return int(self._ctr)

    @_counter.setter
    def _counter(self, value: int) -> None:
        self._ctr.fill_(int(value))

    def _set_key(self) -> None:
        k = _key48(self._torch_gen.initial_seed())
        self._key48 = k
        self._key_words.copy_(torch.tensor([k & 0xFFFFFF, (k >> 24) & 0xFFFFFF], dtype=torch.int32))

    @property
    def device(self) -> str:
        return self._device

    def manual_seed(self, seed: int) -> Generator:
        """Reseed the generator. Resets the counter — starts a fresh logical stream."""
        self._seed = seed
        self._torch_gen.manual_seed(seed)
        self._set_key()
        self._counter = 0  # reseeding restarts the stream; partition context is structural
        return self

    # ── Counter / position API ────────────────────────────────────────────────

    def position(self) -> int:
        """Logical sample count consumed from this generator's stream.

        Returns the number of samples that have been drawn (rounded up to the
        nearest Threefry batch boundary). Use with :meth:`advance_to` for
        checkpoint / resume::

            pos = gen.position()
            # ... experiment runs, crashes, or cluster is reshaped ...
            new_gen = Generator(seed=s, partition_rank=r, partition_size=P)
            new_gen.advance_to(pos)   # resume from the same point
        """
        return self._counter * _BATCH_SIZE

    def advance(self, n_samples: int) -> None:
        """Skip *n_samples* forward in the stream without generating them.

        Increments the internal counter by ``ceil(n_samples / 512)`` batches.
        Cheap — no RNG work is performed.

        Args:
            n_samples: Number of samples to skip (non-negative).
        """
        if n_samples < 0:
            raise ValueError(f"n_samples must be non-negative; got {n_samples}")
        self._counter += math.ceil(n_samples / _BATCH_SIZE)

    def advance_to(self, position: int) -> None:
        """Jump to an absolute sample position for checkpoint / resume.

        Args:
            position: Target sample index (non-negative, as returned by
                :meth:`position`). Rounded down to the nearest batch boundary.
        """
        if position < 0:
            raise ValueError(f"position must be non-negative; got {position}")
        self._counter = position // _BATCH_SIZE

    def _chip_counter_offset(self, n_elements: int) -> int:
        """Counter offset for chip r of P dispatching n_elements samples.

        For partition equivalence, chip r of P uses counter range::

            [r × n_batches + _counter, (r+1) × n_batches + _counter)

        where ``n_batches = ceil(n_elements / 512)``.  Concatenating outputs
        from all P chips in rank order reproduces the single-chip output
        (provided ``n_elements`` is a multiple of 512 × P).
        """
        n_batches = math.ceil(n_elements / _BATCH_SIZE)
        return self._partition_rank * n_batches + self._counter

    def _advance_by_elements(self, n_elements: int) -> None:
        """Advance _counter past one partitioned draw of n_elements samples per chip.

        All P chips together consume ``P × ceil(n_elements / 512)`` batches
        (chip r takes the r-th slice, see :meth:`_chip_counter_offset`), so
        the counter moves by that much: the next draw starts after every
        chip's slice. Advancing by one chip's share instead made chip r's
        next draw reuse chip r+1's counters (with key_mode="counter" the
        values repeated across chips) and broke single-chip equivalence
        from the second draw on. For P = 1 this is ``ceil(n / 512)``.
        """
        self._counter += math.ceil(n_elements / _BATCH_SIZE) * self._partition_size

    # ── State management ──────────────────────────────────────────────────────

    def get_state(self) -> torch.Tensor:
        return self._torch_gen.get_state()

    def set_state(self, state: torch.Tensor):
        self._torch_gen.set_state(state)

    @property
    def torch_generator(self) -> torch.Generator:
        """Access underlying torch.Generator for use with torch ops."""
        return self._torch_gen

    def new_program(self) -> ProgramBuilder:
        """Return a ProgramBuilder seeded from this generator.

        Builds a pre-compiled streaming NEFF program via a fluent API::

            prog = (
                gen.new_program()
                .normal(1_000_000, out="z")
                .uniform(500_000, out="u")
                .build()
            )
            z = torch.empty(1_000_000)
            u = torch.empty(500_000)
            prog.stream_into({"z": z, "u": u})

        On NKI, the first `stream_into` compiles the NEFF; subsequent calls
        with identical buffer shapes reuse the cached kernel. On CPU (no
        neuronxcc), falls back to torch.Generator-based generation.

        Returns:
            ProgramBuilder seeded with this generator's current seed.
        """
        from .nki.program import ProgramBuilder

        return ProgramBuilder(generator=self)


# Module-level default generator
_default_generator = Generator(seed=None)


def manual_seed(seed: int) -> Generator:
    """Seed the default generator."""
    global _default_generator
    _default_generator = Generator(seed=seed)
    return _default_generator


def get_default_generator() -> Generator:
    return _default_generator
