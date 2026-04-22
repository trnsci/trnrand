"""GeneratorProgram: pre-compiled streaming NEFF with fluent builder API.

Provides a ProgramBuilder / GeneratorProgram pair that wraps the streaming
Threefry kernels in `dispatch.py`. The typical usage:

    prog = (
        generator.new_program()
        .normal(1_000_000, out="z")
        .uniform(500_000, out="u")
        .build()
    )
    z = torch.empty(1_000_000)
    u = torch.empty(500_000)
    prog.stream_into({"z": z, "u": u})   # fills in-place, advances counter
    prog.stream_into({"z": z, "u": u})   # independent draw, same shapes

On NKI hardware, `stream_into` reuses the compiled NEFF — no recompilation
for repeated calls with identical buffer shapes. The internal counter advances
automatically so consecutive calls draw from non-overlapping regions of the
Threefry stream.

On CPU (no neuronxcc), both classes are available and fall back to
`torch.Generator`-based generation. The CPU path is not bit-exact with the
NKI path — it exists for testing and development convenience.
"""

from __future__ import annotations

import math

import torch

from .dispatch import _PROGRAM_TILES, HAS_NKI

# Trainium partition-axis limit; same constant as _THREEFRY_LANES in dispatch.py
# (which lives inside if HAS_NKI: and is not importable on dev hosts).
_THREEFRY_LANES = 128
_NORMALS_PER_LAUNCH = _PROGRAM_TILES * _THREEFRY_LANES * 4  # 32 × 128 × 4 = 16,384


class ProgramBuilder:
    """Fluent builder for a GeneratorProgram.

    Accumulates a sequence of distribution steps, then compiles them into a
    `GeneratorProgram` via `.build()`. Each step names an output buffer so
    `stream_into` knows which tensor to fill.

    Example::

        prog = (
            ProgramBuilder(seed=42)
            .normal(1_000_000, out="z")
            .uniform(500_000, out="u")
            .exponential(200_000, rate=2.0, out="e")
            .build()
        )

    `ProgramBuilder` is the internal entry point; the preferred public entry
    point is `Generator.new_program()` which seeds the builder automatically.
    """

    def __init__(self, seed: int = 0, generator=None):
        self._steps: list[tuple] = []
        if generator is not None:
            self._seed = generator.seed if generator.seed is not None else 0
        else:
            self._seed = int(seed)

    def normal(self, n: int, mean: float = 0.0, std: float = 1.0, out: str = "z") -> ProgramBuilder:
        """Queue n N(mean, std) samples into buffer `out`."""
        self._steps.append(("normal", int(n), {"mean": float(mean), "std": float(std)}, out))
        return self

    def uniform(
        self, n: int, low: float = 0.0, high: float = 1.0, out: str = "u"
    ) -> ProgramBuilder:
        """Queue n U(low, high) samples into buffer `out`."""
        self._steps.append(("uniform", int(n), {"low": float(low), "high": float(high)}, out))
        return self

    def exponential(self, n: int, rate: float = 1.0, out: str = "e") -> ProgramBuilder:
        """Queue n Exponential(rate) samples into buffer `out`."""
        self._steps.append(("exponential", int(n), {"rate": float(rate)}, out))
        return self

    def build(self) -> GeneratorProgram:
        """Compile the program.

        Returns a `GeneratorProgram` ready for `stream_into` calls. On NKI,
        the first `stream_into` triggers XLA compilation; subsequent calls
        with identical buffer shapes reuse the cached NEFF.

        Raises:
            ValueError: if no distribution steps have been added.
        """
        if not self._steps:
            raise ValueError(
                "ProgramBuilder has no distribution steps. "
                "Call .normal(), .uniform(), or .exponential() before .build()."
            )
        return GeneratorProgram(steps=list(self._steps), seed=self._seed)


class GeneratorProgram:
    """Pre-compiled streaming NEFF that fills output buffers in-place.

    Each `stream_into` call reuses the NKI compiled NEFF — no recompilation
    for repeated calls with the same buffer shapes (~10μs launch overhead vs
    minutes for first compilation). The internal counter advances automatically
    so consecutive calls draw independent, non-overlapping random streams.

    Do not construct directly; use `ProgramBuilder.build()` or
    `Generator.new_program().…build()`.

    Counter scheme (NKI path):
        Each `stream_into` step for n elements uses
        `n_launches = ceil(n / 16384)` kernel calls, consuming
        `n_launches * _PROGRAM_TILES` counter slots. The counter
        advances monotonically across all steps and all `stream_into` calls.

    CPU fallback:
        On hosts without neuronxcc, `stream_into` uses `torch.Generator`
        with `seed ^ counter` as seed. Output is not bit-exact with the NKI
        path; shape, dtype, and distribution properties are preserved.
    """

    def __init__(self, steps: list[tuple], seed: int = 0):
        self._steps = steps
        self._seed = int(seed)
        # Counter tracks next available batch index in the Threefry stream.
        # Advances by n_launches * _PROGRAM_TILES after each distribution step.
        self._counter: int = 0

    def stream_into(self, buffers: dict[str, torch.Tensor]) -> None:
        """Fill each named buffer in-place with samples from this program.

        Args:
            buffers: Mapping from output name (as declared in the builder)
                     to a pre-allocated float32 tensor. The tensor may have
                     any shape; only `numel()` samples are drawn.

        Each call advances the internal counter so repeated calls produce
        independent, non-overlapping draws from the same random stream.
        """
        if HAS_NKI:
            self._stream_into_nki(buffers)
        else:
            self._stream_into_pytorch(buffers)

    def _stream_into_nki(self, buffers: dict[str, torch.Tensor]) -> None:
        from .dispatch import threefry_stream_normal, threefry_stream_uniform

        for dist, _n, kwargs, out_name in self._steps:
            buf = buffers[out_name]
            n_actual = buf.numel()

            if dist == "normal":
                raw = threefry_stream_normal(
                    n_actual, seed=self._seed, counter_offset=self._counter
                )
                mean = kwargs["mean"]
                std = kwargs["std"]
                if std != 1.0 or mean != 0.0:
                    raw = raw * std + mean
                buf.copy_(raw.reshape(buf.shape))

            elif dist == "uniform":
                raw = threefry_stream_uniform(
                    n_actual, seed=self._seed, counter_offset=self._counter
                )
                low = kwargs["low"]
                high = kwargs["high"]
                if low != 0.0 or high != 1.0:
                    raw = raw * (high - low) + low
                buf.copy_(raw.reshape(buf.shape))

            elif dist == "exponential":
                # Inverse CDF: if U ~ Uniform(0,1) then -log(U)/rate ~ Exp(rate).
                raw = threefry_stream_uniform(
                    n_actual, seed=self._seed, counter_offset=self._counter
                )
                raw = -torch.log(raw) / kwargs["rate"]
                buf.copy_(raw.reshape(buf.shape))

            # Advance counter past the range consumed by this step.
            n_launches = math.ceil(n_actual / _NORMALS_PER_LAUNCH)
            self._counter = (self._counter + n_launches * _PROGRAM_TILES) & 0xFFFFFF

    def _stream_into_pytorch(self, buffers: dict[str, torch.Tensor]) -> None:
        """CPU fallback — uses torch.Generator; not bit-exact with NKI path."""
        gen = torch.Generator()
        gen.manual_seed(self._seed ^ self._counter)

        for dist, _n, kwargs, out_name in self._steps:
            buf = buffers[out_name]
            if dist == "normal":
                buf.normal_(mean=kwargs["mean"], std=kwargs["std"], generator=gen)
            elif dist == "uniform":
                buf.uniform_(kwargs["low"], kwargs["high"], generator=gen)
            elif dist == "exponential":
                buf.exponential_(lambd=kwargs["rate"], generator=gen)

        # Increment counter to distinguish consecutive calls on the CPU path.
        self._counter += 1
