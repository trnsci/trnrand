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
            self._partition_rank = getattr(generator, "_partition_rank", 0)
            self._partition_size = getattr(generator, "_partition_size", 1)
            self._counter_start = getattr(generator, "_counter", 0)
        else:
            self._seed = int(seed)
            self._partition_rank = 0
            self._partition_size = 1
            self._counter_start = 0

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
        return GeneratorProgram(
            steps=list(self._steps),
            seed=self._seed,
            partition_rank=self._partition_rank,
            partition_size=self._partition_size,
            counter_start=self._counter_start,
        )


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
        A partitioned program (rank r of P) draws each step's P × n samples
        and keeps its slice, so ranks concatenate to the single-chip output.
    """

    def __init__(
        self,
        steps: list[tuple],
        seed: int = 0,
        partition_rank: int = 0,
        partition_size: int = 1,
        counter_start: int = 0,
    ):
        self._steps = steps
        self._seed = int(seed)
        self._partition_rank = partition_rank
        self._partition_size = partition_size
        # Counter starts at the generator's snapshot position (0 for single-chip).
        # Advances by n_launches * _PROGRAM_TILES after each distribution step.
        self._counter: int = counter_start

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
        from .dispatch import _float64_to_torch, threefry_stream_normal, threefry_stream_uniform

        # The kernels draw real float32 values. An integer / bool buffer raises on
        # the CPU path too (normal_ / uniform_ / exponential_), and a complex one
        # would get a zero imaginary part, so refuse both before drawing anything
        # (trnsci/trnsci#35).
        for _dist, _n, _kwargs, out_name in self._steps:
            dt = buffers[out_name].dtype
            if not dt.is_floating_point:
                raise NotImplementedError(
                    f"GeneratorProgram.stream_into: buffer {out_name!r} is {dt}; the NKI "
                    "path fills real floating buffers (float32, float64, float16, bfloat16)"
                )
        # float64 buffers (trnrand#105): filled in torch at float64 precision under
        # backend "auto", by the float32 kernel (cast, warned once) under "nki";
        # TRNRAND_REQUIRE_NKI raises here, before anything is drawn.
        in_torch = {
            out_name
            for _dist, _n, _kwargs, out_name in self._steps
            if _float64_to_torch(
                "GeneratorProgram.stream_into", buffers[out_name].dtype, buffers[out_name]
            )
        }

        for dist, _n, kwargs, out_name in self._steps:
            buf = buffers[out_name]
            n_actual = buf.numel()

            n_launches = math.ceil(n_actual / _NORMALS_PER_LAUNCH)
            streaming_batches = n_launches * _PROGRAM_TILES
            # Chip r of P uses [r×step_batches, (r+1)×step_batches) relative to
            # self._counter, so its counter_offset is shifted by partition_rank.
            counter_offset = self._counter + self._partition_rank * streaming_batches

            if out_name in in_torch:
                # Same counter slots as the kernel step, so the other steps'
                # streams do not move; the torch generator is keyed by
                # (seed, counter_offset), so calls and ranks stay independent.
                gen = torch.Generator()
                gen.manual_seed((self._seed * 0x9E3779B97F4A7C15 + counter_offset) % (1 << 63))
                self._fill_torch(buf, dist, kwargs, gen)

            elif dist == "normal":
                raw = threefry_stream_normal(
                    n_actual, seed=self._seed, counter_offset=counter_offset
                )
                mean = kwargs["mean"]
                std = kwargs["std"]
                if std != 1.0 or mean != 0.0:
                    raw = raw * std + mean
                buf.copy_(raw.reshape(buf.shape))

            elif dist == "uniform":
                raw = threefry_stream_uniform(
                    n_actual, seed=self._seed, counter_offset=counter_offset
                )
                low = kwargs["low"]
                high = kwargs["high"]
                if low != 0.0 or high != 1.0:
                    raw = raw * (high - low) + low
                if buf.dtype in (torch.float16, torch.bfloat16) and high > low:
                    from ..distributions import _below_high

                    raw = _below_high(raw.to(buf.dtype), high)  # as uniform() does (#94)
                buf.copy_(raw.reshape(buf.shape))

            elif dist == "exponential":
                # Inverse CDF: if U ~ Uniform(0,1) then -log(U)/rate ~ Exp(rate).
                raw = threefry_stream_uniform(
                    n_actual, seed=self._seed, counter_offset=counter_offset
                )
                # Clamp as exponential() does: the kernel can emit u == 0 (#82).
                raw = -torch.log(torch.clamp(raw, min=1e-10)) / kwargs["rate"]
                buf.copy_(raw.reshape(buf.shape))

            # Advance counter by the FULL single-chip step size so that the
            # next step's base is aligned with what a 1-chip reference program
            # would have consumed.  For P chips each processing n/P samples:
            #   streaming_batches     = batches this chip consumed
            #   streaming_batches * P = batches a 1-chip run would consume
            # For P=1 this is a no-op (× 1).
            self._counter = (self._counter + streaming_batches * self._partition_size) & 0xFFFFFF

    def _stream_into_pytorch(self, buffers: dict[str, torch.Tensor]) -> None:
        """CPU fallback — uses torch.Generator; not bit-exact with NKI path."""
        gen = torch.Generator()
        gen.manual_seed(self._seed ^ self._counter)

        P, r = self._partition_size, self._partition_rank
        for dist, _n, kwargs, out_name in self._steps:
            buf = buffers[out_name]
            # Rank r of P fills its slice of the P × n single-chip draw (#99).
            n = buf.numel()
            t = buf if P == 1 else torch.empty(P * n, dtype=buf.dtype, device=buf.device)
            self._fill_torch(t, dist, kwargs, gen)
            if P > 1:
                buf.copy_(t[r * n : (r + 1) * n].reshape(buf.shape))

        # Increment counter to distinguish consecutive calls on the CPU path.
        self._counter += 1

    @staticmethod
    def _fill_torch(buf: torch.Tensor, dist: str, kwargs: dict, gen: torch.Generator) -> None:
        # A device buffer is filled on the host (the generator is a CPU one);
        # a float64 one on the neuron device warns once (`_note_fp64_host_fill`).
        from .dispatch import _note_fp64_host_fill, _on_neuron_device

        on_host = buf.device.type == "cpu" and not _on_neuron_device(buf)
        if not on_host:
            _note_fp64_host_fill("GeneratorProgram.stream_into", buf)
        out = buf if on_host else torch.empty(buf.shape, dtype=buf.dtype)
        if dist == "normal":
            out.normal_(mean=kwargs["mean"], std=kwargs["std"], generator=gen)
        elif dist == "uniform":
            out.uniform_(kwargs["low"], kwargs["high"], generator=gen)
        elif dist == "exponential":
            out.exponential_(lambd=kwargs["rate"], generator=gen)
        if out is not buf:
            buf.copy_(out)
