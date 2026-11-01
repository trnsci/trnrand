"""
torch.compile support for generator-driven sampling (#61).

Dynamo cannot proxy a ``torch.Generator`` (nor a trnrand ``Generator``) as an
argument to a traced call, so ``tensor.normal_(..., generator=g)`` and
``torch.randint(..., generator=g)`` inside a ``fullgraph=True`` region fail with
``Unsupported: Failed to convert args/kwargs to proxy``. ``torch.compiler.disable``
does not help under ``fullgraph=True`` (calling a disabled function is itself a
graph break).

Instead, the generator-state step is a custom op that takes the generator's
integer handle. Dynamo sees an opaque node with plain int/float arguments; at
run time the op looks the generator up and performs exactly the eager draw,
so values and stream order are bit-identical to eager for every seed. Each op
also declares a mutation of the generator's 0-d ``_compile_token`` tensor (and
bumps it), so AOT/Inductor passes neither dead-code-eliminate a draw whose
output is unused (which would skip a state advance) nor reorder or CSE draws.

- ``trnrand::gen_fill``       PyTorch-backend ``uniform_`` / ``normal_`` fill.
- ``trnrand::gen_nki_state``  NKI path: draw the 24-bit kernel seed, compute the
  partition counter offset, advance the counter; returns int64 (seed, offset).
  The kernel launches that consume it stay in the graph
  (``dispatch.threefry_stream_from_state``): on the native path these are the
  bulk Threefry kernels (#60), with the same ``_bulk_plan`` launches and limb
  variant (``TRNRAND_LIMB_BITS``) as eager, or the streaming kernels under
  ``TRNRAND_BULK_KERNEL=0``. The plan and the limb variant are fixed when the
  graph is traced (they depend only on the static request size and the env).
- ``trnrand::gen_nki_draw``   NKI path where the launches are not traceable
  (XLA lineage, simulator, per-tile kernels): the whole eager draw, opaque.

Lowering limits. ``backend="neuron"`` lowers the whole FX graph to StableHLO;
only aten ops and ``torch_neuronx.nki_op`` kernels survive, so the ``gen_*``
ops above fail there ("failed to legalize operation 'torch.operator'"). They
serve the CPU compile backends (eager / aot_eager / inductor). For
``backend="neuron"``, use a ``Generator(key_mode="counter")`` (or
``TRNRAND_KEY_MODE=counter``): the Threefry key is fixed by the seed and only
the counter advances, held as tensors on the Generator (``_key_words``,
``_ctr``), so the native draw is plain aten ops building the bulk-kernel
inputs (key words from the seed, start tiles from ``_ctr``) plus the nki_op
launches, with the counter advanced by an in-place ``add_`` (functionalized
as an input mutation). The default ``key_mode="draw"`` derives each call's key from a
``torch.randint`` on the Generator's Mersenne Twister, which has no aten-op
equivalent in a graph, so it cannot lower on neuron without changing the
stream. The PyTorch-backend path (``set_backend("pytorch")``) is likewise not
lowerable on neuron.

``torch.library.custom_op`` needs PyTorch 2.4+; on older PyTorch ``AVAILABLE`` is
False and callers keep the eager code (which graph-breaks as before).
"""

from __future__ import annotations

import torch

from .generator import _BATCH_SIZE, _lookup

AVAILABLE = hasattr(torch.library, "custom_op")

if AVAILABLE:
    _TAGS = (torch.Tag.nondeterministic_seeded,)

    @torch.library.custom_op("trnrand::gen_fill", mutates_args=("token",), tags=_TAGS)
    def gen_fill(
        token: torch.Tensor,
        handle: int,
        size: list[int],
        kind: str,
        a: float,
        b: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        token.add_(1)
        g = _lookup(handle).torch_generator
        out = torch.empty(*size, dtype=dtype)
        if kind == "uniform":
            return out.uniform_(a, b, generator=g)
        return out.normal_(a, b, generator=g)

    @gen_fill.register_fake
    def _(token, handle, size, kind, a, b, dtype):
        return torch.empty(*size, dtype=dtype)

    @torch.library.custom_op("trnrand::gen_nki_state", mutates_args=("token",), tags=_TAGS)
    def gen_nki_state(token: torch.Tensor, handle: int, n_elements: int) -> torch.Tensor:
        from .distributions import _nki_seed

        token.add_(1)
        gen = _lookup(handle)
        seed = _nki_seed(gen)
        offset = gen._chip_counter_offset(n_elements)
        gen._advance_by_elements(n_elements)
        return torch.tensor([seed, offset], dtype=torch.int64)

    @gen_nki_state.register_fake
    def _(token, handle, n_elements):
        return torch.empty(2, dtype=torch.int64)

    @torch.library.custom_op("trnrand::gen_nki_draw", mutates_args=("token",), tags=_TAGS)
    def gen_nki_draw(token: torch.Tensor, handle: int, n_elements: int, kind: str) -> torch.Tensor:
        from .distributions import _nki_draw_eager

        token.add_(1)
        return _nki_draw_eager(kind, _lookup(handle), n_elements)

    @gen_nki_draw.register_fake
    def _(token, handle, n_elements, kind):
        return torch.empty(n_elements, dtype=torch.float32)


def fill(gen, size, kind: str, a: float, b: float, dtype: torch.dtype) -> torch.Tensor:
    """Traceable ``torch.empty(*size).{uniform_,normal_}(a, b, generator=gen)``."""
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        size = tuple(size[0])
    return torch.ops.trnrand.gen_fill(
        gen._compile_token, gen._handle, list(size), kind, float(a), float(b), dtype
    )


def nki_draw(gen, kind: str, n_elements: int) -> torch.Tensor:
    """Traceable NKI Threefry draw of ``n_elements`` uniforms or normals (float32)."""
    from .nki import dispatch

    native = (
        dispatch._get_dispatch_path() == "native"
        and not dispatch._use_simulator()
        and dispatch._bulk_streaming()
    )
    if native and gen._key_mode == "counter":
        # Lowerable on backend="neuron": plain aten ops on the generator's tensor
        # counter and key words, plus the nki_op launches. No trnrand custom op.
        n_batches = -(-n_elements // _BATCH_SIZE)
        offset = gen._ctr + gen._partition_rank * n_batches
        out = dispatch.threefry_stream_from_key(kind, n_elements, gen._key_words, offset)
        gen._ctr.add_(n_batches * gen._partition_size)  # every chip's slice
        return out
    if native:
        state = torch.ops.trnrand.gen_nki_state(gen._compile_token, gen._handle, n_elements)
        return dispatch.threefry_stream_from_state(kind, n_elements, state)
    return torch.ops.trnrand.gen_nki_draw(gen._compile_token, gen._handle, n_elements, kind)
