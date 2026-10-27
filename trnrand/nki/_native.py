"""
Native TorchNeuron registration of the trnrand NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel from
`dispatch.py` unchanged; the op schema is inferred from the annotations.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native backend. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on `dispatch` at call time rather than imported
by name, so registration itself does not trace or compile anything; the
first call per input shape does.
"""

import torch
from torch_neuronx import nki_op

from . import dispatch as _d

# Fake (meta) implementations: output shape/dtype only, no kernel trace.
# Used by torch.compile tracing and for meta-device inputs.


def _fake_rows(ref: torch.Tensor, cols: int, dtype: torch.dtype) -> torch.Tensor:
    return ref.new_empty((ref.shape[0], cols), dtype=dtype)


@nki_op(
    "trnrand::philox4x32",
    mutates_args={},
    fake_fn=lambda counter_lo, key_lo, key_hi: _fake_rows(counter_lo, 4, counter_lo.dtype),
)
def philox4x32(
    counter_lo: torch.Tensor, key_lo: torch.Tensor, key_hi: torch.Tensor
) -> torch.Tensor:
    return _d.philox4x32_kernel(counter_lo, key_lo, key_hi)


@nki_op(
    "trnrand::box_muller",
    mutates_args={},
    fake_fn=lambda uniforms: _fake_rows(uniforms, 2, uniforms.dtype),
)
def box_muller(uniforms: torch.Tensor) -> torch.Tensor:
    return _d.box_muller_kernel(uniforms)


@nki_op(
    "trnrand::threefry4x32",
    mutates_args={},
    fake_fn=lambda c0, c1, c2, c3, k0, k1, k2, k3: _fake_rows(c0, 4, torch.float32),
)
def threefry4x32(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry4x32_kernel(c0, c1, c2, c3, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_normal",
    mutates_args={},
    fake_fn=lambda c0, c1, c2, c3, k0, k1, k2, k3: _fake_rows(c0, 4, torch.float32),
)
def threefry_normal(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_normal_kernel(c0, c1, c2, c3, k0, k1, k2, k3)


@nki_op(
    "trnrand::sobol_gray_code",
    mutates_args={},
    fake_fn=lambda gray: _fake_rows(gray, _d._SOBOL_MAX_DIMS, torch.float32),
)
def sobol_gray_code(gray: torch.Tensor) -> torch.Tensor:
    return _d.sobol_gray_code_kernel(gray)


@nki_op(
    "trnrand::halton",
    mutates_args={},
    fake_fn=lambda indices: _fake_rows(indices, _d._HALTON_MAX_DIMS, torch.float32),
)
def halton(indices: torch.Tensor) -> torch.Tensor:
    return _d.halton_kernel(indices)


@nki_op(
    "trnrand::threefry_streaming_normal",
    mutates_args={},
    fake_fn=lambda c0, start_batch, k0, k1, k2, k3: _fake_rows(
        c0, 4 * _d._PROGRAM_TILES, torch.float32
    ),
)
def threefry_streaming_normal(
    c0: torch.Tensor,
    start_batch: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_streaming_normal_kernel(c0, start_batch, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_streaming_uniform",
    mutates_args={},
    fake_fn=lambda c0, start_batch, k0, k1, k2, k3: _fake_rows(
        c0, 4 * _d._PROGRAM_TILES, torch.float32
    ),
)
def threefry_streaming_uniform(
    c0: torch.Tensor,
    start_batch: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_streaming_uniform_kernel(c0, start_batch, k0, k1, k2, k3)


def _fake_bulk(c0, start):
    return c0.new_empty((c0.shape[0], start.shape[1] * 4 * c0.shape[1]), dtype=torch.float32)


@nki_op(
    "trnrand::threefry_bulk_normal",
    mutates_args={},
    fake_fn=lambda c0, start, k0, k1, k2, k3: _fake_bulk(c0, start),
)
def threefry_bulk_normal(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_bulk_normal_kernel(c0, start, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_bulk_uniform",
    mutates_args={},
    fake_fn=lambda c0, start, k0, k1, k2, k3: _fake_bulk(c0, start),
)
def threefry_bulk_uniform(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_bulk_uniform_kernel(c0, start, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_bulk16_normal",
    mutates_args={},
    fake_fn=lambda c0, start, k0, k1, k2, k3: _fake_bulk(c0, start),
)
def threefry_bulk16_normal(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_bulk16_normal_kernel(c0, start, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_bulk16_uniform",
    mutates_args={},
    fake_fn=lambda c0, start, k0, k1, k2, k3: _fake_bulk(c0, start),
)
def threefry_bulk16_uniform(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_bulk16_uniform_kernel(c0, start, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_bulk32_normal",
    mutates_args={},
    fake_fn=lambda c0, start, k0, k1, k2, k3: _fake_bulk(c0, start),
)
def threefry_bulk32_normal(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_bulk32_normal_kernel(c0, start, k0, k1, k2, k3)


@nki_op(
    "trnrand::threefry_bulk32_uniform",
    mutates_args={},
    fake_fn=lambda c0, start, k0, k1, k2, k3: _fake_bulk(c0, start),
)
def threefry_bulk32_uniform(
    c0: torch.Tensor,
    start: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
) -> torch.Tensor:
    return _d.threefry_bulk32_uniform_kernel(c0, start, k0, k1, k2, k3)


# Keyed by the short op name used in `dispatch._launch`.
OPS = {
    "philox4x32": philox4x32,
    "box_muller": box_muller,
    "threefry4x32": threefry4x32,
    "threefry_normal": threefry_normal,
    "sobol_gray_code": sobol_gray_code,
    "halton": halton,
    "threefry_streaming_normal": threefry_streaming_normal,
    "threefry_streaming_uniform": threefry_streaming_uniform,
    "threefry_bulk_normal": threefry_bulk_normal,
    "threefry_bulk_uniform": threefry_bulk_uniform,
    "threefry_bulk16_normal": threefry_bulk16_normal,
    "threefry_bulk16_uniform": threefry_bulk16_uniform,
    "threefry_bulk32_normal": threefry_bulk32_normal,
    "threefry_bulk32_uniform": threefry_bulk32_uniform,
}
