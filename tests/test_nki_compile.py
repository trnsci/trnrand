"""CPU compile check: every NKI kernel lowers to MLIR that passes verification.

`nki.simulate` does not run the MLIR verifier, so a kernel can pass every
simulator test and still fail to compile for hardware. #56 was such a case:
integer `nl.add` / `nl.multiply` lowered to `nisa.tensor_scalar_arith` with a
uint32 operand, which newer NKI versions reject ("'operand0' must be float32,
got 'ui32'"). This module runs the NKI frontend (trace -> MLIR -> verify) on
CPU for every kernel and target. It needs `nki` but no Neuron device and no
`neuronx-cc`; it does not run the backend compiler, so it cannot catch
errors raised after MLIR generation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.nki_simulator

from trnrand.nki.dispatch import HAS_NKI  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None

_P = 128


def _i32(n_args):
    return tuple(np.zeros((_P, 1), dtype=np.int32) for _ in range(n_args))


def _bulk(cols, chunks):
    return (
        np.zeros((_P, cols), dtype=np.int32),
        np.zeros((_P, chunks), dtype=np.int32),
    ) + _i32(4)


# kernel name -> example inputs (shapes/dtypes match the host wrappers)
_KERNEL_INPUTS = {
    "philox4x32_kernel": lambda: _i32(3),
    "box_muller_kernel": lambda: (np.full((_P, 2), 0.5, dtype=np.float32),),
    "threefry4x32_kernel": lambda: _i32(8),
    "threefry_normal_kernel": lambda: _i32(8),
    "sobol_gray_code_kernel": lambda: _i32(1),
    "halton_kernel": lambda: (np.ones((_P, 1), dtype=np.int32),),
    "threefry_streaming_normal_kernel": lambda: _i32(6),
    "threefry_streaming_uniform_kernel": lambda: _i32(6),
    # Bulk kernels (#60): (P, F) lanes, (P, C) chunk starts, 4 × (P, 1) key words.
    # The largest default shape, F = 512 columns × C = 16 chunks.
    "threefry_bulk_normal_kernel": lambda: _bulk(512, 16),
    "threefry_bulk_uniform_kernel": lambda: _bulk(512, 16),
    # Same kernels on 16-bit limbs / full words (TRNRAND_LIMB_BITS=16|32).
    "threefry_bulk16_normal_kernel": lambda: _bulk(512, 16),
    "threefry_bulk16_uniform_kernel": lambda: _bulk(512, 16),
    "threefry_bulk32_normal_kernel": lambda: _bulk(512, 16),
    "threefry_bulk32_uniform_kernel": lambda: _bulk(512, 16),
}


@pytest.fixture(autouse=True)
def _frontend_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")


@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("kernel_name", sorted(_KERNEL_INPUTS))
def test_kernel_mlir_verifies(kernel_name, target):
    from trnrand.nki import dispatch

    kernel = getattr(dispatch, kernel_name)
    result = source_to_mlir(
        kernel, *_KERNEL_INPUTS[kernel_name](), target=target, frontend=FrontendKind.TRACER
    )
    result.verify()


# Full-word kernels split work across the two physical cores of an LNC=2
# logical core (#60): whole chunks per core when C is even, else half the
# columns of every chunk. Trace both cores and verify every split shape.
@pytest.mark.parametrize("cols,chunks", [(512, 16), (512, 5), (512, 1), (32, 1), (64, 2)])
@pytest.mark.parametrize(
    "kernel_name", ["threefry_bulk32_normal_kernel", "threefry_bulk32_uniform_kernel"]
)
def test_bulk32_lnc2_mlir_verifies(kernel_name, cols, chunks):
    from trnrand.nki import dispatch

    kernel = getattr(dispatch, kernel_name)
    result = source_to_mlir(
        kernel, *_bulk(cols, chunks), target="trn2", lnc=2, frontend=FrontendKind.TRACER
    )
    result.verify()
