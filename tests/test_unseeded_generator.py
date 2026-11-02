"""Generator(seed=None) seeds from OS entropy (#98).

Through 0.13.0 a fresh ``torch.Generator`` started at torch's fixed default
seed, so every unseeded trnrand Generator, in every process, drew the same
stream. Seeded generators must stay bit-identical.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

import trnrand as tr


@pytest.fixture(autouse=True)
def _pytorch_backend():
    prev = tr.get_backend()
    tr.set_backend("pytorch")
    yield
    tr.set_backend(prev)


def test_two_unseeded_generators_differ():
    a = tr.uniform(100, generator=tr.Generator())
    b = tr.uniform(100, generator=tr.Generator())
    assert not torch.equal(a, b)


def test_unseeded_generator_not_torch_default_seed():
    assert tr.Generator().torch_generator.initial_seed() != torch.Generator().initial_seed()
    assert tr.Generator().seed is None


def test_unseeded_state_replays():
    g = tr.Generator()
    s = g.get_state()
    a = tr.uniform(100, generator=g)
    h = tr.Generator()
    h.set_state(s)
    assert torch.equal(a, tr.uniform(100, generator=h))
    k = tr.Generator(seed=g.torch_generator.initial_seed())
    assert torch.equal(a, tr.uniform(100, generator=k))


def test_default_generator_differs_across_processes():
    code = "import trnrand as tr; tr.set_backend('pytorch'); print(tr.uniform(8).tolist())"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(outs) == 2


def test_seeded_stream_bit_identical():
    # Pinned from v0.13.0: seeding is unchanged by #98.
    g = tr.Generator(seed=42)
    assert tr.uniform(4, generator=g).tolist() == [
        0.8822692632675171,
        0.9150039553642273,
        0.38286375999450684,
        0.9593056440353394,
    ]
    assert tr.normal(4, generator=g).tolist() == [
        0.23446236550807953,
        0.23033303022384644,
        -1.1228563785552979,
        -0.18632829189300537,
    ]
    assert tr.randint(4, high=1000, generator=g).tolist() == [210, 954, 231, 572]
    assert tr.Generator(seed=42)._key48 == 42
    tr.manual_seed(42)
    assert tr.uniform(1).item() == 0.8822692632675171
