"""Benchmark fixtures."""

import pytest
import torch


@pytest.fixture(params=[128, 256, 512, 1024])
def square_size(request):
    return request.param


@pytest.fixture
def square_matrices(square_size):
    n = square_size
    torch.manual_seed(0)
    return torch.randn(n, n), torch.randn(n, n)
