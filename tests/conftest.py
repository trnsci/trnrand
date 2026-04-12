"""Test configuration."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
