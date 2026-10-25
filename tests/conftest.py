"""Test configuration."""

import os

# Test runs come from source checkouts / editable installs, which the NKI trace
# cache cannot version (it keys on wheel .dist-info). A stale hit would run an
# old kernel, so disable it before anything imports nki. Set it explicitly to
# override.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate(...) on CPU (no hardware)",
    )
