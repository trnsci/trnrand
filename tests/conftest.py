"""Test configuration."""

import atexit
import os
import shutil
import tempfile

# NKI trace cache (aws-neuron/aws-neuron-sdk#1454, trnsci/trnsci#35).
# `import torch_neuronx` turns the cache on (NKI_ENABLE_TRACE_CACHE=1), so
# that is how users run these kernels, and the neuron tests run the same way.
# The cache lives on disk and is versioned by wheel .dist-info, which an
# editable install or source checkout does not change, so a shared cache could
# serve a trace of an older kernel. Each session therefore gets a fresh,
# empty cache directory (NKI_TRACE_CACHE_URL), removed at exit.
# tests/test_trace_cache_safety.py checks that no kernel depends on a value
# the cache key misses (closure or global). Overrides: NKI_ENABLE_TRACE_CACHE=0
# runs with the cache off; an explicit NKI_TRACE_CACHE_URL is kept as is.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")
if os.environ["NKI_ENABLE_TRACE_CACHE"] != "0" and "NKI_TRACE_CACHE_URL" not in os.environ:
    _trace_cache_dir = tempfile.mkdtemp(prefix="trnrand-nki-trace-cache-")
    os.environ["NKI_TRACE_CACHE_URL"] = _trace_cache_dir
    atexit.register(shutil.rmtree, _trace_cache_dir, ignore_errors=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")
    config.addinivalue_line(
        "markers",
        "nki_simulator: runs NKI kernels via nki.simulate(...) on CPU (no hardware)",
    )
