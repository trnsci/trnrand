# Installation

## From PyPI

```bash
pip install trnrand
```

## With Neuron hardware support

```bash
pip install trnrand[neuron]
```

This pulls in `nki` and `torch-neuronx`, which are only needed on
Trainium/Inferentia instances. On CPU or GPU, trnrand falls back to
`torch.Generator` automatically.

`nki` and `torch-neuronx` are not on public PyPI; they ship with the Deep
Learning AMI Neuron or install from `https://pip.repos.neuron.amazonaws.com`
(see `pyproject.toml`).

### Two Neuron PyTorch lineages

`torch-neuronx` ships in two lineages, and trnrand dispatches its NKI
kernels on whichever one is installed:

| Lineage | Instances | Stack | Kernel dispatch |
|---|---|---|---|
| TorchNeuron (native) | Trn2, Trn3 | Neuron SDK 2.33+, Python ≥ 3.12, `torch` 2.14, `torch-neuronx` 2.14.* | Each kernel is a PyTorch custom op (`torch.ops.trnrand.philox4x32`, `threefry4x32`, ...) registered with `torch_neuronx.nki_op`; tensors live on `device="neuron"`. Works in eager mode and inside `torch.compile(backend="neuron")`. |
| PyTorch/XLA | Trn1 (also Inf2) | `torch-neuronx` 2.9 lineage, `torch_xla` | Unchanged: kernels run on the `torch_xla` device. |

Selection is automatic, on the first kernel launch:

1. **native** — `torch_neuronx` exports `nki_op` and a `neuron` device is
   available (`torch_neuronx.device_count() > 0`). This path never imports
   `torch_xla`.
2. **xla** — `torch_xla` is installed.
3. otherwise there is no device path. Without `nki`, `trnrand.*` uses
   `torch.Generator`; with `nki` but no device, use
   `set_backend("pytorch")` or `TRNRAND_USE_SIMULATOR=1` (unchanged from
   earlier releases).

The `trnrand::` custom ops are registered only when the native path is
selected, so importing trnrand on CPU or on the XLA lineage has no
TorchNeuron side effects. `pip install trnrand[neuron]` does not choose a
lineage for you: install the `torch-neuronx` that matches the instance, or
use the matching Deep Learning AMI Neuron.

### Editable installs and the NKI kernel cache

TorchNeuron enables the NKI trace cache by default
(`NKI_ENABLE_TRACE_CACHE=1`), which persists compiled kernels across
processes. The cache only versions source that belongs to an installed
wheel (a `site-packages` tree with `.dist-info`); an editable install is
not tracked and can get a stale kernel after you edit `dispatch.py`. When
developing with `pip install -e`, turn the cache off before starting
Python:

```bash
export NKI_ENABLE_TRACE_CACHE=0
```

See the TorchNeuron
[environment variables](https://github.com/aws-neuron/torch-neuronx/blob/main/docs/environment_variables.md)
reference.

## From source

```bash
git clone https://github.com/trnsci/trnrand
cd trnrand
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- `torch >= 2.1`
- `numpy >= 1.24`
- `nki >= 0.3.0` and `torch-neuronx` (optional, for on-hardware NKI
  kernels; native dispatch needs `torch-neuronx` 2.14.* on Python ≥ 3.12)
