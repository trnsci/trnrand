# Installation

## From PyPI

```bash
pip install trnrand
```

## With Neuron hardware support

```bash
pip install trnrand[neuron]
```

This pulls in `neuronxcc` and `torch-neuronx`, which are only needed on
Trainium/Inferentia instances. On CPU or GPU, trnrand falls back to
`torch.Generator` automatically.

`neuronxcc` is not on public PyPI; it ships with the Deep Learning AMI
Neuron. We pin `neuronxcc>=2.24` to match the NKI calling convention used
by sibling repos (see `pyproject.toml`).

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
- `neuronxcc >= 2.24` (optional, for on-hardware NKI kernels)
