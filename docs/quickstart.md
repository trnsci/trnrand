# Quickstart

```python
import trnrand

# Seeded, reproducible generation
g = trnrand.Generator(seed=42)

# Standard distributions
x = trnrand.normal(1000, mean=0.0, std=1.0, generator=g)
u = trnrand.uniform(1000, low=-1.0, high=1.0, generator=g)
e = trnrand.exponential(1000, rate=2.0, generator=g)
b = trnrand.bernoulli(1000, p=0.3, generator=g)

# Quasi-random sequences (better convergence for MC integration)
sobol_pts = trnrand.sobol(1024, n_dims=5, seed=42)
halton_pts = trnrand.halton(1024, n_dims=3)
lhs_pts = trnrand.latin_hypercube(100, n_dims=4)

# Module-level seeding
trnrand.manual_seed(42)
x = trnrand.standard_normal(256)
```

## Backend selection

```python
import trnrand

trnrand.set_backend("auto")     # NKI on Trainium, PyTorch elsewhere (default)
trnrand.set_backend("pytorch")  # force PyTorch
trnrand.set_backend("nki")      # force NKI (requires Neuron hardware)
```

## MC vs QMC example

```bash
python examples/mc_integration.py
```

Compares pseudo-random vs Sobol quasi-random for estimating the volume of a
5-D hypersphere. QMC converges O(1/N) vs O(1/√N).
