# Generator

Reproducible seeding and state management. The `Generator` wraps
`torch.Generator` for CPU/CUDA today; on Neuron hardware it will dispatch
to a Philox counter-based RNG running on the GpSimd engine.

## `Generator(seed=None, device="cpu")`

Construct a seeded generator. `seed=None` leaves the underlying
`torch.Generator` unseeded (random state).

## `Generator.manual_seed(seed) -> Generator`

Re-seed in place. Returns `self` so you can chain.

## `Generator.get_state() / set_state(state)`

Save and restore the underlying `torch.Generator` state for checkpointing.

## `Generator.torch_generator`

Access the underlying `torch.Generator` for use with raw PyTorch ops.

## `manual_seed(seed) -> Generator`

Module-level helper that re-seeds (and replaces) the default generator.

## `get_default_generator() -> Generator`

Return the module-level default generator used when `generator=None` is
passed to a distribution function.
