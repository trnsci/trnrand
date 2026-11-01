# Generator

Reproducible seeding and state management. The `Generator` wraps a
`torch.Generator` (the PyTorch path) and a Threefry counter, which the NKI
path draws from on the GpSimd engine.

## `Generator(seed=None, device="cpu", partition_rank=0, partition_size=1, key_mode=None)`

Construct a seeded generator. `seed=None` leaves the underlying
`torch.Generator` unseeded (random state).

- `partition_rank` / `partition_size`: this generator is rank `r` of `P`
  in a counter-partitioned stream; concatenating the ranks' draws in order
  gives the single-generator stream on the NKI path. Fixed for P > 1 in
  v0.12.0 (0.7.0–0.11.0 reused counters from the second draw on).
- `key_mode`: `"draw"` (default, or `TRNRAND_KEY_MODE`) takes a fresh
  Threefry key from the `torch.Generator` on each NKI draw. `"counter"`
  fixes the key by the seed and only advances the counter, which makes
  `uniform` / `normal` traceable by `torch.compile(backend="neuron")`.
  The two modes give different streams.

## `Generator.manual_seed(seed) -> Generator`

Re-seed in place. Returns `self` so you can chain.

## `Generator.get_state() / set_state(state)`

Save and restore the generator for checkpointing: the `torch.Generator`
state followed by the 8-byte Threefry counter. Use this to checkpoint and
resume in either key mode. Before v0.12.1 the state held only the
`torch.Generator` state, so a restored generator drew different values on
the NKI path (#74); a bare `torch.Generator` state is still accepted and
leaves the counter unchanged.

## `Generator.position() / advance(n) / advance_to(position)`

`position()` is the number of samples drawn (in 512-sample Threefry
batches); `advance` and `advance_to` move the Threefry counter without
drawing. They resume a stream only with `key_mode="counter"` (for
`uniform` / `normal`) on the NKI path. With the default `key_mode="draw"`
the per-draw keys come from the `torch.Generator`, which they don't move,
and on the PyTorch path they have no effect. Use `get_state` / `set_state`
for checkpoint/resume (#76).

## `Generator.torch_generator`

Access the underlying `torch.Generator` for use with raw PyTorch ops.

## `manual_seed(seed) -> Generator`

Module-level helper that re-seeds (and replaces) the default generator.

## `get_default_generator() -> Generator`

Return the module-level default generator used when `generator=None` is
passed to a distribution function.
