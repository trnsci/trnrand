# RFC: Counter-partitioned multi-chip RNG

**Status:** Design · **Phase:** 4 · **Tracker:** [#20](https://github.com/trnsci/trnrand/issues/20)

This RFC describes Phase 4's shape: scaling RNG across NeuronCores /
chips without sacrificing the property that makes trnrand valuable for
scientific workloads — *bit-exact reproducibility regardless of
cluster topology*.

## The Trainium shape of the problem

Scaling an RNG library across chips is usually done one of two ways:

1. **Per-chip independent seeds** (PyTorch `torch.Generator` default,
   NumPy's typical usage) — each chip gets a derived seed, streams
   are statistically independent. Simple, but a 4-chip run and a
   1-chip run produce *different* sample sequences for the same
   logical seed. Scientifically meaningful reproducibility across
   cluster reshape is lost.
2. **Key-splitting protocols** (JAX `random.split`) — deterministic
   key derivation, threaded through user code. Reproducible if the
   user plumbs keys correctly, but cluster-reshape consistency
   requires the user to preserve the split tree exactly.

Trainium's Philox kernel (Phase 1) is counter-based: `(counter, key) →
output` is a pure function. That enables a third option that neither
GPU approach supports as a first-class API feature:

**Counter-space partitioning.** The 128-bit Philox counter space
(`2^128` tile addresses) is divided across `P` chips as disjoint
ranges. Each chip independently emits its subrange. Concatenating in
rank order produces the same stream as a 1-chip run of the same total
length. **Zero coordination. Zero synchronization. Bit-exact.**

This is the Phase 4 thesis: a bit-exactness guarantee across chip
counts, shipped as a first-class API property, tested in CI on CPU
reference and on-hardware.

## Core idea

### Counter arithmetic

For a user request of `N` total samples with `partition_size = P`
chips, each chip `r ∈ [0, P)` emits `N/P` samples from counter range:

```
chip r starts at counter = seed_base + r * (N/P) * counters_per_sample
chip r emits              N/P samples
```

Where `counters_per_sample = 1` for uniform (one Philox block → 4
uniforms → 4 samples, counter advances by 1 per 4 samples), and
proportional for transforms (Box-Muller consumes 2 uniforms per
normal, so 1 counter increment produces 2 normals).

The critical invariant: **the mapping from (seed, global_sample_index)
to sample value is independent of P.** Sample `k` in the logical
stream has the same value whether `P=1` and chip 0 produces it, or
`P=32` and chip `k mod 32` produces it (with appropriate
partitioning).

### Generator API

Two new optional kwargs on `Generator.__init__`:

```python
import torch_xla.core.xla_model as xm
import trnrand

gen = trnrand.Generator(
    seed=42,
    partition_rank=xm.get_ordinal(),      # 0 ≤ rank < size
    partition_size=xm.xrt_world_size(),   # total chips in cluster
)

# Subsequent calls take this chip's subrange of the logical stream:
samples = gen.uniform(1_000_000)   # emits N/P samples on chip rank
```

Defaults (`rank=0, size=1`) preserve the single-chip behavior
identically. Users who never use multi-chip pay nothing.

### `Generator.advance(n)` — checkpoint primitive

A cheap way to skip forward in the logical stream:

```python
gen.advance(1_000_000)   # skip 1M samples; counter jumps, no work done
gen.uniform(1_000)       # emits samples (1M+1) through (1M+1000)
```

This enables:
- **Warm-starting** an experiment at a specific point in the stream.
- **Checkpoint/resume** across cluster topology changes — persist the
  logical sample index, resume at any `P`.
- **Deterministic sub-sampling** — draw samples 0..N, advance, draw
  samples 2N..3N, etc., without materializing the middle.

## Composition with Phase 3 (SBUF-resident streaming)

Phase 3's `GeneratorProgram` / `stream_into` composes cleanly. What
changes:

- **`GeneratorProgram` construction** — unchanged. The program
  describes distributions and sizes; counter arithmetic is a
  `Generator`-level concern.
- **Kernel entry** — the NKI kernel loads the counter from HBM into
  SBUF, offset by the partition arithmetic. Each chip gets a
  different starting counter; kernel code is identical.
- **NEFF cache** — one NEFF per program shape, shared across all
  chips. Rank is a runtime argument, not part of the compiled
  kernel. No per-rank recompilation.

That last point matters: if every chip needed its own NEFF, Phase 3's
cache-reuse guarantees would break at multi-chip. Because rank lives
in HBM arguments, not in the compile unit, the Phase 3 and Phase 4
wins multiply cleanly.

## Bit-exactness contract

Three guarantees, stated as testable properties:

### 1. Partition equivalence

For any `(seed, N, dist, params)` and any `P` dividing cleanly:

```python
# 1-chip run
g1 = trnrand.Generator(seed=s)
ref = g1.sample(dist, N, **params)

# P-chip run, concatenated in rank order
per_chip = [
    trnrand.Generator(seed=s, partition_rank=r, partition_size=P)
      .sample(dist, N // P, **params)
    for r in range(P)
]
multi = torch.cat(per_chip)

assert torch.equal(ref, multi)   # byte-for-byte
```

### 2. Seed invariance

```python
g16 = trnrand.Generator(seed=42, partition_rank=3, partition_size=16)
g32 = trnrand.Generator(seed=42, partition_rank=6, partition_size=32)
# Both see the same logical range [6*N/32, 7*N/32) of the seed=42 stream.
# g32's output equals g16.uniform(N/16)[:N/32] for the appropriate N.
```

The same seed produces the same logical stream regardless of
`partition_size`. Resharding a cluster from 16 → 32 chips mid-
experiment gives the user a consistent stream, not a different one.

### 3. Checkpoint resumption

```python
# Record position
position = gen.position()   # returns the current counter

# ... run, crash, restart on different chip count ...

new_gen = trnrand.Generator(seed=s, partition_rank=r, partition_size=P_new)
new_gen.advance_to(position)   # resume from the same logical index
```

Position is a single integer (the global counter index), portable
across cluster topologies.

## What GPU libraries typically can't do

Honest contrast:

| Library | Cross-chip bit-exactness | Reshape consistency | First-class API? |
|---|---|---|---|
| cuRAND (default per-thread seeds) | ❌ | ❌ | — |
| cuRAND (`curandSetGeneratorOffset`) | ⚠️ manual | ⚠️ manual | Offset API exists, not cluster-aware |
| JAX `random.split` | ✅ if keys threaded | ⚠️ requires user plumbing | Key-splitting is the API |
| NumPy `default_rng` (PCG64) | — (single process) | — | No cluster API |
| PyTorch `torch.Generator` (Mersenne Twister) | ❌ | ❌ | Not counter-based |
| **trnrand (this RFC)** | ✅ | ✅ | ✅ `partition_rank` / `partition_size` on `Generator.__init__` |

The first-class-ness is what matters: users don't have to choose the
right option or thread state through their code. The default path in
a multi-chip context is reproducible.

## Data-parallel vs model-parallel training

The same API covers both interpretations:

- **Data-parallel (per-chip noise).** Each chip sees different input
  batches and wants independent RNG (e.g., dropout masks, data
  augmentation). `partition_rank` / `partition_size` gives exactly
  that — each rank sees a disjoint counter range, so samples are
  iid and uncorrelated across ranks by construction.
- **Model-parallel (sharded weight init).** A logically-single RNG
  draw is sharded across chips for memory reasons (e.g., a 100GB
  weight tensor). The user wants the chip pieces to combine into
  the single logical draw the non-sharded version would produce.
  Partition API gives this too — each chip emits its contiguous slice
  of the logical stream, concatenation yields the original.

The design elegance is that the library doesn't have to know which
interpretation the user intends. The bit-exactness contract holds in
both cases; the user's code context determines the semantics.

## Validation claims

Phase 4 hardware testing must prove:

1. **Partition equivalence.** For each of the 5 distribution families
   (uniform, normal, exponential, gamma, beta), 1-chip vs 32-chip run
   with same seed produces bit-identical concatenated output. Assert
   via `torch.equal`, not `torch.allclose` — exact match is the
   contract.
2. **Zero cross-chip coordination.** Neuron profiler trace shows no
   collective ops during the RNG kernel. Each chip runs independently.
3. **Near-linear strong scaling** at sample counts above the
   dispatch-bound regime (~10M samples / chip). 32-chip throughput
   ≥ 28× 1-chip throughput (≥87% efficiency target; scales with tile
   size).
4. **Checkpoint/resume consistency.** Advance on chip 0 to counter
   `N`, stop, re-init with new rank/size, resume — the stream from
   counter `N` forward is bit-identical to the uninterrupted run.
5. **Composition with Phase 3.** A `GeneratorProgram` built once and
   invoked across chips via `stream_into` produces the Phase 4
   bit-exactness contract across all distributions in the program.
   Validates that the streaming kernel (Phase 3) and the partition
   API (Phase 4) don't conflict.

### CPU-side validation, now

Partition equivalence is testable *now* on CPU, no hardware required.
`philox_uniform_cpu(n, seed, counter_offset)` already accepts an
offset parameter. A future test can assert:

```python
N = 4096
full = philox_uniform_cpu(N, seed=42, counter_offset=0)
for P in [2, 4, 8, 16]:
    chunks = [
        philox_uniform_cpu(N // P, seed=42, counter_offset=r * (N // P) // 4)
        for r in range(P)
    ]
    assert torch.equal(full, torch.cat(chunks))
```

This isn't in-scope for this RFC (implementation is out-of-scope for
the RFC), but landing the test before the hardware work starts is
cheap insurance against spec drift.

## Open questions

Deliberately deferred to implementation:

- **Counter exhaustion.** `2^128` counter space is vast — a run
  generating 10^12 samples/sec for 100 years still only uses ~10^22
  counters, far below `2^128 ≈ 3.4×10^38`. But the API should have a
  clear story: wraparound, error, or undefined? Suggestion: error
  loudly, since exhaustion always indicates a bug (or a truly
  exceptional workload).
- **Dynamic cluster resize.** If `partition_size` changes mid-run
  (chip failure, elastic cluster), what's the semantics? Probably
  user-level — they're responsible for explicit checkpoint/restart
  with the new `size`.
- **Framework integration.** `partition_rank` / `partition_size`
  plays nicely with PyTorch/XLA's `xm.get_ordinal` /
  `xm.xrt_world_size` but the RFC doesn't commit to an integration
  wrapper (`trnrand.Generator.from_xla(seed)`). Defer to
  implementation once the rough edges are visible.
- **Interaction with Generator state mutation.** If a user calls
  `gen.manual_seed(new_seed)` mid-run, does that reset the partition
  context? Suggestion: yes — partition kwargs are structural, not
  mutable; reseeding is a fresh logical stream.

## Cross-suite alignment

- **Phase 1** (#18): assumes Philox + Box-Muller kernels validated.
  No changes to the per-block kernel; Phase 4 only changes the
  counter loaded at entry.
- **Phase 3** (#19, [RFC](sbuf_resident_generator.md)): `GeneratorProgram`
  composes. Partition is a runtime HBM argument, not a compile unit
  — NEFF caching works unchanged.
- **Phase 5** (#21): trn2 wider-PSUM path is orthogonal. The partition
  API works identically on trn1 and trn2; Phase 5's tuning happens
  inside the per-chip kernel.
- **Umbrella [nki_validation_status](https://trnsci.dev/nki_validation_status/)**:
  Phase 4 closure updates trnrand's validation status accordingly.

## References

- [trnsci ROADMAP — Phase 4](https://trnsci.dev/roadmap/#phase-4-model-parallel-multi-chip) — suite-wide framing.
- [trnrand Phase 4 tracker (#20)](https://github.com/trnsci/trnrand/issues/20) — issue-level acceptance criteria.
- [Phase 3 RFC — SBUF-resident streaming Generator](sbuf_resident_generator.md) — composition target.
- [Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11)](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) — Philox paper; the counter-based design is exactly what makes this RFC possible.
