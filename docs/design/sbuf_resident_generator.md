# RFC: SBUF-resident streaming Generator

**Status:** Design · **Phase:** 3 · **Tracker:** [#19](https://github.com/trnsci/trnrand/issues/19)

This RFC describes the architectural shape of trnrand's Phase 3 work.
It intentionally frames the design in terms of what Trainium uniquely
enables rather than as an optimization of the v0.2.0 per-call kernel
path.

## The Trainium shape of the problem

Scientific RNG workloads — Monte Carlo integration, Hutchinson trace
estimators, MCMC samplers, neural-net weight init — generate 10⁶–10⁹
samples per batch, consuming them immediately. The inner loop is
dispatch-bound on any accelerator: kernel-launch and host-device
transfer overhead dominate the arithmetic.

A GPU library like cuRAND optimizes the kernel body well but can't
escape per-call dispatch — every `curandGenerateNormal()` pays the
full launch cost. Techniques like persistent threads help but don't
generalize across distributions.

Trainium has a structurally different escape route:

1. **SBUF** is 24 MB of directly-addressable on-chip memory that
   persists for the lifetime of a kernel invocation. Per-lane state
   (Philox counter/key, rejection accumulators) can live there
   indefinitely at near-register latency.
2. **NEFF** (Neuron Executable Format) caches compiled kernels across
   Python invocations. A pre-compiled generator kernel invoked
   repeatedly pays only launch latency (~tens of microseconds on
   trn1), not compile cost.
3. **Four independent compute engines** — Tensor, Vector, Scalar,
   GpSimd — can run concurrently within a single kernel. RNG
   distributions naturally decompose across them: integer Philox on
   GpSimd, transcendentals on Vector, scalar fused multiply-add on
   Scalar.

Composed: one kernel launch that emits millions of samples across
multiple distributions while keeping Generator state SBUF-resident and
running three engines in parallel. **This shape doesn't have a direct
GPU analog.**

## Core idea

A `GeneratorProgram` is a pre-compiled sequence of distribution calls.
The user describes the program once (via a Python builder); trnrand
compiles it into a single NKI kernel that:

- Loads Generator state (counter, key) into SBUF at entry.
- Emits samples for each distribution in the program, advancing the
  counter in SBUF between distributions.
- Writes only final outputs to HBM — no per-distribution intermediate
  spills.
- Writes the advanced counter back to HBM at exit.

The kernel is cached via NEFF keyed on the program shape. Repeat
invocations skip compilation entirely.

## Engine orchestration

Concrete role for each engine in a `(uniform, normal, scale+shift)`
program:

| Engine | Role |
|---|---|
| **GpSimd** | Philox 4×32-10 rounds. Integer multiply-XOR, 10 rounds per 4-sample block. Also serves as the uniform-producing path for rejection samplers. |
| **Vector** | `log`, `exp`, `cos`, `sin`, `sqrt` for Box-Muller and exponential transforms. Rejection-check predicates (`log(u) < bound`) for gamma/beta. |
| **Scalar** | Accumulator for rejection counts. Scale-and-shift (`μ + σ·z`) for parameterized normals. |
| **Tensor** | **Not used.** RNG is not matmul-shaped. Deliberately leaving it idle is the right choice — it's available for a concurrent workload that wants it. |

Calling out the "Tensor is intentionally idle" is important: it
demonstrates the design understood the workload rather than reaching
for the highest-throughput engine by default.

## Pipeline pattern

Steady-state schedule for a fused `uniform → Box-Muller → scale+shift`
sequence:

```
tick  | GpSimd                  | Vector                  | Scalar
------+-------------------------+-------------------------+-------------------
  0   | Philox block 0          | idle                    | idle
  1   | Philox block 1          | Box-Muller on block 0   | idle
  2   | Philox block 2          | Box-Muller on block 1   | scale+shift block 0
  3   | Philox block 3          | Box-Muller on block 2   | scale+shift block 1
  ...
```

After a two-tick startup, all three engines run concurrently every
cycle. GPU hardware can't pipeline like this — its "engines" (Tensor
Cores, SFU, scalar issue ports) are unified under one warp scheduler,
so the equivalent logical pipeline serializes in practice.

## NEFF cache story

Python side:

```python
gen = trnrand.Generator(seed=42)
program = gen.new_program() \
    .uniform(n=1024, out="u") \
    .normal(n=4096, mu=mu, sigma=sigma, out="z") \
    .build()                      # first call: compile + cache NEFF
```

Repeat invocations:

```python
for step in range(n_steps):
    program.stream_into({"u": u_buf, "z": z_buf})   # ~10s of μs
```

The expensive compile happens once. Per-step cost is kernel-launch
latency plus actual compute — the same floor GPU libraries can't beat
because they recompile per shape.

## API shape

New public surface:

- `Generator.new_program() -> ProgramBuilder` — start a builder.
- `ProgramBuilder.uniform(n, low, high, out)` / `.normal(n, mu, sigma, out)` / etc — fluent accumulation of distribution calls.
- `ProgramBuilder.build() -> GeneratorProgram` — compile to NEFF.
- `Generator.stream_into(program, buffers)` — zero-alloc execution
  into pre-allocated output buffers.

Back-compat:

- Existing `trnrand.normal(size, ...)` / `trnrand.uniform(...)` stay
  unchanged. Under the hood, ad-hoc calls compile a single-distribution
  program on first use, cache the NEFF, reuse on subsequent same-shape
  calls.

## What this unlocks downstream

Concrete consumers that get qualitative improvements, not just bench
numbers:

- **trnfft speech-enhancement pipelines.** Frame-wise noise injection
  for cIRM training generates one noise tensor per frame. Today each
  is a dispatch; a streaming program folds 1000+ frames into one
  launch, eliminating the per-frame dispatch tax.
- **trnblas stochastic trace estimation.** Hutchinson / Hutch++
  require N mat-vecs with iid Rademacher or Gaussian vectors. The
  full vector ensemble comes from one streaming call. The outer
  Hutchinson loop becomes dispatch-free for the RNG portion.
- **trnsolver MCMC / randomized solvers.** The inner loop needs a
  batch of standard normals per iteration. One launch per step, no
  per-step RNG dispatch.

## Validation claims

Phase 3 hardware testing must prove these specific numbers. The
numbers aren't the point — they're evidence that the architectural
story is real.

1. **Single-launch latency floor (dispatch-bound regime).**
   `Generator.uniform(1_000_000)` on trn1.2xlarge: < 1 ms.
2. **Streaming throughput.** A program emitting `(1M uniform, 1M
   normal, 1M exponential)` in a single launch: ≥ 10⁹ samples/s. The
   ratio-to-CUDA-dispatch-cost is the more interesting number than
   raw parity.
3. **NEFF cache reuse.** Second call of a cached program: < 100 μs
   end-to-end (compile cost amortized to zero).
4. **Cross-distribution bit-exactness.** A program containing
   `u1, u2, z = BoxMuller(u1,u2)` yields the same `z` as invoking
   Philox then Box-Muller piecewise. Proves the SBUF counter
   bookkeeping is sound and fusion doesn't alter the sample stream.
5. **Engine parallelism evidence.** Profiler trace showing non-zero
   concurrent time on GpSimd + Vector + Scalar for the pipelined
   program. Without this, the pipeline is only nominal.

## Open questions

Deliberately unsettled for implementation time:

- **Rejection samplers with fixed output size.** Gamma/beta's accept
  rate is < 1, so "emit exactly N samples" requires either (a)
  oversample + compact on Scalar, (b) variable-length output, or
  (c) bound pre-sampling by expected rate + loop. Each has SBUF
  residency implications.
- **Counter persistence.** `Generator` state advances across Python
  `stream_into` calls (so reseeding isn't needed). Implementation
  choice: counter lives in HBM between launches but is loaded to
  SBUF at entry; alternative is a persistent-kernel model (more
  complex, probably not worth it).
- **Mixed dtypes in one program.** Can the program emit FP32
  uniforms and BF16 normals in the same launch? Depends on SBUF
  dtype conventions — defer until shape work on BF16 lands
  (scheduled for v0.4.0+).
- **NEFF cache keying.** Parameterizing shape in the NEFF (so N=1024
  and N=2048 hit the same cache entry) would be ideal. If not
  possible, users get N distinct NEFFs — a pragmatic fallback.
- **Multi-chip interaction.** Phase 4 (#20) will handle
  counter-partitioned multi-chip. The streaming kernel should
  compose with that — implementation detail is which axis the
  counter split lives on.

## Cross-suite alignment

The design is compatible with suite-wide patterns:

- **Phase 1** (#18): this RFC assumes #18's Philox + Box-Muller
  kernels are the building blocks. No conflict.
- **Phase 4** (#20): counter partitioning across chips. The
  streaming kernel receives a counter subrange at entry; nothing
  else changes.
- **[trnsci/trnsci#3](https://github.com/trnsci/trnsci/issues/3)**
  (autograd wrappers): trnrand outputs remain non-differentiable per
  `docs/stability.md#differentiability`. Streaming doesn't change
  that.
- **Umbrella [nki_validation_status](https://trnsci.dev/nki_validation_status/)**:
  Phase 3 closure updates trnrand's row.

## References

- [trnsci ROADMAP — Phase 3](https://trnsci.dev/roadmap/#phase-3-single-chip-performance) — suite-wide framing.
- [trnrand Phase 3 tracker (#19)](https://github.com/trnsci/trnrand/issues/19) — issue-level acceptance criteria.
- [Salmon, Moraes, Dror, Shaw — "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11)](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) — Philox spec; informs SBUF state layout.
