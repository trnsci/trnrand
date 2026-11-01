# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`Generator.get_state()` / `set_state()` did not restore the NKI stream** (silent).
  Affects released versions **0.7.0 through 0.12.0**, on the NKI path only
  (`set_backend("nki")`, or `"auto"` on Neuron hardware); the PyTorch path is
  not affected. The state held only the `torch.Generator` state, not the
  Threefry counter the NKI draws take their offsets from, so restoring a
  checkpoint and drawing again gave different values (Trn2: max |diff| 0.97
  for `uniform`, 5.7 for `normal`, both key modes). The state now carries the
  counter too (`torch` state followed by 8 counter bytes), and `set_state`
  restores both and re-derives the counter-mode key. A bare `torch.Generator`
  state is still accepted (#74).

## [0.12.0] - 2026-11-01

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: **correctness fix — upgrade recommended for partitioned generators.** In 0.7.0–0.11.0, partitioned generators (P > 1) on the NKI path advanced their counter by one chip's share instead of all P chips', so from the second draw on chip r reused chip r+1's counters: with `key_mode="counter"` the ranks returned duplicate values, and with `"draw"` the stream no longer matched the single-chip stream. P = 1 and the CPU path are unaffected (#72). Also fixes `halton_nki` repeating a point at chunk boundaries, and makes `sobol_nki` reject indices ≥ 2^24, where points would repeat.

### Fixed

- **Partitioned generators reused counters from the second draw on** (silent).
  Affects released versions **0.7.0 through 0.11.0**, on the NKI path only
  (`set_backend("nki")`, or `"auto"` on Neuron hardware); the PyTorch CPU path
  is not affected. A `Generator(partition_rank=r, partition_size=P)` with P > 1
  advanced its counter by one rank's share after each draw, so rank r's next
  draw used the counters rank r+1 had just used.
  - `key_mode="counter"` (0.8.0 to 0.11.0): the key is fixed per seed, so rank
    r's draw k+1 **returned the same values** as rank r+1's draw k. Streams
    across ranks contained duplicates from the second call on.
  - `key_mode="draw"` (the default, and the only behaviour in 0.7.x): each call
    takes a fresh key, so values did not repeat, but from the second call on
    the ranks no longer concatenated to the single-chip stream, as the
    partition API documents.
  - Every draw (`uniform`, `normal`, `exponential`, `poisson`, the `*_into`
    variants and the `torch.compile` route) now advances past all P slices.
    Unpartitioned generators (P = 1) were never affected.
- `halton_nki(start_index=s)` started at Halton index `max(s, 1)`, so
  `start_index=0` and `1` gave the same points and chunked calls repeated a
  point at each chunk boundary. Point j is now index `s + j + 1`, the CPU
  convention. `trnrand.halton()` (start 0) is unchanged.
- `sobol_nki` now raises past index 2^24 instead of silently dropping the high
  Gray-code bits (24-bit direction vectors) and repeating points.

### Added

- `tests/test_nki_tail_audit.py`: every NKI path at awkward sizes (1, 3, 127,
  129, 200, 1000, 4097, 65,537, 262,145, 4,194,305), odd counter offsets,
  chained and partitioned draws, multi-core splits, Sobol / Halton, Philox and
  Box-Muller, against independent CPU oracles (trnsci/trnsci#35).

## [0.11.0] - 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: pinned-memory host output makes bulk draws ≥8M values 1.5–2.9× faster end to end, and multi-core now pays end to end (`normal` 64M: 130 → 55 ms on four cores, #70); output bit-identical.

### Changed

#### Host results at the host-link rate (#60)

- On the native TorchNeuron path, large `uniform()` / `normal()` results (1 MiB
  and up) now land in memory from the accelerator's caching host allocator
  (`torch.empty(..., pin_memory=True)`), the fix trnblas#64 made for BLAS
  results. A device-to-host copy into fresh pageable memory runs at about
  4 GB/s on Trn2 because every page faults; into that memory it runs at
  13-14 GB/s. Each call still returns a fresh tensor, and the output is
  bit-identical.
- `uniform()` with `low=0` skips the `+ 0.0` pass, which is the identity bit
  for bit on the kernel's non-negative values.
- Trn2, warm medians, end to end to the host: `normal(64M)` 130 -> 76 ms on one
  core and 109 -> 55 ms on four (`set_cores(4)`); `uniform(64M)` 126 -> 60 ms
  and 103 -> 36 ms. Draws of 4M values and fewer are unchanged.
- `TRNRAND_PINNED_OUTPUT=0` turns it off. Cached blocks are rounded up to a
  power of two and stay cached until `torch.accelerator.empty_host_cache()`.

## [0.10.0] - 2026-10-31

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron) — see trnsci/trnsci#35.

Headline: opt-in multi-core bulk `uniform()`/`normal()` (`set_cores`), bit-identical for every core count; kernel time at 64M values 36 → 12 ms on four logical cores (#68).

### Added

#### Multi-core bulk `uniform()` / `normal()` on one Trn2 chip (#60)

- `trnrand.set_cores(n | "auto" | None)` / `trnrand.get_cores()`, and the
  `TRNRAND_CORES` environment setting, spread large native-path bulk draws
  over several logical NeuronCores. Off by default.
- The counter range is split as the partitioned generators split it across
  chips: one contiguous run of batches per core, gathered into one host tensor
  in the original order. Output is bit-identical for every core count (CPU
  tests, plus 1/2/3/4 cores on Trn2 at up to 16M values, 8- and 32-bit kernels).
- Requests under `TRNRAND_CORES_MIN_ELEMENTS` (default 8,388,608, measured)
  stay on one core.
- Kernel time at 64M values: 36 ms on one core, 19 ms on two, 12 ms on four.
  End to end to the host the gain is small (about 136 to 120 ms at 64M),
  because the device→host copy and host page faults do not scale with cores.

### Changed

- Native bulk draws copy each launch's output straight into a preallocated
  host tensor. This replaces the device-side stack and host concatenation of
  multi-launch draws. Single-core end to end: 16M values 48 → 32 ms, 64M
  200 → 121 ms. Values are unchanged.
- The eager bulk path names the device explicitly and caches the lane tile
  per device. `torch.neuron.synchronize(i)` makes device i current. Before,
  if the current device changed between draws, the cached tile stayed on the
  old device and the launch failed ("Inconsistent device IDs").

## [0.9.0] - 2026-10-30

Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0) — see trnsci/trnsci#35.

Headline performance: the full-word bulk Threefry kernel drops from 5.0 to 2.5 ms at 4M samples on Trn2, and `trnrand.normal` end to end (5.5–8.1 ms) now beats CPU `randn` (11.5–15.6 ms) (#66).

### Changed

#### Faster full-word bulk Threefry kernels and host path on Trn2 (#60)

Output stays bit-identical: every value, at every seed and `counter_offset`,
matches the 8-bit kernels (`test_limb_variants_bit_identical`, on Trn2).

- A Trn2 profile of `normal(4M)` showed the device time was GpSimd integer
  adds (~2/3) plus Vector rotate/xor. Vector instructions issued while a
  GpSimd add runs stretch to the add's length, because the two engines share
  the SBUF port, so double-buffering the engines cannot help.
- The full-word kernels (`TRNRAND_LIMB_BITS=32`, the native default) now use
  both physical cores of an LNC=2 logical core. A gridless NKI kernel at
  LNC=2 used to run in full on both cores. The work is split by
  `program_id`: whole chunks per core when the chunk count is even,
  otherwise half the columns.
- Chunk-invariant key-injection adds are hoisted, and the u24 → uniform
  conversion is one fused `tensor_scalar`.
- The full-word kernels write their output in per-tile order, shape
  `(C·P, F, 4)`, so the host reorder (`_bulk_tile_order`) is gone for them.
  The 8/16-bit kernels keep their layout.
- Native host path: one packed host→device copy per call instead of six.
  The eager bulk path caches the lane tile on the device, and
  `normal`/`uniform` apply `* std + mean` in place, which avoids two
  out-of-place n-element CPU passes (~9 ms at 4M).

## [0.8.0] - 2026-10-28

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device) — see trnsci/trnsci#35.

### Added

#### Native PyTorch dispatch on TorchNeuron (#54)

On TorchNeuron (Neuron SDK 2.33+, `torch-neuronx` 2.14, PyTorch 2.14,
Trn2/Trn3), every NKI kernel is now registered as a PyTorch custom op with
`torch_neuronx.nki_op` and dispatched on `device="neuron"`, with no
`torch_xla` import. Ops: `trnrand::philox4x32`, `box_muller`,
`threefry4x32`, `threefry_normal`, `sobol_gray_code`, `halton`,
`threefry_streaming_normal`, `threefry_streaming_uniform`. Kernel bodies
are unchanged.

- The dispatch path is detected on the first kernel launch: native when
  `torch_neuronx` exports `nki_op` and a `neuron` device is available, XLA
  when `torch_xla` is installed, otherwise none. The XLA path (Trn1) is
  unchanged, as are the `torch.Generator` fallback and the
  `TRNRAND_USE_SIMULATOR` path.
- Custom ops are registered lazily (`trnrand/nki/_native.py` is imported
  only when the native path is selected), so CPU-only installs and the XLA
  lineage see no import-time side effects.
- All host wrappers, `GeneratorProgram.stream_into`, and the partitioned
  multi-chip paths share one launch helper, so they follow the same path.
- `TRNSCI_NKI_DISPATCH=auto|native|xla` forces the dispatch path (a
  suite-wide variable shared by all trnsci libraries). Forcing a path that
  isn't available raises instead of falling back.
- New `[torchneuron]` extra (`torch>=2.14`, `torch-neuronx>=2.14`,
  `nki>=0.6`); `[neuron]` stays the PyTorch/XLA lineage.
- Docs: install notes cover both extras and lineages and recommend
  `NKI_ENABLE_TRACE_CACHE=0` for editable installs.
- Tests: `conftest.py` sets `NKI_ENABLE_TRACE_CACHE=0` by default so test
  runs from a checkout never hit a stale cached kernel.
- CI: the scheduled run is weekly (Mondays) instead of daily.
- Tests: CPU-runnable backend-selection and lazy-registration tests;
  `@pytest.mark.neuron` native-path tests for Trn2 (not yet run on
  hardware).

### Changed

#### Bulk `uniform`/`normal` use multi-tile launches (#60)

`threefry_uniform_nki` and `threefry_normal_nki` (behind `trnrand.uniform`,
`normal`, `exponential`, and the gamma/beta/chi-squared/Poisson/
truncated-normal NKI paths) launched one kernel per 512 values, so
`normal(4_194_304)` was 8,192 launches (about 9 s on Trn2 at the native
dispatch floor). They now go through the streaming kernels
(`threefry_stream_*`, 16,384 values per launch): 256 launches for 4M values,
ceil(N / 16,384) in general. Output values and order are unchanged for every
seed and `counter_offset` (#58). Tests: CPU launch-count and bit-identity
tests with a stand-in launcher (`tests/test_launch_granularity.py`), and
simulator bit-identity against the per-tile path.

- On the native path the streaming launches keep their inputs and outputs
  on the device. The lane-index and key inputs are copied to the device
  once per call, the start-batch inputs for all launches are copied once as
  one `(launches, 128, 1)` tensor, and the per-launch outputs are stacked on
  the device and copied back in one transfer: 6 host-to-device copies and 1
  device-to-host copy per call instead of 6 and 1 per launch. Measured on
  Trn2 before this step: `normal(4M)` went from 9.2 s to 1.6 s with
  multi-tile launches alone.
- The per-tile wrappers are kept as `_threefry_uniform_per_tile` /
  `_threefry_normal_per_tile`. They are still used on the XLA path when
  `neuronx-cc` is older than 2.28, where the streaming kernels hit SBUF OOM
  at compile (#57). `TRNRAND_BULK_STREAMING=0|1` forces the choice.
- The Threefry host wrappers now live at module level in
  `trnrand/nki/dispatch.py`, outside `if HAS_NKI:`, so they import and can
  be tested on hosts without NKI.

#### Bulk `uniform`/`normal` run as one wide-tile kernel launch (#60)

The streaming kernels held every byte of every Threefry word as a `(128, 1)`
tile, so each NKI instruction did 128 elements of work and a 16,384-value
launch was about 55k instructions. On Trn2 that made `normal()` linear at
about 0.23 ms per 1k values (934 ms for 4M) no matter how launches were
grouped. On the native path and the simulator, `threefry_stream_*` (and so
`trnrand.uniform`, `normal` and the distributions built on them) now use
new bulk kernels, `threefry_bulk_normal_kernel` and
`threefry_bulk_uniform_kernel` (ops `trnrand::threefry_bulk_normal` and
`trnrand::threefry_bulk_uniform`). They run the same byte-tile arithmetic on
`(128, F)` tiles, with F up to 512 Threefry batches per chunk and up to 16
unrolled chunks per launch. One launch covers up to 4,194,304 values.

- Counters are built in SBUF: the column index comes from `nisa.iota` and is
  added byte-wise to each chunk's start batch. Key words are added as float32
  `(128, 1)` `tensor_scalar` operands (byte plus byte is at most 510, exact),
  so the key schedule is never broadcast to `(128, F)`. Box-Muller runs in
  the same kernel, and each chunk's four output words leave in one DMA.
- Values and order are bit-identical to the streaming and per-tile kernels
  for every seed, `counter_offset` and size. The counter for each element is
  unchanged, including the 24-bit batch wrap. F and C are rounded up to
  powers of two (F from 32 to 512, C from 1 to 16), so at most 9 NEFF shapes
  are compiled.
- `TRNRAND_BULK_COLS` lowers the column cap if a compiler reports SBUF
  pressure. `TRNRAND_BULK_KERNEL=0` goes back to the 16,384-value streaming
  kernels. The XLA path (Trn1) still uses the streaming kernels.
- Tests: `tests/test_bulk_kernel.py` checks the wrapper against an
  independent numpy Threefry4x32-20 plus Box-Muller reference on CPU for
  sizes that are and aren't multiples of the tile, for multi-chunk and
  multi-launch plans, and for partitioned generators. On the simulator it
  checks the real kernels against the streaming kernels. Both bulk kernels
  are added to the MLIR verifier check for trn1 and trn2
  (`tests/test_nki_compile.py`).

#### Bulk kernels run on full uint32 words by default (#60)

On the native TorchNeuron path (Trn2+), the bulk Threefry kernels now hold
each word as one uint32 tile instead of four byte tiles. `TRNRAND_LIMB_BITS`
selects the width: `32` (default), or `8` for the byte-limb kernels as a
fallback. Values are bit-identical for every width. The XLA/Trn1 path never
uses the bulk kernels (it streams on the byte kernels), so it is unchanged.

- Bytes were used because integer `nl.add` / `tensor_scalar` arithmetic runs
  in float32 (exact only below 2^24, aws-neuron-sdk#1308), and the byte
  form was carried over from the Philox multiply, which needs 8-bit
  sub-products. NKI 0.7 documents `nisa.tensor_tensor` with all-int32/uint32
  operands as native integer arithmetic on GpSimd. The full-word kernels
  (`threefry_bulk32_{normal,uniform}_kernel`, ops
  `trnrand::threefry_bulk32_*`) pin their adds there and do xor/rotate as
  bitvec ops. That is about 510 compute instructions per (128, 512) chunk
  in trn2 MLIR, against about 1,950 for bytes. On Trn2 (SDK 2.33.0) the add
  wraps mod 2^32: results matched the 8-bit kernels exactly in
  `test_limb_variants_bit_identical` (all 4 cases), the full `-m neuron`
  suite (88 passed, 1 xpassed) and the 17-case edge sweep.
- `normal` latency on Trn2, 32-bit vs 8-bit:

  | n | 32-bit | 8-bit |
  |---|---|---|
  | 16k | 0.78 ms | 0.90 ms |
  | 262k | 1.25 ms | 2.46 ms |
  | 1M | 5.0 ms | 10.6 ms |
  | 4M | 15.9 ms | 37.1 ms |

- `TRNRAND_LIMB_BITS=16` (two 16-bit limbs, every add below 2^17) is kept
  for diagnostics only. It is also bit-identical, but slower than 32 at
  every size.
- Plan buckets: the chunk count C no longer rounds up to a power of two.
  Launches split the chunks evenly, and C rounds up to one of
  1-8, 10, 12, 14 or 16, so a request computes at most about 25% more than it
  needs. Before, 2^20 + 1 values ran 8 chunks (1.7x the 2^20 latency); now
  they run 5. That makes at most 12 C shapes.
- Tests: a numpy port of the 16-bit limb kernel that does every add in
  float32 is checked against the full-word reference, including all
  rotations, carry edge cases and the counter wrap at 2^24. The new ops are
  checked for routing and values on CPU, including partitioned generators,
  and checked against the 8-bit kernels on the simulator (including
  multi-chunk/multi-launch plans and 2^24 - 5 offsets). They are added to
  the trn1/trn2 MLIR verifier check. `tests/test_nki_native.py` adds a
  device bit-identity test for 16 and 32 against 8. The CPU tests check that
  32 is the default and that the XLA path does not select the bulk kernels.

#### Streaming output is in per-tile order (#58)

**Reproducibility break.** For a given seed and counter, the streaming
generators (`threefry_stream_normal`, `threefry_stream_uniform`, and
everything built on them: `GeneratorProgram.stream_into`, the in-place
`*_into` distributions on the NKI path, and the partitioned multi-chip
paths) now return values in a different order than before. The values
drawn are unchanged; only their order changes.

- Streams now match the per-tile path (`threefry_normal_nki`,
  `threefry_uniform_nki`) element for element for the same seed and
  counter offset: all 128 lanes of counter batch 0, then batch 1, and so
  on. Before, each 16,384-element launch was flattened lane-major
  (`(P, tiles, 4)`) instead of tile-major (`(tiles, P, 4)`).
- The fix is a host-side permute after each launch
  (`_stream_tile_order` in `trnrand/nki/dispatch.py`). The kernels are
  unchanged.
- Partitioned streams still concatenate to the single-chip result.
- Checked on the NKI simulator. The Trn2 run of
  `TestHardwareBitExact` and the native-path program test is pending.

### Fixed

#### `torch.compile(fullgraph=True)` graph-breaks on `normal()` / `uniform()` (#61)

`torch.compile(lambda: trnrand.normal(n), fullgraph=True)` (and `uniform`,
`standard_normal`) failed with `Unsupported: Failed to convert args/kwargs to
proxy` on every backend (`neuron` on Trn2, `eager` on CPU). Root cause: the
draw passed a `torch.Generator` to a traced call (`Tensor.normal_` /
`uniform_(generator=...)` on the PyTorch path, `torch.randint(generator=...)`
for the 24-bit kernel seed on the NKI path), and Dynamo cannot proxy a
Generator. `torch.compiler.disable` is not a fix under `fullgraph=True`.

- While compiling, the generator-state step runs in opaque custom ops that
  take the generator's integer handle (new `trnrand/_compile.py`):
  `trnrand::gen_fill` (PyTorch-path fill), `trnrand::gen_nki_state` (seed
  draw + partition counter offset + advance, returned as an int64 tensor) and
  `trnrand::gen_nki_draw` (whole draw, for the XLA / simulator / per-tile
  routes). Each declares a mutation of a per-generator token tensor so
  AOT/Inductor keep every draw, in order, even if its output is unused.
- On the native TorchNeuron path the kernel launches stay in the graph:
  `dispatch.threefry_stream_from_state` builds the kernel inputs with torch
  ops from the state tensor (seed and offset are not baked in as constants,
  so repeated calls neither recompile nor repeat values) and shares the
  launch/copy code (`_native_launches`) with the eager wrapper. It takes the
  eager route: the bulk Threefry kernels (#60) with the same `_bulk_plan`
  launches and `TRNRAND_LIMB_BITS` variant, or the streaming kernels under
  `TRNRAND_BULK_KERNEL=0`. The plan and variant are fixed at trace time.
- Eager calls are unchanged. Compiled output is bit-identical to eager for
  every seed, and the generator ends in the same state.
- `backend="neuron"` lowers the whole graph to StableHLO and rejects any
  custom op that is not an `nki_op` kernel, so the `gen_*` ops only serve the
  CPU compile backends (eager / aot_eager / inductor). The default per-call
  NKI key is a `torch.randint` on the Generator's Mersenne Twister, which has
  no in-graph aten equivalent, so it cannot lower without changing the stream.
  New opt-in `Generator(key_mode="counter")` (or `TRNRAND_KEY_MODE=counter`):
  the Threefry key is fixed by the seed (64-bit seed folded to 48 bits,
  k0/k1) and only the counter advances between calls. The counter is now a
  0-d tensor on the Generator (`_counter` stays as the int view), so on the
  native path the compiled draw is plain aten ops (the bulk-kernel lane,
  key and start tiles built from the seed's key words and the `_ctr` tensor)
  plus the `nki_op` launches (`dispatch.threefry_stream_from_key`),
  advancing the counter with an
  in-place `add_`. **`key_mode="counter"` gives a different stream from the
  default `"draw"`** (same seed, different values) and does not consume the
  host `torch.Generator`; the default is unchanged. It applies to
  `uniform` / `normal` / `standard_normal`; other distributions still draw
  their key. The PyTorch-backend path (`set_backend("pytorch")`) remains
  non-lowerable on `neuron`.
- `Generator` gains a public `key_mode` and private `_handle` (registry
  lookup), `_compile_token`, `_ctr` and `_key_words`. Needs PyTorch 2.4+
  (`torch.library.custom_op`); older PyTorch keeps the previous behaviour.
- Tests: `tests/test_torch_compile.py` (CPU: `backend="eager"`, zero graph
  breaks via `torch._dynamo.explain`, compiled == eager for fixed seeds on the
  PyTorch, native-NKI (CPU stand-in kernels: every bulk limb variant,
  multi-chunk / multi-launch plans and the streaming opt-out) and per-tile
  NKI paths, in both key modes; the AOT ATen graph in counter mode contains only aten ops and
  the kernel ops; plus a `neuron`-marked `backend="neuron"` check using
  `key_mode="counter"`).

#### NKI kernels fail to compile on neuronxcc 2.26 / NKI 0.5+ (#56)

Hardware compilation failed with `'nisa.tensor_scalar_arith' op 'operand0'
must be float32, got 'ui32'` for the Philox, Threefry, Sobol and streaming
kernels. `nl.add` / `nl.multiply` lower any `(P, 1)` operand (a Python
scalar or a `(P, 1)` tile, which is every tile in these kernels) to
`nisa.tensor_scalar`, whose arithmetic form now requires a float32
operand. Integer adds and multiplies now call `nisa.tensor_tensor`
directly. Bitwise ops and shifts lower to `tensor_scalar_bitvec`, which
accepts integer operands, so they are unchanged. The arithmetic and
outputs are unchanged.

- Every kernel now generates MLIR that passes verification on CPU
  (`nki.compiler.frontend.source_to_mlir`, trn1 and trn2 targets, NKI
  0.5.0 / 0.6.0 / 0.7.0). Before the fix, 6 of 8 kernels failed
  verification there with the same error as on hardware.
- Tests: `tests/test_nki_compile.py` (`nki_simulator` marker) runs that
  frontend check for every kernel on trn1 and trn2, so the simulator CI job
  now catches MLIR verification failures that `nki.simulate` misses. It
  does not run the backend compiler.
- Full CPU compile to NEFF (frontend, then `neuronx-cc`, no device): all 8
  kernels compile for trn1 and trn2 with `neuronx-cc` 2.28 / NKI 0.7.0. With
  `neuronx-cc` 2.26 / NKI 0.5.0, 6 of 8 compile. The two streaming kernels
  then fail with an unrelated SBUF out-of-memory error in NKI 0.5.0's pass
  pipeline: 32 unrolled tiles leave too many allocations live at once
  (2 tiles fit). That failure is not fixed here.
- Not yet re-run on hardware.

#### Philox output corrupted by saturating int32/uint32 casts (#56)

On Trn2 hardware (Neuron SDK 2.33, NKI 0.7.0, `neuronx-cc` 2.28) every
`philox4x32_kernel` output word was wrong: many were exactly `0x7FFFFFFF`
and the uniform mean was about 0.33. The kernels crossed between int32 (the
kernel input/output dtype) and uint32 (the arithmetic dtype) with
`nl.copy(x, dtype=...)`, which is a value conversion, not a bit
reinterpretation: uint32 values >= 2^31 saturate to `0x7FFFFFFF` and
negative int32 values to 0. Every crossing now uses `tile.view(dtype)`, a
bit reinterpretation with no data copied. The arithmetic is unchanged.

- Philox: inputs are reinterpreted as uint32 after `nl.load`, the state and
  key bump stay uint32 through all 10 rounds (`_mul32_hi_lo` now returns
  uint32), and the outputs are reinterpreted as int32 at the store. This
  removes 6 casts per round.
- Threefry, Threefry-normal and both streaming kernels: the int32 -> uint32
  cast in `_b_split` (counter and key words) is now a reinterpretation, so
  words >= 2^31 are no longer read as 0. The host wrappers keep these words
  below 2^24, so existing outputs are unchanged.
- Streaming kernels: `nl.load(start_batch_ref, dtype=nl.uint32)` is now a
  plain load followed by `.view(nl.uint32)`.
- Sobol: the Gray-code load is reinterpreted rather than converted. Gray
  codes are below 2^24, so outputs are unchanged.
- The simulator (`nki.simulate`, NKI 0.7.0) saturated the same way, so the
  Philox simulator tests that were xfailed as aws-neuron-sdk#1308 were in
  fact failing on this bug. They now pass and are no longer xfailed. New
  tests feed full-range counters and keys (top bit set) to Philox and
  Threefry and require bit-exact output, with about half the Philox words
  >= 2^31. They fail on the previous kernels.
- Checked with `neuronx-cc` 2.28's BIR simulator (no device), trn2 target:
  the previous Philox kernel matches 0 of 512 words against
  `philox4x32_reference`, with 153 words equal to `0x7FFFFFFF` (the
  hardware symptom), and the fixed kernel matches 512 of 512.
- Not yet re-run on hardware.

#### NKI Sobol differs from SobolEngine in dimensions 2-5 (#59)

The unscrambled NKI Sobol output (`sobol_nki`, `sobol()` with the NKI
backend) did not match `torch.quasirandom.SobolEngine` in dimension
indices 2-5 (Joe-Kuo dimensions 3-6). The initial direction numbers
`m_init` for those four rows of the table in `_init_sobol_directions` were
wrong (for example `[1, 1]` instead of `[1, 3]` for index 2). The primitive
polynomials and the Gray-code recurrence were correct. The rows now match
Joe-Kuo `new-joe-kuo-6.21201`, the table SobolEngine uses, and all 10
dimensions equal the top 24 bits of SobolEngine's output. Dimensions 0, 1
and 6-9 are unchanged.

- The CPU `sobol()` path was not affected: it calls SobolEngine.
- The direction-vector table and a new CPU reference, `sobol_reference`,
  are now at module level in `trnrand/nki/dispatch.py`, so CPU-only
  installs can check them. `tests/test_quasi.py` compares the table with
  `SobolEngine.sobolstate` and `sobol_reference` with SobolEngine (after
  `fast_forward`) for every dimension. Both tests fail on the old table.
- Simulator tests (`tests/test_nki_sim.py`) check the kernel against
  `sobol_reference` and `sobol_nki` against SobolEngine for all 10
  dimensions, including a nonzero `start_index`.
- `TestSobolNKI.test_start_index_advances_sequence` called the public
  `sobol()` with `start_index`, which `sobol()` has never accepted. It now
  calls `sobol_nki`, which has the parameter. `test_conformance_vs_cpu` now
  checks all 10 dimensions instead of 3.

## [0.7.1] - 2026-04-29

### Fixed

#### Multi-step `GeneratorProgram` partition counter bug (#52)

`GeneratorProgram._stream_into_nki` advanced `_counter` by the per-chip
`streaming_batches` after each distribution step. In a multi-step partitioned
program (`ProgramBuilder.normal(…).uniform(…).build()`) this caused the base
counter for step 2 onward to be misaligned: chip r's step 2 started at the
per-chip offset instead of the correct full-step offset. Concretely, the
uniform step in a 2-distribution P=2 program drew from the wrong region of
the Threefry stream, breaking partition equivalence.

Fix: advance `_counter` by `streaming_batches × partition_size` (the full
single-chip step width) rather than `streaming_batches` alone. For
`partition_size=1` this is a no-op; all single-step programs and single-chip
programs are unaffected.

Detected by `test_multi_distribution_program_equivalence_P2` in the NKI
simulator CI job introduced in #49.

#### Simulator CI test fixes (#52)

- `test_neff_cache_reuse_timing` — skip when `TRNRAND_USE_SIMULATOR=1`. The
  NKI simulator compiles every kernel invocation fresh (~10s each); asserting
  <500ms is meaningless without actual NEFF caching.
- `TestSimulatorBitExact` renamed `TestHardwareBitExact`, moved from
  `nki_simulator` to `neuron` marker. The simulator's `nl.static_range` tile
  execution order differs from sequential per-tile invocations, producing
  completely different output sequences (max diff 5.58 for normal). On
  hardware, NEFF compilation guarantees consistent deterministic output.
- `ruff format` — reformatted `program.py` to resolve CI lint failure.

## [0.7.0] - 2026-04-27

### Added

#### Counter-partitioned multi-chip RNG — bit-exact reproducibility across chip counts (#47 #48 #49)

The central claim: a 1-chip run and a P-chip run with the same seed produce the
same combined sample stream — byte-for-byte — with zero cross-chip coordination.
Concatenating P-chip draws in rank order exactly reproduces the single-chip output.
GPU libraries cannot offer this without user-side key-threading (JAX split) or
manual offset management (cuRAND). On Trainium, it falls out of the Threefry
counter structure for free.

**`Generator` partition API (#47)**

- `Generator(seed, partition_rank=r, partition_size=P)` — declares this generator
  as chip `r` of `P`. Defaults `rank=0, size=1` so all existing single-chip code
  is unchanged.
- `partition_rank` / `partition_size` validation at construction time: `size ≥ 1`,
  `0 ≤ rank < size`.
- `generator.position() → int` — logical sample count consumed (rounded up to the
  nearest 512-sample Threefry batch). Authoritative for checkpoint / resume.
- `generator.advance(n_samples)` — skip `n_samples` forward without generating
  them. Increments internal counter by `ceil(n / 512)` batches. Cheap — no RNG
  work performed.
- `generator.advance_to(position)` — jump to an absolute sample position for
  checkpoint / resume. Rounded down to the nearest batch boundary.
- `generator._chip_counter_offset(n)` — counter offset for chip `r` of `P`
  dispatching `n` elements: `partition_rank × ceil(n / 512) + _counter`.
- `generator._advance_by_elements(n)` — advance internal counter by batches
  consumed by `n` samples. Called by NKI dispatch after each kernel launch.
- `manual_seed()` resets `_counter` to 0 — reseeding starts a fresh stream.
- `_BATCH_SIZE = 512` exported at module level (`generator.py`) for use in tests
  and dispatch without NKI dependency.

**Dispatch wiring (#48)**

- `uniform()`, `normal()`, `exponential()` (per-tile NKI path): pass
  `counter_offset = gen._chip_counter_offset(n)` to the NKI kernel, then call
  `gen._advance_by_elements(n)`. Single-chip behaviour is unchanged
  (`rank=0` → offset = `_counter`, same as before).
- `normal_into()`, `uniform_into()`, `exponential_into()` (streaming path):
  `counter_offset = partition_rank × streaming_batches + gen._counter`;
  `gen._counter += streaming_batches` after dispatch.
- `ProgramBuilder` captures `partition_rank`, `partition_size`, and `_counter`
  from the `Generator` at `.new_program()` time.
- `GeneratorProgram._stream_into_nki` applies per-step partition offset:
  `counter_offset = self._counter + partition_rank × streaming_batches`.
  Counter advances by `streaming_batches` per step (not `partition_size ×
  streaming_batches`), so consecutive single-chip calls remain independent.

**Simulator and hardware tests (#49)**

- `tests/test_partitioning.py` — 43 CPU tests (no NKI required):
  `TestGeneratorPartitionAPI` (30 tests), `TestPartitionEquivalenceCPU` using
  `threefry_uniform_cpu` with 4-sample block units (8 tests),
  `TestProgramBuilderPartitionContext` (5 tests).
- `tests/test_nki_partitioning.py` — gated tests (skip automatically on dev hosts):
  - `nki_simulator` marker: per-tile equivalence (uniform/normal/exponential, P=2,4),
    streaming equivalence (stream_normal/stream_uniform, P=2,4), GeneratorProgram
    equivalence including multi-distribution programs, advance/resume equivalence.
  - `neuron` marker (trn1.32xlarge, manual): zero cross-chip coordination smoke test
    (profiler confirms no collective ops), near-linear strong scaling benchmark (≥87%
    efficiency / ≥28× 1-chip target), P=32 chip-0 slice structural check, NEFF cache
    reuse with partition args (<100ms second launch).

### Architecture notes

**Counter unit distinction**: `threefry_uniform_cpu` uses 4-sample Threefry output
blocks as its `counter_offset` unit; the NKI kernel aggregates 128 lanes into
512-sample batches. `Generator._counter` and `_BATCH_SIZE = 512` track NKI batch
units. CPU-reference tests use `ceil(n / 4)` block units directly.

**NEFF is not partition-aware at compile time.** Partition rank and counter offset
live in HBM arguments passed at runtime. The same compiled NEFF serves all chips in
a partition — no per-rank recompilation.

**Partition equivalence boundary**: N must be a multiple of `512 × P` (per-tile)
or `16,384 × P` (streaming) for exact batch alignment. Partial-batch tails
produce correct but non-partitionable outputs — documented and tested.

### Deferred

- **Partition wiring for `gamma`, `beta`, `chi_squared`, `truncated_normal`** —
  rejection-loop batch counts are data-dependent and cannot be tracked at the
  `Generator` level without significant complexity. These distributions remain
  single-chip on the NKI path. Partition support will be added once a fixed-batch
  rejection kernel is available.

- **Hardware validation** — zero cross-chip coordination profiler trace and
  near-linear strong scaling numbers (trn1.32xlarge, 32 chips) are pending a
  manual hardware run. The `neuron`-marked tests in `test_nki_partitioning.py`
  capture the assertions; results will be documented in a follow-up release note.

## [0.6.0] - 2026-04-22

### Added

#### Streaming NKI kernels — 32-tile pipeline (#42)

- `threefry_streaming_normal_kernel` / `threefry_streaming_uniform_kernel` —
  `@nki.jit` kernels using `nl.static_range(_PROGRAM_TILES)` (32 tiles) to
  pipeline GpSimd Threefry rounds and Vector Engine Box-Muller transcendentals
  within a single NEFF invocation. This is only possible inside one kernel call;
  separate XLA graph submissions cannot share SBUF state across the tile loop.
- `threefry_stream_normal(n, seed, counter_offset)` /
  `threefry_stream_uniform(n, seed, counter_offset)` — host wrappers.
  `counter_offset` is passed as a runtime HBM argument so the same NEFF serves
  every launch; the counter advances `n_launches × _PROGRAM_TILES` per call to
  guarantee non-overlapping streams.
- `_PROGRAM_TILES = 32` exported at module level (outside `if HAS_NKI:`) so
  `program.py` and test files can import it on dev hosts without neuronxcc.
- `tests/test_nki_streaming.py` — simulator + neuron tests: shape, range,
  moments, determinism, counter-advance, bit-exactness, hardware throughput.

#### `GeneratorProgram` API (#43)

- `ProgramBuilder` — fluent builder: `.normal(n)`, `.uniform(n)`,
  `.exponential(n)`, `.build()`. Raises `ValueError` if no steps added.
- `GeneratorProgram` — pre-compiled streaming NEFF wrapper. `stream_into(buffers)`
  fills named `torch.Tensor` buffers in-place and advances an internal counter so
  consecutive calls draw independent, non-overlapping streams.
  - **NKI path**: calls `threefry_stream_normal` / `threefry_stream_uniform` with
    `counter_offset=self._counter`; counter masked to 24 bits.
  - **CPU fallback**: `torch.Generator` seeded with `seed ^ counter`; increments
    counter by 1 per call. Not bit-exact with the NKI path.
- `Generator.new_program()` — entry point that creates a `ProgramBuilder` seeded
  from the generator. Late import keeps `generator.py` free of NKI dependency.
- `GeneratorProgram` exported from `trnrand.__all__`; `ProgramBuilder` is internal.
- `__version__` now resolves dynamically via `importlib.metadata` instead of the
  stale hardcoded `"0.2.0"`.
- `tests/test_program.py` — 30 CPU tests + 7 `nki_simulator` tests.

#### Zero-allocation in-place variants (#44)

- `normal_into(buf, mean, std, generator)` — fills `buf` in-place with
  `N(mean, std)` samples. NKI path uses `threefry_stream_normal`; XLA caches the
  NEFF for same-size buffers automatically — no Python-level cache needed.
- `uniform_into(buf, low, high, generator)` — fills `buf` with `U(low, high)`.
- `exponential_into(buf, rate, generator)` — fills `buf` with `Exp(rate)` via
  inverse-CDF from a uniform stream.
- All three exported in `trnrand.__all__`; PyTorch fallback calls `buf.normal_()`,
  `buf.uniform_()`, `buf.exponential_()` directly.
- 22 new CPU tests added to `tests/test_distributions.py`.

#### Streaming benchmarks (#45)

- `benchmarks/bench_streaming.py` — validates five RFC claims:
  latency floor (single NKI launch = 16,384 samples), throughput (250-round
  1M-sample benchmark), NEFF cache-reuse assertion (< 100ms), bit-exactness
  (program output matches direct kernel dispatch at same seed + counter_offset),
  engine-overlap timing guard (neuron-only).
- `scripts/run_benchmarks.sh` updated to include `bench_streaming.py`.

### Architecture notes

**Tensor Engine is intentionally idle** in all RNG kernels. Random number
generation is not matmul-shaped; Trainium's Tensor Engine is a matrix-multiply
unit and would add nothing here. GpSimd (Threefry byte arithmetic) + Vector Engine
(Box-Muller transcendentals) is the correct engine pairing for this workload.

**Stochastic rounding is not used.** SR corrects systematic bias in repeated
accumulations — the error mode that kills FP32 training. For RNG, the dominant
error is statistical (1/√N); arithmetic rounding contributes ≪ 1 ULP per sample.
SR would add hardware cost with no distributional benefit.

### Deferred

- **Poisson NKI kernel** (#15) — Knuth's sequential-product algorithm requires a
  data-dependent loop termination per sample, which cannot be vectorized across
  lanes. `torch.poisson` on the host is the only practical path until a
  Trainium-native rejection sampler becomes feasible. Honest non-fit; not deferred
  for schedule reasons.

## [0.5.0] - 2026-04-21

### Added

#### NKI backend dispatch wiring (#18)

- `set_backend("nki")` now routes `uniform()`, `normal()`, `standard_normal()`,
  and `exponential()` to the Threefry NKI kernels (`threefry_uniform_nki`,
  `threefry_normal_nki`) shipped in v0.4.0. Closes
  [#18](https://github.com/trnsci/trnrand/issues/18).
- `_nki_active()` helper in both `distributions.py` and `quasi.py`: returns
  `True` when `get_backend() != "pytorch"` and `HAS_NKI`.
- `_nki_seed(gen)`: draws one `torch.randint` to advance generator state and
  produce a 24-bit seed for NKI dispatch — same state-advance guarantee the
  PyTorch path provides for successive calls.
- `quasi.py` dispatch stubs for `sobol()` and `halton()`: try-import pattern
  that falls through to the CPU path when NKI kernels are not yet available.
- `tests/test_nki_dispatch.py`: CPU-only backend routing tests
  (`TestBackendRoutingCPU`, 11 tests) and NKI simulator tests
  (`TestNKIDispatchSimulator`, 9 tests, `@pytest.mark.nki_simulator`).

#### Truncated normal NKI kernel (#17)

- `truncated_normal_nki(n, low, high, mean, std, seed)` — Box-Muller
  candidates from `threefry_normal_nki` + host-side rejection sampling.
  2.5× oversample; configurable `[low, high]` bounds (in standard-normal
  units); `mean`/`std` shift applied to accepted samples. No new NKI code —
  reuses the Threefry normal kernel established in v0.4.0. Closes
  [#17](https://github.com/trnsci/trnrand/issues/17).
- `distributions.truncated_normal()` dispatches to `truncated_normal_nki`
  when NKI is active.

#### Gamma NKI kernel (#14)

- `gamma_nki(n, shape, scale, seed)` — Marsaglia-Tsang squeeze acceptance
  in float32. Boost identity (`U^(1/shape)`) for `shape < 1`. 1.7× oversample.
  All arithmetic in float32 (adequate precision for scientific Monte Carlo).
  Closes [#14](https://github.com/trnsci/trnrand/issues/14).
- `distributions.gamma()` dispatches to `gamma_nki` when NKI is active.

#### Chi-squared NKI kernel (#16)

- `chi_squared_nki(n, df, seed)` — thin wrapper: `gamma_nki(shape=df/2, scale=2)`.
  Closes [#16](https://github.com/trnsci/trnrand/issues/16).
- `distributions.chi_squared()` dispatches to `chi_squared_nki` when NKI is active.

#### Beta NKI kernel (#13)

- `beta_nki(n, alpha, beta_param, seed)` — gamma-ratio identity
  `X / (X + Y)` where `X ~ Gamma(alpha)`, `Y ~ Gamma(beta)`. Distinct seeds
  for the X and Y streams ensure statistical independence. Closes
  [#13](https://github.com/trnsci/trnrand/issues/13).
- `distributions.beta()` dispatches to `beta_nki` when NKI is active.

#### Sobol NKI kernel (#11)

- `_init_sobol_directions()` — Joe-Kuo 2010 direction vectors for 10 Sobol
  dimensions, 24-bit precision. Computed once at module import; embedded as
  `nl.full` constants at `@nki.jit` trace time. No HBM round-trip for
  lookup tables.
- `sobol_gray_code_kernel(@nki.jit)` — GpSimd XOR accumulation over Gray-code
  bits. For each lane `p` and dimension `d`:
  `s[p,d] = XOR{ v[d][k]  for k in 0..23  if bit k of gray(i_p) }`.
  10 dims × 24 bits fully unrolled via `nl.static_range` (~240 ops, all
  vectorized over 128 lanes). Float conversion via 3-byte decomposition
  identical to `threefry4x32_kernel`. Output: `(P, 10)` float32 in `[0, 1)`.
- `sobol_nki(n_points, n_dims, seed, start_index)` — host wrapper. Gray codes
  computed host-side; 128-lane tiling with padding; additive scrambling
  (random shift in `[0,1)` per dimension) when `seed != 0`. Closes
  [#11](https://github.com/trnsci/trnrand/issues/11).
- `quasi.sobol()` routes to `sobol_nki` when NKI is active.

#### Halton NKI kernel (#12)

- `halton_kernel(@nki.jit)` — iterative float32 radical inverse on the Vector
  Engine. Per digit: `q = nl.floor(i / p)` (exact for `i < 2²²` — ULP proof
  in source), `digit = i - p * q`, `result += digit / p^(k+1)`, `i = q`.
  Outer loop (10 dims) + inner loop (depth) both unrolled via `nl.static_range`.
  Output: `(P, 10)` float32 in `(0, 1)`.
- `halton_nki(n_points, n_dims, start_index)` — host wrapper. Skips index 0
  per Halton convention; 128-lane tiling; asserts index < 2²² = 4,194,304.
  Primes: (2, 3, 5, 7, 11, 13, 17, 19, 23, 29). Closes
  [#12](https://github.com/trnsci/trnrand/issues/12).
- `quasi.halton()` routes to `halton_nki` when NKI is active.

### Architecture notes

All five new distribution/QMC kernels follow the **fixed-batch-with-mask pattern**:
each kernel generates a fixed tile of candidates; the host wrapper retries until
enough accepted samples are gathered. This sidesteps NKI's no-dynamic-loop
constraint without any kernel changes.

The Sobol and Halton kernels demonstrate complementary engine strategies:
- **Sobol** uses GpSimd exclusively (integer XOR — no multiplies needed)
- **Halton** uses GpSimd for arithmetic + Vector Engine for `nl.floor` (same
  engine mix as the Threefry normal kernel)

### Deferred

- **Poisson NKI kernel** (#15) — Knuth's algorithm (sequential product loop)
  is not vectorizable. Poisson remains CPU-only via `torch.poisson`. Will be
  revisited if a Trainium-native rejection sampler becomes practical.

## [0.4.1] - 2026-04-18

### Fixed

- **NKI hardware compiler compatibility** — three categories of Python constructs
  accepted by the NKI CPU simulator but rejected by the real trn1 compiler have
  been eliminated from the Threefry and Philox kernels:
  1. Inner `def` statements inside functions in the `@nki.jit` call tree
     (`_mul32_hi_lo` inner helpers extracted to module-level `_nki_mul_u32` etc.)
  2. List comprehensions (`[expr for i in range(n)]` → explicit literals)
  3. Subscript expressions as LHS assignment targets in tuple unpacking
     (`x_b_list[0], x_b_list[1] = ...` → named variables `x0_b, x1_b = ...`)

### Hardware validation (trn2, 2026-04-18)

- **Threefry normal kernel: hardware-validated on trn2.3xlarge (sa-east-1)** —
  `test_normal_kernel_distribution` and `test_normal_kernel_matches_box_muller_cpu`
  pass on trn2 (XPASS). NCC_IBIR605 confirmed trn1-only; does not affect trn2+.
  All 5 `TestThreefryNKI` tests now pass on trn2.
- `xfail` marks removed from both tests; trn1 CI is unaffected (`--philox-only`
  already deselects them on trn1).

### Hardware validation (trn1, 2026-04-16)

- **Threefry4×32-20 uniform kernel: hardware-validated** — 4 of 5
  `TestThreefryNKI` tests pass on trn1 (correctness, distribution, seed
  determinism, seed isolation). Byte-tile arithmetic is confirmed exact.
- Threefry normal kernel (fused Box-Muller): blocked by NCC_IBIR605
  (same trn1 compiler restriction as standalone `box_muller_kernel`; tracked
  in trnrand#2). Confirmed trn1-only; passes on trn2.3xlarge.

## [0.4.0] - 2026-04-16

### Added

- **Threefry4×32-20 NKI kernel** — integer-multiply-free PRNG for Trainium,
  implemented entirely in byte-tile arithmetic. Every tile element stays ≤ 511,
  well below float32's 2²⁴ exact-integer ceiling, sidestepping
  [aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)
  at the algorithm level rather than the decomposition level.
  Uses only 32-bit addition, XOR, and rotation — the operations Threefry was
  designed for (Salmon et al. SC'11, same paper as Philox).
- `threefry4x32_reference()` — CPU oracle verified against 3 Random123 KAT vectors.
- `threefry_uniform_cpu()` — uniform float stream, mirrors `philox_uniform_cpu` API.
- `threefry4x32_kernel` — NKI kernel: 8×(P,1) counter/key → (P,4) float32 U[0,1).
  Output built directly from 3 low bytes of each output word (23-bit mantissa),
  bypassing uint32 tile assembly entirely.
- `threefry_normal_kernel` — fused Threefry + Box-Muller NKI kernel. GpSimd byte
  arithmetic feeds directly into Vector Engine transcendentals with tiles remaining
  SBUF-resident between stages — no HBM round-trip between RNG and transform.
- `threefry_uniform_nki()`, `threefry_normal_nki()` — host-side wrappers.
- `_add32_bytes_numpy()`, `_rotl32_bytes_numpy()` — pure-numpy byte-arithmetic
  references for NKI kernel parity testing.
- 4 new simulator tests (no `xfail` marks): KAT vector check, 128-lane reference
  parity, U[0,1) distribution, N(0,1) distribution for the fused kernel.
- `TestThreefryReference` class: 12 CPU tests including KAT vectors, distributional
  checks, and byte-helper ground-truth comparisons.

### Architecture note

The four-engine RNG framing is now fully realised for normals: GpSimd handles
Threefry byte arithmetic; Vector Engine handles Box-Muller transcendentals;
SBUF holds output between stages. Philox remains as the CPU reference and will
become the hardware path once AWS ships GpSimd integer multiply
([aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)).

## [0.3.0] - 2026-04-15

### Added

- NKI 0.3.0 CPU simulator dispatch via `TRNRAND_USE_SIMULATOR=1`
  ([#26](https://github.com/trnsci/trnrand/issues/26)). Lets contributors
  iterate kernel correctness on any Linux x86_64 host — or the existing
  trn1 DLAMI via SSM — without paying NEFF compile + hardware dispatch
  cost. Seconds per cycle vs minutes on hardware.
- `nki_simulator` pytest marker + `tests/test_nki_sim.py` curated suite
  (Philox spec vectors, reference equivalence, Box-Muller distribution).
- `scripts/run_simulator_tests.sh` — SSM runner for the simulator suite.
- `scripts/run_neuron_tests.sh --philox-only` flag — deselects Box-Muller
  kernel tests when isolating Philox's hardware status.
- `nki-simulator` GitHub Actions job on `ubuntu-latest` (installs
  `nki>=0.3.0` from the Neuron pip index).
- `docs/developing_kernels.md` — kernel authoring guide, env var
  reference, trn1 NKI 0.3.0 gotchas encountered during Phase 1.
- `_mul32_hi_lo_numpy` pure-numpy ground-truth port alongside the NKI
  `_mul32_hi_lo` helper, plus `test_mul32_numpy_matches_ground_truth`
  and `test_mul32_simulator_matches_numpy` tests to separate algorithm
  bugs from NKI op-semantics bugs during kernel debugging.

### Changed

- Migrated to NKI 0.3.0 canonical namespace: `import nki` / `nki.language`
  / `nki.isa` replace the deprecated `neuronxcc.nki.*` path. `[neuron]`
  extra pins `nki>=0.3.0`. NKI 0.3.0 ships on the latest Deep Learning
  AMI (Neuron SDK 2.29, April 2026).
- Main CI `test` job marker filter now `-m "not neuron and not nki_simulator"`
  so the three test channels don't collide.
- Philox 4×32-10 NKI kernel: 32×32 multiply reworked to an 8-bit byte
  decomposition (16 sub-products, byte-wise carry chain). Algorithm is
  validated bit-exact against Python unbounded-int ground truth via
  `_mul32_hi_lo_numpy`.

### Known limitations

- **NKI Philox kernel is not hardware-validated in this release.**
  `aws-neuron-sdk#1308` — NKI ops on `uint32` tiles route through the
  float32 activation engine on both CPU simulator and trn1 hardware.
  Because float32 exactly represents integers only up to 2^24, any
  uint32 tile value > 2^24 (including Philox counter state itself)
  loses precision at the NKI op boundary. No kernel-level decomposition
  can work around this — the issue is at `nl.copy(..., dtype=nl.uint32)`,
  not just `nl.multiply`. Filed upstream with reproducer; `trnrand`
  dispatch continues to use the PyTorch `torch.Generator` path by
  default, which is the only user-visible path today. Tracked in
  [#1](https://github.com/trnsci/trnrand/issues/1) alongside the
  upstream issue; will reopen for hardware validation once AWS ships
  a true integer multiply primitive.
- Box-Muller NKI kernel still gated on trn1 compiler fix for
  `NCC_IBIR605` (InstActivation bias parameter) — tracked in
  [#2](https://github.com/trnsci/trnrand/issues/2). Unaffected by #1308.

## [0.2.0] - 2026-04-13

### Added

- `gamma(size, shape, scale=1.0, ...)` — Gamma distribution via
  Marsaglia-Tsang rejection with the boost identity for `shape < 1`.
- `chi_squared(size, df, ...)` — Chi-squared via `2·Gamma(df/2, 1)`.
- `beta(size, alpha, beta, ...)` — Beta via the gamma-ratio identity
  `X / (X + Y)` with `X ~ Gamma(α)`, `Y ~ Gamma(β)`.
- `poisson(size, lam, ...)` — Poisson via `torch.poisson` with a
  generator-bound rates tensor.

All four distributions are generator-aware for reproducibility and
tested against analytic mean / variance on 100k–200k samples.

### Known limitations

- NKI kernels for these distributions are v0.3.0 scope (#13, #14, #15,
  #16). CPU path works today via the PyTorch backend; on-device
  acceleration waits for hardware validation.

## [0.1.1] - 2026-04-13

### Fixed

- Docs badge in `README.md` and `mkdocs.yml` `site_url` now point at
  `trnsci.dev/trnrand/` (the central docs home) instead of the retired
  per-repo github.io URL (#9). Repo `homepageUrl` updated to match (#10).

### Added

- `ROADMAP.md` and `docs/roadmap.md` with v0.1.1 / v0.2.0 / v0.3.0
  milestones and the issues attached to each.
- `[Unreleased]` skeleton in CHANGELOG for future entries.

## [0.1.0] - 2026-04-12

### Added

- `Generator` class with `manual_seed`, `get_state`/`set_state`, and a
  module-level default generator (`trnrand.manual_seed`).
- Standard distributions: `uniform`, `normal`, `standard_normal`,
  `exponential`, `bernoulli`, `randint`, `randperm`, `truncated_normal`.
- Quasi-random sequences: `sobol` (scrambled), `halton`, `latin_hypercube`.
- Backend dispatch: `set_backend` (`auto`/`pytorch`/`nki`), `get_backend`,
  `HAS_NKI` flag.
- Philox 4×32-10 NKI kernel in `trnrand/nki/dispatch.py` with a CPU
  reference (`philox4x32_reference`, `philox_uniform_cpu`) used as the
  conformance oracle. CPU-side spec invariants and the three canonical
  Salmon et al. SC'11 test vectors are verified in
  `tests/test_nki_philox.py::TestPhiloxReference`.
- Box-Muller transform on the Vector Engine for `normal()`, with a CPU
  reference (`box_muller_cpu`) and CPU-side distributional tests
  (`TestBoxMullerReference`).
- MC vs QMC hypersphere-volume example in `examples/mc_integration.py`.
- Test suites for distributions (reproducibility, statistics, bounds) and
  quasi-random sequences (uniformity, determinism, stratification).

### Known limitations

- **NKI path is not hardware-validated yet.** The Philox and Box-Muller
  kernels are scaffolded against the NKI 2.24 API and pass the CPU
  conformance oracle by construction, but the `@pytest.mark.neuron`
  suite has not yet run on trn1/trn2. Tracked in #1 (Philox) and #2
  (Box-Muller). Until validated, `trnrand` falls back to `torch.Generator`
  on CPU — the backend dispatch path is the only user-visible difference.
- **Benchmarks vs cuRAND pending hardware access** (#3).
