# API Stability

trnrand follows [Semantic Versioning](https://semver.org/). This page
documents which parts of the API are stable under semver and which are
experimental and may change in minor releases.

## Stable surface

The following public API is covered by semver: breaking changes require
a major version bump (currently gated at 1.0.0 — all 0.x.y → 0.x.z
releases preserve stable-API compatibility).

**Generator and seeding:**

- `trnrand.Generator` — `__init__`, `manual_seed`, `get_state`,
  `set_state`, `torch_generator`
- `trnrand.manual_seed`
- `trnrand.get_default_generator`

**Distributions** (module: `trnrand.distributions`, all re-exported at
top level):

- `uniform`, `normal`, `standard_normal`, `exponential`, `bernoulli`,
  `randint`, `randperm`, `truncated_normal`
- `gamma`, `chi_squared`, `beta`, `poisson` *(added in v0.2.0)*

**Quasi-random sequences** (module: `trnrand.quasi`, all re-exported):

- `sobol`, `halton`, `latin_hypercube`

**Backend control:**

- `trnrand.HAS_NKI` (read-only bool)
- `trnrand.set_backend("auto" | "pytorch" | "nki")`
- `trnrand.get_backend()`

Signature changes to any of the above in a 0.x.y release are bugs.

## Experimental surface

Everything under `trnrand.nki.*` is internal wiring that may change
without a major bump. Users should only interact with the NKI path via
`set_backend("nki")`. Directly importing kernels, reference functions, or
constants from `trnrand.nki.dispatch` is not supported.

Examples of what's experimental:

- `trnrand.nki.dispatch.philox4x32_kernel`, `philox4x32_reference`,
  `box_muller_kernel`, `box_muller_cpu`, constants like `PHILOX_M0`
- Any future kernel-level helpers added for performance work

These are tested and documented for contributors, not for downstream use.

## Deprecation process

When a stable API needs to change in a way that would affect users:

1. The new behavior lands alongside the old, with the old path emitting
   `DeprecationWarning` on first call.
2. The CHANGELOG `### Deprecated` section notes the change and the
   replacement.
3. After **at least one minor release** of grace (e.g., deprecated in
   v0.3.0 → removable in v0.4.0), the old path can be removed.
4. Pre-1.0 removals happen in minor releases; post-1.0 removals require
   a major bump.

Experimental APIs are exempt — they can change or disappear in any
minor release without deprecation.

## Version semantics

- **Patch (0.x.y → 0.x.z):** bug fixes, documentation, CI changes,
  internal refactors with no user-visible surface change. Safe to
  upgrade without reading the CHANGELOG.
- **Minor (0.x.0 → 0.y.0):** additive changes to the stable API; may
  change or remove experimental APIs. Skim the CHANGELOG before
  upgrading if you use `trnrand.nki.*` directly.
- **Major (x.0.0 → y.0.0):** breaking changes to the stable API.
  Read the migration guide before upgrading. Currently no 1.0.0 planned.

## Reproducibility guarantees

A fixed seed produces the same output sequence across patch releases of
the same minor version (e.g., v0.2.0 and v0.2.1 yield identical
`Generator(seed=42).uniform(...)` output).

Across minor releases, seed → output mappings may change when an
algorithm is upgraded (e.g., switching a distribution from rejection
sampling to an analytic method). Such changes will be flagged in the
CHANGELOG under `### Changed` with a note about reproducibility impact.
Workaround: pin to a specific minor version for reproducible research
runs.
