# Architecture Roadmap

## Contributor Note

Before performing architectural or refactoring work, contributors and coding
agents should consult this document. If an implementation requires changing an
accepted architectural decision or a migration invariant, update this document
explicitly rather than allowing the code and roadmap to diverge silently.

## 1. Motivation and Scope

This project began as research and benchmark software supporting a scientific
paper on high-performance CUDA evaluation of monovariate population balance
equations (PBEs) discretized with the fixed-pivot sectional method. The current
code emphasizes efficient CPU/CUDA aggregation and breakage RHS calculations,
analytical verification cases, and standalone 0D examples using simple Euler or
RK4 time integration.

The long-term objective is to evolve the codebase into a modern, robust,
embeddable C++/CUDA scientific library suitable for CFD and multiphysics
applications. The core library responsibility is evaluation of the semi-discrete
PBE right-hand side:

```text
dN/dt = R(N, ...)
```

The library should not impose a time-integration strategy. Standalone Euler/RK4
drivers may remain as examples or verification utilities, but they should not
define the core architecture.

## 2. Design Goals

Primary goals:

- Preserve numerical correctness, conservation behavior, and validated results.
- Preserve the existing fixed-pivot aggregation and breakage formulations.
- Preserve or improve CPU and CUDA performance.
- Provide a clean, stable, strongly typed public API.
- Make the library easy to embed in CFD and multiphysics codes.
- Support repeated RHS calls with externally owned state and RHS storage.
- Separate configuration, evaluation conditions, storage, workspace, and backend.
- Avoid unnecessary allocations, copies, host/device transfers, and synchronization
  during RHS evaluation.
- Make future PBE processes easier to add, including nucleation, growth,
  condensation/evaporation, sintering, oxidation, and shrinkage.
- Improve maintainability and testability through clearer component boundaries.

Explicit non-goals:

- The core library is not an ODE solver library.
- The core API should not require Euler, RK4, CVODE, or any specific integration
  strategy.
- The redesign should not force all future physical processes into one overly
  generic abstraction.
- The migration should not rewrite validated numerical algorithms without an
  explicit numerical-change decision and regression evidence.

## 3. Architectural Principles

- Separate public API from backend implementation details.
- Separate physical model definitions from fixed-pivot/numerical machinery where
  practical.
- Prefer composition and small concrete responsibilities over deep inheritance
  hierarchies.
- Runtime model selection/configuration may occur at setup or dispatch level.
  Performance-critical CUDA paths should retain compile-time specialization where
  useful.
- Users or external solvers should normally own population and RHS arrays.
- Model objects should own configuration and reusable/precomputed model metadata.
- Workspace/backend objects should own reusable temporary resources.
- No unnecessary allocations, transfers, synchronization, or virtual dispatch
  inside hot RHS loops.
- Configuration errors should be detected primarily during setup/construction,
  not inside O(N^2) loops.
- Backend-specific APIs should make CPU versus CUDA memory expectations clear.

The architecture must distinguish:

1. **Model configuration:** relatively immutable information such as grid,
   enabled processes, selected submodels, quadrature choices, and constants that
   truly do not vary.
2. **Evaluation conditions:** local/time-varying physical inputs such as
   temperature, pressure, viscosity, density, turbulence dissipation, shear rate,
   supersaturation, gas composition, or other CFD fields.

Future physical kernels may depend on local conditions:

```text
beta_ij = beta_ij(T, p, mu, epsilon, rho, ...)
```

Therefore the design must not assume that all physical parameters are
construction-time constants. A mature CFD-facing API should allow one grid/model
configuration to be reused for many local PBE states and many changing local
condition sets.

## 4. Target Architecture

The proposed target architecture is layered:

```text
User / external solver
  |
  v
Top-level RHS evaluator / PBE model
  |
  +-- model configuration
  +-- sectional grid and grid metadata
  +-- enabled physical process models
  +-- backend plan / precomputed data
  +-- reusable workspace
  |
  v
RHS assembler
  |
  +-- fixed-pivot numerical utilities
  +-- aggregation implementation
  +-- breakage implementation
  +-- future process implementations
  |
  v
CPU backend or CUDA backend
```

High-level component concepts:

- **Sectional grid / grid views:** validated pivots, optional bin boundaries,
  geometric-grid metadata, host views, and eventually device representations.
- **Fixed-pivot numerical utilities:** bracketing, interpolation weights, boundary
  clipping policy, and birth redistribution helpers.
- **Aggregation:** physical aggregation kernels and backend-specific RHS assembly.
- **Breakage:** selection functions, daughter distributions/quadrature, and
  backend-specific RHS assembly.
- **Model configuration:** selected processes, submodels, numerical choices, and
  immutable/precomputed metadata.
- **Evaluation conditions:** per-call/per-cell physical conditions that may vary
  without rebuilding the model.
- **Backend execution:** CPU serial execution and CUDA execution with explicit
  memory semantics.
- **Reusable workspace:** temporary arrays, CUDA streams or scratch buffers, and
  generated quadrature/device data that must not be allocated inside hot RHS
  calls.
- **Top-level RHS evaluator/model:** stable public entry point coordinating
  enabled process contributions.

This document intentionally does not freeze class names or final interfaces that
have not yet been implemented.

## 5. Public API Direction

Conceptual final usage:

```cpp
SectionalGrid grid = SectionalGrid::geometric(n, v_min, ratio);

PBEModelConfig config;
config.grid = grid;
config.processes.aggregation =
    AggregationModel::brownian_continuum_shear(/* coefficients/config */);
config.processes.breakage =
    BreakageModel{/* selection */, /* daughter distribution */};

PBEModel model(config);          // reusable for many cells/states
PBEWorkspace workspace(model);   // reusable scratch/resources

for (int cell = 0; cell < n_cells; ++cell) {
    EvaluationContext ctx;
    ctx.temperature = T[cell];
    ctx.pressure    = p[cell];
    ctx.viscosity   = mu[cell];
    ctx.shear_rate  = G[cell];

    model.compute_rhs(N[cell], rhs[cell], ctx, workspace);
}
```

The exact API remains subject to refinement during migration. Important intended
properties are:

- one model can serve many local states/cells;
- state and RHS storage are externally owned;
- changing local conditions do not require rebuilding the model;
- CPU and CUDA memory expectations are explicit;
- no hidden time integration is performed.

## 6. Numerical and Performance Invariants

Refactoring must preserve:

- fixed-pivot sectional formulation;
- current aggregation and breakage numerical behavior;
- birth/death handling and boundary clipping policy unless explicitly changed;
- conservation properties for interior events;
- existing analytical verification cases;
- CPU/CUDA agreement within defined tolerances;
- CUDA compile-time specialization where it matters for hot kernels;
- absence of hidden allocations, transfers, or synchronization in hot RHS calls.

Bitwise reproducibility is not a universal requirement. Where operation ordering
or backend differences make bitwise equality unrealistic, tests should use
well-defined numerical tolerances and document the expected error scale.

## 7. Staged Migration Roadmap

### Phase 0: C++17 Baseline

- **Objective:** Make the language standard policy explicit and uniform.
- **Principal changes:** Use C++17 consistently for library, tests, examples, and
  CUDA compilation where practical. Avoid `std::span`; introduce a small C++17
  view type later.
- **Invariants:** Existing launch APIs and numerical results unchanged.
- **Required tests:** CPU-only build; CUDA build where available; existing unit and
  analytical tests.
- **API implications:** None initially.
- **Main risks:** CMake/toolchain standard mismatches.
- **Status:** Completed.

### Phase 1: Baseline Regression Lock

- **Objective:** Freeze current validated behavior before structural changes.
- **Principal changes:** Add compact CPU/CUDA agreement tests, formula-level
  tests, and fixed-pivot reference tests.
- **Invariants:** Current RHS values preserved within established tolerances.
- **Required tests:** Current CPU tests; CUDA tests where available; new
  regression tests.
- **API implications:** None.
- **Main risks:** Existing boundary-clipping behavior must be documented rather
  than accidentally changed.
- **Status:** Completed.

### Phase 2: Lightweight Array Views

- **Objective:** Introduce non-owning array views without changing kernels.
- **Principal changes:** Add simple C++17 `ArrayView<T>` /
  `ArrayView<const T>` or equivalent.
- **Invariants:** Raw-pointer launch APIs continue to work.
- **Required tests:** View construction tests; existing RHS tests.
- **API implications:** Additive/internal at first.
- **Main risks:** Views must remain pointer-size wrappers and avoid overhead.
- **Status:** Completed.

### Phase 3: SectionalGrid

- **Objective:** Make the sectional grid a reusable validated object.
- **Principal changes:** Add grid construction/validation, pivot ownership,
  monotonicity checks, geometric metadata, and lightweight grid views.
- **Invariants:** Existing kernels still receive equivalent `x`, `n`, `log_x0`,
  and `inv_log_r`.
- **Required tests:** Invalid grids; geometric detection; lookup metadata
  equivalence with current examples.
- **API implications:** Additive; low-level launch APIs remain.
- **Main risks:** Off-by-one or roundoff differences in geometric lookup.
- **Status:** Completed.

### Phase 4: Fixed-Pivot Utility Extraction

- **Objective:** Isolate fixed-pivot lookup/interpolation from physical kernels.
- **Principal changes:** Add CPU and CUDA helper utilities for bracketing,
  interpolation, birth allocation, and clipping.
- **Invariants:** Aggregation and breakage RHS behavior unchanged.
- **Required tests:** Interior birth allocation; clipping; conservation; CPU/CUDA
  agreement.
- **API implications:** None publicly.
- **Main risks:** Small bracket differences can perturb results.
- **Status:** Completed.

### Phase 5: Configuration vs Evaluation Conditions

- **Objective:** Introduce the model-configuration/evaluation-condition split.
- **Principal changes:** Add early `PBEModelConfig` and `EvaluationContext`
  concepts. Current constant coefficients may remain configuration values.
- **Invariants:** Current kernels may ignore `EvaluationContext`.
- **Required tests:** Empty context works; validation hooks ready for future
  condition-dependent models.
- **API implications:** Additive concepts for later high-level API.
- **Main risks:** Overpopulating the context too early.
- **Status:** Completed.

### Phase 6: Strongly Typed Aggregation Configuration

- **Objective:** Make aggregation selection clearer and harder to misuse.
- **Principal changes:** Add typed aggregation model configs, likely backed by
  `std::variant` or a similarly explicit C++17 mechanism. Map once to current
  internal params.
- **Invariants:** Current aggregation RHS results unchanged.
- **Required tests:** Every config maps to expected kernel/coefficients; invalid
  coefficients rejected at setup.
- **API implications:** Preferred high-level config begins; low-level API remains.
- **Main risks:** Do not visit variants inside O(N^2) loops.
- **Status:** Not started.

### Phase 7: Strongly Typed Breakage Configuration

- **Objective:** Separate breakage selection models and daughter distributions.
- **Principal changes:** Add typed selection configs and daughter distribution
  configs; generate quadrature once during setup.
- **Invariants:** Current breakage RHS behavior unchanged.
- **Required tests:** Daughter quadrature generation; symmetric/uniform
  verification; invalid daughter parameters.
- **API implications:** Preferred high-level config for breakage.
- **Main risks:** Preserve an expert path for user-supplied quadrature.
- **Status:** Not started.

### Phase 8: Backend Resource Layer

- **Objective:** Prepare explicit CPU/CUDA memory and resource semantics.
- **Principal changes:** Add CPU/CUDA backend concepts, device grid ownership,
  CUDA stream handling, and reusable workspaces.
- **Invariants:** Existing low-level CUDA launches continue to work.
- **Required tests:** Device grid upload once; workspace reuse; CUDA smoke tests.
- **API implications:** Add explicit backend/resource types.
- **Main risks:** Avoid implicit transfers and synchronization.
- **Status:** Not started.

### Phase 9: CPU PBEModel

- **Objective:** Introduce the first high-level CPU RHS model.
- **Principal changes:** Add top-level CPU `compute_rhs` coordinating enabled
  processes, external state/RHS views, evaluation context, and workspace.
- **Invariants:** Results match manual aggregation/breakage calls.
- **Required tests:** Aggregation-only, breakage-only, combined RHS, repeated
  calls with changing contexts.
- **API implications:** New high-level CPU API.
- **Main risks:** Avoid allocations or expensive validation in `compute_rhs`.
- **Status:** Not started.

### Phase 10: CUDA PBEModel

- **Objective:** Add high-level CUDA RHS model without performance loss.
- **Principal changes:** Add CUDA model execution using explicit device views,
  device grid, stream/workspace resources, and existing specialized kernels.
- **Invariants:** CUDA RHS matches old launch functions and CPU references within
  tolerance.
- **Required tests:** CUDA model tests; CPU/CUDA agreement; repeated calls;
  stream behavior where practical.
- **API implications:** New high-level CUDA API.
- **Main risks:** Accidental synchronization, transfers, or dispatch overhead.
- **Status:** Not started.

### Phase 11: Multi-Cell / Batched Readiness

- **Objective:** Ensure CFD usage does not require one model per cell.
- **Principal changes:** Validate repeated single-state calls with changing
  contexts; design storage/context APIs to allow future batch evaluation.
- **Invariants:** Single-state API remains clean and efficient.
- **Required tests:** Many repeated calls; workspace reuse; no per-call
  allocations.
- **API implications:** Possible future batch API, not required initially.
- **Main risks:** Premature batching could distort the core API.
- **Status:** Not started.

### Phase 12: Move Integrators Out of Core

- **Objective:** Prevent Euler/RK4 utilities from driving architecture.
- **Principal changes:** Move or keep integrators in examples/utilities and make
  them consume the high-level RHS API.
- **Invariants:** Verification examples produce the same errors.
- **Required tests:** Existing analytical examples.
- **API implications:** Optional utilities only.
- **Main risks:** Accidentally creating an unsupported ODE solver layer.
- **Status:** Not started.

### Phase 13: Project Reorganization

- **Objective:** Separate core library, tests, verification, benchmarks, and
  examples.
- **Principal changes:** Reorganize directories and CMake targets after
  abstractions stabilize.
- **Invariants:** Build options, install target, and validation behavior remain.
- **Required tests:** Full CTest matrix; example builds.
- **API implications:** None intended.
- **Main risks:** CMake churn.
- **Status:** Not started.

### Phase 14: Future Process Interface

- **Objective:** Make nucleation, growth, sintering, and related processes natural
  additions.
- **Principal changes:** Document or implement a minimal internal process
  contribution convention.
- **Invariants:** Aggregation and breakage behavior unchanged.
- **Required tests:** A trivial constant source process may be used as an
  architectural test before adding physical models.
- **API implications:** Future extension point.
- **Main risks:** Growth-like processes may require numerical schemes distinct
  from aggregation/breakage; avoid forcing all processes into one hierarchy.
- **Status:** Not started.

## 8. Architectural Decisions

| Decision | Current recommendation | Status | Rationale |
|---|---|---|---|
| C++ standard | Use C++17 consistently across host code, tests, examples, and CUDA-facing code | Accepted | C++17 maximizes CUDA/HPC compatibility and is sufficient for the planned design |
| CPU/CUDA backend strategy | Keep both first-class; make memory/backend expectations explicit | Accepted | Avoids ambiguous host/device pointer usage and preserves performance |
| Runtime vs compile-time model selection | Runtime at setup/wrapper level; compile-time specialization in hot CUDA paths | Accepted | Usable configuration without hot-loop overhead |
| State/RHS ownership | External solver/user owns `N` and `rhs` | Accepted | Enables embedding and avoids copies |
| Workspace ownership | Workspace/backend objects own reusable scratch/resources | Accepted | Avoids allocations inside repeated RHS calls |
| Model ownership | Model owns configuration and precomputed model/grid metadata | Accepted | One model can be reused for many CFD cells |
| Evaluation conditions | Per-call context distinct from model configuration | Accepted | Future physical kernels depend on local CFD conditions |
| Configuration strategy | Strongly typed C++ configuration first; optional parsers outside core | Accepted | Reduces misuse in embedded scientific codes |
| Error handling | Detect configuration errors during setup; return/report lightweight runtime errors from compute/backend calls | Open | Need final choice between exceptions, status codes, or mixed policy |
| Backward compatibility | Reasonable breaking changes allowed; preserve numerical behavior | Accepted | Current API is research-oriented; clean architecture is higher priority |
| Low-level launch APIs | Keep during migration as compatibility/internal validation layer | Revisit | Useful for incremental tests; may become advanced/internal API later |

## 9. Future Extensions

Anticipated future processes and model families include:

- nucleation;
- surface growth;
- condensation/evaporation;
- sintering;
- oxidation/shrinkage;
- turbulent aggregation;
- Fuchs/transition-regime Brownian kernels;
- temperature-, pressure-, viscosity-, and composition-dependent rates;
- more advanced breakage selection and daughter distributions.

These processes have different mathematical structures. Aggregation, nucleation,
and internal-coordinate growth should not be forced prematurely into one generic
process hierarchy. The architecture should first provide clear seams for adding
new process contributions while preserving process-specific numerical algorithms.

## 10. Change Log

| Date | Change |
|---|---|
| 2026-08-17 | Initial architecture roadmap created. |
| 2026-08-17 | Completed Phase 0: project C++ standard policy aligned to C++17. |
| 2026-08-17 | Completed Phase 1: added baseline formula, fixed-pivot reference, and CPU/CUDA agreement regression tests. |
| 2026-08-17 | Completed Phase 2: added lightweight C++17 `ArrayView` and construction/behavior tests without changing raw-pointer launch APIs. |
| 2026-08-17 | Completed Phase 3: added validated host `SectionalGrid` with owned pivots, geometric metadata, grid views, and tests while preserving raw-pointer launch APIs. |
| 2026-08-17 | Completed Phase 4: extracted shared fixed-pivot bracket and birth-allocation helpers for CPU/CUDA paths with direct utility tests and unchanged RHS launch APIs. |
| 2026-08-17 | Completed Phase 5: added early `PBEModelConfig` and per-call `EvaluationContext` types with setup/context validation tests and no RHS dispatch changes. |
