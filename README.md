# pbe-fixed-pivot-cuda

**GPU-accelerated Fixed Pivot Method for Population Balance Equations (PBE) — aggregation and breakage CUDA kernels**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21428566.svg)](https://doi.org/10.5281/zenodo.21428566)

> **Associated publication:**
> A. Cuoci, *GPU-Accelerated Fixed-Pivot Population Balance Simulations: Enabling High-Resolution Modeling of Aggregation–Breakage Systems*, submitted to Next Chemical Engineering, 2026.

---

## Overview

This repository provides a **C++/CUDA library** implementing the aggregation and breakage source terms of the Population Balance Equation (PBE) discretised with the **Fixed Pivot Sectional Method** of Kumar and Ramkrishna (1996). The kernels are designed for high-resolution simulations with thousands of size sections, achieving GPU speedups of one to two orders of magnitude relative to optimised CPU implementations.

The library computes the right-hand side (RHS) of the semi-discrete PBE:

```
dN/dt = R_agg(N) + R_br(N) + R_src(...)
```

where `N` is the vector of section-integrated number concentrations. The caller is responsible for time integration, allowing the library to be embedded in any ODE solver (explicit Euler, RK4, CVODE, etc.) or CFD framework.

### Key features

- **Aggregation** — eight kernels: constant, sum, product, Brownian continuum, Brownian free-molecular, shear, and both Brownian/shear combinations
- **Breakage** — four selection functions (constant, linear, power-law, threshold) and four daughter distributions (uniform, symmetric binary, power-law, erosion)
- **Additive source contribution** — minimal constant-source extension point for future process models
- **Fixed-pivot redistribution** — exact conservation of particle volume by construction
- **O(1) bin lookup** for geometric grids; O(log N) binary search fallback for general grids
- **Tiled shared-memory accumulation** — reduces global atomic contention at high resolution
- **Compile-time kernel dispatch** — zero warp divergence for uniform kernel flags
- **C++ public API** — no CUDA syntax exposed to the caller; drop-in integration into any C++ or CFD code

### Performance (NVIDIA A30, CUDA 12.6)

| Kernel | GPU crossover vs Numba | Speedup vs Numba (N=16384) | Speedup vs NumPy (N=16384) |
|---|---|---|---|
| Constant / Sum / Product | N ≈ 64 | ~45–50× | ~70–80× |
| Brownian continuum / Shear | N ≈ 16 | ~120× | ~105× |
| Brownian continuum + shear | N ≈ 16 | ~122× | ~136× |

For the full aggregation–breakage flocculation case study (N=3840, 30 min simulation): **GPU 37.6 s vs Numba 4064 s vs NumPy 9756 s** (~108× and ~260× speedup respectively).

---

## Requirements

| Component | Minimum version |
|---|---|
| CUDA Toolkit | 12.0 |
| CMake | 3.18 |
| C++ standard | C++17 (host), C++17 (device) |
| GPU compute capability | sm_80 (A100, A30) or sm_86 (RTX 30xx, A40) |
| Host compiler | GCC 9+ or equivalent |

---

## Quick build

```bash
git clone git@github.com:YOUR_USERNAME/pbe-fixed-pivot-cuda.git
cd pbe-fixed-pivot-cuda
mkdir build && cd build

# Configure — set your GPU architecture:
# sm_80 = A100, A30   |   sm_86 = RTX 3090, A40   |   sm_90 = H100
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86" \
    -DPBE_BUILD_EXAMPLES=ON \
    -DPBE_BUILD_TESTS=ON

cmake --build . -j$(nproc)
```

### Build options

| CMake option | Default | Description |
|---|---|---|
| `ENABLE_CUDA` | ON | Build CUDA kernels and examples. Set OFF for a pure serial CPU build with no CUDA toolkit or GPU required |
| `PBE_BUILD_STATIC` | ON | Build static library (`libpbe_cuda.a`) |
| `PBE_BUILD_SHARED` | OFF | Build shared library (`libpbe_cuda.so`) |
| `PBE_BUILD_EXAMPLES` | ON | Build worked examples |
| `PBE_BUILD_TESTS` | ON | Build unit/regression tests |
| `PBE_BUILD_BENCHMARKS` | OFF | Build optional performance benchmarks |
| `PBE_ENABLE_WARNINGS` | ON | Enable extra compiler warnings |
| `CMAKE_CUDA_ARCHITECTURES` | `80 86` | Target GPU architectures (override for your hardware) |

### CPU-only build

The library can also be built as a pure serial C++ implementation:

```bash
cmake -S . -B build-cpu \
    -DENABLE_CUDA=OFF \
    -DPBE_BUILD_TESTS=ON

cmake --build build-cpu -j$(nproc)
ctest --test-dir build-cpu --output-on-failure -V
```

In this mode no CUDA toolkit, `nvcc`, CUDA runtime, or GPU is required. CUDA examples are skipped, and `launch_aggregation_rhs()` / `launch_breakage_rhs()` operate on host memory by forwarding to the serial CPU implementations. The explicit host-memory entry points `launch_aggregation_rhs_cpu()` and `launch_breakage_rhs_cpu()` are always available, including in CUDA builds for side-by-side comparisons.

---

## Quick usage

```cpp
#include <pbe_cuda/pbe_cuda.cuh>
#include <cuda_runtime.h>
#include <cmath>

// --- Aggregation example ---
pbe_cuda::AggregationParams agg;
agg.n           = 256;                          // number of size sections
agg.log_x0      = std::log(x_host[0]);          // pre-computed grid parameter
agg.inv_log_r   = 1.0 / std::log(x_host[1] / x_host[0]);  // geometric ratio
agg.kernel_type = pbe_cuda::AggregationKernel::BrownianContinuumShear;
agg.beta_bc     = 6.73e-18;                     // continuum Brownian prefactor [m³/s]
agg.beta_bfm    = 0.0;                          // free-molecular Brownian prefactor
agg.beta_sh     = 4.0 / 3.0 * G;               // shear prefactor [m³/s]

cudaMemset(d_rhs, 0, n * sizeof(double));        // rhs MUST be pre-zeroed
pbe_cuda::launch_aggregation_rhs(d_N, d_x, d_rhs, agg, stream);

// --- Breakage example ---
pbe_cuda::BreakageParams br;
br.n         = 256;
br.n_quad    = 16;                              // Gauss-Legendre quadrature points
br.selection = pbe_cuda::BreakageSelection::Threshold;
br.S0        = 1.0e-3;
br.v_min     = v_crit;

// cudaMemset not needed if rhs was already zeroed above:
pbe_cuda::launch_breakage_rhs(d_N, d_x, d_t_q, d_bw_q, d_rhs, br, stream);
```

Both functions **accumulate** into `rhs` — zero it once before calling either or both.

---

## Repository structure

```
pbe-fixed-pivot-cuda/
├── include/pbe_cuda/
│   ├── pbe_cuda.cuh              ← umbrella header
│   ├── aggregation*.{cuh,hpp}    ← aggregation launch/configuration API
│   ├── breakage*.{cuh,hpp}       ← breakage launch/configuration API
│   ├── source*.{cuh,hpp}         ← additive source contribution API
│   ├── cpu_pbe_model.hpp         ← high-level serial RHS model
│   ├── cuda_pbe_model.hpp        ← high-level CUDA RHS model
│   └── detail/                   ← shared fixed-pivot helpers
├── src/
│   ├── aggregation_cpu.cpp       ← serial aggregation RHS
│   ├── aggregation.cu            ← CUDA aggregation wrapper + dispatch
│   ├── breakage_cpu.cpp          ← serial breakage RHS
│   ├── breakage.cu               ← CUDA breakage wrapper + dispatch
│   ├── source_cpu.cpp            ← serial additive source RHS
│   ├── source.cu                 ← CUDA additive source wrapper
│   └── *_kernels.cuh             ← internal CUDA kernels
├── examples/
│   ├── common/                  ← example-only utilities and ODE steppers
│   ├── aggregation_*/           ← analytical aggregation examples
│   ├── breakage_*/              ← analytical breakage examples
│   └── flocculation_two_stage/  ← combined application case
├── tests/
│   ├── unit/                    ← focused API/configuration/kernel tests
│   └── verification/            ← analytical/regression verification tests
├── benchmarks/                  ← optional performance benchmark targets
├── docs/
├── cmake/
│   └── pbe_cudaConfig.cmake.in
└── CMakeLists.txt
```

CTest labels mirror this layout: use `ctest -L unit`,
`ctest -L verification`, or, in CUDA builds, `ctest -L cuda`.

---

## Examples

### Aggregation — Smoluchowski validation (constant kernel)

```bash
./build/examples/homogeneous_batch/example_homogeneous_batch
```

Simulates pure aggregation with a constant kernel and monodisperse initial condition. Validates the total number concentration against the analytical Smoluchowski solution `N_tot(t) = N₀ / (1 + t/t_half)`. Achieves relative error < 0.02% at t = 2000 s with 256 sections and dt = 1 s.

### Breakage — volume conservation validation (linear selection, symmetric binary daughter)

```bash
./build/examples/homogeneous_batch_breakage/example_homogeneous_batch_breakage
```

Simulates pure breakage with linear selection S(v) = S₀ v/v_ref and symmetric binary fragmentation. Validates against two independent analytical quantities: N_tot(t) = N₀ exp(S₀ t) and exact volume conservation V_tot = const. Volume is conserved to machine precision (~10⁻¹⁵) throughout.

---

## API reference

### Aggregation

```cpp
// Kernel types
enum class AggregationKernel : int {
    Constant      = 0,   // β = β₀
    Sum           = 1,   // β = β₀ (u + v)
    Product       = 2,   // β = β₀ u v
    BrownianContinuum          = 3,
    BrownianFreeMolecular      = 4,
    Shear                      = 5,
    BrownianContinuumShear     = 6,
    BrownianFreeMolecularShear = 7
};

struct AggregationParams {
    int    n;             // number of sections
    double log_x0;        // log(x[0]) — pre-computed on host
    double inv_log_r;     // 1/log(x[1]/x[0]) — 0.0 for non-geometric grids
    AggregationKernel kernel_type;
    double beta0;         // prefactor for Constant/Sum/Product
    double beta_bc;       // continuum Brownian prefactor
    double beta_bfm;      // free-molecular Brownian prefactor
    double beta_sh;       // shear prefactor
    int    block_size;    // CUDA threads per block (default 256)
};

cudaError_t launch_aggregation_rhs(
    const double* N,      // [in]  device pointer, length n
    const double* x,      // [in]  device pointer, pivot volumes, length n
    double*       rhs,    // [out] device pointer, length n — must be pre-zeroed
    const AggregationParams& params,
    cudaStream_t stream = 0);
```

### Breakage

```cpp
// Selection functions
enum class BreakageSelection : int {
    Constant  = 0,   // S(v) = S₀
    Linear    = 1,   // S(v) = S₀ v/v_ref
    PowerLaw  = 2,   // S(v) = S₀ (v/v_ref)^alpha
    Threshold = 3    // S(v) = S₀ if v > v_min, else 0
};

struct BreakageParams {
    int    n;          // number of sections
    int    n_quad;     // quadrature points per parent section
    BreakageSelection selection;
    double S0;         // selection prefactor
    double v_ref;      // reference volume (Linear, PowerLaw)
    double alpha;      // exponent (PowerLaw)
    double v_min;      // threshold volume (Threshold)
    int    block_size; // CUDA threads per block (default 256)
};

cudaError_t launch_breakage_rhs(
    const double* N,     // [in]  device pointer, length n
    const double* x,     // [in]  device pointer, pivot volumes, length n
    const double* t_q,   // [in]  device pointer, quadrature abscissae, length n_quad
    const double* bw_q,  // [in]  device pointer, quadrature weights, length n_quad
    double*       rhs,   // [out] device pointer, length n — must be pre-zeroed
    const BreakageParams& params,
    cudaStream_t stream = 0);
```

**Quadrature conventions for supported daughter distributions:**

| Distribution | n_quad | t_q | bw_q |
|---|---|---|---|
| Symmetric binary | 1 | [0.5] | [2.0] |
| Erosion (fraction ε) | 2 | [1−ε, ε] | [1.0, 1.0] |
| Uniform | user (Gauss-Legendre on (0,1)) | GL abscissae | GL weights × 2/v |
| Power-law | user (2-subinterval GL) | GL abscissae | GL weights × C(v) vq |

---

## Integration into your own project

### Option A — CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(pbe_cuda
    GIT_REPOSITORY https://github.com/YOUR_USERNAME/pbe-fixed-pivot-cuda.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(pbe_cuda)

target_link_libraries(your_target PRIVATE pbe_cuda::pbe_cuda)
```

### Option B — CMake find_package (after install)

```bash
cmake --install build --prefix /path/to/install
```

```cmake
find_package(pbe_cuda REQUIRED)
target_link_libraries(your_target PRIVATE pbe_cuda::pbe_cuda)
```

### Option C — Direct inclusion

Copy `include/` and link against `libpbe_cuda.a`. Pass `-I/path/to/include` and link with `-lpbe_cuda -lcudart`.

---

## Mathematical background

The fixed-pivot method (Kumar & Ramkrishna, 1996) discretises the PBE internal coordinate into N sections with pivot volumes x₁ < x₂ < ⋯ < xₙ. A newly formed particle of volume v* is assigned to the two neighbouring pivots xᵢ and xᵢ₊₁ with weights:

```
ηᵢ   = (xᵢ₊₁ − v*) / (xᵢ₊₁ − xᵢ)
ηᵢ₊₁ = (v* − xᵢ)   / (xᵢ₊₁ − xᵢ)
```

This two-point redistribution preserves both particle number and particle volume simultaneously.

The discrete aggregation operator evaluates all N(N+1)/2 unique collision pairs (j ≤ k), with each GPU thread assigned to one pair. The discrete breakage operator decomposes into a local death kernel (one thread per section) and a birth kernel (one thread per parent-section/quadrature-point pair).

Full mathematical details, CUDA implementation specifics, convergence analyses, and a representative flocculation case study are provided in the associated publication.

---

## Verification

The implementation was verified against:

- **Aggregation:** Scott (constant), Golovin (sum), and Smoluchowski (product) analytical solutions; Richardson extrapolation for Brownian continuum, shear, and combined Brownian/shear kernels. Normalized L₁ error converges at approximately second order (1.97–2.11 for the product kernel).
- **Breakage:** Ziff–McGrady analytical solution (linear selection + uniform daughter distribution). PSD error < 10⁻³ for dimensionless times τ = S₀t ∈ [1, 5]; moment errors < 10⁻⁵.
- **Volume conservation:** First moment M₁ preserved to machine precision for all tested configurations.

---

## Citing this work

If you use this library in your research, please cite both the software and the associated journal article:

**Software (this repository):**
```bibtex
@software{cuoci2026pbe_cuda,
  author    = {Cuoci, Alberto},
  title     = {pbe-fixed-pivot-cuda: GPU-accelerated Fixed Pivot Method for Population Balance Equations},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21428566},
  url       = {https://github.com/acuoci/pbe-fixed-pivot-cuda}
}
```

**Journal article:**
```bibtex
@article{cuoci2026gpu,
  author  = {Cuoci, Alberto},
  title   = {GPU-Accelerated Fixed-Pivot Population Balance Simulations:
             Enabling High-Resolution Modeling of Aggregation--Breakage Systems},
  journal = {Next Chemical Engineering},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Alberto Cuoci**
CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering
Politecnico di Milano, P.zza Leonardo da Vinci 32, 20133 Milano, Italy
