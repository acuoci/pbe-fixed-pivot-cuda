# Examples Guide

The `examples/` directory contains five self-contained validation programs, each comparing the GPU solver against a known analytical solution. All examples share a common utilities header (`examples/common/pbe_examples_utils.hpp`) that provides grid construction, moment computation, RAII device array management, and analytical solution structs.

---

## Building and running

```bash
# Build all examples
cmake .. -DPBE_BUILD_EXAMPLES=ON
cmake --build . -j$(nproc)

# Run individual examples from the build directory
./examples/homogeneous_batch/example_homogeneous_batch
./examples/aggregation_sum_kernel/example_aggregation_sum_kernel
./examples/aggregation_product_kernel/example_aggregation_product_kernel
./examples/homogeneous_batch_breakage/example_homogeneous_batch_breakage
./examples/breakage_ziff_mcgrady/example_breakage_ziff_mcgrady

# Build a single example
cmake --build . --target example_homogeneous_batch
```

---

## Example 1 — Constant kernel aggregation (Smoluchowski)

**Directory:** `examples/homogeneous_batch/`
**Target:** `example_homogeneous_batch`

### Physical problem

Pure aggregation with a constant kernel β(u,v) = β₀ and a monodisperse initial condition (all particles in the smallest bin).

### Analytical solution

For a monodisperse IC, the total number concentration satisfies the Smoluchowski result:

```
N_tot(t) = N₀ / (1 + t/t_half),   t_half = 2/(β₀ N₀)
M₁(t)    = N₀ v_min               (conserved)
```

### Configuration

| Parameter | Value |
|---|---|
| Bins | 256, geometric, r = 2^(1/3) |
| v_min | 1×10⁻¹⁸ m³ |
| β₀ | 1×10⁻¹⁷ m³/s |
| N₀ | 1×10¹⁴ #/m³ |
| Integrator | Explicit Euler, dt = 1 s |
| t_end | 2000 s |

### Expected output

```
  N_tot error : ~1.7e-04  PASS
  M1 error    : ~0.0e+00  PASS (machine precision)
```

### What it demonstrates

- Basic usage of `launch_aggregation_rhs` with the Constant kernel
- Pre-zeroing of `rhs` before each call
- How to pre-compute `log_x0` and `inv_log_r` from the host grid
- Integration into a simple Euler time loop

---

## Example 2 — Sum kernel aggregation (Golovin)

**Directory:** `examples/aggregation_sum_kernel/`
**Target:** `example_aggregation_sum_kernel`

### Physical problem

Pure aggregation with the sum kernel β(u,v) = β₀(u+v) and an exponential initial condition n(v,0) = (N₀/vc) exp(-v/vc).

### Analytical solution

```
s     = β₀ N₀ vc t
M₀(t) = N₀ exp(-s)     (exponential decay)
M₁(t) = N₀ vc          (conserved)
```

### Configuration

| Parameter | Value |
|---|---|
| Bins | 300, geometric, v ∈ [10⁻⁴, 10⁴] |
| β₀, N₀, vc | 1.0 (dimensionless) |
| Integrator | RK4, dt = 10⁻⁴ |
| t_end | 0.3 |

### Expected output

```
  M0 error : ~1.4e-04  PASS
  M1 error : ~1.6e-04  PASS
```

### What it demonstrates

- Sum kernel usage
- Exponential initial condition using the exact cell-integral formula in `make_exponential_ic`
- RK4 integration pattern (4 RHS evaluations per step)
- `DeviceArray<T>` RAII wrapper for clean device memory management

---

## Example 3 — Product kernel aggregation (Smoluchowski)

**Directory:** `examples/aggregation_product_kernel/`
**Target:** `example_aggregation_product_kernel`

### Physical problem

Pure aggregation with the product kernel β(u,v) = β₀ uv and an exponential initial condition. This kernel approaches gelation at finite time t_gel = 1/(2β₀ N₀ vc²).

### Analytical solution

```
T     = β₀ N₀ vc² t
M₀(t) = N₀ (1 − T/2)            (linear decay)
M₁(t) = N₀ vc                   (conserved)
M₂(t) = 2 N₀ vc² / (1 − 2T)    (diverges at T = 0.5: gelation)
```

### Configuration

| Parameter | Value |
|---|---|
| Bins | 300, geometric, v ∈ [10⁻⁴, 10⁴] |
| β₀, N₀, vc | 1.0 (dimensionless) |
| t_end | 0.3 (60% of t_gel = 0.5) |
| Integrator | RK4, dt = 10⁻⁴ |

### Expected output

```
  M0 error : ~1.7e-04  PASS
  M2 error : ~4.8e-03  PASS
  M1 error : ~1.6e-04  PASS
```

### What it demonstrates

- Product kernel usage near gelation conditions
- Validation through both M₀ (linear decay) and M₂ (diverging second moment)
- How increasing M₂ error near gelation is expected and physically meaningful

---

## Example 4 — Linear selection + symmetric binary breakage

**Directory:** `examples/homogeneous_batch_breakage/`
**Target:** `example_homogeneous_batch_breakage`

### Physical problem

Pure breakage with linear selection S(v) = S₀ v/v_ref and symmetric binary daughter distribution (two fragments of v/2 each). Monodisperse initial condition in bin 384 of a 512-bin grid.

### Analytical solution

For linear selection with any daughter distribution and any IC:

```
N_tot(t) = N₀ (1 + S₀ t)    (linear growth)
V_tot(t) = N₀ v₀            (conserved)
```

> **Note:** The linear growth formula `N₀(1+S₀t)` is correct for **linear** selection. For **constant** selection S(v) = S₀ with ν fragments, the growth is exponential: `N₀ exp((ν−1) S₀ t)`.

### Configuration

| Parameter | Value |
|---|---|
| Bins | 512, geometric, r = 2^(1/3) |
| IC bin | 384 (near top — fragments propagate downward) |
| S₀ | 1×10⁻³ s⁻¹ |
| Quadrature | 1 point: t_q = [0.5], bw_q = [2.0] |
| Integrator | Explicit Euler, dt = 0.01 s |
| t_end | 100 s |

### Expected output

```
  N_tot error : ~1e-04  PASS
  V_tot error : ~0.0    PASS (machine precision)
```

### What it demonstrates

- Basic usage of `launch_breakage_rhs` with linear selection
- Symmetric binary quadrature setup (n_quad=1)
- Volume conservation to machine precision — a key property of the fixed-pivot method
- Why volume conservation is a stronger test than N_tot for breakage

---

## Example 5 — Ziff-McGrady breakage benchmark

**Directory:** `examples/breakage_ziff_mcgrady/`
**Target:** `example_breakage_ziff_mcgrady`

### Physical problem

Pure breakage with linear selection S(v) = S₀ v/vc and uniform daughter distribution b(v|v′) = 2/v′. Exponential initial condition n(v,0) = (N₀/vc) exp(-v/vc). This is the classical benchmark of Ziff & McGrady (1985), used as the primary breakage validation case in the associated paper.

### Analytical solution

```
τ     = S₀ t
n(v,t) = (N₀/vc) (1+τ)² exp(−(1+τ)v/vc)

M₀(t) = N₀ (1 + τ)               (linear growth)
M₁(t) = N₀ vc                    (conserved)
M₂(t) = 2 N₀ vc² / (1 + τ)      (decaying)
```

### Configuration

| Parameter | Value |
|---|---|
| Bins | 200, geometric, v ∈ [10⁻⁴, 10¹] |
| S₀, N₀, vc | 1.0 (dimensionless) |
| Quadrature | 8-point Gauss-Legendre on (0,1) |
| bw_q | 2 × GL weights (uniform daughter) |
| Integrator | Explicit Euler, dt = 10⁻³ |
| t_end | 5.0 (τ = 5) |

### Expected output

```
  M0 error : ~2.2e-04  PASS
  M1 error : ~1.8e-04  PASS
```

These values are consistent with Table 6 of the associated paper (PSD errors < 10⁻³, moment errors < 10⁻⁵ for 200 bins).

### What it demonstrates

- Gauss-Legendre quadrature setup for a continuous daughter distribution
- How to set `bw_q` for the uniform daughter: `bw_q[q] = 2 * GL_weight[q]`
- The Ziff-McGrady benchmark used in the paper for breakage validation
- Validation at large dimensionless times (τ up to 5)

---

## Shared utilities reference

All examples include `examples/common/pbe_examples_utils.hpp`. Key items:

### `DeviceArray<T>`

RAII device array — allocates on construction, frees on destruction. Eliminates boilerplate `cudaMalloc`/`cudaFree` pairs.

```cpp
DeviceArray<double> d_N(n);    // allocate
d_N.upload(N_host);             // host → device
d_N.download(N_host);           // device → host
d_N.zero();                     // cudaMemset to 0
double* ptr = d_N.get();        // raw pointer for API calls
```

### Grid construction

```cpp
// Geometric grid: x[i] = v_min * r^i
auto x = make_geometric_grid(n, v_min, r);

// Geometric grid spanning [v_min, v_max]
auto x = make_geometric_grid_range(n, v_min, v_max);
```

### Initial conditions

```cpp
// Exponential IC: exact integral of (N0/vc)*exp(-v/vc) over each cell
auto N = make_exponential_ic(x, N0, vc);
```

### Moments

```cpp
double M0 = compute_M0(N);           // total number
double M1 = compute_M1(N, x);        // total volume (M1 = sum N[i]*x[i])
double M2 = compute_M2(N, x);        // second moment
```

### Analytical solutions

```cpp
ConstantKernelAnalytical  ana{N0, vc, beta0};   // Scott, exponential IC
SumKernelAnalytical       ana{N0, vc, beta0};   // Golovin, exponential IC
ProductKernelAnalytical   ana{N0, vc, beta0};   // product, exponential IC
ZiffMcGradyAnalytical     ana{N0, vc, S0};      // Ziff-McGrady

// Access moments
double m0 = ana.M0(t);
double m1 = ana.M1(t);
double m2 = ana.M2(t);
```

---

## Adding your own example

1. Create a new directory under `examples/`:
   ```bash
   mkdir examples/my_example
   ```

2. Write `examples/my_example/CMakeLists.txt`:
   ```cmake
   find_package(CUDAToolkit REQUIRED)
   add_executable(example_my_example main.cpp)
   target_link_libraries(example_my_example PRIVATE
       pbe_cuda::pbe_cuda CUDA::cudart pbe_examples_common)
   set_target_properties(example_my_example PROPERTIES
       CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
   ```

3. Add to root `CMakeLists.txt`:
   ```cmake
   if(PBE_BUILD_EXAMPLES)
       # ... existing entries ...
       add_subdirectory(examples/my_example)
   endif()
   ```

4. Include the shared utilities:
   ```cpp
   #include <pbe_cuda/pbe_cuda.cuh>
   #include "pbe_examples_utils.hpp"
   ```
