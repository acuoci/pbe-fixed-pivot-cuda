# API Reference

All public symbols live in the `pbe_cuda` namespace. Include the umbrella header:

```cpp
#include <pbe_cuda/pbe_cuda.cuh>
```

Or include individual module headers:

```cpp
#include <pbe_cuda/aggregation.cuh>
#include <pbe_cuda/breakage.cuh>
```

---

## Aggregation

### `enum class AggregationKernel`

Selects the aggregation frequency function β(u,v).

```cpp
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
```

### `struct AggregationParams`

All scalar parameters for one aggregation RHS evaluation.

```cpp
struct AggregationParams {
    // Grid
    int    n         = 0;     // Number of pivot sections
    double log_x0    = 0.0;   // log(x[0]) — pre-computed once on the host
    double inv_log_r = 0.0;   // 1/log(x[1]/x[0]) for geometric grids
                               // Pass 0.0 for non-geometric grids (binary search fallback)

    // Kernel type and coefficients
    AggregationKernel kernel_type = AggregationKernel::Constant;
    double beta0   = 1.0;     // Prefactor for Constant / Sum / Product kernels
    double beta_bc  = 0.0;    // Continuum Brownian prefactor
    double beta_bfm = 0.0;    // Free-molecular Brownian prefactor
    double beta_sh  = 0.0;    // Shear prefactor

    // Launch configuration
    int block_size = 256;     // CUDA threads per block (must be power of 2)
};
```

**Pre-computing grid parameters on the host:**

```cpp
double log_x0    = std::log(x_host[0]);
double inv_log_r = 1.0 / std::log(x_host[1] / x_host[0]);  // geometric grid only
```

These are passed as scalar kernel arguments to avoid recomputing them per thread.

### `launch_aggregation_rhs`

Computes the aggregation contribution to the PBE right-hand side on the GPU.

```cpp
cudaError_t launch_aggregation_rhs(
    const double*            N,       // [in]  device pointer, length n
    const double*            x,       // [in]  device pointer, pivot volumes, length n
    double*                  rhs,     // [out] device pointer, length n
    const AggregationParams& params,
    cudaStream_t             stream = 0);
```

**Parameters:**

| Parameter | Direction | Description |
|---|---|---|
| `N` | in | Device pointer to number distribution array, length `n` |
| `x` | in | Device pointer to pivot volume array, length `n`. Must be strictly positive and monotonically increasing |
| `rhs` | out | Device pointer to output RHS array, length `n`. **Must be pre-zeroed by the caller** before this call |
| `params` | in | Aggregation parameters (see `AggregationParams`) |
| `stream` | in | CUDA stream for asynchronous execution (default: 0) |

**Returns:** `cudaSuccess` on success, or a CUDA error code on failure.

**Important:** `rhs` must be zeroed before calling, e.g.:
```cpp
cudaMemset(rhs, 0, n * sizeof(double));
```
This design allows safe accumulation of multiple source terms (aggregation + breakage) into the same `rhs` array without interference.

**Thread safety:** Multiple calls on different streams with non-overlapping `rhs` arrays are safe. Concurrent calls sharing the same `rhs` array produce undefined behaviour.

**Error detection:** The return value reflects kernel launch errors. Asynchronous execution errors (from the kernel body itself) are only visible after `cudaStreamSynchronize()` or `cudaDeviceSynchronize()`.

**Minimal example:**

```cpp
pbe_cuda::AggregationParams p;
p.n           = 256;
p.log_x0      = std::log(x_host[0]);
p.inv_log_r   = 1.0 / std::log(x_host[1] / x_host[0]);
p.kernel_type = pbe_cuda::AggregationKernel::BrownianContinuumShear;
p.beta_bc     = 6.73e-18;   // continuum Brownian prefactor
p.beta_bfm    = 0.0;        // free-molecular Brownian prefactor
p.beta_sh     = (4.0/3.0) * G * std::pow(3.0/(4.0*M_PI), 1.0/3.0);

cudaMemset(d_rhs, 0, p.n * sizeof(double));
cudaError_t err = pbe_cuda::launch_aggregation_rhs(d_N, d_x, d_rhs, p);
```

---

## Breakage

### `enum class BreakageSelection`

Selects the breakage selection function S(v).

```cpp
enum class BreakageSelection : int {
    Constant  = 0,   // S(v) = S₀
    Linear    = 1,   // S(v) = S₀ v / v_ref
    PowerLaw  = 2,   // S(v) = S₀ (v / v_ref)^alpha
    Threshold = 3    // S(v) = S₀ if v > v_min, else 0
};
```

### `struct BreakageParams`

All scalar parameters for one breakage RHS evaluation.

```cpp
struct BreakageParams {
    // Grid
    int    n      = 0;    // Number of pivot sections
    int    n_quad = 0;    // Quadrature points per parent section

    // Selection function
    BreakageSelection selection = BreakageSelection::Constant;
    double S0    = 1.0;   // Selection prefactor [1/s]
    double v_ref = 1.0;   // Reference volume for Linear and PowerLaw [m³]
    double alpha = 1.0;   // Power-law exponent (PowerLaw only)
    double v_min = 0.0;   // Threshold volume below which S=0 (Threshold only) [m³]

    // Launch configuration
    int block_size = 256; // CUDA threads per block
};
```

### `launch_breakage_rhs`

Computes the breakage contribution to the PBE right-hand side on the GPU. Internally launches two kernels sequentially on the same stream:
1. **Death kernel** — `rhs[j] -= S(x[j]) N[j]` (one thread per bin, no atomics)
2. **Birth kernel** — daughter fragment redistribution via fixed-pivot interpolation

```cpp
cudaError_t launch_breakage_rhs(
    const double*         N,      // [in]  device pointer, length n
    const double*         x,      // [in]  device pointer, pivot volumes, length n
    const double*         t_q,    // [in]  device pointer, quadrature abscissae, length n_quad
    const double*         bw_q,   // [in]  device pointer, quadrature weights, length n_quad
    double*               rhs,    // [out] device pointer, length n
    const BreakageParams& params,
    cudaStream_t          stream = 0);
```

**Parameters:**

| Parameter | Direction | Description |
|---|---|---|
| `N` | in | Device pointer to number distribution array, length `n` |
| `x` | in | Device pointer to pivot volume array, length `n` |
| `t_q` | in | Quadrature abscissae in (0,1): `v_frag = x[j] * t_q[q]`, length `n_quad` |
| `bw_q` | in | Quadrature weights including daughter distribution value: `bw_q[q] = b(v_frag|x[j]) * dv_frag`, length `n_quad` |
| `rhs` | out | Device pointer to output RHS array, length `n`. **Must be pre-zeroed by the caller** |
| `params` | in | Breakage parameters (see `BreakageParams`) |
| `stream` | in | CUDA stream (default: 0) |

**Quadrature setup for supported daughter distributions:**

| Distribution | `n_quad` | `t_q` | `bw_q` |
|---|---|---|---|
| Symmetric binary | 1 | `{0.5}` | `{2.0}` |
| Erosion (fraction ε) | 2 | `{1-ε, ε}` | `{1.0, 1.0}` |
| Uniform | GL order (e.g. 8) | GL abscissae on (0,1) | `2 * GL_weights` |
| Power-law (exponent q) | 2 × GL order | 2-subinterval GL abscissae | `C(v′) v^q * GL_weights` |

**Minimal example (symmetric binary daughter):**

```cpp
// Quadrature: symmetric binary
std::vector<double> t_q  = { 0.5 };
std::vector<double> bw_q = { 2.0 };

// Upload to device
double *d_t_q, *d_bw_q;
cudaMalloc(&d_t_q,  sizeof(double));
cudaMalloc(&d_bw_q, sizeof(double));
cudaMemcpy(d_t_q,  t_q.data(),  sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_bw_q, bw_q.data(), sizeof(double), cudaMemcpyHostToDevice);

pbe_cuda::BreakageParams p;
p.n         = 256;
p.n_quad    = 1;
p.selection = pbe_cuda::BreakageSelection::Linear;
p.S0        = 1.0e-3;
p.v_ref     = x_host[128];   // reference volume

cudaMemset(d_rhs, 0, p.n * sizeof(double));
pbe_cuda::launch_breakage_rhs(d_N, d_x, d_t_q, d_bw_q, d_rhs, p);
```

---

## Combined aggregation and breakage

Both functions **accumulate** into `rhs`. Zero it once before calling either or both:

```cpp
// One RK4 stage: evaluate full RHS = aggregation + breakage
cudaMemset(d_rhs, 0, n * sizeof(double));

pbe_cuda::launch_aggregation_rhs(d_N, d_x, d_rhs, agg_params, stream);
pbe_cuda::launch_breakage_rhs(d_N, d_x, d_t_q, d_bw_q, d_rhs, br_params, stream);

cudaStreamSynchronize(stream);
// d_rhs now contains R_agg + R_br
```

---

## Error handling

All launch functions return `cudaError_t`. Recommended pattern:

```cpp
cudaError_t err = pbe_cuda::launch_aggregation_rhs(d_N, d_x, d_rhs, p);
if (err != cudaSuccess) {
    fprintf(stderr, "Aggregation kernel failed: %s\n", cudaGetErrorString(err));
    // handle error
}
// Synchronise to catch asynchronous errors:
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    fprintf(stderr, "Device error: %s\n", cudaGetErrorString(err));
}
```

The following argument checks are performed before kernel launch and return `cudaErrorInvalidValue` or `cudaErrorInvalidDevicePointer`:
- `n <= 0` or `n_quad <= 0`
- Any required pointer is `nullptr`
- Invalid kernel flag (internal dispatch error)

---

## Shared utilities (`examples/common/pbe_examples_utils.hpp`)

The examples ship a convenience header with grid construction, moment computation, RAII device array management, and analytical solutions. It is not installed with the library but is available for users to copy into their own projects.

Key items:

```cpp
// RAII device array — allocates on construction, frees on destruction
template<typename T> class DeviceArray;

// Grid construction
std::vector<double> make_geometric_grid(int n, double v_min, double r);
std::vector<double> make_geometric_grid_range(int n, double v_min, double v_max);

// Exponential IC: exact integral over each cell
std::vector<double> make_exponential_ic(const std::vector<double>& x,
                                         double N0, double vc);

// Moment computation
double compute_M0(const std::vector<double>& N);
double compute_M1(const std::vector<double>& N, const std::vector<double>& x);
double compute_M2(const std::vector<double>& N, const std::vector<double>& x);

// Analytical solutions (see docs/theory.md for formulas)
struct ConstantKernelAnalytical;   // Scott solution, exponential IC
struct SumKernelAnalytical;        // Golovin solution, exponential IC
struct ProductKernelAnalytical;    // Smoluchowski product, exponential IC
struct ZiffMcGradyAnalytical;      // Linear selection + uniform daughter
struct ConstantSymmetricBreakageAnalytical;  // Constant selection + symmetric binary
```

See `examples/common/pbe_examples_utils.hpp` for full documentation of each item.
