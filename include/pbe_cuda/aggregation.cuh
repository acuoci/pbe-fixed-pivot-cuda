// =============================================================================
// aggregation.cuh  —  Public API for aggregation RHS computation
//
// Part of pbe_cuda: Fast CUDA implementation of aggregation and breakage
// terms in Population Balance Equations (PBE) discretised using the
// fixed pivot sectional method.
//
// Usage:
//   #include <pbe_cuda/aggregation.cuh>
//
//   // Device pointers must already be allocated and initialised.
//   // rhs must be pre-zeroed before calling.
//   pbe_cuda::launch_aggregation_rhs(N_d, x_d, rhs_d, params, stream);
// =============================================================================

#pragma once

#include <cuda_runtime.h>

namespace pbe_cuda {

// ---------------------------------------------------------------------------
// AggregationKernel — selects the aggregation frequency function β(xⱼ, xₖ).
//
// Constant      β = β₀
// Sum           β = β₀ (xⱼ + xₖ)
// Product       β = β₀  xⱼ xₖ
// Brownian      β = β_br (xⱼ^(1/3)/xₖ^(1/3) + xₖ^(1/3)/xⱼ^(1/3) + 2)
// Shear         β = β_sh (xⱼ^(1/3) + xₖ^(1/3))³
// BrownianShear β = Brownian + Shear   (β_br and β_sh both active)
// ---------------------------------------------------------------------------
enum class AggregationKernel : int {
    Constant      = 0,
    Sum           = 1,
    Product       = 2,
    Brownian      = 3,
    Shear         = 4,
    BrownianShear = 5
};

// ---------------------------------------------------------------------------
// AggregationParams — all scalar parameters for one RHS evaluation.
//
// Grid geometry:
//   n          Number of pivot bins (length of N, x, rhs arrays).
//   log_x0     log(x[0]) — pre-computed once per grid, passed by caller
//              to avoid redundant device-side log() calls.
//   inv_log_r  1 / log(x[1]/x[0]) for a geometric grid; pass 0.0 for a
//              generic (non-geometric) grid to fall back to binary search.
//
// Kernel coefficients:
//   beta0      Prefactor for Constant / Sum / Product kernels.
//   beta_br    Brownian prefactor (used by Brownian and BrownianShear).
//   beta_sh    Shear prefactor    (used by Shear    and BrownianShear).
//
// Launch tuning:
//   block_size CUDA threads per block (default 256; must be a power of 2).
// ---------------------------------------------------------------------------
struct AggregationParams {
    // Grid
    int    n          = 0;
    double log_x0     = 0.0;
    double inv_log_r  = 0.0;

    // Kernel type and coefficients
    AggregationKernel kernel_type = AggregationKernel::Constant;
    double beta0      = 1.0;
    double beta_br    = 0.0;
    double beta_sh    = 0.0;

    // Launch configuration
    int    block_size = 256;
};

// ---------------------------------------------------------------------------
// launch_aggregation_rhs
//
// Computes the aggregation contribution to the PBE right-hand side using
// the fixed pivot sectional method on the GPU.
//
// All pointer arguments must be valid device pointers. The caller is
// responsible for allocation, initialisation, and deallocation.
//
// Parameters:
//   N      [in]  Device pointer to number distribution array, length n.
//   x      [in]  Device pointer to pivot volume array, length n.
//                Must be strictly positive and monotonically increasing.
//   rhs    [out] Device pointer to output RHS array, length n.
//                MUST be pre-zeroed by the caller before this call, e.g.:
//                  cudaMemset(rhs, 0, n * sizeof(double));
//                This allows safe accumulation from multiple source terms.
//   params [in]  Aggregation parameters (see AggregationParams).
//   stream [in]  CUDA stream for asynchronous execution (default: 0).
//
// Returns:
//   cudaSuccess on success, or a CUDA error code on failure.
//   Errors can arise from invalid launch parameters or device-side faults.
//
// Thread safety:
//   Multiple calls on different streams with non-overlapping rhs arrays
//   are safe.  Concurrent calls sharing the same rhs array produce
//   undefined behaviour.
//
// Example:
//   pbe_cuda::AggregationParams p;
//   p.n           = 100;
//   p.log_x0      = std::log(x_host[0]);
//   p.inv_log_r   = 1.0 / std::log(x_host[1] / x_host[0]);
//   p.kernel_type = pbe_cuda::AggregationKernel::Brownian;
//   p.beta_br     = 6.73e-18;  // SI units
//
//   cudaMemset(rhs_d, 0, p.n * sizeof(double));
//   pbe_cuda::launch_aggregation_rhs(N_d, x_d, rhs_d, p);
// ---------------------------------------------------------------------------
cudaError_t launch_aggregation_rhs(
    const double*            N,
    const double*            x,
    double*                  rhs,
    const AggregationParams& params,
    cudaStream_t             stream = 0);

} // namespace pbe_cuda