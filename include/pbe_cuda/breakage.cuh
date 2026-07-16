// =============================================================================
// breakage.cuh  —  Public API for breakage RHS computation
//
// Part of pbe_cuda: Fast CUDA implementation of aggregation and breakage
// terms in Population Balance Equations (PBE) discretised using the
// fixed pivot sectional method.
//
// Usage:
//   #include <pbe_cuda/breakage.cuh>   // or <pbe_cuda/pbe_cuda.cuh>
//
//   // Device pointers must be allocated and initialised by the caller.
//   // rhs must be pre-zeroed before calling (cudaMemset).
//   pbe_cuda::launch_breakage_rhs(N_d, x_d, t_q_d, bw_q_d, rhs_d, params, stream);
// =============================================================================

#pragma once

#include <cuda_runtime.h>

namespace pbe_cuda {

// ---------------------------------------------------------------------------
// BreakageSelection — selects the breakage selection function S(v).
//
// Constant   S(v) = S0
// Linear     S(v) = S0 * v / v_ref
// PowerLaw   S(v) = S0 * (v / v_ref)^alpha
// Threshold  S(v) = S0 if v > v_min, else 0
// ---------------------------------------------------------------------------
enum class BreakageSelection : int {
    Constant  = 0,
    Linear    = 1,
    PowerLaw  = 2,
    Threshold = 3
};

// ---------------------------------------------------------------------------
// BreakageParams — all scalar parameters for one breakage RHS evaluation.
//
// Grid:
//   n          Number of pivot bins (length of N, x, rhs arrays).
//
// Quadrature:
//   n_quad     Number of quadrature points per parent bin.
//              t_q and bw_q arrays must have length n_quad.
//              Supported daughter distributions and their n_quad values:
//                symmetric : n_quad = 1  (t_q = [0.5], bw_q = [2])
//                erosion   : n_quad = 2  (t_q = [1-eps, eps], bw_q = [1,1])
//                uniform   : n_quad = Gauss-Legendre order (user-chosen)
//                powerlaw  : n_quad = 2 * Gauss-Legendre order
//
// Selection function:
//   selection  BreakageSelection enum value.
//   S0         Selection prefactor (all models).
//   v_ref      Reference volume (Linear and PowerLaw).
//   alpha      Power-law exponent (PowerLaw only).
//   v_min      Threshold volume below which S = 0 (Threshold only).
//
// Launch tuning:
//   block_size CUDA threads per block for birth kernel (default 256).
//              Death kernel always uses block_size threads per block.
// ---------------------------------------------------------------------------
struct BreakageParams {
    // Grid
    int    n      = 0;
    int    n_quad = 0;

    // Selection function
    BreakageSelection selection = BreakageSelection::Constant;
    double S0     = 1.0;
    double v_ref  = 1.0;
    double alpha  = 1.0;
    double v_min  = 0.0;

    // Launch configuration
    int    block_size = 256;
};

// ---------------------------------------------------------------------------
// launch_breakage_rhs
//
// Computes the breakage contribution to the PBE right-hand side using the
// fixed pivot sectional method on the GPU. Launches two kernels internally:
//   1. breakage_death_kernel : rhs[j] -= S(x[j]) * N[j]   (no atomics)
//   2. breakage_birth_kernel : birth via quadrature + tiled shared memory
//
// All pointer arguments must be valid device pointers. The caller is
// responsible for allocation, initialisation, and deallocation.
//
// Parameters:
//   N      [in]  Device pointer to number distribution array, length n.
//   x      [in]  Device pointer to pivot volume array, length n.
//                Must be strictly positive and monotonically increasing.
//   t_q    [in]  Device pointer to quadrature abscissae, length n_quad.
//                Values in (0, 1): v_frag = x[j] * t_q[q].
//   bw_q   [in]  Device pointer to quadrature weights, length n_quad.
//                Includes the daughter distribution value:
//                  bw_q[q] = b(v_frag | x[j]) * dv_frag
//                (pre-multiplied on the host, j-independent).
//   rhs    [out] Device pointer to output RHS array, length n.
//                MUST be pre-zeroed by the caller before this call:
//                  cudaMemset(rhs, 0, n * sizeof(double));
//   params [in]  Breakage parameters (see BreakageParams).
//   stream [in]  CUDA stream (default: 0). Both kernels are launched on
//                the same stream and execute sequentially.
//
// Returns:
//   cudaSuccess, or a CUDA error code from either kernel launch.
//
// Example:
//   pbe_cuda::BreakageParams p;
//   p.n         = 100;
//   p.n_quad    = 16;
//   p.selection = pbe_cuda::BreakageSelection::PowerLaw;
//   p.S0        = 1.0;
//   p.v_ref     = x_host[50];
//   p.alpha     = 2.0;
//
//   cudaMemset(rhs_d, 0, p.n * sizeof(double));
//   pbe_cuda::launch_breakage_rhs(N_d, x_d, t_q_d, bw_q_d, rhs_d, p);
// ---------------------------------------------------------------------------
cudaError_t launch_breakage_rhs(
    const double*          N,
    const double*          x,
    const double*          t_q,
    const double*          bw_q,
    double*                rhs,
    const BreakageParams&  params,
    cudaStream_t           stream = 0);

} // namespace pbe_cuda