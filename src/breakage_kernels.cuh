// =============================================================================
// breakage_kernels.cuh
//
// Internal device-side code for the breakage RHS kernels.
// This header is NOT part of the public API and is NOT installed.
// It is included only by breakage.cu.
//
// Contents:
//   - eval_selection_t<FLAG>  : templated selection function evaluators
//   - dispatch_selection()    : runtime dispatcher (uniform branch)
//   - fixed-pivot helpers     : O(log n) birth lookup and interpolation
//   - breakage_death_kernel   : death term (one thread per bin, no atomics)
//   - breakage_birth_kernel   : birth term (tiled shared-memory accumulation)
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <math.h>

#include "pbe_cuda/detail/fixed_pivot.cuh"

namespace pbe_cuda {
namespace detail  {

// ---------------------------------------------------------------------------
// Tile width — shared across aggregation and breakage kernels.
// Defined here independently so breakage_kernels.cuh has no dependency on
// aggregation_kernels.cuh. Both must use the same value; if you change one,
// change the other. 512 doubles = 4 KB per block.
// ---------------------------------------------------------------------------
#ifndef PBE_TILE
#  define PBE_TILE 512
#endif
static constexpr int BTILE = PBE_TILE;

// ---------------------------------------------------------------------------
// Templated selection function evaluators.
//
// FLAG mapping (must match BreakageSelection enum in breakage.cuh):
//   0 = Constant   S(v) = S0
//   1 = Linear     S(v) = S0 * v / v_ref
//   2 = PowerLaw   S(v) = S0 * (v/v_ref)^alpha
//   3 = Threshold  S(v) = S0 if v > v_min else 0
//
// alpha and v_min are carried in all specialisations for a uniform call
// signature; unused arguments are eliminated by the compiler.
// ---------------------------------------------------------------------------
template<int FLAG>
__device__ __forceinline__ double eval_selection_t(
    double v, double S0, double v_ref, double alpha, double v_min);

template<> __device__ __forceinline__ double
eval_selection_t<0>(double v, double S0, double, double, double)
{ return S0; }

template<> __device__ __forceinline__ double
eval_selection_t<1>(double v, double S0, double v_ref, double, double)
{ return S0 * v / v_ref; }

template<> __device__ __forceinline__ double
eval_selection_t<2>(double v, double S0, double v_ref, double alpha, double)
{ return S0 * pow(v / v_ref, alpha); }

template<> __device__ __forceinline__ double
eval_selection_t<3>(double v, double S0, double, double, double v_min)
{ return (v > v_min) ? S0 : 0.0; }

// ---------------------------------------------------------------------------
// Runtime dispatcher — uniform branch across all threads in a launch.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double dispatch_selection(
    double v, int flag,
    double S0, double v_ref, double alpha, double v_min)
{
    switch (flag) {
        case 0: return eval_selection_t<0>(v, S0, v_ref, alpha, v_min);
        case 1: return eval_selection_t<1>(v, S0, v_ref, alpha, v_min);
        case 2: return eval_selection_t<2>(v, S0, v_ref, alpha, v_min);
        case 3: return eval_selection_t<3>(v, S0, v_ref, alpha, v_min);
        default: return 0.0;
    }
}

// ---------------------------------------------------------------------------
// KERNEL 1: breakage_death_kernel
//
// Death contribution to the PBE RHS: rhs[j] -= S(x[j]) * N[j]
//
// One thread per bin — no atomics needed (bins are independent).
// Lightweight kernel, launched with n threads total.
// ---------------------------------------------------------------------------
template<int FLAG>
__global__ void breakage_death_kernel(
    const double* __restrict__ N,
    const double* __restrict__ x,
    double*       __restrict__ rhs,
    double S0, double v_ref, double alpha, double v_min,
    int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    double xj = __ldg(&x[j]);
    double Nj = __ldg(&N[j]);
    double Sj = eval_selection_t<FLAG>(xj, S0, v_ref, alpha, v_min);

    // Guard: skip zero selection or effectively zero population
    if (Sj != 0.0 && Nj > 2.2e-308)
        rhs[j] -= Sj * Nj;
}

// ---------------------------------------------------------------------------
// KERNEL 2: breakage_birth_kernel
//
// Birth contribution to the PBE RHS using fixed-pivot redistribution and
// Gauss-Legendre quadrature over the daughter size distribution.
//
// Thread layout: one thread per (j, q) pair
//   idx = j * n_quad + q
//   j   = idx / n_quad   (parent bin index)
//   q   = idx % n_quad   (quadrature point index)
//
// t_q[q]  : quadrature abscissa in (0,1), so v_frag = x[j] * t_q[q]
// bw_q[q] : quadrature weight * daughter distribution value at t_q[q]
//            (pre-multiplied on the host, j-independent for all supported
//             daughter models on log-spaced grids)
//
// Tiled shared-memory accumulation (same strategy as aggregation kernel):
//   Each output bin can receive O(n) birth contributions (one from each
//   parent), making naive global atomicAdd a bottleneck. The n output
//   bins are swept in tiles of BTILE bins; each tile accumulates into a
//   shared buffer before flushing to global memory, reducing global
//   atomic traffic from O(n * n_quad) to O(BTILE) per block per tile.
//
// Shared memory: BTILE * 8 = 4 KB per block (independent of n).
// ---------------------------------------------------------------------------
template<int FLAG>
__global__ void breakage_birth_kernel(
    const double* __restrict__ N,
    const double* __restrict__ x,
    const double* __restrict__ t_q,
    const double* __restrict__ bw_q,
    double*                    rhs,       // NOT __restrict__: written via atomicAdd
    double S0, double v_ref, double alpha, double v_min,
    int n, int n_quad)
{
    extern __shared__ double shared_rhs[];   // [BTILE] doubles

    // ------------------------------------------------------------------
    // Pre-computation: decode (j, q), evaluate selection, find birth bins.
    // All results stored in registers and reused across tile passes.
    // ------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M   = n * n_quad;

    bool   active   = false;
    int    birth_lo = -1;
    int    birth_hi = -1;    // -1 = boundary clip (only birth_lo active)
    double contrib  = 0.0;
    double eta_u    = 0.0;   // interpolation weight for birth_hi

    if (idx < M) {
        int j = idx / n_quad;
        int q = idx % n_quad;

        double xj  = __ldg(&x[j]);
        double Nj  = __ldg(&N[j]);
        double tq  = __ldg(&t_q[q]);
        double bwq = __ldg(&bw_q[q]);
        double Sj  = eval_selection_t<FLAG>(xj, S0, v_ref, alpha, v_min);

        if (Sj != 0.0 && Nj > 2.2e-308) {
            active  = true;
            contrib = Sj * Nj * bwq;

            double v_frag = xj * tq;
            FixedPivotBirthAllocation birth =
                fixed_pivot_birth_allocation(x, n, v_frag);
            birth_lo = birth.lower;
            birth_hi = birth.upper;
            eta_u = birth.upper_weight;
        }
    }

    // ------------------------------------------------------------------
    // Tile loop: sweep output bins in chunks of BTILE.
    // ------------------------------------------------------------------
    for (int tile_start = 0; tile_start < n; tile_start += BTILE) {
        int tile_end = min(tile_start + BTILE, n);

        // Step 1: zero shared buffer cooperatively
        for (int i = threadIdx.x; i < BTILE; i += blockDim.x)
            shared_rhs[i] = 0.0;
        __syncthreads();

        // Step 2: accumulate into shared memory
        if (active) {
            if (birth_hi < 0) {
                if (birth_lo >= tile_start && birth_lo < tile_end)
                    atomicAdd(&shared_rhs[birth_lo - tile_start], contrib);
            } else {
                if (birth_lo >= tile_start && birth_lo < tile_end)
                    atomicAdd(&shared_rhs[birth_lo - tile_start],
                              (1.0 - eta_u) * contrib);
                if (birth_hi >= tile_start && birth_hi < tile_end)
                    atomicAdd(&shared_rhs[birth_hi - tile_start],
                              eta_u * contrib);
            }
        }
        __syncthreads();

        // Step 3: flush shared tile to global rhs[]
        for (int i = threadIdx.x; i < tile_end - tile_start; i += blockDim.x)
            atomicAdd(&rhs[tile_start + i], shared_rhs[i]);
        __syncthreads();
    }
}

} // namespace detail
} // namespace pbe_cuda
