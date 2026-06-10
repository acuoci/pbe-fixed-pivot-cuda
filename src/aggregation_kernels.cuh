// =============================================================================
// aggregation_kernels.cuh
//
// Internal device-side code for the aggregation RHS kernel.
// This header is NOT part of the public API and is NOT installed.
// It is included only by aggregation.cu.
//
// Contents:
//   - eval_kernel_t<FLAG>   : templated aggregation kernel evaluators
//   - dispatch_kernel()     : runtime dispatcher (uniform branch, no divergence)
//   - bin_index()           : O(1) geometric / O(log n) generic bin lookup
//   - aggregation_rhs<FLAG> : tiled shared-memory CUDA kernel
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_60_atomic_functions.h>   // double atomicAdd — requires sm_60+
#include <math.h>

namespace pbe_cuda {
namespace detail  {

// ---------------------------------------------------------------------------
// Tile width for shared-memory accumulation.
// 512 doubles = 4 KB per block, well within the 48 KB shared memory limit.
// Users needing different occupancy can recompile with -DPBE_TILE=<N>.
// ---------------------------------------------------------------------------
#ifndef PBE_TILE
#  define PBE_TILE 512
#endif
static constexpr int TILE = PBE_TILE;

// ---------------------------------------------------------------------------
// Templated aggregation kernel evaluators.
//
// One specialisation per AggregationKernel enum value.  Since kernel_flag
// is uniform across the entire launch, the switch in dispatch_kernel() is a
// uniform branch: every thread takes the same path, giving zero warp
// divergence.  Each specialisation compiles to a straight-line, branch-free
// device function that the compiler inlines fully at the call site.
//
// FLAG mapping (must match AggregationKernel enum in aggregation.cuh):
//   0 = Constant
//   1 = Sum
//   2 = Product
//   3 = Brownian
//   4 = Shear
//   5 = BrownianShear
// ---------------------------------------------------------------------------
template<int FLAG>
__device__ double eval_kernel_t(double xj, double xk,
                                double b0, double b_br, double b_sh);

template<> __device__ inline double
eval_kernel_t<0>(double xj, double xk, double b0, double, double)
{ return b0; }

template<> __device__ inline double
eval_kernel_t<1>(double xj, double xk, double b0, double, double)
{ return b0 * (xj + xk); }

template<> __device__ inline double
eval_kernel_t<2>(double xj, double xk, double b0, double, double)
{ return b0 * xj * xk; }

template<> __device__ inline double
eval_kernel_t<3>(double xj, double xk, double b0, double b_br, double)
{
    double xj3 = cbrt(xj), xk3 = cbrt(xk);
    return b_br * (xj3 / xk3 + xk3 / xj3 + 2.0);
}

template<> __device__ inline double
eval_kernel_t<4>(double xj, double xk, double b0, double, double b_sh)
{
    double s = cbrt(xj) + cbrt(xk);
    return b_sh * s * s * s;
}

template<> __device__ inline double
eval_kernel_t<5>(double xj, double xk, double, double b_br, double b_sh)
{
    double xj3 = cbrt(xj), xk3 = cbrt(xk);
    double br  = b_br * (xj3 / xk3 + xk3 / xj3 + 2.0);
    double sh  = b_sh * (xj3 + xk3) * (xj3 + xk3) * (xj3 + xk3);
    return br + sh;
}

// ---------------------------------------------------------------------------
// Runtime dispatcher — uniform branch across all threads in a launch.
// ---------------------------------------------------------------------------
__device__ inline double dispatch_kernel(double xj, double xk,
                                         double b0, double b_br, double b_sh,
                                         int flag)
{
    switch (flag) {
        case 0: return eval_kernel_t<0>(xj, xk, b0, b_br, b_sh);
        case 1: return eval_kernel_t<1>(xj, xk, b0, b_br, b_sh);
        case 2: return eval_kernel_t<2>(xj, xk, b0, b_br, b_sh);
        case 3: return eval_kernel_t<3>(xj, xk, b0, b_br, b_sh);
        case 4: return eval_kernel_t<4>(xj, xk, b0, b_br, b_sh);
        case 5: return eval_kernel_t<5>(xj, xk, b0, b_br, b_sh);
        default: return 0.0;
    }
}

// ---------------------------------------------------------------------------
// Birth bin lookup — supports both geometric and generic grids.
//
//   inv_log_r != 0  →  geometric grid: O(1) formula-based lookup
//   inv_log_r == 0  →  generic grid:   O(log n) binary search
//
// Precondition: v_new must be strictly interior to (x[0], x[n-1]).
//               Boundary clips are resolved before this is called.
// ---------------------------------------------------------------------------
__device__ inline int bin_index(const double* x, int n, double v_new,
                                 double log_x0, double inv_log_r)
{
    if (inv_log_r != 0.0) {
        double pos = (log(v_new) - log_x0) * inv_log_r;
        int hi = static_cast<int>(floor(pos)) + 1;
        return min(n - 1, max(1, hi));
    } else {
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (x[mid] <= v_new) lo = mid;
            else                 hi = mid;
        }
        return hi;
    }
}

// ---------------------------------------------------------------------------
// Tiled aggregation RHS kernel — templated on kernel FLAG.
//
// Each thread handles one (j,k) pair from the upper triangle (j <= k).
// Shared memory tile of TILE doubles accumulates partial RHS sums,
// avoiding per-thread global atomics for the inner loop.
//
// See aggregation.cuh for full parameter documentation.
// ---------------------------------------------------------------------------
template<int FLAG>
__global__ void aggregation_rhs_kernel(
    const double* __restrict__ N,
    const double* __restrict__ x,
    double*       __restrict__ rhs,
    double        beta0,
    double        beta_br,
    double        beta_sh,
    int           n,
    double        log_x0,
    double        inv_log_r)
{
    extern __shared__ double shared_rhs[];

    // ---- Decode (j, k) from linearised upper-triangle index ----
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long M   = static_cast<long long>(n) * (n + 1) / 2;

    bool   active   = false;
    bool   diagonal = false;
    int    tj = -1, tk = -1;
    int    birth_lo = -1, birth_hi = -1;
    double rate = 0.0, w_upper = 0.0;

    if (idx < M) {
        int j = static_cast<int>(
            (2*n + 1 - sqrt(static_cast<double>((2*n+1)*(2*n+1)) - 8.0*idx)) / 2);
        if      (static_cast<long long>(j)   * (2*n - j + 1) / 2 >  idx) j--;
        else if (static_cast<long long>(j+1) * (2*n - j)     / 2 <= idx) j++;
        int k = static_cast<int>(idx - static_cast<long long>(j) * (2*n - j + 1) / 2) + j;

        if (j >= 0 && k < n && j <= k) {
            active   = true;
            diagonal = (j == k);
            tj = j; tk = k;

            double xj = __ldg(&x[j]);
            double xk = __ldg(&x[k]);
            double Nj = __ldg(&N[j]);
            double Nk = __ldg(&N[k]);

            double beta_jk = eval_kernel_t<FLAG>(xj, xk, beta0, beta_br, beta_sh);
            rate = beta_jk * Nj * Nk;
            if (diagonal) rate *= 0.5;

            double v_new = xj + xk;
            if (v_new >= __ldg(&x[n - 1])) {
                birth_lo = n - 1; birth_hi = -1;
            } else if (v_new <= __ldg(&x[0])) {
                birth_lo = 0;     birth_hi = -1;
            } else {
                int hi  = bin_index(x, n, v_new, log_x0, inv_log_r);
                int lo  = hi - 1;
                w_upper = (v_new - __ldg(&x[lo])) / (__ldg(&x[hi]) - __ldg(&x[lo]));
                birth_lo = lo; birth_hi = hi;
            }
        }
    }

    // ---- Tiled shared-memory accumulation ----
    for (int tile_start = 0; tile_start < n; tile_start += TILE) {
        int tile_end = min(tile_start + TILE, n);

        for (int i = threadIdx.x; i < TILE; i += blockDim.x)
            shared_rhs[i] = 0.0;
        __syncthreads();

        if (active) {
            if (diagonal) {
                if (tj >= tile_start && tj < tile_end)
                    atomicAdd(&shared_rhs[tj - tile_start], -2.0 * rate);
            } else {
                if (tj >= tile_start && tj < tile_end)
                    atomicAdd(&shared_rhs[tj - tile_start], -rate);
                if (tk >= tile_start && tk < tile_end)
                    atomicAdd(&shared_rhs[tk - tile_start], -rate);
            }

            if (birth_hi < 0) {
                if (birth_lo >= tile_start && birth_lo < tile_end)
                    atomicAdd(&shared_rhs[birth_lo - tile_start], rate);
            } else {
                if (birth_lo >= tile_start && birth_lo < tile_end)
                    atomicAdd(&shared_rhs[birth_lo - tile_start], (1.0 - w_upper) * rate);
                if (birth_hi >= tile_start && birth_hi < tile_end)
                    atomicAdd(&shared_rhs[birth_hi - tile_start], w_upper * rate);
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < tile_end - tile_start; i += blockDim.x)
            atomicAdd(&rhs[tile_start + i], shared_rhs[i]);
        __syncthreads();
    }
}

} // namespace detail
} // namespace pbe_cuda