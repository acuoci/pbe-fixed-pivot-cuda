// =============================================================================
// fixed_pivot.cuh  --  Internal fixed-pivot lookup and birth allocation helpers
//
// Shared by CPU and CUDA RHS implementations. This header is intentionally kept
// small and allocation-free; backend code remains responsible for accumulation.
// =============================================================================

#pragma once

#include <math.h>

namespace pbe_cuda {
namespace detail {

#if defined(__CUDACC__)
#  define PBE_CUDA_HD __host__ __device__ __forceinline__
#else
#  define PBE_CUDA_HD inline
#endif

struct FixedPivotBirthAllocation {
    int lower = -1;
    int upper = -1;
    double upper_weight = 0.0;
};

template <typename T>
PBE_CUDA_HD T fixed_pivot_load(const T* data, int index)
{
#if defined(__CUDA_ARCH__)
    return __ldg(&data[index]);
#else
    return data[index];
#endif
}

PBE_CUDA_HD int fixed_pivot_right_bracket(const double* x,
                                          int n,
                                          double v,
                                          double log_x0,
                                          double inv_log_r)
{
    if (inv_log_r != 0.0) {
        const double pos = (log(v) - log_x0) * inv_log_r;
        int hi = static_cast<int>(floor(pos)) + 1;
        if (hi < 1) hi = 1;
        if (hi > n - 1) hi = n - 1;
        return hi;
    }

    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (fixed_pivot_load(x, mid) <= v) lo = mid;
        else                               hi = mid;
    }
    return hi;
}

PBE_CUDA_HD int fixed_pivot_left_bracket(const double* x, int n, double v)
{
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (fixed_pivot_load(x, mid) <= v) lo = mid;
        else                               hi = mid;
    }
    return lo;
}

PBE_CUDA_HD FixedPivotBirthAllocation fixed_pivot_birth_allocation(
    const double* x,
    int n,
    double v,
    double log_x0,
    double inv_log_r)
{
    FixedPivotBirthAllocation allocation;

    if (v <= fixed_pivot_load(x, 0)) {
        allocation.lower = 0;
        return allocation;
    }
    if (v >= fixed_pivot_load(x, n - 1)) {
        allocation.lower = n - 1;
        return allocation;
    }

    const int hi = fixed_pivot_right_bracket(x, n, v, log_x0, inv_log_r);
    const int lo = hi - 1;
    const double xlo = fixed_pivot_load(x, lo);
    const double xhi = fixed_pivot_load(x, hi);

    allocation.lower = lo;
    allocation.upper = hi;
    allocation.upper_weight = (v - xlo) / (xhi - xlo);
    return allocation;
}

PBE_CUDA_HD FixedPivotBirthAllocation fixed_pivot_birth_allocation(
    const double* x,
    int n,
    double v)
{
    FixedPivotBirthAllocation allocation;

    if (v <= fixed_pivot_load(x, 0)) {
        allocation.lower = 0;
        return allocation;
    }
    if (v >= fixed_pivot_load(x, n - 1)) {
        allocation.lower = n - 1;
        return allocation;
    }

    const int lo = fixed_pivot_left_bracket(x, n, v);
    const int hi = lo + 1;
    const double xlo = fixed_pivot_load(x, lo);
    const double xhi = fixed_pivot_load(x, hi);

    allocation.lower = lo;
    allocation.upper = hi;
    allocation.upper_weight = (v - xlo) / (xhi - xlo);
    return allocation;
}

#undef PBE_CUDA_HD

} // namespace detail
} // namespace pbe_cuda
