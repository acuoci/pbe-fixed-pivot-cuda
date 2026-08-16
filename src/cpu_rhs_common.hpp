// =============================================================================
// cpu_rhs_common.hpp  —  Shared serial fixed-pivot helpers
//
// Internal host-side utilities used by the CPU aggregation and breakage RHS
// implementations.  This header is not part of the installed public API.
// =============================================================================

#pragma once

#include "pbe_cuda/aggregation.cuh"
#include "pbe_cuda/breakage.cuh"

#include <algorithm>
#include <cmath>

namespace pbe_cuda {
namespace detail {

inline double eval_aggregation_kernel_cpu(AggregationKernel kernel,
                                          double xj,
                                          double xk,
                                          double beta0,
                                          double beta_br,
                                          double beta_sh)
{
    switch (kernel) {
        case AggregationKernel::Constant:
            return beta0;
        case AggregationKernel::Sum:
            return beta0 * (xj + xk);
        case AggregationKernel::Product:
            return beta0 * xj * xk;
        case AggregationKernel::Brownian: {
            const double xj3 = std::cbrt(xj);
            const double xk3 = std::cbrt(xk);
            return beta_br * (xj3 / xk3 + xk3 / xj3 + 2.0);
        }
        case AggregationKernel::Shear: {
            const double s = std::cbrt(xj) + std::cbrt(xk);
            return beta_sh * s * s * s;
        }
        case AggregationKernel::BrownianShear: {
            const double xj3 = std::cbrt(xj);
            const double xk3 = std::cbrt(xk);
            const double br  = beta_br * (xj3 / xk3 + xk3 / xj3 + 2.0);
            const double s   = xj3 + xk3;
            return br + beta_sh * s * s * s;
        }
    }
    return 0.0;
}

inline double eval_breakage_selection_cpu(BreakageSelection selection,
                                          double v,
                                          double S0,
                                          double v_ref,
                                          double alpha,
                                          double v_min)
{
    switch (selection) {
        case BreakageSelection::Constant:
            return S0;
        case BreakageSelection::Linear:
            return S0 * v / v_ref;
        case BreakageSelection::PowerLaw:
            return S0 * std::pow(v / v_ref, alpha);
        case BreakageSelection::Threshold:
            return (v > v_min) ? S0 : 0.0;
    }
    return 0.0;
}

inline int fixed_pivot_hi_index_cpu(const double* x,
                                    int n,
                                    double v,
                                    double log_x0,
                                    double inv_log_r)
{
    if (inv_log_r != 0.0) {
        const double pos = (std::log(v) - log_x0) * inv_log_r;
        const int hi = static_cast<int>(std::floor(pos)) + 1;
        return std::min(n - 1, std::max(1, hi));
    }

    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (x[mid] <= v) lo = mid;
        else             hi = mid;
    }
    return hi;
}

inline int fixed_pivot_lo_index_cpu(const double* x, int n, double v)
{
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (x[mid] <= v) lo = mid;
        else             hi = mid;
    }
    return lo;
}

inline void add_fixed_pivot_birth_cpu(double* rhs,
                                      const double* x,
                                      int n,
                                      double v,
                                      double amount,
                                      double log_x0,
                                      double inv_log_r)
{
    if (v >= x[n - 1]) {
        rhs[n - 1] += amount;
    } else if (v <= x[0]) {
        rhs[0] += amount;
    } else {
        const int hi = fixed_pivot_hi_index_cpu(x, n, v, log_x0, inv_log_r);
        const int lo = hi - 1;
        const double w_upper = (v - x[lo]) / (x[hi] - x[lo]);
        rhs[lo] += (1.0 - w_upper) * amount;
        rhs[hi] += w_upper * amount;
    }
}

inline void add_fixed_pivot_birth_cpu(double* rhs,
                                      const double* x,
                                      int n,
                                      double v,
                                      double amount)
{
    if (v <= x[0]) {
        rhs[0] += amount;
    } else if (v >= x[n - 1]) {
        rhs[n - 1] += amount;
    } else {
        const int lo = fixed_pivot_lo_index_cpu(x, n, v);
        const int hi = lo + 1;
        const double eta_u = (v - x[lo]) / (x[hi] - x[lo]);
        rhs[lo] += (1.0 - eta_u) * amount;
        rhs[hi] += eta_u * amount;
    }
}

} // namespace detail
} // namespace pbe_cuda
