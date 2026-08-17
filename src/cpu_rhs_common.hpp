// =============================================================================
// cpu_rhs_common.hpp  —  Shared serial fixed-pivot helpers
//
// Internal host-side utilities used by the CPU aggregation and breakage RHS
// implementations.  This header is not part of the installed public API.
// =============================================================================

#pragma once

#include "pbe_cuda/aggregation.cuh"
#include "pbe_cuda/breakage.cuh"
#include "pbe_cuda/detail/fixed_pivot.cuh"

#include <cmath>

namespace pbe_cuda {
namespace detail {

inline double eval_aggregation_kernel_cpu(AggregationKernel kernel,
                                          double xj,
                                          double xk,
                                          double beta0,
                                          double beta_bc,
                                          double beta_bfm,
                                          double beta_sh)
{
    switch (kernel) {
        case AggregationKernel::Constant:
            return beta0;
        case AggregationKernel::Sum:
            return beta0 * (xj + xk);
        case AggregationKernel::Product:
            return beta0 * xj * xk;
        case AggregationKernel::BrownianContinuum: {
            const double xj3 = std::cbrt(xj);
            const double xk3 = std::cbrt(xk);
            return beta_bc * (xj3 / xk3 + xk3 / xj3 + 2.0);
        }
        case AggregationKernel::BrownianFreeMolecular: {
            const double xj3 = std::cbrt(xj);
            const double xk3 = std::cbrt(xk);
            const double s = xj3 + xk3;
            return beta_bfm * s * s * std::sqrt(1.0 / xj + 1.0 / xk);
        }
        case AggregationKernel::Shear: {
            const double s = std::cbrt(xj) + std::cbrt(xk);
            return beta_sh * s * s * s;
        }
        case AggregationKernel::BrownianContinuumShear: {
            const double xj3 = std::cbrt(xj);
            const double xk3 = std::cbrt(xk);
            const double br  = beta_bc * (xj3 / xk3 + xk3 / xj3 + 2.0);
            const double s   = xj3 + xk3;
            return br + beta_sh * s * s * s;
        }
        case AggregationKernel::BrownianFreeMolecularShear: {
            const double xj3 = std::cbrt(xj);
            const double xk3 = std::cbrt(xk);
            const double s = xj3 + xk3;
            const double bfm = beta_bfm * s * s * std::sqrt(1.0 / xj + 1.0 / xk);
            return bfm + beta_sh * s * s * s;
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

inline void add_fixed_pivot_birth_cpu(double* rhs,
                                      const double* x,
                                      int n,
                                      double v,
                                      double amount,
                                      double log_x0,
                                      double inv_log_r)
{
    const FixedPivotBirthAllocation allocation =
        fixed_pivot_birth_allocation(x, n, v, log_x0, inv_log_r);

    if (allocation.upper < 0) {
        rhs[allocation.lower] += amount;
    } else {
        rhs[allocation.lower] += (1.0 - allocation.upper_weight) * amount;
        rhs[allocation.upper] += allocation.upper_weight * amount;
    }
}

inline void add_fixed_pivot_birth_cpu(double* rhs,
                                      const double* x,
                                      int n,
                                      double v,
                                      double amount)
{
    const FixedPivotBirthAllocation allocation =
        fixed_pivot_birth_allocation(x, n, v);

    if (allocation.upper < 0) {
        rhs[allocation.lower] += amount;
    } else {
        rhs[allocation.lower] += (1.0 - allocation.upper_weight) * amount;
        rhs[allocation.upper] += allocation.upper_weight * amount;
    }
}

} // namespace detail
} // namespace pbe_cuda
