// =============================================================================
// aggregation_cpu.cpp  —  Serial CPU aggregation RHS implementation
// =============================================================================

#include "pbe_cuda/aggregation.cuh"
#include "cpu_rhs_common.hpp"

#include <limits>

namespace pbe_cuda {

cudaError_t launch_aggregation_rhs_cpu(
    const double*            N,
    const double*            x,
    double*                  rhs,
    const AggregationParams& p)
{
    if (p.n <= 0) return cudaErrorInvalidValue;
    if (!N || !x || !rhs) return cudaErrorInvalidDevicePointer;

    switch (p.kernel_type) {
        case AggregationKernel::Constant:
        case AggregationKernel::Sum:
        case AggregationKernel::Product:
        case AggregationKernel::BrownianContinuum:
        case AggregationKernel::BrownianFreeMolecular:
        case AggregationKernel::Shear:
        case AggregationKernel::BrownianContinuumShear:
        case AggregationKernel::BrownianFreeMolecularShear:
            break;
        default:
            return cudaErrorInvalidValue;
    }

    for (int j = 0; j < p.n; ++j) {
        const double xj = x[j];
        const double Nj = N[j];

        for (int k = j; k < p.n; ++k) {
            const double xk = x[k];
            const double Nk = N[k];
            const double beta_jk = detail::eval_aggregation_kernel_cpu(
                p.kernel_type, xj, xk, p.beta0, p.beta_bc, p.beta_bfm, p.beta_sh);

            double rate = beta_jk * Nj * Nk;
            if (j == k) rate *= 0.5;

            if (j == k) {
                rhs[j] -= 2.0 * rate;
            } else {
                rhs[j] -= rate;
                rhs[k] -= rate;
            }

            detail::add_fixed_pivot_birth_cpu(
                rhs, x, p.n, xj + xk, rate, p.log_x0, p.inv_log_r);
        }
    }

    return cudaSuccess;
}

#if !defined(PBE_ENABLE_CUDA)
cudaError_t launch_aggregation_rhs(
    const double*            N,
    const double*            x,
    double*                  rhs,
    const AggregationParams& params,
    cudaStream_t)
{
    return launch_aggregation_rhs_cpu(N, x, rhs, params);
}
#endif

} // namespace pbe_cuda
