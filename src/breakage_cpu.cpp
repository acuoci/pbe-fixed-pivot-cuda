// =============================================================================
// breakage_cpu.cpp  —  Serial CPU breakage RHS implementation
// =============================================================================

#include "pbe_cuda/breakage.cuh"
#include "cpu_rhs_common.hpp"

#include <limits>

namespace pbe_cuda {

cudaError_t launch_breakage_rhs_cpu(
    const double*         N,
    const double*         x,
    const double*         t_q,
    const double*         bw_q,
    double*               rhs,
    const BreakageParams& p)
{
    if (p.n <= 0 || p.n_quad <= 0) return cudaErrorInvalidValue;
    if (!N || !x || !t_q || !bw_q || !rhs) return cudaErrorInvalidDevicePointer;

    switch (p.selection) {
        case BreakageSelection::Constant:
        case BreakageSelection::Linear:
        case BreakageSelection::PowerLaw:
        case BreakageSelection::Threshold:
            break;
        default:
            return cudaErrorInvalidValue;
    }

    for (int j = 0; j < p.n; ++j) {
        const double xj = x[j];
        const double Nj = N[j];
        const double Sj = detail::eval_breakage_selection_cpu(
            p.selection, xj, p.S0, p.v_ref, p.alpha, p.v_min);

        if (Sj != 0.0 && Nj > std::numeric_limits<double>::min())
            rhs[j] -= Sj * Nj;
    }

    for (int j = 0; j < p.n; ++j) {
        const double xj = x[j];
        const double Nj = N[j];
        const double Sj = detail::eval_breakage_selection_cpu(
            p.selection, xj, p.S0, p.v_ref, p.alpha, p.v_min);

        if (Sj == 0.0 || Nj <= std::numeric_limits<double>::min())
            continue;

        for (int q = 0; q < p.n_quad; ++q) {
            const double contrib = Sj * Nj * bw_q[q];
            const double v_frag = xj * t_q[q];
            detail::add_fixed_pivot_birth_cpu(rhs, x, p.n, v_frag, contrib);
        }
    }

    return cudaSuccess;
}

#if !defined(PBE_ENABLE_CUDA)
cudaError_t launch_breakage_rhs(
    const double*         N,
    const double*         x,
    const double*         t_q,
    const double*         bw_q,
    double*               rhs,
    const BreakageParams& params,
    cudaStream_t)
{
    return launch_breakage_rhs_cpu(N, x, t_q, bw_q, rhs, params);
}
#endif

} // namespace pbe_cuda
