// =============================================================================
// source_cpu.cpp  --  Serial CPU additive source RHS implementation
// =============================================================================

#include "pbe_cuda/source.cuh"

namespace pbe_cuda {

cudaError_t launch_constant_source_rhs_cpu(
    const double*              rates,
    double*                    rhs,
    const ConstantSourceParams& p)
{
    if (p.n <= 0) return cudaErrorInvalidValue;
    if (!rates || !rhs) return cudaErrorInvalidDevicePointer;

    for (int i = 0; i < p.n; ++i)
        rhs[i] += rates[i];

    return cudaSuccess;
}

#if !defined(PBE_ENABLE_CUDA)
cudaError_t launch_constant_source_rhs(
    const double*              rates,
    double*                    rhs,
    const ConstantSourceParams& params,
    cudaStream_t)
{
    return launch_constant_source_rhs_cpu(rates, rhs, params);
}
#endif

} // namespace pbe_cuda
