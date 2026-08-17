// =============================================================================
// source.cu  --  CUDA wrapper for additive source RHS contribution
// =============================================================================

#include "pbe_cuda/source.cuh"

namespace pbe_cuda {
namespace detail {

__global__ void constant_source_rhs_kernel(const double* rates,
                                           double* rhs,
                                           int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rhs[i] += rates[i];
}

} // namespace detail

cudaError_t launch_constant_source_rhs(
    const double*              rates,
    double*                    rhs,
    const ConstantSourceParams& p,
    cudaStream_t               stream)
{
    if (p.n <= 0) return cudaErrorInvalidValue;
    if (!rates || !rhs) return cudaErrorInvalidDevicePointer;

    const int block_size = (p.block_size > 0) ? p.block_size : 256;
    const int grid_size = (p.n + block_size - 1) / block_size;

    detail::constant_source_rhs_kernel<<<grid_size, block_size, 0, stream>>>(
        rates, rhs, p.n);

    return cudaGetLastError();
}

} // namespace pbe_cuda
