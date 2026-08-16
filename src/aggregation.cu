// =============================================================================
// aggregation.cu  —  C++ wrapper for the aggregation RHS CUDA kernel
//
// Translates the public AggregationParams struct into a typed kernel launch,
// resolving the AggregationKernel enum to a compile-time template parameter.
// The user never interacts with kernel launch syntax or shared memory sizing.
// =============================================================================

#include "pbe_cuda/aggregation.cuh"
#include "aggregation_kernels.cuh"   // internal, not installed

#include <stdexcept>

namespace pbe_cuda {

cudaError_t launch_aggregation_rhs(
    const double*            N,
    const double*            x,
    double*                  rhs,
    const AggregationParams& p,
    cudaStream_t             stream)
{
    if (p.n <= 0)      return cudaErrorInvalidValue;
    if (!N || !x || !rhs) return cudaErrorInvalidDevicePointer;

    // ---- Launch geometry ------------------------------------------------
    // M = total upper-triangle pairs (j <= k), one thread per pair.
    const long long M          = static_cast<long long>(p.n) * (p.n + 1) / 2;
    const int       block_size = (p.block_size > 0) ? p.block_size : 256;
    const int       grid_size  = static_cast<int>((M + block_size - 1) / block_size);
    const size_t    smem_bytes = detail::TILE * sizeof(double);  // 4 KB

    // ---- Dispatch to compile-time kernel specialisation -----------------
    // The switch maps the runtime enum value to a template parameter.
    // Each case launches a fully specialised, branch-free kernel.
    const int flag = static_cast<int>(p.kernel_type);

    switch (flag) {
#define LAUNCH(F)                                                       \
    detail::aggregation_rhs_kernel<F>                                   \
        <<<grid_size, block_size, smem_bytes, stream>>>(                \
            N, x, rhs,                                                  \
            p.beta0, p.beta_bc, p.beta_bfm, p.beta_sh,                  \
            p.n, p.log_x0, p.inv_log_r);                                \
    break;

        case 0: LAUNCH(0)
        case 1: LAUNCH(1)
        case 2: LAUNCH(2)
        case 3: LAUNCH(3)
        case 4: LAUNCH(4)
        case 5: LAUNCH(5)
        case 6: LAUNCH(6)
        case 7: LAUNCH(7)
#undef LAUNCH
        default: return cudaErrorInvalidValue;
    }

    // Return the status of the kernel launch (catches invalid config errors).
    // Note: this does NOT synchronise — errors from kernel execution itself
    // are only visible after a cudaStreamSynchronize() or cudaDeviceSynchronize().
    return cudaGetLastError();
}

} // namespace pbe_cuda
