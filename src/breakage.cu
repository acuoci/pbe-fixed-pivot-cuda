// =============================================================================
// breakage.cu  —  C++ wrapper for the breakage RHS CUDA kernels
//
// Translates BreakageParams into typed kernel launches, resolving the
// BreakageSelection enum to compile-time template parameters.
// Launches death and birth kernels sequentially on the same stream.
// =============================================================================

#include "pbe_cuda/breakage.cuh"
#include "breakage_kernels.cuh"    // internal, not installed

namespace pbe_cuda {

cudaError_t launch_breakage_rhs(
    const double*         N,
    const double*         x,
    const double*         t_q,
    const double*         bw_q,
    double*               rhs,
    const BreakageParams& p,
    cudaStream_t          stream)
{
    if (p.n <= 0 || p.n_quad <= 0)    return cudaErrorInvalidValue;
    if (!N || !x || !t_q || !bw_q || !rhs) return cudaErrorInvalidDevicePointer;

    const int flag = static_cast<int>(p.selection);

    // ---- Kernel 1: death term ----------------------------------------
    // One thread per bin, no shared memory.
    {
        const int grid = (p.n + p.block_size - 1) / p.block_size;

        switch (flag) {
#define LAUNCH_DEATH(F)                                           \
    detail::breakage_death_kernel<F>                              \
        <<<grid, p.block_size, 0, stream>>>(                      \
            N, x, rhs,                                            \
            p.S0, p.v_ref, p.alpha, p.v_min, p.n);               \
    break;

            case 0: LAUNCH_DEATH(0)
            case 1: LAUNCH_DEATH(1)
            case 2: LAUNCH_DEATH(2)
            case 3: LAUNCH_DEATH(3)
#undef LAUNCH_DEATH
            default: return cudaErrorInvalidValue;
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // ---- Kernel 2: birth term ----------------------------------------
    // One thread per (j, q) pair, tiled shared memory accumulation.
    {
        const long long M    = static_cast<long long>(p.n) * p.n_quad;
        const int grid       = static_cast<int>((M + p.block_size - 1) / p.block_size);
        const size_t smem    = detail::BTILE * sizeof(double);   // 4 KB

        switch (flag) {
#define LAUNCH_BIRTH(F)                                           \
    detail::breakage_birth_kernel<F>                              \
        <<<grid, p.block_size, smem, stream>>>(                   \
            N, x, t_q, bw_q, rhs,                                \
            p.S0, p.v_ref, p.alpha, p.v_min,                     \
            p.n, p.n_quad);                                       \
    break;

            case 0: LAUNCH_BIRTH(0)
            case 1: LAUNCH_BIRTH(1)
            case 2: LAUNCH_BIRTH(2)
            case 3: LAUNCH_BIRTH(3)
#undef LAUNCH_BIRTH
            default: return cudaErrorInvalidValue;
        }

        return cudaGetLastError();
    }
}

} // namespace pbe_cuda