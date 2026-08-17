// =============================================================================
// source.cuh  --  Public API for simple additive source RHS contributions
//
// This small contribution is an architectural foothold for future PBE processes
// with mathematical structures distinct from aggregation and breakage.
// =============================================================================

#pragma once

#include "pbe_cuda/cuda_compat.cuh"

namespace pbe_cuda {

struct ConstantSourceParams {
    int n = 0;
    int block_size = 256;
};

cudaError_t launch_constant_source_rhs(
    const double*              rates,
    double*                    rhs,
    const ConstantSourceParams& params,
    cudaStream_t               stream = 0);

cudaError_t launch_constant_source_rhs_cpu(
    const double*              rates,
    double*                    rhs,
    const ConstantSourceParams& params);

} // namespace pbe_cuda
