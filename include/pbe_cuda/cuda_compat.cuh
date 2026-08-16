// =============================================================================
// cuda_compat.cuh  —  Minimal CUDA type compatibility for CPU-only builds
//
// When PBE_ENABLE_CUDA is enabled this header includes cuda_runtime.h.  When
// CUDA support is disabled it provides the small subset of CUDA runtime types
// and constants used by the public pbe_cuda API.
// =============================================================================

#pragma once

#if defined(PBE_ENABLE_CUDA)
#  include <cuda_runtime.h>
#else
using cudaStream_t = void*;
using cudaError_t  = int;

static constexpr cudaError_t cudaSuccess                   = 0;
static constexpr cudaError_t cudaErrorInvalidValue         = 1;
static constexpr cudaError_t cudaErrorInvalidDevicePointer = 17;

inline const char* cudaGetErrorString(cudaError_t error)
{
    switch (error) {
        case cudaSuccess:                   return "cudaSuccess";
        case cudaErrorInvalidValue:         return "cudaErrorInvalidValue";
        case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
        default:                            return "cudaErrorUnknown";
    }
}
#endif
