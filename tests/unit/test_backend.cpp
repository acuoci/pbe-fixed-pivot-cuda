// =============================================================================
// tests/unit/test_backend.cpp
//
// Tests for backend tags and CPU reusable workspace resources.
// =============================================================================

#include <pbe_cuda/backend.hpp>
#include <pbe_cuda/pbe_cuda.cuh>

#include <gtest/gtest.h>

#include <type_traits>

TEST(Backend, ExposesBackendKinds)
{
    EXPECT_EQ(pbe_cuda::CpuBackend::kind, pbe_cuda::BackendKind::Cpu);
    EXPECT_EQ(pbe_cuda::CudaBackend::kind, pbe_cuda::BackendKind::Cuda);
}

TEST(Backend, CudaAvailabilityMatchesCompileDefinition)
{
#if defined(PBE_ENABLE_CUDA)
    EXPECT_TRUE(pbe_cuda::cuda_backend_available);
#else
    EXPECT_FALSE(pbe_cuda::cuda_backend_available);
#endif
}

TEST(CpuWorkspace, ReusesScratchStorage)
{
    pbe_cuda::CpuWorkspace workspace;
    EXPECT_EQ(workspace.scratch_size(), 0u);
    EXPECT_TRUE(workspace.scratch().empty());

    workspace.ensure_scratch_size(8);
    ASSERT_EQ(workspace.scratch_size(), 8u);

    auto scratch = workspace.scratch();
    ASSERT_EQ(scratch.size(), 8u);
    scratch[0] = 3.0;
    scratch[7] = 4.0;

    const double* data = scratch.data();
    workspace.ensure_scratch_size(4);

    EXPECT_EQ(workspace.scratch().data(), data);
    EXPECT_EQ(workspace.scratch_size(), 8u);
    EXPECT_DOUBLE_EQ(workspace.scratch()[0], 3.0);
    EXPECT_DOUBLE_EQ(workspace.scratch()[7], 4.0);
}

TEST(CpuWorkspace, GrowsScratchStorageWhenRequired)
{
    pbe_cuda::CpuWorkspace workspace(2);

    EXPECT_EQ(workspace.scratch_size(), 2u);
    workspace.ensure_scratch_size(5);

    EXPECT_EQ(workspace.scratch_size(), 5u);
    EXPECT_EQ(workspace.scratch().size(), 5u);
}

TEST(Backend, CpuWorkspaceIsMovable)
{
    static_assert(std::is_move_constructible<pbe_cuda::CpuWorkspace>::value,
                  "CpuWorkspace should be movable");
    static_assert(std::is_move_assignable<pbe_cuda::CpuWorkspace>::value,
                  "CpuWorkspace should be move assignable");
}
