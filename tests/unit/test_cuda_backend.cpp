// =============================================================================
// tests/unit/test_cuda_backend.cpp
//
// CUDA resource smoke tests. Built only when ENABLE_CUDA=ON.
// =============================================================================

#include <pbe_cuda/backend.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

namespace {

bool has_cuda_device()
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

} // namespace

TEST(CudaBackend, StreamCanOwnOrWrapStream)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    auto owned = pbe_cuda::CudaStream::create();
    EXPECT_NE(owned.get(), nullptr);
    EXPECT_TRUE(owned.owns_stream());

    auto external = pbe_cuda::CudaStream::external(owned.get());
    EXPECT_EQ(external.get(), owned.get());
    EXPECT_FALSE(external.owns_stream());
}

TEST(CudaBackend, DeviceGridUploadsPivotsOnce)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    const auto grid = pbe_cuda::SectionalGrid::geometric(4, 1.0, 2.0);
    auto stream = pbe_cuda::CudaStream::create();
    const pbe_cuda::CudaDeviceGrid device_grid(grid, stream.get());

    std::vector<double> pivots(grid.size(), 0.0);
    cudaError_t err = cudaMemcpyAsync(
        pivots.data(), device_grid.data(), pivots.size() * sizeof(double),
        cudaMemcpyDeviceToHost, stream.get());
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

    EXPECT_EQ(device_grid.n(), grid.n());
    EXPECT_DOUBLE_EQ(device_grid.log_x0(), grid.log_x0());
    EXPECT_DOUBLE_EQ(device_grid.inv_log_r(), grid.inv_log_r());
    for (std::size_t i = 0; i < pivots.size(); ++i)
        EXPECT_DOUBLE_EQ(pivots[i], grid[i]);
}

TEST(CudaBackend, WorkspaceReusesScratchBuffer)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::CudaWorkspace workspace;
    workspace.ensure_scratch_size(8);
    auto* data = workspace.scratch().data();

    workspace.ensure_scratch_size(4);

    EXPECT_EQ(workspace.scratch().data(), data);
    EXPECT_EQ(workspace.scratch().size(), 8u);
}
