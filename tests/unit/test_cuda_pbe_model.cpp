// =============================================================================
// tests/unit/test_cuda_pbe_model.cpp
//
// Tests for the high-level CUDA RHS model. Built only when ENABLE_CUDA=ON.
// =============================================================================

#include <pbe_cuda/cpu_pbe_model.hpp>
#include <pbe_cuda/cuda_pbe_model.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

bool has_cuda_device()
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

pbe_cuda::SectionalGrid make_grid()
{
    return pbe_cuda::SectionalGrid::geometric(5, 1.0, 2.0);
}

pbe_cuda::ConstRealView view(const std::vector<double>& values)
{
    return pbe_cuda::ConstRealView(values.data(), values.size());
}

pbe_cuda::RealView view(std::vector<double>& values)
{
    return pbe_cuda::RealView(values.data(), values.size());
}

pbe_cuda::ConstDeviceRealView device_view(
    const pbe_cuda::CudaDeviceBuffer<double>& values)
{
    return pbe_cuda::ConstDeviceRealView(values.data(), values.size());
}

pbe_cuda::DeviceRealView device_view(pbe_cuda::CudaDeviceBuffer<double>& values)
{
    return pbe_cuda::DeviceRealView(values.data(), values.size());
}

void upload(pbe_cuda::CudaDeviceBuffer<double>& device,
            const std::vector<double>& host,
            cudaStream_t stream)
{
    device.resize(host.size());
    device.upload(view(host), stream);
}

std::vector<double> download(const pbe_cuda::CudaDeviceBuffer<double>& device,
                             cudaStream_t stream)
{
    std::vector<double> host(device.size(), 0.0);
    device.download(view(host), stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    return host;
}

void expect_vectors_near(const std::vector<double>& actual,
                         const std::vector<double>& expected,
                         double tol)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tol);
}

} // namespace

TEST(CudaPBEModel, AggregationOnlyMatchesLowLevelLaunchAndCpuReference)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model = pbe_cuda::AggregationModel::constant(2.0);

    pbe_cuda::CudaWorkspace workspace;
    const pbe_cuda::CudaPBEModel model(config, 256, workspace.stream());

    const std::vector<double> N = {3.0, 1.0, 0.5, 0.0, 0.0};
    pbe_cuda::CudaDeviceBuffer<double> d_N;
    pbe_cuda::CudaDeviceBuffer<double> d_rhs_model(N.size());
    pbe_cuda::CudaDeviceBuffer<double> d_rhs_manual(N.size());
    upload(d_N, N, workspace.stream());

    const auto params = config.aggregation_model->to_params(*config.grid);
    d_rhs_manual.zero(workspace.stream());
    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs(
                  d_N.data(), model.device_grid().data(),
                  d_rhs_manual.data(), params, workspace.stream()),
              cudaSuccess);

    ASSERT_EQ(model.compute_rhs(device_view(d_N),
                                device_view(d_rhs_model),
                                workspace),
              cudaSuccess);

    const auto cuda_model = download(d_rhs_model, workspace.stream());
    const auto cuda_manual = download(d_rhs_manual, workspace.stream());

    pbe_cuda::CpuPBEModel cpu_model(config);
    pbe_cuda::CpuWorkspace cpu_workspace;
    std::vector<double> cpu_rhs(N.size(), 0.0);
    ASSERT_EQ(cpu_model.compute_rhs(view(N), view(cpu_rhs), cpu_workspace),
              cudaSuccess);

    expect_vectors_near(cuda_model, cuda_manual, 0.0);
    expect_vectors_near(cuda_model, cpu_rhs, 1.0e-12);
}

TEST(CudaPBEModel, BreakageOnlyMatchesLowLevelLaunchAndCpuReference)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.breakage_model = pbe_cuda::BreakageModel::linear_symmetric(1.5, 2.0);

    pbe_cuda::CudaWorkspace workspace;
    const pbe_cuda::CudaPBEModel model(config, 256, workspace.stream());

    const std::vector<double> N = {0.0, 2.0, 1.0, 0.5, 0.0};
    pbe_cuda::CudaDeviceBuffer<double> d_N;
    pbe_cuda::CudaDeviceBuffer<double> d_rhs_model(N.size());
    pbe_cuda::CudaDeviceBuffer<double> d_rhs_manual(N.size());
    pbe_cuda::CudaDeviceBuffer<double> d_t_q;
    pbe_cuda::CudaDeviceBuffer<double> d_bw_q;
    upload(d_N, N, workspace.stream());

    const auto params = config.breakage_model->to_params(*config.grid);
    const auto& q = config.breakage_model->quadrature();
    d_t_q.resize(q.t_q.size());
    d_bw_q.resize(q.bw_q.size());
    d_t_q.upload(q.t_view(), workspace.stream());
    d_bw_q.upload(q.bw_view(), workspace.stream());

    d_rhs_manual.zero(workspace.stream());
    ASSERT_EQ(pbe_cuda::launch_breakage_rhs(
                  d_N.data(), model.device_grid().data(), d_t_q.data(),
                  d_bw_q.data(), d_rhs_manual.data(), params,
                  workspace.stream()),
              cudaSuccess);

    ASSERT_EQ(model.compute_rhs(device_view(d_N),
                                device_view(d_rhs_model),
                                workspace),
              cudaSuccess);

    const auto cuda_model = download(d_rhs_model, workspace.stream());
    const auto cuda_manual = download(d_rhs_manual, workspace.stream());

    pbe_cuda::CpuPBEModel cpu_model(config);
    pbe_cuda::CpuWorkspace cpu_workspace;
    std::vector<double> cpu_rhs(N.size(), 0.0);
    ASSERT_EQ(cpu_model.compute_rhs(view(N), view(cpu_rhs), cpu_workspace),
              cudaSuccess);

    expect_vectors_near(cuda_model, cuda_manual, 0.0);
    expect_vectors_near(cuda_model, cpu_rhs, 1.0e-12);
}

TEST(CudaPBEModel, CombinedRhsMatchesCpuReference)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model = pbe_cuda::AggregationModel::sum(0.25);
    config.breakage_model = pbe_cuda::BreakageModel::constant_symmetric(0.5);

    pbe_cuda::CudaWorkspace workspace;
    const pbe_cuda::CudaPBEModel model(config, 256, workspace.stream());

    const std::vector<double> N = {1.0, 2.0, 0.0, 1.0, 0.0};
    pbe_cuda::CudaDeviceBuffer<double> d_N;
    pbe_cuda::CudaDeviceBuffer<double> d_rhs(N.size());
    upload(d_N, N, workspace.stream());

    ASSERT_EQ(model.compute_rhs(device_view(d_N), device_view(d_rhs),
                                workspace),
              cudaSuccess);
    const auto cuda_rhs = download(d_rhs, workspace.stream());

    pbe_cuda::CpuPBEModel cpu_model(config);
    pbe_cuda::CpuWorkspace cpu_workspace;
    std::vector<double> cpu_rhs(N.size(), 0.0);
    ASSERT_EQ(cpu_model.compute_rhs(view(N), view(cpu_rhs), cpu_workspace),
              cudaSuccess);

    expect_vectors_near(cuda_rhs, cpu_rhs, 1.0e-12);
}

TEST(CudaPBEModel, RepeatedCallsWithChangingContextReuseWorkspace)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model =
        pbe_cuda::AggregationModel::brownian_free_molecular(1.0e-4);

    pbe_cuda::CudaWorkspace workspace;
    const pbe_cuda::CudaPBEModel model(config, 256, workspace.stream());

    const std::vector<double> N = {1.0, 0.5, 0.25, 0.0, 0.0};
    pbe_cuda::CudaDeviceBuffer<double> d_N;
    pbe_cuda::CudaDeviceBuffer<double> d_rhs_a(N.size());
    pbe_cuda::CudaDeviceBuffer<double> d_rhs_b(N.size());
    upload(d_N, N, workspace.stream());

    pbe_cuda::EvaluationContext ctx_a;
    ctx_a.temperature = 300.0;
    ctx_a.viscosity = 1.0e-3;

    pbe_cuda::EvaluationContext ctx_b;
    ctx_b.temperature = 350.0;
    ctx_b.viscosity = 2.0e-3;
    ctx_b.shear_rate = 10.0;

    ASSERT_EQ(model.compute_rhs(device_view(d_N), device_view(d_rhs_a),
                                ctx_a, workspace),
              cudaSuccess);
    ASSERT_EQ(model.compute_rhs(device_view(d_N), device_view(d_rhs_b),
                                ctx_b, workspace),
              cudaSuccess);

    const auto rhs_a = download(d_rhs_a, workspace.stream());
    const auto rhs_b = download(d_rhs_b, workspace.stream());
    expect_vectors_near(rhs_a, rhs_b, 0.0);
    EXPECT_TRUE(workspace.scratch().empty());
}

TEST(CudaPBEModel, ManyLocalStatesReuseOneModelDeviceBuffersAndWorkspace)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model =
        pbe_cuda::AggregationModel::brownian_continuum_shear(2.0e-5, 1.0e-3);
    config.breakage_model =
        pbe_cuda::BreakageModel::threshold_erosion(0.75, 2.0, 0.05);

    pbe_cuda::CudaWorkspace workspace;
    const pbe_cuda::CudaPBEModel model(config, 256, workspace.stream());
    const double* const device_grid_data = model.device_grid().data();

    pbe_cuda::CpuPBEModel cpu_model(config);
    pbe_cuda::CpuWorkspace cpu_workspace;

    pbe_cuda::CudaDeviceBuffer<double> d_N(config.grid->size());
    pbe_cuda::CudaDeviceBuffer<double> d_rhs(config.grid->size());
    const double* const d_N_data = d_N.data();
    double* const d_rhs_data = d_rhs.data();

    constexpr int n_cells = 64;
    for (int cell = 0; cell < n_cells; ++cell) {
        const double c = static_cast<double>(cell);
        const std::vector<double> N = {1.0 + 0.01 * c,
                                       0.5 + 0.02 * c,
                                       0.25 + 0.005 * c,
                                       0.1 + 0.001 * c,
                                       0.0};

        pbe_cuda::EvaluationContext context;
        context.temperature = 290.0 + c;
        context.pressure = 101325.0 + 10.0 * c;
        context.viscosity = 1.0e-3 + 1.0e-6 * c;
        context.shear_rate = 0.5 * c;

        upload(d_N, N, workspace.stream());
        ASSERT_EQ(model.compute_rhs(device_view(d_N), device_view(d_rhs),
                                    context, workspace),
                  cudaSuccess);
        const auto cuda_rhs = download(d_rhs, workspace.stream());

        std::vector<double> cpu_rhs(config.grid->size(), 0.0);
        ASSERT_EQ(cpu_model.compute_rhs(view(N), view(cpu_rhs),
                                        context, cpu_workspace),
                  cudaSuccess);
        expect_vectors_near(cuda_rhs, cpu_rhs, 1.0e-12);

        EXPECT_EQ(model.device_grid().data(), device_grid_data);
        EXPECT_EQ(d_N.data(), d_N_data);
        EXPECT_EQ(d_rhs.data(), d_rhs_data);
        EXPECT_TRUE(workspace.scratch().empty());
    }
}

TEST(CudaPBEModel, RejectsInvalidConfigurationAndInputs)
{
    if (!has_cuda_device())
        GTEST_SKIP() << "No CUDA device available";

    pbe_cuda::CudaWorkspace workspace;

    pbe_cuda::PBEModelConfig missing_aggregation_model;
    missing_aggregation_model.grid = make_grid();
    missing_aggregation_model.aggregation_enabled = true;
    EXPECT_THROW((void)pbe_cuda::CudaPBEModel{
                     missing_aggregation_model, 256, workspace.stream()},
                 std::invalid_argument);

    pbe_cuda::PBEModelConfig valid;
    valid.grid = make_grid();
    valid.aggregation_model = pbe_cuda::AggregationModel::constant(1.0);
    const pbe_cuda::CudaPBEModel model(valid, 256, workspace.stream());

    pbe_cuda::CudaDeviceBuffer<double> d_N(2);
    pbe_cuda::CudaDeviceBuffer<double> d_rhs(5);
    EXPECT_THROW((void)model.compute_rhs(device_view(d_N), device_view(d_rhs),
                                         workspace),
                 std::invalid_argument);

    pbe_cuda::CudaDeviceBuffer<double> d_good_N(5);
    pbe_cuda::EvaluationContext bad_context;
    bad_context.temperature = -1.0;
    EXPECT_THROW((void)model.compute_rhs(device_view(d_good_N),
                                         device_view(d_rhs),
                                         bad_context,
                                         workspace),
                 std::invalid_argument);
}
