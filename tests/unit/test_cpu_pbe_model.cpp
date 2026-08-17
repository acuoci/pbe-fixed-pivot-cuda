// =============================================================================
// tests/unit/test_cpu_pbe_model.cpp
//
// Tests for the high-level serial CPU RHS model.
// =============================================================================

#include <pbe_cuda/cpu_pbe_model.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

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

void expect_vectors_near(const std::vector<double>& actual,
                         const std::vector<double>& expected,
                         double tol)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tol);
}

} // namespace

TEST(CpuPBEModel, AggregationOnlyMatchesManualLaunch)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model = pbe_cuda::AggregationModel::constant(2.0);

    const pbe_cuda::CpuPBEModel model(config);
    pbe_cuda::CpuWorkspace workspace;

    const std::vector<double> N = {3.0, 1.0, 0.5, 0.0, 0.0};
    std::vector<double> rhs(5, 99.0);
    std::vector<double> expected(5, 0.0);

    auto params = config.aggregation_model->to_params(*config.grid);
    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                  N.data(), config.grid->data(), expected.data(), params),
              cudaSuccess);

    ASSERT_EQ(model.compute_rhs(view(N), view(rhs), workspace), cudaSuccess);
    expect_vectors_near(rhs, expected, 0.0);
    EXPECT_EQ(workspace.scratch_size(), 0u);
}

TEST(CpuPBEModel, BreakageOnlyMatchesManualLaunch)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.breakage_model = pbe_cuda::BreakageModel::linear_symmetric(1.5, 2.0);

    const pbe_cuda::CpuPBEModel model(config);
    pbe_cuda::CpuWorkspace workspace;

    const std::vector<double> N = {0.0, 2.0, 1.0, 0.5, 0.0};
    std::vector<double> rhs(5, 7.0);
    std::vector<double> expected(5, 0.0);

    const auto params = config.breakage_model->to_params(*config.grid);
    const auto& q = config.breakage_model->quadrature();
    ASSERT_EQ(pbe_cuda::launch_breakage_rhs_cpu(
                  N.data(), config.grid->data(), q.t_q.data(), q.bw_q.data(),
                  expected.data(), params),
              cudaSuccess);

    ASSERT_EQ(model.compute_rhs(view(N), view(rhs), workspace), cudaSuccess);
    expect_vectors_near(rhs, expected, 0.0);
    EXPECT_EQ(workspace.scratch_size(), 0u);
}

TEST(CpuPBEModel, CombinedRhsMatchesManualAggregationPlusBreakage)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model = pbe_cuda::AggregationModel::sum(0.25);
    config.breakage_model = pbe_cuda::BreakageModel::constant_symmetric(0.5);

    const pbe_cuda::CpuPBEModel model(config);
    pbe_cuda::CpuWorkspace workspace;

    const std::vector<double> N = {1.0, 2.0, 0.0, 1.0, 0.0};
    std::vector<double> rhs(5, -3.0);
    std::vector<double> expected(5, 0.0);

    const auto aggregation_params =
        config.aggregation_model->to_params(*config.grid);
    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                  N.data(), config.grid->data(), expected.data(),
                  aggregation_params),
              cudaSuccess);

    const auto breakage_params = config.breakage_model->to_params(*config.grid);
    const auto& q = config.breakage_model->quadrature();
    ASSERT_EQ(pbe_cuda::launch_breakage_rhs_cpu(
                  N.data(), config.grid->data(), q.t_q.data(), q.bw_q.data(),
                  expected.data(), breakage_params),
              cudaSuccess);

    ASSERT_EQ(model.compute_rhs(view(N), view(rhs), workspace), cudaSuccess);
    expect_vectors_near(rhs, expected, 0.0);
}

TEST(CpuPBEModel, RepeatedCallsWithChangingContextReuseWorkspace)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model =
        pbe_cuda::AggregationModel::brownian_free_molecular(1.0e-4);

    const pbe_cuda::CpuPBEModel model(config);
    pbe_cuda::CpuWorkspace workspace;

    const std::vector<double> N = {1.0, 0.5, 0.25, 0.0, 0.0};
    std::vector<double> rhs_a(5, 0.0);
    std::vector<double> rhs_b(5, 0.0);

    pbe_cuda::EvaluationContext ctx_a;
    ctx_a.temperature = 300.0;
    ctx_a.viscosity = 1.0e-3;

    pbe_cuda::EvaluationContext ctx_b;
    ctx_b.temperature = 350.0;
    ctx_b.viscosity = 2.0e-3;
    ctx_b.shear_rate = 10.0;

    ASSERT_EQ(model.compute_rhs(view(N), view(rhs_a), ctx_a, workspace),
              cudaSuccess);
    ASSERT_EQ(model.compute_rhs(view(N), view(rhs_b), ctx_b, workspace),
              cudaSuccess);

    expect_vectors_near(rhs_a, rhs_b, 0.0);
    EXPECT_EQ(workspace.scratch_size(), 0u);
}

TEST(CpuPBEModel, RejectsInvalidConfigurationAndInputs)
{
    pbe_cuda::PBEModelConfig missing_aggregation_model;
    missing_aggregation_model.grid = make_grid();
    missing_aggregation_model.aggregation_enabled = true;
    EXPECT_THROW((void)pbe_cuda::CpuPBEModel{missing_aggregation_model},
                 std::invalid_argument);

    pbe_cuda::PBEModelConfig valid;
    valid.grid = make_grid();
    valid.aggregation_model = pbe_cuda::AggregationModel::constant(1.0);
    const pbe_cuda::CpuPBEModel model(valid);

    pbe_cuda::CpuWorkspace workspace;
    const std::vector<double> N = {1.0, 2.0};
    std::vector<double> rhs(5, 0.0);

    EXPECT_THROW((void)model.compute_rhs(view(N), view(rhs), workspace),
                 std::invalid_argument);

    std::vector<double> good_N(5, 1.0);
    pbe_cuda::EvaluationContext bad_context;
    bad_context.temperature = -1.0;
    EXPECT_THROW((void)model.compute_rhs(
                     view(good_N), view(rhs), bad_context, workspace),
                 std::invalid_argument);
}
