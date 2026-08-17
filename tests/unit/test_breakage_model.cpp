// =============================================================================
// tests/unit/test_breakage_model.cpp
//
// Tests for strongly typed breakage selection and daughter configuration.
// =============================================================================

#include <pbe_cuda/breakage_model.hpp>
#include <pbe_cuda/model_config.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

pbe_cuda::SectionalGrid make_grid()
{
    return pbe_cuda::SectionalGrid::geometric(6, 1.0, 2.0);
}

void expect_grid_metadata(const pbe_cuda::BreakageParams& params,
                          const pbe_cuda::SectionalGrid& grid)
{
    EXPECT_EQ(params.n, grid.n());
}

} // namespace

TEST(BreakageModel, MapsSelectionModelsToLowLevelParams)
{
    const auto grid = make_grid();

    {
        const auto model = pbe_cuda::BreakageModel::constant_symmetric(2.0);
        const auto params = model.to_params(grid, 128);
        EXPECT_EQ(params.selection, pbe_cuda::BreakageSelection::Constant);
        EXPECT_DOUBLE_EQ(params.S0, 2.0);
        EXPECT_EQ(params.n_quad, 1);
        EXPECT_EQ(params.block_size, 128);
        expect_grid_metadata(params, grid);
    }

    {
        const auto model = pbe_cuda::BreakageModel::linear_symmetric(3.0, 4.0);
        const auto params = model.to_params(grid);
        EXPECT_EQ(params.selection, pbe_cuda::BreakageSelection::Linear);
        EXPECT_DOUBLE_EQ(params.S0, 3.0);
        EXPECT_DOUBLE_EQ(params.v_ref, 4.0);
        EXPECT_EQ(params.block_size, 256);
        expect_grid_metadata(params, grid);
    }

    {
        const auto model = pbe_cuda::BreakageModel::power_law(
            5.0, 6.0, 1.5, pbe_cuda::SymmetricBinaryDaughter{});
        const auto params = model.to_params(grid);
        EXPECT_EQ(params.selection, pbe_cuda::BreakageSelection::PowerLaw);
        EXPECT_DOUBLE_EQ(params.S0, 5.0);
        EXPECT_DOUBLE_EQ(params.v_ref, 6.0);
        EXPECT_DOUBLE_EQ(params.alpha, 1.5);
    }

    {
        const auto model = pbe_cuda::BreakageModel::threshold_erosion(
            7.0, 8.0, 0.05);
        const auto params = model.to_params(grid);
        EXPECT_EQ(params.selection, pbe_cuda::BreakageSelection::Threshold);
        EXPECT_DOUBLE_EQ(params.S0, 7.0);
        EXPECT_DOUBLE_EQ(params.v_min, 8.0);
        EXPECT_EQ(params.n_quad, 2);
    }
}

TEST(BreakageModel, GeneratesSymmetricBinaryQuadrature)
{
    const auto model = pbe_cuda::BreakageModel::constant_symmetric(1.0);
    const auto& q = model.quadrature();

    ASSERT_EQ(q.size(), 1);
    EXPECT_DOUBLE_EQ(q.t_q[0], 0.5);
    EXPECT_DOUBLE_EQ(q.bw_q[0], 2.0);
    EXPECT_EQ(q.t_view().data(), q.t_q.data());
    EXPECT_EQ(q.bw_view().data(), q.bw_q.data());
}

TEST(BreakageModel, GeneratesErosionQuadrature)
{
    const auto model = pbe_cuda::BreakageModel::threshold_erosion(
        1.0, 0.2, 0.05);
    const auto& q = model.quadrature();

    ASSERT_EQ(q.size(), 2);
    EXPECT_DOUBLE_EQ(q.t_q[0], 0.95);
    EXPECT_DOUBLE_EQ(q.t_q[1], 0.05);
    EXPECT_DOUBLE_EQ(q.bw_q[0], 1.0);
    EXPECT_DOUBLE_EQ(q.bw_q[1], 1.0);
}

TEST(BreakageModel, GeneratesUniformQuadratureMatchingExistingEightPointTable)
{
    const double gl8_t[] = {
        0.019855071751232, 0.101666761293187, 0.237233795041836,
        0.408282678752175, 0.591717321247825, 0.762766204958164,
        0.898333238706813, 0.980144928248768
    };
    const double gl8_w[] = {
        0.050614268145188, 0.111190517226687, 0.156853322938943,
        0.181341891689181, 0.181341891689181, 0.156853322938943,
        0.111190517226687, 0.050614268145188
    };

    const auto model = pbe_cuda::BreakageModel::linear_uniform(1.0, 1.0, 8);
    const auto& q = model.quadrature();

    ASSERT_EQ(q.size(), 8);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(q.t_q[static_cast<std::size_t>(i)], gl8_t[i], 2.0e-14);
        EXPECT_NEAR(q.bw_q[static_cast<std::size_t>(i)], 2.0 * gl8_w[i],
                    2.0e-14);
    }

    const double daughter_count =
        std::accumulate(q.bw_q.begin(), q.bw_q.end(), 0.0);
    double daughter_volume = 0.0;
    for (std::size_t i = 0; i < q.t_q.size(); ++i)
        daughter_volume += q.t_q[i] * q.bw_q[i];

    EXPECT_NEAR(daughter_count, 2.0, 4.0e-14);
    EXPECT_NEAR(daughter_volume, 1.0, 4.0e-14);
}

TEST(BreakageModel, PreservesUserQuadratureExpertPath)
{
    pbe_cuda::UserQuadratureDaughter daughter;
    daughter.t_q = {0.25, 0.75};
    daughter.bw_q = {1.0, 1.0};

    const auto model = pbe_cuda::BreakageModel(
        pbe_cuda::LinearBreakageSelection{1.0, 2.0}, daughter);
    const auto& q = model.quadrature();

    ASSERT_EQ(q.size(), 2);
    EXPECT_DOUBLE_EQ(q.t_q[0], 0.25);
    EXPECT_DOUBLE_EQ(q.t_q[1], 0.75);
    EXPECT_DOUBLE_EQ(q.bw_q[0], 1.0);
    EXPECT_DOUBLE_EQ(q.bw_q[1], 1.0);
}

TEST(BreakageModel, RejectsInvalidSelectionOrDaughterConfiguration)
{
    EXPECT_THROW(pbe_cuda::BreakageModel::constant_symmetric(-1.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::BreakageModel::linear_symmetric(1.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::BreakageModel::threshold_erosion(1.0, 0.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::BreakageModel::linear_uniform(1.0, 1.0, 0),
                 std::invalid_argument);

    pbe_cuda::UserQuadratureDaughter invalid;
    invalid.t_q = {0.0};
    invalid.bw_q = {1.0};
    EXPECT_THROW(pbe_cuda::BreakageModel(
                     pbe_cuda::ConstantBreakageSelection{1.0}, invalid),
                 std::invalid_argument);
}

TEST(BreakageModel, RejectsInvalidBlockSizeDuringParamMapping)
{
    const auto grid = make_grid();
    const auto model = pbe_cuda::BreakageModel::constant_symmetric(1.0);

    EXPECT_THROW((void)model.to_params(grid, 0), std::invalid_argument);
    EXPECT_THROW((void)model.to_params(grid, 192), std::invalid_argument);
}

TEST(BreakageModel, IntegratesWithEarlyPBEModelConfig)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.breakage_model = pbe_cuda::BreakageModel::linear_uniform(1.0, 1.0, 8);

    EXPECT_TRUE(config.has_grid());
    EXPECT_TRUE(config.has_enabled_process());
    EXPECT_NO_THROW(config.validate());
}
