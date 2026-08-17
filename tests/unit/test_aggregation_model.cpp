// =============================================================================
// tests/unit/test_aggregation_model.cpp
//
// Tests for strongly typed aggregation configuration mapped to AggregationParams.
// =============================================================================

#include <pbe_cuda/aggregation_model.hpp>
#include <pbe_cuda/model_config.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

pbe_cuda::SectionalGrid make_grid()
{
    return pbe_cuda::SectionalGrid::geometric(5, 1.0, 2.0);
}

void expect_grid_metadata(const pbe_cuda::AggregationParams& params,
                          const pbe_cuda::SectionalGrid& grid)
{
    EXPECT_EQ(params.n, grid.n());
    EXPECT_DOUBLE_EQ(params.log_x0, grid.log_x0());
    EXPECT_DOUBLE_EQ(params.inv_log_r, grid.inv_log_r());
}

} // namespace

TEST(AggregationModel, MapsSimpleKernelsToLowLevelParams)
{
    const auto grid = make_grid();

    {
        const auto params =
            pbe_cuda::AggregationModel::constant(2.0).to_params(grid, 128);
        EXPECT_EQ(params.kernel_type, pbe_cuda::AggregationKernel::Constant);
        EXPECT_DOUBLE_EQ(params.beta0, 2.0);
        EXPECT_EQ(params.block_size, 128);
        expect_grid_metadata(params, grid);
    }

    {
        const auto params = pbe_cuda::AggregationModel::sum(3.0).to_params(grid);
        EXPECT_EQ(params.kernel_type, pbe_cuda::AggregationKernel::Sum);
        EXPECT_DOUBLE_EQ(params.beta0, 3.0);
        EXPECT_EQ(params.block_size, 256);
        expect_grid_metadata(params, grid);
    }

    {
        const auto params =
            pbe_cuda::AggregationModel::product(4.0).to_params(grid);
        EXPECT_EQ(params.kernel_type, pbe_cuda::AggregationKernel::Product);
        EXPECT_DOUBLE_EQ(params.beta0, 4.0);
        expect_grid_metadata(params, grid);
    }
}

TEST(AggregationModel, MapsBrownianAndShearKernelsToLowLevelParams)
{
    const auto grid = make_grid();

    {
        const auto params =
            pbe_cuda::AggregationModel::brownian_continuum(1.0e-18)
                .to_params(grid);
        EXPECT_EQ(params.kernel_type,
                  pbe_cuda::AggregationKernel::BrownianContinuum);
        EXPECT_DOUBLE_EQ(params.beta_bc, 1.0e-18);
        expect_grid_metadata(params, grid);
    }

    {
        const auto params =
            pbe_cuda::AggregationModel::brownian_free_molecular(2.0e-9)
                .to_params(grid);
        EXPECT_EQ(params.kernel_type,
                  pbe_cuda::AggregationKernel::BrownianFreeMolecular);
        EXPECT_DOUBLE_EQ(params.beta_bfm, 2.0e-9);
        expect_grid_metadata(params, grid);
    }

    {
        const auto params = pbe_cuda::AggregationModel::shear(3.0).to_params(grid);
        EXPECT_EQ(params.kernel_type, pbe_cuda::AggregationKernel::Shear);
        EXPECT_DOUBLE_EQ(params.beta_sh, 3.0);
        expect_grid_metadata(params, grid);
    }
}

TEST(AggregationModel, MapsCombinedKernelsToLowLevelParams)
{
    const auto grid = make_grid();

    {
        const auto params =
            pbe_cuda::AggregationModel::brownian_continuum_shear(1.0e-18, 2.0)
                .to_params(grid);
        EXPECT_EQ(params.kernel_type,
                  pbe_cuda::AggregationKernel::BrownianContinuumShear);
        EXPECT_DOUBLE_EQ(params.beta_bc, 1.0e-18);
        EXPECT_DOUBLE_EQ(params.beta_sh, 2.0);
    }

    {
        const auto params =
            pbe_cuda::AggregationModel::brownian_free_molecular_shear(3.0e-9,
                                                                      4.0)
                .to_params(grid);
        EXPECT_EQ(params.kernel_type,
                  pbe_cuda::AggregationKernel::BrownianFreeMolecularShear);
        EXPECT_DOUBLE_EQ(params.beta_bfm, 3.0e-9);
        EXPECT_DOUBLE_EQ(params.beta_sh, 4.0);
    }
}

TEST(AggregationModel, RejectsInvalidCoefficientsAtSetup)
{
    EXPECT_THROW(pbe_cuda::AggregationModel::constant(-1.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::AggregationModel::brownian_continuum(
                     std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::AggregationModel::brownian_free_molecular_shear(
                     1.0, -2.0),
                 std::invalid_argument);
}

TEST(AggregationModel, RejectsInvalidBlockSizeDuringParamMapping)
{
    const auto grid = make_grid();
    const auto model = pbe_cuda::AggregationModel::constant(1.0);

    EXPECT_THROW((void)model.to_params(grid, 0), std::invalid_argument);
    EXPECT_THROW((void)model.to_params(grid, 192), std::invalid_argument);
}

TEST(AggregationModel, IntegratesWithEarlyPBEModelConfig)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = make_grid();
    config.aggregation_model = pbe_cuda::AggregationModel::shear(5.0);

    EXPECT_TRUE(config.has_grid());
    EXPECT_TRUE(config.has_enabled_process());
    EXPECT_NO_THROW(config.validate());
}
