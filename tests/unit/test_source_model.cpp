// =============================================================================
// tests/unit/test_source_model.cpp
//
// Tests for the minimal additive constant-source process configuration.
// =============================================================================

#include <pbe_cuda/source_model.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

TEST(ConstantSourceModel, UniformSourceBuildsPerSectionRates)
{
    const auto grid = pbe_cuda::SectionalGrid::geometric(4, 1.0, 2.0);
    const auto source = pbe_cuda::ConstantSourceModel::uniform(grid, 0.25);

    const auto rates = source.rates();
    ASSERT_EQ(rates.size(), grid.size());
    for (std::size_t i = 0; i < rates.size(); ++i)
        EXPECT_DOUBLE_EQ(rates[i], 0.25);

    const auto params = source.to_params(grid);
    EXPECT_EQ(params.n, grid.n());
    EXPECT_EQ(params.block_size, 256);
}

TEST(ConstantSourceModel, PreservesUserProvidedPerSectionRates)
{
    const auto grid = pbe_cuda::SectionalGrid::geometric(3, 1.0, 2.0);
    const pbe_cuda::ConstantSourceModel source({1.0, -2.0, 0.5});

    const auto rates = source.rates();
    ASSERT_EQ(rates.size(), 3u);
    EXPECT_DOUBLE_EQ(rates[0], 1.0);
    EXPECT_DOUBLE_EQ(rates[1], -2.0);
    EXPECT_DOUBLE_EQ(rates[2], 0.5);
    EXPECT_EQ(source.to_params(grid, 128).block_size, 128);
}

TEST(ConstantSourceModel, RejectsInvalidConfiguration)
{
    EXPECT_THROW((void)pbe_cuda::ConstantSourceModel({}),
                 std::invalid_argument);

    EXPECT_THROW(
        (void)pbe_cuda::ConstantSourceModel(
            {1.0, std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument);

    const auto grid = pbe_cuda::SectionalGrid::geometric(3, 1.0, 2.0);
    const pbe_cuda::ConstantSourceModel source({1.0, 2.0});

    EXPECT_THROW((void)source.to_params(grid), std::invalid_argument);
    EXPECT_THROW((void)source.to_params(grid, 192), std::invalid_argument);
}
