// =============================================================================
// tests/unit/test_sectional_grid.cpp
//
// Unit tests for the validated host sectional pivot grid.
// =============================================================================

#include <pbe_cuda/sectional_grid.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(SectionalGrid, BuildsFromExplicitPivots)
{
    const std::vector<double> pivots = {1.0, 2.0, 4.0, 8.0};
    const auto grid = pbe_cuda::SectionalGrid::from_pivots(
        pbe_cuda::ConstRealView(pivots.data(), pivots.size()));

    EXPECT_EQ(grid.size(), pivots.size());
    EXPECT_EQ(grid.n(), static_cast<int>(pivots.size()));
    EXPECT_EQ(grid.data(), grid.pivots().data());
    EXPECT_FALSE(grid.empty());

    for (std::size_t i = 0; i < pivots.size(); ++i)
        EXPECT_DOUBLE_EQ(grid[i], pivots[i]);
}

TEST(SectionalGrid, DetectsGeometricMetadata)
{
    constexpr std::size_t n = 6;
    constexpr double x0 = 1.0e-18;
    constexpr double ratio = 1.1220184543019633;

    const auto grid = pbe_cuda::SectionalGrid::geometric(n, x0, ratio);

    EXPECT_TRUE(grid.is_geometric());
    EXPECT_DOUBLE_EQ(grid.ratio(), ratio);
    EXPECT_DOUBLE_EQ(grid.log_x0(), std::log(x0));
    EXPECT_DOUBLE_EQ(grid.inv_log_r(), 1.0 / std::log(ratio));

    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(grid[i], x0 * std::pow(ratio, static_cast<double>(i)),
                    8.0 * std::numeric_limits<double>::epsilon() * grid[i]);
}

TEST(SectionalGrid, BuildsGeometricRangeMetadataMatchingExamples)
{
    constexpr std::size_t n = 9;
    constexpr double x_min = 1.0e-4;
    constexpr double x_max = 1.0e4;
    const double ratio = std::pow(x_max / x_min, 1.0 / static_cast<double>(n - 1));

    const auto grid = pbe_cuda::SectionalGrid::geometric_range(n, x_min, x_max);

    EXPECT_TRUE(grid.is_geometric());
    EXPECT_DOUBLE_EQ(grid.ratio(), ratio);
    EXPECT_DOUBLE_EQ(grid.log_x0(), std::log(x_min));
    EXPECT_DOUBLE_EQ(grid.inv_log_r(), 1.0 / std::log(ratio));
    EXPECT_NEAR(grid[n - 1], x_max,
                16.0 * std::numeric_limits<double>::epsilon() * x_max);
}

TEST(SectionalGrid, MarksNonGeometricGridForGenericLookup)
{
    const std::vector<double> pivots = {1.0, 2.0, 5.0, 11.0};
    const auto grid = pbe_cuda::SectionalGrid::from_pivots(
        pbe_cuda::ConstRealView(pivots.data(), pivots.size()));

    EXPECT_FALSE(grid.is_geometric());
    EXPECT_DOUBLE_EQ(grid.ratio(), 0.0);
    EXPECT_DOUBLE_EQ(grid.log_x0(), std::log(pivots.front()));
    EXPECT_DOUBLE_EQ(grid.inv_log_r(), 0.0);
}

TEST(SectionalGrid, RejectsInvalidPivots)
{
    EXPECT_THROW(pbe_cuda::SectionalGrid::from_pivots(pbe_cuda::ConstRealView()),
                 std::invalid_argument);

    const double one[] = {1.0};
    EXPECT_THROW(pbe_cuda::SectionalGrid::from_pivots(one), std::invalid_argument);

    const double non_positive[] = {0.0, 1.0};
    EXPECT_THROW(pbe_cuda::SectionalGrid::from_pivots(non_positive),
                 std::invalid_argument);

    const double non_monotone[] = {1.0, 1.0, 2.0};
    EXPECT_THROW(pbe_cuda::SectionalGrid::from_pivots(non_monotone),
                 std::invalid_argument);

    const double nan_grid[] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    EXPECT_THROW(pbe_cuda::SectionalGrid::from_pivots(nan_grid),
                 std::invalid_argument);
}

TEST(SectionalGrid, RejectsInvalidGeometricInputs)
{
    EXPECT_THROW(pbe_cuda::SectionalGrid::geometric(1, 1.0, 2.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::SectionalGrid::geometric(2, 0.0, 2.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::SectionalGrid::geometric(2, 1.0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(pbe_cuda::SectionalGrid::geometric_range(2, 1.0, 1.0),
                 std::invalid_argument);
}
