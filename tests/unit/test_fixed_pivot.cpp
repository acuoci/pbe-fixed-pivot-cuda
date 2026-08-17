// =============================================================================
// tests/unit/test_fixed_pivot.cpp
//
// Direct tests for internal fixed-pivot lookup and birth allocation helpers.
// =============================================================================

#include <pbe_cuda/detail/fixed_pivot.cuh>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

TEST(FixedPivot, GeometricRightBracketMatchesExistingLookupMetadata)
{
    const std::vector<double> x = {1.0, 2.0, 4.0, 8.0};
    const double log_x0 = std::log(x.front());
    const double inv_log_r = 1.0 / std::log(x[1] / x[0]);

    EXPECT_EQ(pbe_cuda::detail::fixed_pivot_right_bracket(
                  x.data(), static_cast<int>(x.size()), 3.0, log_x0, inv_log_r),
              2);
    EXPECT_EQ(pbe_cuda::detail::fixed_pivot_right_bracket(
                  x.data(), static_cast<int>(x.size()), 4.0, log_x0, inv_log_r),
              3);
}

TEST(FixedPivot, GenericLeftAndRightBracketsMatchBinarySearchConventions)
{
    const std::vector<double> x = {1.0, 2.0, 5.0, 11.0};

    EXPECT_EQ(pbe_cuda::detail::fixed_pivot_left_bracket(
                  x.data(), static_cast<int>(x.size()), 4.0),
              1);
    EXPECT_EQ(pbe_cuda::detail::fixed_pivot_right_bracket(
                  x.data(), static_cast<int>(x.size()), 4.0, 0.0, 0.0),
              2);
}

TEST(FixedPivot, InteriorBirthAllocationInterpolates)
{
    const std::vector<double> x = {1.0, 2.0, 5.0, 11.0};

    const auto allocation = pbe_cuda::detail::fixed_pivot_birth_allocation(
        x.data(), static_cast<int>(x.size()), 3.5);

    EXPECT_EQ(allocation.lower, 1);
    EXPECT_EQ(allocation.upper, 2);
    EXPECT_DOUBLE_EQ(allocation.upper_weight, 0.5);
}

TEST(FixedPivot, BoundaryBirthAllocationClips)
{
    const std::vector<double> x = {1.0, 2.0, 5.0};

    const auto low = pbe_cuda::detail::fixed_pivot_birth_allocation(
        x.data(), static_cast<int>(x.size()), 0.5);
    EXPECT_EQ(low.lower, 0);
    EXPECT_EQ(low.upper, -1);
    EXPECT_DOUBLE_EQ(low.upper_weight, 0.0);

    const auto high = pbe_cuda::detail::fixed_pivot_birth_allocation(
        x.data(), static_cast<int>(x.size()), 7.0);
    EXPECT_EQ(high.lower, 2);
    EXPECT_EQ(high.upper, -1);
    EXPECT_DOUBLE_EQ(high.upper_weight, 0.0);
}
