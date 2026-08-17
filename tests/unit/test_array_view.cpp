// =============================================================================
// tests/unit/test_array_view.cpp
//
// Unit tests for the C++17 lightweight non-owning ArrayView utility.
// =============================================================================

#include <pbe_cuda/array_view.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

TEST(ArrayView, DefaultConstructsEmpty)
{
    pbe_cuda::ArrayView<double> view;

    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.size(), 0u);
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.begin(), view.end());
}

TEST(ArrayView, WrapsMutablePointerAndSize)
{
    double data[] = {1.0, 2.0, 3.0};
    pbe_cuda::ArrayView<double> view(data, 3);

    ASSERT_EQ(view.data(), data);
    ASSERT_EQ(view.size(), 3u);
    EXPECT_FALSE(view.empty());

    view[1] = 4.0;
    EXPECT_DOUBLE_EQ(data[1], 4.0);
}

TEST(ArrayView, WrapsCArray)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    pbe_cuda::ArrayView<double> view(data);

    EXPECT_EQ(view.data(), data);
    EXPECT_EQ(view.size(), 4u);
    EXPECT_DOUBLE_EQ(view[3], 4.0);
}

TEST(ArrayView, ConvertsMutableViewToConstView)
{
    double data[] = {5.0, 6.0};
    pbe_cuda::ArrayView<double> mutable_view(data);
    pbe_cuda::ArrayView<const double> const_view(mutable_view);

    EXPECT_EQ(const_view.data(), data);
    EXPECT_EQ(const_view.size(), mutable_view.size());
    EXPECT_DOUBLE_EQ(const_view[0], 5.0);
}

TEST(ArrayView, SupportsIteration)
{
    double data[] = {1.0, 2.0, 3.0};
    pbe_cuda::ArrayView<const double> view(data);

    double sum = 0.0;
    for (double value : view)
        sum += value;

    EXPECT_DOUBLE_EQ(sum, 6.0);
}

TEST(ArrayView, CreatesSubviewAndFirstView)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    pbe_cuda::ArrayView<double> view(data);

    const auto first = view.first(2);
    EXPECT_EQ(first.data(), data);
    EXPECT_EQ(first.size(), 2u);
    EXPECT_DOUBLE_EQ(first[1], 2.0);

    const auto middle = view.subview(1, 2);
    EXPECT_EQ(middle.data(), data + 1);
    EXPECT_EQ(middle.size(), 2u);

    middle[0] = 8.0;
    EXPECT_DOUBLE_EQ(data[1], 8.0);
}

TEST(ArrayView, RemainsTrivialPointerSizeWrapper)
{
    static_assert(std::is_trivially_copyable<pbe_cuda::ArrayView<double>>::value,
                  "ArrayView must remain trivially copyable");
    static_assert(std::is_standard_layout<pbe_cuda::ArrayView<double>>::value,
                  "ArrayView must remain standard layout");

    EXPECT_LE(sizeof(pbe_cuda::ArrayView<double>),
              sizeof(double*) + sizeof(std::size_t));
}
