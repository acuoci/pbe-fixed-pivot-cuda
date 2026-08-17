// =============================================================================
// array_view.hpp  --  Lightweight non-owning array view for C++17
//
// This is an intentionally small span-like wrapper used during the architecture
// migration. It does not own memory, allocate, or validate lifetimes.
// =============================================================================

#pragma once

#include <cstddef>
#include <type_traits>

namespace pbe_cuda {

template <typename T>
class ArrayView {
public:
    using element_type = T;
    using value_type = typename std::remove_cv<T>::type;
    using size_type = std::size_t;
    using pointer = T*;
    using reference = T&;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr ArrayView() noexcept = default;

    constexpr ArrayView(pointer data, size_type size) noexcept
        : data_(data), size_(size)
    {}

    template <std::size_t N>
    constexpr ArrayView(T (&data)[N]) noexcept
        : data_(data), size_(N)
    {}

    template <
        typename U,
        typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    constexpr ArrayView(const ArrayView<U>& other) noexcept
        : data_(other.data()), size_(other.size())
    {}

    [[nodiscard]] constexpr pointer data() const noexcept { return data_; }
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr reference operator[](size_type index) const noexcept
    {
        return data_[index];
    }

    [[nodiscard]] constexpr iterator begin() const noexcept { return data_; }
    [[nodiscard]] constexpr iterator end() const noexcept { return data_ + size_; }

    [[nodiscard]] constexpr ArrayView<T> first(size_type count) const noexcept
    {
        return ArrayView<T>(data_, count);
    }

    [[nodiscard]] constexpr ArrayView<T> subview(size_type offset,
                                                 size_type count) const noexcept
    {
        return ArrayView<T>(data_ + offset, count);
    }

private:
    pointer data_ = nullptr;
    size_type size_ = 0;
};

using RealView = ArrayView<double>;
using ConstRealView = ArrayView<const double>;

} // namespace pbe_cuda
