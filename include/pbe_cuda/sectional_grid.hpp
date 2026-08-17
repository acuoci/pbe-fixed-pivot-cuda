// =============================================================================
// sectional_grid.hpp  --  Validated sectional pivot grid
//
// Owns host pivot coordinates and precomputes the grid metadata currently
// passed to low-level fixed-pivot RHS launch functions.
// =============================================================================

#pragma once

#include "pbe_cuda/array_view.hpp"

#include <cstddef>
#include <vector>

namespace pbe_cuda {

class SectionalGrid {
public:
    explicit SectionalGrid(std::vector<double> pivots);

    static SectionalGrid from_pivots(ConstRealView pivots);
    static SectionalGrid geometric(std::size_t n, double x0, double ratio);
    static SectionalGrid geometric_range(std::size_t n,
                                         double x_min,
                                         double x_max);

    [[nodiscard]] ConstRealView pivots() const noexcept;
    [[nodiscard]] const double* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] int n() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] double operator[](std::size_t i) const noexcept;

    [[nodiscard]] bool is_geometric() const noexcept;
    [[nodiscard]] double ratio() const noexcept;
    [[nodiscard]] double log_x0() const noexcept;
    [[nodiscard]] double inv_log_r() const noexcept;

private:
    void validate_and_update_metadata();

    std::vector<double> pivots_;
    bool is_geometric_ = false;
    double ratio_ = 0.0;
    double log_x0_ = 0.0;
    double inv_log_r_ = 0.0;
};

} // namespace pbe_cuda
