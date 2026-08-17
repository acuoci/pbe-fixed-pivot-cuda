// =============================================================================
// sectional_grid.cpp  --  Validated sectional pivot grid
// =============================================================================

#include "pbe_cuda/sectional_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace pbe_cuda {

namespace {

void require_finite_positive(double value, const char* name)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string("SectionalGrid: ") + name +
                                    " must be finite and positive");
}

bool ratios_match(double ratio, double reference)
{
    const double scale = std::max(std::fabs(reference), 1.0);
    const double tol = 128.0 * std::numeric_limits<double>::epsilon() * scale;
    return std::fabs(ratio - reference) <= tol;
}

} // namespace

SectionalGrid::SectionalGrid(std::vector<double> pivots)
    : pivots_(std::move(pivots))
{
    validate_and_update_metadata();
}

SectionalGrid SectionalGrid::from_pivots(ConstRealView pivots)
{
    return SectionalGrid(std::vector<double>(pivots.begin(), pivots.end()));
}

SectionalGrid SectionalGrid::geometric(std::size_t n, double x0, double ratio)
{
    if (n < 2)
        throw std::invalid_argument("SectionalGrid: at least two pivots are required");
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("SectionalGrid: pivot count exceeds int range");
    require_finite_positive(x0, "x0");
    require_finite_positive(ratio, "ratio");
    if (ratio <= 1.0)
        throw std::invalid_argument("SectionalGrid: geometric ratio must be greater than one");

    std::vector<double> pivots(n);
    pivots[0] = x0;
    for (std::size_t i = 1; i < n; ++i)
        pivots[i] = pivots[i - 1] * ratio;

    return SectionalGrid(std::move(pivots));
}

SectionalGrid SectionalGrid::geometric_range(std::size_t n,
                                             double x_min,
                                             double x_max)
{
    if (n < 2)
        throw std::invalid_argument("SectionalGrid: at least two pivots are required");
    require_finite_positive(x_min, "x_min");
    require_finite_positive(x_max, "x_max");
    if (x_max <= x_min)
        throw std::invalid_argument("SectionalGrid: x_max must be greater than x_min");

    const double ratio =
        std::pow(x_max / x_min, 1.0 / static_cast<double>(n - 1));
    return geometric(n, x_min, ratio);
}

ConstRealView SectionalGrid::pivots() const noexcept
{
    return ConstRealView(pivots_.data(), pivots_.size());
}

const double* SectionalGrid::data() const noexcept
{
    return pivots_.data();
}

std::size_t SectionalGrid::size() const noexcept
{
    return pivots_.size();
}

int SectionalGrid::n() const noexcept
{
    return static_cast<int>(pivots_.size());
}

bool SectionalGrid::empty() const noexcept
{
    return pivots_.empty();
}

double SectionalGrid::operator[](std::size_t i) const noexcept
{
    return pivots_[i];
}

bool SectionalGrid::is_geometric() const noexcept
{
    return is_geometric_;
}

double SectionalGrid::ratio() const noexcept
{
    return ratio_;
}

double SectionalGrid::log_x0() const noexcept
{
    return log_x0_;
}

double SectionalGrid::inv_log_r() const noexcept
{
    return inv_log_r_;
}

void SectionalGrid::validate_and_update_metadata()
{
    if (pivots_.size() < 2)
        throw std::invalid_argument("SectionalGrid: at least two pivots are required");
    if (pivots_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("SectionalGrid: pivot count exceeds int range");

    require_finite_positive(pivots_[0], "pivot");
    for (std::size_t i = 1; i < pivots_.size(); ++i) {
        require_finite_positive(pivots_[i], "pivot");
        if (pivots_[i] <= pivots_[i - 1])
            throw std::invalid_argument(
                "SectionalGrid: pivots must be strictly increasing");
    }

    log_x0_ = std::log(pivots_[0]);
    ratio_ = pivots_[1] / pivots_[0];
    is_geometric_ = true;

    for (std::size_t i = 2; i < pivots_.size(); ++i) {
        const double ratio = pivots_[i] / pivots_[i - 1];
        if (!ratios_match(ratio, ratio_)) {
            is_geometric_ = false;
            break;
        }
    }

    inv_log_r_ = is_geometric_ ? 1.0 / std::log(ratio_) : 0.0;
    if (!is_geometric_)
        ratio_ = 0.0;
}

} // namespace pbe_cuda
