// =============================================================================
// source_model.hpp  --  Minimal additive source process configuration
//
// The constant source model is intentionally small. It records the contribution
// convention for process terms that add directly to rhs without forcing future
// nucleation, growth, or sintering models into one generic hierarchy.
// =============================================================================

#pragma once

#include "pbe_cuda/array_view.hpp"
#include "pbe_cuda/sectional_grid.hpp"
#include "pbe_cuda/source.cuh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pbe_cuda {

class ConstantSourceModel {
public:
    static ConstantSourceModel uniform(const SectionalGrid& grid, double rate)
    {
        return ConstantSourceModel(std::vector<double>(grid.size(), rate));
    }

    explicit ConstantSourceModel(std::vector<double> rates)
        : rates_(std::move(rates))
    {
        validate();
    }

    [[nodiscard]] ConstRealView rates() const noexcept
    {
        return ConstRealView(rates_.data(), rates_.size());
    }

    [[nodiscard]] ConstantSourceParams to_params(const SectionalGrid& grid,
                                                 int block_size = 256) const
    {
        if (rates_.size() != grid.size())
            throw std::invalid_argument(
                "ConstantSourceModel: rate vector size must match grid size");
        if (block_size <= 0 || (block_size & (block_size - 1)) != 0)
            throw std::invalid_argument(
                "ConstantSourceModel: block_size must be a positive power of two");

        ConstantSourceParams params;
        params.n = grid.n();
        params.block_size = block_size;
        return params;
    }

    void validate() const
    {
        if (rates_.empty())
            throw std::invalid_argument(
                "ConstantSourceModel: rate vector must not be empty");
        for (double rate : rates_) {
            if (!std::isfinite(rate))
                throw std::invalid_argument(
                    "ConstantSourceModel: rates must be finite");
        }
    }

private:
    std::vector<double> rates_;
};

} // namespace pbe_cuda
