// =============================================================================
// cpu_pbe_model.hpp  --  High-level serial CPU RHS evaluator
//
// Coordinates configured aggregation and breakage contributions using existing
// serial fixed-pivot RHS implementations. State and RHS storage are externally
// owned by the caller.
// =============================================================================

#pragma once

#include "pbe_cuda/aggregation.cuh"
#include "pbe_cuda/array_view.hpp"
#include "pbe_cuda/backend.hpp"
#include "pbe_cuda/breakage.cuh"
#include "pbe_cuda/model_config.hpp"
#include "pbe_cuda/sectional_grid.hpp"
#include "pbe_cuda/source.cuh"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace pbe_cuda {

class CpuPBEModel {
public:
    explicit CpuPBEModel(PBEModelConfig config, int block_size = 256)
        : grid_(*require_grid(config))
    {
        config.validate();

        if (config.aggregation_enabled && !config.aggregation_model)
            throw std::invalid_argument(
                "CpuPBEModel: aggregation_enabled requires aggregation_model");
        if (config.breakage_enabled && !config.breakage_model)
            throw std::invalid_argument(
                "CpuPBEModel: breakage_enabled requires breakage_model");
        if (config.constant_source_enabled && !config.constant_source_model)
            throw std::invalid_argument(
                "CpuPBEModel: constant_source_enabled requires constant_source_model");

        if (config.aggregation_model)
            aggregation_params_ = config.aggregation_model->to_params(
                grid_, block_size);
        if (config.breakage_model) {
            breakage_params_ = config.breakage_model->to_params(
                grid_, block_size);
            breakage_quadrature_ = config.breakage_model->quadrature();
        }
        if (config.constant_source_model) {
            constant_source_model_ = *config.constant_source_model;
            constant_source_params_ = constant_source_model_->to_params(
                grid_, block_size);
        }
    }

    [[nodiscard]] const SectionalGrid& grid() const noexcept { return grid_; }
    [[nodiscard]] bool has_aggregation() const noexcept
    {
        return aggregation_params_.has_value();
    }
    [[nodiscard]] bool has_breakage() const noexcept
    {
        return breakage_params_.has_value();
    }
    [[nodiscard]] bool has_constant_source() const noexcept
    {
        return constant_source_params_.has_value();
    }

    [[nodiscard]] cudaError_t compute_rhs(ConstRealView N,
                                          RealView rhs,
                                          CpuWorkspace& workspace) const
    {
        return compute_rhs(N, rhs, EvaluationContext{}, workspace);
    }

    [[nodiscard]] cudaError_t compute_rhs(ConstRealView N,
                                          RealView rhs,
                                          const EvaluationContext& context,
                                          CpuWorkspace& workspace) const
    {
        (void)workspace;

        context.validate();
        validate_state_views(N, rhs);
        std::fill(rhs.begin(), rhs.end(), 0.0);

        if (aggregation_params_) {
            const cudaError_t err = launch_aggregation_rhs_cpu(
                N.data(), grid_.data(), rhs.data(), *aggregation_params_);
            if (err != cudaSuccess)
                return err;
        }

        if (breakage_params_) {
            const cudaError_t err = launch_breakage_rhs_cpu(
                N.data(), grid_.data(), breakage_quadrature_.t_q.data(),
                breakage_quadrature_.bw_q.data(), rhs.data(),
                *breakage_params_);
            if (err != cudaSuccess)
                return err;
        }

        if (constant_source_params_) {
            const cudaError_t err = launch_constant_source_rhs_cpu(
                constant_source_model_->rates().data(), rhs.data(),
                *constant_source_params_);
            if (err != cudaSuccess)
                return err;
        }

        return cudaSuccess;
    }

private:
    static const SectionalGrid* require_grid(const PBEModelConfig& config)
    {
        if (!config.grid)
            throw std::invalid_argument("CpuPBEModel: grid is required");
        return &*config.grid;
    }

    void validate_state_views(ConstRealView N, RealView rhs) const
    {
        const std::size_t n = grid_.size();
        if (N.size() != n)
            throw std::invalid_argument("CpuPBEModel: N size mismatch");
        if (rhs.size() != n)
            throw std::invalid_argument("CpuPBEModel: rhs size mismatch");
        if (!N.data())
            throw std::invalid_argument("CpuPBEModel: N data is null");
        if (!rhs.data())
            throw std::invalid_argument("CpuPBEModel: rhs data is null");
    }

    SectionalGrid grid_;
    std::optional<AggregationParams> aggregation_params_;
    std::optional<BreakageParams> breakage_params_;
    std::optional<ConstantSourceModel> constant_source_model_;
    std::optional<ConstantSourceParams> constant_source_params_;
    BreakageQuadrature breakage_quadrature_;
};

} // namespace pbe_cuda
