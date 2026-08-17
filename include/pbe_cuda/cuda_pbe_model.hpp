// =============================================================================
// cuda_pbe_model.hpp  --  High-level CUDA RHS evaluator
//
// Coordinates configured aggregation and breakage contributions using existing
// CUDA fixed-pivot RHS launch functions. State and RHS storage are externally
// owned device memory supplied by the caller.
// =============================================================================

#pragma once

#include "pbe_cuda/aggregation.cuh"
#include "pbe_cuda/array_view.hpp"
#include "pbe_cuda/backend.hpp"
#include "pbe_cuda/breakage.cuh"
#include "pbe_cuda/model_config.hpp"
#include "pbe_cuda/sectional_grid.hpp"
#include "pbe_cuda/source.cuh"

#include <optional>
#include <stdexcept>
#include <utility>

namespace pbe_cuda {

#if defined(PBE_ENABLE_CUDA)

using DeviceRealView = ArrayView<double>;
using ConstDeviceRealView = ArrayView<const double>;

class CudaPBEModel {
public:
    explicit CudaPBEModel(const PBEModelConfig& config,
                          int block_size = 256,
                          cudaStream_t setup_stream = 0)
        : CudaPBEModel(validated_grid_copy(config),
                       config,
                       block_size,
                       setup_stream)
    {}

    [[nodiscard]] const SectionalGrid& grid() const noexcept { return grid_; }
    [[nodiscard]] const CudaDeviceGrid& device_grid() const noexcept
    {
        return device_grid_;
    }

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

    [[nodiscard]] cudaError_t compute_rhs(ConstDeviceRealView N,
                                          DeviceRealView rhs,
                                          CudaWorkspace& workspace) const
    {
        return compute_rhs(N, rhs, EvaluationContext{}, workspace);
    }

    [[nodiscard]] cudaError_t compute_rhs(ConstDeviceRealView N,
                                          DeviceRealView rhs,
                                          const EvaluationContext& context,
                                          CudaWorkspace& workspace) const
    {
        context.validate();
        validate_state_views(N, rhs);

        const cudaError_t zero_error =
            cudaMemsetAsync(rhs.data(), 0, grid_.size() * sizeof(double),
                            workspace.stream());
        if (zero_error != cudaSuccess)
            return zero_error;

        if (aggregation_params_) {
            const cudaError_t err = launch_aggregation_rhs(
                N.data(), device_grid_.data(), rhs.data(),
                *aggregation_params_, workspace.stream());
            if (err != cudaSuccess)
                return err;
        }

        if (breakage_params_) {
            const cudaError_t err = launch_breakage_rhs(
                N.data(), device_grid_.data(), t_q_device_.data(),
                bw_q_device_.data(), rhs.data(), *breakage_params_,
                workspace.stream());
            if (err != cudaSuccess)
                return err;
        }

        if (constant_source_params_) {
            const cudaError_t err = launch_constant_source_rhs(
                source_rates_device_.data(), rhs.data(),
                *constant_source_params_, workspace.stream());
            if (err != cudaSuccess)
                return err;
        }

        return cudaSuccess;
    }

private:
    static SectionalGrid validated_grid_copy(const PBEModelConfig& config)
    {
        config.validate();

        if (config.aggregation_enabled && !config.aggregation_model)
            throw std::invalid_argument(
                "CudaPBEModel: aggregation_enabled requires aggregation_model");
        if (config.breakage_enabled && !config.breakage_model)
            throw std::invalid_argument(
                "CudaPBEModel: breakage_enabled requires breakage_model");
        if (config.constant_source_enabled && !config.constant_source_model)
            throw std::invalid_argument(
                "CudaPBEModel: constant_source_enabled requires constant_source_model");

        return *config.grid;
    }

    CudaPBEModel(SectionalGrid grid,
                 const PBEModelConfig& config,
                 int block_size,
                 cudaStream_t setup_stream)
        : grid_(std::move(grid)),
          device_grid_(grid_, setup_stream)
    {
        if (config.aggregation_model)
            aggregation_params_ = config.aggregation_model->to_params(
                grid_, block_size);

        if (config.breakage_model) {
            breakage_params_ = config.breakage_model->to_params(
                grid_, block_size);
            breakage_quadrature_ = config.breakage_model->quadrature();

            t_q_device_.resize(breakage_quadrature_.t_q.size());
            bw_q_device_.resize(breakage_quadrature_.bw_q.size());
            t_q_device_.upload(breakage_quadrature_.t_view(), setup_stream);
            bw_q_device_.upload(breakage_quadrature_.bw_view(), setup_stream);
        }

        if (config.constant_source_model) {
            constant_source_params_ = config.constant_source_model->to_params(
                grid_, block_size);
            source_rates_device_.resize(
                config.constant_source_model->rates().size());
            source_rates_device_.upload(
                config.constant_source_model->rates(), setup_stream);
        }
    }

    void validate_state_views(ConstDeviceRealView N,
                              DeviceRealView rhs) const
    {
        const std::size_t n = grid_.size();
        if (N.size() != n)
            throw std::invalid_argument("CudaPBEModel: N size mismatch");
        if (rhs.size() != n)
            throw std::invalid_argument("CudaPBEModel: rhs size mismatch");
        if (!N.data())
            throw std::invalid_argument("CudaPBEModel: N data is null");
        if (!rhs.data())
            throw std::invalid_argument("CudaPBEModel: rhs data is null");
    }

    SectionalGrid grid_;
    CudaDeviceGrid device_grid_;
    std::optional<AggregationParams> aggregation_params_;
    std::optional<BreakageParams> breakage_params_;
    std::optional<ConstantSourceParams> constant_source_params_;
    BreakageQuadrature breakage_quadrature_;
    CudaDeviceBuffer<double> t_q_device_;
    CudaDeviceBuffer<double> bw_q_device_;
    CudaDeviceBuffer<double> source_rates_device_;
};

#endif // defined(PBE_ENABLE_CUDA)

} // namespace pbe_cuda
