// =============================================================================
// aggregation_model.hpp  --  Strongly typed aggregation model configuration
//
// Converts named aggregation model choices into the existing low-level
// AggregationParams once during setup. No runtime model dispatch is introduced
// inside RHS pair loops.
// =============================================================================

#pragma once

#include "pbe_cuda/aggregation.cuh"
#include "pbe_cuda/sectional_grid.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>

namespace pbe_cuda {

struct ConstantAggregation {
    double beta0 = 1.0;
};

struct SumAggregation {
    double beta0 = 1.0;
};

struct ProductAggregation {
    double beta0 = 1.0;
};

struct BrownianContinuumAggregation {
    double beta_bc = 0.0;
};

struct BrownianFreeMolecularAggregation {
    double beta_bfm = 0.0;
};

struct ShearAggregation {
    double beta_sh = 0.0;
};

struct BrownianContinuumShearAggregation {
    double beta_bc = 0.0;
    double beta_sh = 0.0;
};

struct BrownianFreeMolecularShearAggregation {
    double beta_bfm = 0.0;
    double beta_sh = 0.0;
};

class AggregationModel {
public:
    using Variant = std::variant<
        ConstantAggregation,
        SumAggregation,
        ProductAggregation,
        BrownianContinuumAggregation,
        BrownianFreeMolecularAggregation,
        ShearAggregation,
        BrownianContinuumShearAggregation,
        BrownianFreeMolecularShearAggregation>;

    static AggregationModel constant(double beta0)
    {
        return AggregationModel(ConstantAggregation{beta0});
    }

    static AggregationModel sum(double beta0)
    {
        return AggregationModel(SumAggregation{beta0});
    }

    static AggregationModel product(double beta0)
    {
        return AggregationModel(ProductAggregation{beta0});
    }

    static AggregationModel brownian_continuum(double beta_bc)
    {
        return AggregationModel(BrownianContinuumAggregation{beta_bc});
    }

    static AggregationModel brownian_free_molecular(double beta_bfm)
    {
        return AggregationModel(BrownianFreeMolecularAggregation{beta_bfm});
    }

    static AggregationModel shear(double beta_sh)
    {
        return AggregationModel(ShearAggregation{beta_sh});
    }

    static AggregationModel brownian_continuum_shear(double beta_bc,
                                                    double beta_sh)
    {
        return AggregationModel(BrownianContinuumShearAggregation{
            beta_bc, beta_sh});
    }

    static AggregationModel brownian_free_molecular_shear(double beta_bfm,
                                                         double beta_sh)
    {
        return AggregationModel(BrownianFreeMolecularShearAggregation{
            beta_bfm, beta_sh});
    }

    explicit AggregationModel(Variant model)
        : model_(model)
    {
        validate();
    }

    [[nodiscard]] const Variant& variant() const noexcept { return model_; }

    [[nodiscard]] AggregationKernel kernel_type() const
    {
        AggregationParams params;
        fill_params(params);
        return params.kernel_type;
    }

    [[nodiscard]] AggregationParams to_params(const SectionalGrid& grid,
                                              int block_size = 256) const
    {
        if (block_size <= 0 || (block_size & (block_size - 1)) != 0)
            throw std::invalid_argument(
                "AggregationModel: block_size must be a positive power of two");

        AggregationParams params;
        params.n = grid.n();
        params.log_x0 = grid.log_x0();
        params.inv_log_r = grid.inv_log_r();
        params.block_size = block_size;

        fill_params(params);
        return params;
    }

    void validate() const
    {
        std::visit([](const auto& model) { validate_model(model); }, model_);
    }

private:
    static void validate_coefficient(double value, const char* name)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string("AggregationModel: ") +
                                        name + " must be finite");
        if (value < 0.0)
            throw std::invalid_argument(std::string("AggregationModel: ") +
                                        name + " must be nonnegative");
    }

    static void validate_model(const ConstantAggregation& model)
    {
        validate_coefficient(model.beta0, "beta0");
    }

    static void validate_model(const SumAggregation& model)
    {
        validate_coefficient(model.beta0, "beta0");
    }

    static void validate_model(const ProductAggregation& model)
    {
        validate_coefficient(model.beta0, "beta0");
    }

    static void validate_model(const BrownianContinuumAggregation& model)
    {
        validate_coefficient(model.beta_bc, "beta_bc");
    }

    static void validate_model(const BrownianFreeMolecularAggregation& model)
    {
        validate_coefficient(model.beta_bfm, "beta_bfm");
    }

    static void validate_model(const ShearAggregation& model)
    {
        validate_coefficient(model.beta_sh, "beta_sh");
    }

    static void validate_model(const BrownianContinuumShearAggregation& model)
    {
        validate_coefficient(model.beta_bc, "beta_bc");
        validate_coefficient(model.beta_sh, "beta_sh");
    }

    static void validate_model(const BrownianFreeMolecularShearAggregation& model)
    {
        validate_coefficient(model.beta_bfm, "beta_bfm");
        validate_coefficient(model.beta_sh, "beta_sh");
    }

    static void fill_one(AggregationParams& params,
                         const ConstantAggregation& model)
    {
        params.kernel_type = AggregationKernel::Constant;
        params.beta0 = model.beta0;
    }

    static void fill_one(AggregationParams& params, const SumAggregation& model)
    {
        params.kernel_type = AggregationKernel::Sum;
        params.beta0 = model.beta0;
    }

    static void fill_one(AggregationParams& params,
                         const ProductAggregation& model)
    {
        params.kernel_type = AggregationKernel::Product;
        params.beta0 = model.beta0;
    }

    static void fill_one(AggregationParams& params,
                         const BrownianContinuumAggregation& model)
    {
        params.kernel_type = AggregationKernel::BrownianContinuum;
        params.beta_bc = model.beta_bc;
    }

    static void fill_one(AggregationParams& params,
                         const BrownianFreeMolecularAggregation& model)
    {
        params.kernel_type = AggregationKernel::BrownianFreeMolecular;
        params.beta_bfm = model.beta_bfm;
    }

    static void fill_one(AggregationParams& params, const ShearAggregation& model)
    {
        params.kernel_type = AggregationKernel::Shear;
        params.beta_sh = model.beta_sh;
    }

    static void fill_one(AggregationParams& params,
                         const BrownianContinuumShearAggregation& model)
    {
        params.kernel_type = AggregationKernel::BrownianContinuumShear;
        params.beta_bc = model.beta_bc;
        params.beta_sh = model.beta_sh;
    }

    static void fill_one(AggregationParams& params,
                         const BrownianFreeMolecularShearAggregation& model)
    {
        params.kernel_type = AggregationKernel::BrownianFreeMolecularShear;
        params.beta_bfm = model.beta_bfm;
        params.beta_sh = model.beta_sh;
    }

    void fill_params(AggregationParams& params) const
    {
        std::visit([&params](const auto& model) { fill_one(params, model); },
                   model_);
    }

    Variant model_;
};

} // namespace pbe_cuda
