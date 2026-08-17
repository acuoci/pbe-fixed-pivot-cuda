// =============================================================================
// model_config.hpp  --  Early model configuration and evaluation context types
//
// These additive types introduce the architectural distinction between mostly
// immutable model configuration and per-call local physical conditions.
// =============================================================================

#pragma once

#include "pbe_cuda/sectional_grid.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace pbe_cuda {

// Per-call/local physical conditions. Current kernels may ignore this context,
// but future condition-dependent kernels can consume it without rebuilding the
// model configuration for every CFD cell or time step.
struct EvaluationContext {
    std::optional<double> temperature;
    std::optional<double> pressure;
    std::optional<double> viscosity;
    std::optional<double> density;
    std::optional<double> turbulence_dissipation;
    std::optional<double> shear_rate;

    [[nodiscard]] bool empty() const noexcept
    {
        return !temperature && !pressure && !viscosity && !density &&
               !turbulence_dissipation && !shear_rate;
    }

    void validate() const
    {
        validate_positive(temperature, "temperature");
        validate_positive(pressure, "pressure");
        validate_positive(viscosity, "viscosity");
        validate_positive(density, "density");
        validate_nonnegative(turbulence_dissipation, "turbulence_dissipation");
        validate_nonnegative(shear_rate, "shear_rate");
    }

private:
    static void validate_finite(double value, const char* name)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string("EvaluationContext: ") +
                                        name + " must be finite");
    }

    static void validate_positive(const std::optional<double>& value,
                                  const char* name)
    {
        if (!value) return;
        validate_finite(*value, name);
        if (*value <= 0.0)
            throw std::invalid_argument(std::string("EvaluationContext: ") +
                                        name + " must be positive");
    }

    static void validate_nonnegative(const std::optional<double>& value,
                                     const char* name)
    {
        if (!value) return;
        validate_finite(*value, name);
        if (*value < 0.0)
            throw std::invalid_argument(std::string("EvaluationContext: ") +
                                        name + " must be nonnegative");
    }
};

// Early top-level configuration shell. Detailed process submodel configuration
// is intentionally left to later migration phases.
struct PBEModelConfig {
    std::optional<SectionalGrid> grid;
    bool aggregation_enabled = false;
    bool breakage_enabled = false;

    [[nodiscard]] bool has_grid() const noexcept { return grid.has_value(); }

    [[nodiscard]] bool has_enabled_process() const noexcept
    {
        return aggregation_enabled || breakage_enabled;
    }

    void validate() const
    {
        if (!grid)
            throw std::invalid_argument("PBEModelConfig: grid is required");
        if (!has_enabled_process())
            throw std::invalid_argument(
                "PBEModelConfig: at least one process must be enabled");
    }
};

} // namespace pbe_cuda
