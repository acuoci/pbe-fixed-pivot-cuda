// =============================================================================
// tests/unit/test_model_config.cpp
//
// Tests for the early model-configuration/evaluation-condition split.
// =============================================================================

#include <pbe_cuda/model_config.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

TEST(EvaluationContext, EmptyContextIsValid)
{
    const pbe_cuda::EvaluationContext ctx;

    EXPECT_TRUE(ctx.empty());
    EXPECT_NO_THROW(ctx.validate());
}

TEST(EvaluationContext, AcceptsFiniteLocalConditions)
{
    pbe_cuda::EvaluationContext ctx;
    ctx.temperature = 300.0;
    ctx.pressure = 101325.0;
    ctx.viscosity = 1.8e-5;
    ctx.density = 1.2;
    ctx.turbulence_dissipation = 0.0;
    ctx.shear_rate = 10.0;

    EXPECT_FALSE(ctx.empty());
    EXPECT_NO_THROW(ctx.validate());
}

TEST(EvaluationContext, RejectsInvalidLocalConditions)
{
    pbe_cuda::EvaluationContext negative_temperature;
    negative_temperature.temperature = -1.0;
    EXPECT_THROW(negative_temperature.validate(), std::invalid_argument);

    pbe_cuda::EvaluationContext nan_pressure;
    nan_pressure.pressure = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(nan_pressure.validate(), std::invalid_argument);

    pbe_cuda::EvaluationContext negative_shear;
    negative_shear.shear_rate = -0.1;
    EXPECT_THROW(negative_shear.validate(), std::invalid_argument);
}

TEST(PBEModelConfig, RequiresGridAndEnabledProcess)
{
    pbe_cuda::PBEModelConfig missing_grid;
    missing_grid.aggregation_enabled = true;
    EXPECT_FALSE(missing_grid.has_grid());
    EXPECT_TRUE(missing_grid.has_enabled_process());
    EXPECT_THROW(missing_grid.validate(), std::invalid_argument);

    pbe_cuda::PBEModelConfig missing_process;
    missing_process.grid = pbe_cuda::SectionalGrid::geometric(4, 1.0, 2.0);
    EXPECT_TRUE(missing_process.has_grid());
    EXPECT_FALSE(missing_process.has_enabled_process());
    EXPECT_THROW(missing_process.validate(), std::invalid_argument);
}

TEST(PBEModelConfig, ValidatesMinimalCurrentModelConfiguration)
{
    pbe_cuda::PBEModelConfig config;
    config.grid = pbe_cuda::SectionalGrid::geometric(4, 1.0, 2.0);
    config.aggregation_enabled = true;

    EXPECT_TRUE(config.has_grid());
    EXPECT_TRUE(config.has_enabled_process());
    EXPECT_NO_THROW(config.validate());
}
