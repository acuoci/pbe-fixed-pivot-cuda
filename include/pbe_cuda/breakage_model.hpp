// =============================================================================
// breakage_model.hpp  --  Strongly typed breakage model configuration
//
// Separates selection-function configuration from daughter-distribution
// quadrature and maps once to the existing low-level BreakageParams.
// =============================================================================

#pragma once

#include "pbe_cuda/array_view.hpp"
#include "pbe_cuda/breakage.cuh"
#include "pbe_cuda/sectional_grid.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pbe_cuda {

struct ConstantBreakageSelection {
    double S0 = 1.0;
};

struct LinearBreakageSelection {
    double S0 = 1.0;
    double v_ref = 1.0;
};

struct PowerLawBreakageSelection {
    double S0 = 1.0;
    double v_ref = 1.0;
    double alpha = 1.0;
};

struct ThresholdBreakageSelection {
    double S0 = 1.0;
    double v_min = 0.0;
};

struct SymmetricBinaryDaughter {};

struct ErosionDaughter {
    double epsilon = 0.05;
};

struct UniformDaughter {
    int quadrature_order = 8;
};

struct UserQuadratureDaughter {
    std::vector<double> t_q;
    std::vector<double> bw_q;
};

struct BreakageQuadrature {
    std::vector<double> t_q;
    std::vector<double> bw_q;

    [[nodiscard]] int size() const noexcept
    {
        return static_cast<int>(t_q.size());
    }

    [[nodiscard]] ConstRealView t_view() const noexcept
    {
        return ConstRealView(t_q.data(), t_q.size());
    }

    [[nodiscard]] ConstRealView bw_view() const noexcept
    {
        return ConstRealView(bw_q.data(), bw_q.size());
    }
};

class BreakageModel {
public:
    using Selection = std::variant<
        ConstantBreakageSelection,
        LinearBreakageSelection,
        PowerLawBreakageSelection,
        ThresholdBreakageSelection>;

    using Daughter = std::variant<
        SymmetricBinaryDaughter,
        ErosionDaughter,
        UniformDaughter,
        UserQuadratureDaughter>;

    static BreakageModel constant_symmetric(double S0)
    {
        return BreakageModel(ConstantBreakageSelection{S0},
                             SymmetricBinaryDaughter{});
    }

    static BreakageModel linear_symmetric(double S0, double v_ref)
    {
        return BreakageModel(LinearBreakageSelection{S0, v_ref},
                             SymmetricBinaryDaughter{});
    }

    static BreakageModel linear_uniform(double S0,
                                        double v_ref,
                                        int quadrature_order)
    {
        return BreakageModel(LinearBreakageSelection{S0, v_ref},
                             UniformDaughter{quadrature_order});
    }

    static BreakageModel threshold_erosion(double S0,
                                           double v_min,
                                           double epsilon)
    {
        return BreakageModel(ThresholdBreakageSelection{S0, v_min},
                             ErosionDaughter{epsilon});
    }

    static BreakageModel power_law(double S0,
                                   double v_ref,
                                   double alpha,
                                   Daughter daughter)
    {
        return BreakageModel(PowerLawBreakageSelection{S0, v_ref, alpha},
                             std::move(daughter));
    }

    BreakageModel(Selection selection, Daughter daughter)
        : selection_(std::move(selection)),
          daughter_(std::move(daughter)),
          quadrature_(make_quadrature(daughter_))
    {
        validate();
    }

    [[nodiscard]] const Selection& selection() const noexcept
    {
        return selection_;
    }

    [[nodiscard]] const Daughter& daughter() const noexcept
    {
        return daughter_;
    }

    [[nodiscard]] const BreakageQuadrature& quadrature() const noexcept
    {
        return quadrature_;
    }

    [[nodiscard]] BreakageSelection selection_type() const
    {
        BreakageParams params;
        fill_selection(params);
        return params.selection;
    }

    [[nodiscard]] BreakageParams to_params(const SectionalGrid& grid,
                                           int block_size = 256) const
    {
        if (block_size <= 0 || (block_size & (block_size - 1)) != 0)
            throw std::invalid_argument(
                "BreakageModel: block_size must be a positive power of two");

        BreakageParams params;
        params.n = grid.n();
        params.n_quad = quadrature_.size();
        params.block_size = block_size;
        fill_selection(params);
        return params;
    }

    void validate() const
    {
        std::visit([](const auto& model) { validate_selection(model); },
                   selection_);
        std::visit([](const auto& model) { validate_daughter(model); },
                   daughter_);
        validate_quadrature(quadrature_);
    }

private:
    static void validate_finite(double value, const char* owner, const char* name)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string(owner) + ": " + name +
                                        " must be finite");
    }

    static void validate_nonnegative(double value,
                                     const char* owner,
                                     const char* name)
    {
        validate_finite(value, owner, name);
        if (value < 0.0)
            throw std::invalid_argument(std::string(owner) + ": " + name +
                                        " must be nonnegative");
    }

    static void validate_positive(double value,
                                  const char* owner,
                                  const char* name)
    {
        validate_finite(value, owner, name);
        if (value <= 0.0)
            throw std::invalid_argument(std::string(owner) + ": " + name +
                                        " must be positive");
    }

    static void validate_selection(const ConstantBreakageSelection& model)
    {
        validate_nonnegative(model.S0, "BreakageModel", "S0");
    }

    static void validate_selection(const LinearBreakageSelection& model)
    {
        validate_nonnegative(model.S0, "BreakageModel", "S0");
        validate_positive(model.v_ref, "BreakageModel", "v_ref");
    }

    static void validate_selection(const PowerLawBreakageSelection& model)
    {
        validate_nonnegative(model.S0, "BreakageModel", "S0");
        validate_positive(model.v_ref, "BreakageModel", "v_ref");
        validate_finite(model.alpha, "BreakageModel", "alpha");
    }

    static void validate_selection(const ThresholdBreakageSelection& model)
    {
        validate_nonnegative(model.S0, "BreakageModel", "S0");
        validate_nonnegative(model.v_min, "BreakageModel", "v_min");
    }

    static void validate_daughter(const SymmetricBinaryDaughter&) {}

    static void validate_daughter(const ErosionDaughter& model)
    {
        validate_positive(model.epsilon, "BreakageModel", "epsilon");
        if (model.epsilon >= 1.0)
            throw std::invalid_argument(
                "BreakageModel: epsilon must be less than one");
    }

    static void validate_daughter(const UniformDaughter& model)
    {
        if (model.quadrature_order <= 0)
            throw std::invalid_argument(
                "BreakageModel: quadrature_order must be positive");
    }

    static void validate_daughter(const UserQuadratureDaughter& model)
    {
        validate_quadrature(BreakageQuadrature{model.t_q, model.bw_q});
    }

    static BreakageQuadrature make_quadrature(const Daughter& daughter)
    {
        return std::visit([](const auto& model) { return make_quadrature(model); },
                          daughter);
    }

    static BreakageQuadrature make_quadrature(const SymmetricBinaryDaughter&)
    {
        return BreakageQuadrature{{0.5}, {2.0}};
    }

    static BreakageQuadrature make_quadrature(const ErosionDaughter& model)
    {
        return BreakageQuadrature{{1.0 - model.epsilon, model.epsilon},
                                  {1.0, 1.0}};
    }

    static BreakageQuadrature make_quadrature(const UniformDaughter& model)
    {
        BreakageQuadrature quadrature;
        quadrature.t_q.resize(static_cast<std::size_t>(model.quadrature_order));
        quadrature.bw_q.resize(static_cast<std::size_t>(model.quadrature_order));

        fill_gauss_legendre_01(model.quadrature_order,
                               quadrature.t_q,
                               quadrature.bw_q);
        for (double& weight : quadrature.bw_q)
            weight *= 2.0;
        return quadrature;
    }

    static BreakageQuadrature make_quadrature(const UserQuadratureDaughter& model)
    {
        return BreakageQuadrature{model.t_q, model.bw_q};
    }

    static void fill_gauss_legendre_01(int order,
                                       std::vector<double>& nodes,
                                       std::vector<double>& weights)
    {
        constexpr double pi = 3.141592653589793238462643383279502884;
        constexpr double tol = 1.0e-14;
        const int m = (order + 1) / 2;

        for (int i = 0; i < m; ++i) {
            double z = std::cos(pi * (static_cast<double>(i) + 0.75) /
                                (static_cast<double>(order) + 0.5));
            double p1 = 0.0;
            double p2 = 0.0;
            double pp = 0.0;

            for (;;) {
                p1 = 1.0;
                p2 = 0.0;
                for (int j = 1; j <= order; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) /
                         static_cast<double>(j);
                }
                pp = order * (z * p1 - p2) / (z * z - 1.0);
                const double z_old = z;
                z = z_old - p1 / pp;
                if (std::fabs(z - z_old) <= tol)
                    break;
            }

            const double node_lo = 0.5 * (1.0 - z);
            const double node_hi = 0.5 * (1.0 + z);
            const double weight = 1.0 / ((1.0 - z * z) * pp * pp);

            nodes[static_cast<std::size_t>(i)] = node_lo;
            weights[static_cast<std::size_t>(i)] = weight;
            nodes[static_cast<std::size_t>(order - 1 - i)] = node_hi;
            weights[static_cast<std::size_t>(order - 1 - i)] = weight;
        }
    }

    static void validate_quadrature(const BreakageQuadrature& quadrature)
    {
        if (quadrature.t_q.empty())
            throw std::invalid_argument("BreakageModel: quadrature is empty");
        if (quadrature.t_q.size() != quadrature.bw_q.size())
            throw std::invalid_argument(
                "BreakageModel: quadrature arrays must have equal length");
        if (quadrature.t_q.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument(
                "BreakageModel: quadrature size exceeds int range");

        for (std::size_t i = 0; i < quadrature.t_q.size(); ++i) {
            validate_finite(quadrature.t_q[i], "BreakageModel", "t_q");
            validate_finite(quadrature.bw_q[i], "BreakageModel", "bw_q");
            if (quadrature.t_q[i] <= 0.0 || quadrature.t_q[i] >= 1.0)
                throw std::invalid_argument(
                    "BreakageModel: t_q values must lie in (0, 1)");
            if (quadrature.bw_q[i] < 0.0)
                throw std::invalid_argument(
                    "BreakageModel: bw_q values must be nonnegative");
        }
    }

    void fill_selection(BreakageParams& params) const
    {
        std::visit([&params](const auto& model) { fill_one(params, model); },
                   selection_);
    }

    static void fill_one(BreakageParams& params,
                         const ConstantBreakageSelection& model)
    {
        params.selection = BreakageSelection::Constant;
        params.S0 = model.S0;
    }

    static void fill_one(BreakageParams& params,
                         const LinearBreakageSelection& model)
    {
        params.selection = BreakageSelection::Linear;
        params.S0 = model.S0;
        params.v_ref = model.v_ref;
    }

    static void fill_one(BreakageParams& params,
                         const PowerLawBreakageSelection& model)
    {
        params.selection = BreakageSelection::PowerLaw;
        params.S0 = model.S0;
        params.v_ref = model.v_ref;
        params.alpha = model.alpha;
    }

    static void fill_one(BreakageParams& params,
                         const ThresholdBreakageSelection& model)
    {
        params.selection = BreakageSelection::Threshold;
        params.S0 = model.S0;
        params.v_min = model.v_min;
    }

    Selection selection_;
    Daughter daughter_;
    BreakageQuadrature quadrature_;
};

} // namespace pbe_cuda
