// =============================================================================
// tests/unit/test_cpu_rhs.cpp
//
// Regression tests for the serial CPU RHS implementation.  These tests use only
// host memory and must pass when the project is configured with ENABLE_CUDA=OFF.
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

std::vector<double> make_geometric_grid(int n, double v_min, double r)
{
    std::vector<double> x(n);
    x[0] = v_min;
    for (int i = 1; i < n; ++i)
        x[i] = x[i - 1] * r;
    return x;
}

std::vector<double> make_geometric_grid_range(int n, double v_min, double v_max)
{
    const double r = std::pow(v_max / v_min, 1.0 / (n - 1));
    return make_geometric_grid(n, v_min, r);
}

std::vector<double> make_exponential_ic(const std::vector<double>& x,
                                        double N0,
                                        double vc)
{
    const int n = static_cast<int>(x.size());
    const double r = x[1] / x[0];
    const double sqrt_r = std::sqrt(r);
    std::vector<double> N(n);

    for (int i = 0; i < n; ++i) {
        const double v_lo = x[i] / sqrt_r;
        const double v_hi = x[i] * sqrt_r;
        N[i] = N0 * (std::exp(-v_lo / vc) - std::exp(-v_hi / vc));
    }

    return N;
}

double compute_moment(const std::vector<double>& N,
                      const std::vector<double>& x,
                      int k)
{
    double M = 0.0;
    for (std::size_t i = 0; i < N.size(); ++i)
        M += N[i] * std::pow(x[i], k);
    return M;
}

double compute_M0(const std::vector<double>& N)
{
    return std::accumulate(N.begin(), N.end(), 0.0);
}

double compute_M1(const std::vector<double>& N, const std::vector<double>& x)
{
    return compute_moment(N, x, 1);
}

double compute_M2(const std::vector<double>& N, const std::vector<double>& x)
{
    return compute_moment(N, x, 2);
}

void euler_step_aggregation(std::vector<double>& N,
                            const std::vector<double>& x,
                            const pbe_cuda::AggregationParams& params,
                            double dt)
{
    std::vector<double> rhs(N.size(), 0.0);
    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                  N.data(), x.data(), rhs.data(), params),
              cudaSuccess);

    for (std::size_t i = 0; i < N.size(); ++i)
        N[i] += dt * rhs[i];
}

void euler_step_breakage(std::vector<double>& N,
                         const std::vector<double>& x,
                         const std::vector<double>& t_q,
                         const std::vector<double>& bw_q,
                         const pbe_cuda::BreakageParams& params,
                         double dt)
{
    std::vector<double> rhs(N.size(), 0.0);
    ASSERT_EQ(pbe_cuda::launch_breakage_rhs_cpu(
                  N.data(), x.data(), t_q.data(), bw_q.data(),
                  rhs.data(), params),
              cudaSuccess);

    for (std::size_t i = 0; i < N.size(); ++i)
        N[i] += dt * rhs[i];
}

void rk4_step_aggregation(std::vector<double>& N,
                          const std::vector<double>& x,
                          const pbe_cuda::AggregationParams& params,
                          double dt)
{
    const int n = static_cast<int>(N.size());
    std::vector<double> k1(n), k2(n), k3(n), k4(n), stage(n);

    auto rhs = [&](const std::vector<double>& in, std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                      in.data(), x.data(), out.data(), params),
                  cudaSuccess);
    };

    rhs(N, k1);
    for (int i = 0; i < n; ++i) stage[i] = N[i] + 0.5 * dt * k1[i];
    rhs(stage, k2);
    for (int i = 0; i < n; ++i) stage[i] = N[i] + 0.5 * dt * k2[i];
    rhs(stage, k3);
    for (int i = 0; i < n; ++i) stage[i] = N[i] + dt * k3[i];
    rhs(stage, k4);

    for (int i = 0; i < n; ++i)
        N[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

} // namespace

TEST(CpuAggregation, ConstantKernelSmoluchowskiAnalytical)
{
    constexpr int n = 256;
    constexpr double v_min = 1.0e-18;
    constexpr double r = 1.122018;
    constexpr double beta0 = 1.0e-17;
    constexpr double N0 = 1.0e14;
    constexpr double t_end = 2.0e3;
    constexpr int n_steps = 2000;

    const auto x = make_geometric_grid(n, v_min, r);
    std::vector<double> N(n, 0.0);
    N[0] = N0;

    pbe_cuda::AggregationParams params;
    params.n = n;
    params.log_x0 = std::log(x[0]);
    params.inv_log_r = 1.0 / std::log(x[1] / x[0]);
    params.kernel_type = pbe_cuda::AggregationKernel::Constant;
    params.beta0 = beta0;

    const double M1_ref = compute_M1(N, x);
    const double dt = t_end / n_steps;
    for (int step = 0; step < n_steps; ++step)
        euler_step_aggregation(N, x, params, dt);

    const double t_half = 2.0 / (beta0 * N0);
    const double M0_exact = N0 / (1.0 + t_end / t_half);
    const double err_M0 = std::abs(compute_M0(N) - M0_exact) / M0_exact;
    const double err_M1 = std::abs(compute_M1(N, x) - M1_ref) / M1_ref;

    std::printf("CPU constant aggregation errors: M0=%.6e M1=%.6e\n",
                err_M0, err_M1);
    EXPECT_LT(err_M0, 1.0e-2);
    EXPECT_LT(err_M1, 1.0e-10);
}

TEST(CpuAggregation, SumKernelGolovinAnalytical)
{
    constexpr int n = 300;
    constexpr double N0 = 1.0;
    constexpr double vc = 1.0;
    constexpr double beta0 = 1.0;
    constexpr double t_end = 0.3;
    constexpr int n_steps = 3000;

    const auto x = make_geometric_grid_range(n, 1.0e-4, 1.0e4);
    auto N = make_exponential_ic(x, N0, vc);

    pbe_cuda::AggregationParams params;
    params.n = n;
    params.log_x0 = std::log(x[0]);
    params.inv_log_r = 1.0 / std::log(x[1] / x[0]);
    params.kernel_type = pbe_cuda::AggregationKernel::Sum;
    params.beta0 = beta0;

    const double M1_ref = N0 * vc;
    const double dt = t_end / n_steps;
    for (int step = 0; step < n_steps; ++step)
        rk4_step_aggregation(N, x, params, dt);

    const double M0_exact = N0 * std::exp(-beta0 * N0 * vc * t_end);
    const double err_M0 = std::abs(compute_M0(N) - M0_exact) / M0_exact;
    const double err_M1 = std::abs(compute_M1(N, x) - M1_ref) / M1_ref;

    std::printf("CPU sum aggregation errors: M0=%.6e M1=%.6e\n",
                err_M0, err_M1);
    EXPECT_LT(err_M0, 1.0e-2);
    EXPECT_LT(err_M1, 1.0e-3);
}

TEST(CpuAggregation, ProductKernelAnalytical)
{
    constexpr int n = 300;
    constexpr double N0 = 1.0;
    constexpr double vc = 1.0;
    constexpr double beta0 = 1.0;
    constexpr double t_end = 0.3;
    constexpr int n_steps = 3000;

    const auto x = make_geometric_grid_range(n, 1.0e-4, 1.0e4);
    auto N = make_exponential_ic(x, N0, vc);

    pbe_cuda::AggregationParams params;
    params.n = n;
    params.log_x0 = std::log(x[0]);
    params.inv_log_r = 1.0 / std::log(x[1] / x[0]);
    params.kernel_type = pbe_cuda::AggregationKernel::Product;
    params.beta0 = beta0;

    const double M1_ref = N0 * vc;
    const double dt = t_end / n_steps;
    for (int step = 0; step < n_steps; ++step)
        rk4_step_aggregation(N, x, params, dt);

    const double tau = beta0 * N0 * vc * vc * t_end;
    const double M0_exact = N0 * (1.0 - 0.5 * tau);
    const double M2_exact = 2.0 * N0 * vc * vc / (1.0 - 2.0 * tau);
    const double err_M0 = std::abs(compute_M0(N) - M0_exact) / M0_exact;
    const double err_M1 = std::abs(compute_M1(N, x) - M1_ref) / M1_ref;
    const double err_M2 = std::abs(compute_M2(N, x) - M2_exact) / M2_exact;

    std::printf("CPU product aggregation errors: M0=%.6e M1=%.6e M2=%.6e\n",
                err_M0, err_M1, err_M2);
    EXPECT_LT(err_M0, 1.0e-2);
    EXPECT_LT(err_M1, 1.0e-3);
    EXPECT_LT(err_M2, 5.0e-2);
}

TEST(CpuAggregation, BrownianAndShearVariantsConserveVolume)
{
    constexpr int n = 64;
    const auto x = make_geometric_grid(n, 1.0e-18, 1.122018);

    const pbe_cuda::AggregationKernel kernels[] = {
        pbe_cuda::AggregationKernel::BrownianContinuum,
        pbe_cuda::AggregationKernel::BrownianFreeMolecular,
        pbe_cuda::AggregationKernel::Shear,
        pbe_cuda::AggregationKernel::BrownianContinuumShear,
        pbe_cuda::AggregationKernel::BrownianFreeMolecularShear
    };

    for (auto kernel : kernels) {
        std::vector<double> N(n);
        for (int i = 0; i < n / 3; ++i)
            N[i] = 1.0e9 * std::exp(-static_cast<double>(i) / 12.0);

        std::vector<double> rhs(n, 0.0);
        pbe_cuda::AggregationParams params;
        params.n = n;
        params.log_x0 = std::log(x[0]);
        params.inv_log_r = 1.0 / std::log(x[1] / x[0]);
        params.kernel_type = kernel;
        params.beta_bc = 1.2e-8;
        params.beta_bfm = 2.5e-9;
        params.beta_sh = 7.5e-10;

        ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                      N.data(), x.data(), rhs.data(), params),
                  cudaSuccess);

        double dM1 = 0.0;
        double M1 = 0.0;
        for (int i = 0; i < n; ++i) {
            dM1 += rhs[i] * x[i];
            M1 += N[i] * x[i];
        }
        EXPECT_NEAR(dM1, 0.0, 1.0e-6 * std::abs(M1))
            << "kernel=" << static_cast<int>(kernel);
    }
}

TEST(CpuAggregation, BrownianFreeMolecularSelfCollisionReference)
{
    const std::vector<double> x = {1.0, 2.0, 4.0};
    const std::vector<double> N = {3.0, 0.0, 0.0};
    std::vector<double> rhs(3, 0.0);

    constexpr double beta_bfm = 2.5;
    pbe_cuda::AggregationParams params;
    params.n = static_cast<int>(x.size());
    params.log_x0 = std::log(x[0]);
    params.inv_log_r = 1.0 / std::log(x[1] / x[0]);
    params.kernel_type = pbe_cuda::AggregationKernel::BrownianFreeMolecular;
    params.beta_bfm = beta_bfm;

    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                  N.data(), x.data(), rhs.data(), params),
              cudaSuccess);

    const double ri = std::cbrt(x[0]);
    const double s = ri + ri;
    const double beta = beta_bfm * s * s * std::sqrt(1.0 / x[0] + 1.0 / x[0]);
    const double rate = 0.5 * beta * N[0] * N[0];

    EXPECT_DOUBLE_EQ(rhs[0], -2.0 * rate);
    EXPECT_DOUBLE_EQ(rhs[1], rate);
    EXPECT_DOUBLE_EQ(rhs[2], 0.0);
}

TEST(CpuBreakage, LinearSymmetricBinaryAnalytical)
{
    constexpr int n = 512;
    constexpr double v_min = 1.0e-18;
    constexpr double r = 1.122018;
    constexpr int ic_bin = 384;
    constexpr double S0 = 1.0e-3;
    constexpr double N0 = 1.0e14;
    constexpr double t_end = 100.0;
    constexpr int n_steps = 10000;

    const auto x = make_geometric_grid(n, v_min, r);
    std::vector<double> N(n, 0.0);
    N[ic_bin] = N0;

    const std::vector<double> t_q = {0.5};
    const std::vector<double> bw_q = {2.0};

    pbe_cuda::BreakageParams params;
    params.n = n;
    params.n_quad = 1;
    params.selection = pbe_cuda::BreakageSelection::Linear;
    params.S0 = S0;
    params.v_ref = x[ic_bin];

    const double M1_ref = compute_M1(N, x);
    const double dt = t_end / n_steps;
    for (int step = 0; step < n_steps; ++step)
        euler_step_breakage(N, x, t_q, bw_q, params, dt);

    const double M0_exact = N0 * (1.0 + S0 * t_end);
    const double err_M0 = std::abs(compute_M0(N) - M0_exact) / M0_exact;
    const double err_M1 = std::abs(compute_M1(N, x) - M1_ref) / M1_ref;

    std::printf("CPU symmetric breakage errors: M0=%.6e M1=%.6e\n",
                err_M0, err_M1);
    EXPECT_LT(err_M0, 1.0e-3);
    EXPECT_LT(err_M1, 1.0e-10);
}

TEST(CpuBreakage, ZiffMcGradyLinearUniformAnalytical)
{
    constexpr int n = 200;
    constexpr double N0 = 1.0;
    constexpr double vc = 1.0;
    constexpr double S0 = 1.0;
    constexpr double t_end = 5.0;
    constexpr int n_steps = 5000;

    const double gl8_t[] = {
        0.019855071751232, 0.101666761293187, 0.237233795041836,
        0.408282678752175, 0.591717321247825, 0.762766204958164,
        0.898333238706813, 0.980144928248768
    };
    const double gl8_w[] = {
        0.050614268145188, 0.111190517226687, 0.156853322938943,
        0.181341891689181, 0.181341891689181, 0.156853322938943,
        0.111190517226687, 0.050614268145188
    };

    const auto x = make_geometric_grid_range(n, 1.0e-4, 1.0e1);
    auto N = make_exponential_ic(x, N0, vc);
    std::vector<double> t_q(8), bw_q(8);
    for (int q = 0; q < 8; ++q) {
        t_q[q] = gl8_t[q];
        bw_q[q] = 2.0 * gl8_w[q];
    }

    pbe_cuda::BreakageParams params;
    params.n = n;
    params.n_quad = 8;
    params.selection = pbe_cuda::BreakageSelection::Linear;
    params.S0 = S0;
    params.v_ref = vc;

    const double M1_ref = N0 * vc;
    const double dt = t_end / n_steps;
    for (int step = 0; step < n_steps; ++step)
        euler_step_breakage(N, x, t_q, bw_q, params, dt);

    const double M0_exact = N0 * (1.0 + S0 * t_end);
    const double err_M0 = std::abs(compute_M0(N) - M0_exact) / M0_exact;
    const double err_M1 = std::abs(compute_M1(N, x) - M1_ref) / M1_ref;

    std::printf("CPU Ziff-McGrady breakage errors: M0=%.6e M1=%.6e\n",
                err_M0, err_M1);
    EXPECT_LT(err_M0, 1.0e-2);
    EXPECT_LT(err_M1, 1.0e-3);
}

TEST(CpuRhs, CpuOnlyLaunchAliasesCallSerialImplementation)
{
#if !defined(PBE_ENABLE_CUDA)
    const std::vector<double> x = {1.0, 2.0, 4.0};
    const std::vector<double> N = {3.0, 0.0, 0.0};
    std::vector<double> rhs_alias(3, 0.0);
    std::vector<double> rhs_cpu(3, 0.0);

    pbe_cuda::AggregationParams params;
    params.n = 3;
    params.log_x0 = std::log(x[0]);
    params.inv_log_r = 1.0 / std::log(x[1] / x[0]);
    params.kernel_type = pbe_cuda::AggregationKernel::Constant;
    params.beta0 = 2.0;

    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                  N.data(), x.data(), rhs_cpu.data(), params),
              cudaSuccess);
    ASSERT_EQ(pbe_cuda::launch_aggregation_rhs(
                  N.data(), x.data(), rhs_alias.data(), params),
              cudaSuccess);

    for (std::size_t i = 0; i < rhs_cpu.size(); ++i)
        EXPECT_DOUBLE_EQ(rhs_alias[i], rhs_cpu[i]);
#else
    GTEST_SKIP() << "Default launch path expects device pointers when CUDA is enabled";
#endif
}
