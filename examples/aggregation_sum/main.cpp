// =============================================================================
// examples/aggregation_sum_kernel/main.cpp
//
// Validation: pure aggregation with sum kernel β(u,v) = β₀(u+v)
// against the Golovin analytical solution (exponential IC).
//
// Analytical moments:
//   τ    = β₀ M₁(0) t = β₀ N₀ vc t
//   M₀(t) = N₀ exp(-τ/2)
//   M₁(t) = N₀ vc             (conserved)
//   M₂(t) = 2 N₀ vc² exp(τ)
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>
#include "pbe_examples_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cfg {
    constexpr int    n        = 300;
    constexpr double v_min    = 1.0e-4;
    constexpr double v_max    = 1.0e4;
    constexpr double N0       = 1.0;
    constexpr double vc       = 1.0;
    constexpr double beta0    = 1.0;
    constexpr double t_end    = 0.3;
    constexpr int    n_steps  = 3000;
    constexpr int    n_print  = 300;
}

int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Aggregation example: sum kernel (Golovin)\n");
    std::printf("=============================================================\n\n");

    cudaDeviceProp prop{};
    PBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Grid
    const auto x_host = make_geometric_grid_range(cfg::n, cfg::v_min, cfg::v_max);
    const double r       = x_host[1] / x_host[0];
    const double log_x0  = std::log(x_host[0]);
    const double inv_logr= 1.0 / std::log(r);

    // Exponential IC
    auto N_host = make_exponential_ic(x_host, cfg::N0, cfg::vc);

    // Analytical solution
    SumKernelAnalytical ana{cfg::N0, cfg::vc, cfg::beta0};

    // Device arrays
    DeviceArray<double> d_x(cfg::n), d_N(cfg::n), d_rhs(cfg::n);
    d_x.upload(x_host);
    d_N.upload(N_host);

    pbe_cuda::AggregationParams p;
    p.n           = cfg::n;
    p.log_x0      = log_x0;
    p.inv_log_r   = inv_logr;
    p.kernel_type = pbe_cuda::AggregationKernel::Sum;
    p.beta0       = cfg::beta0;
    p.block_size  = 256;

    const double dt      = cfg::t_end / cfg::n_steps;
    const double M1_ref  = ana.M1(0.0);

    std::printf("Grid    : n=%d, v=[%.1e, %.1e], r=%.4f\n",
                cfg::n, cfg::v_min, cfg::v_max, r);
    std::printf("Physics : β₀=%.2f, N₀=%.2f, vc=%.2f\n",
                cfg::beta0, cfg::N0, cfg::vc);
    std::printf("Time    : t_end=%.2f, dt=%.2e, steps=%d (RK4)\n\n",
                cfg::t_end, dt, cfg::n_steps);

    std::printf("Validating M₀ (Golovin: N₀exp(-β₀N₀vc·t)) and M₁ conservation:\n\n");
    print_table_header("t", "M0 num", "M0 exact", "Err M0", "M1 num", "Err M1");


    // ---- RK4 time integration -------------------------------------------
    // Requires 4 RHS evaluations per step.
    // k1 = R(N^n)
    // k2 = R(N^n + dt/2 * k1)
    // k3 = R(N^n + dt/2 * k2)
    // k4 = R(N^n + dt   * k3)
    // N^(n+1) = N^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    // Extra device arrays for RK4 stages
    DeviceArray<double> d_N_stage(cfg::n);   // N^n + dt/2 * ki
    DeviceArray<double> d_k1(cfg::n), d_k2(cfg::n),
                        d_k3(cfg::n), d_k4(cfg::n);

    // Host arrays for RK4 arithmetic (done on host for clarity)
    std::vector<double> k1(cfg::n), k2(cfg::n),
                        k3(cfg::n), k4(cfg::n);
    std::vector<double> N_stage(cfg::n);

    double t = 0.0;

    // Helper lambda: evaluate RHS into d_rhs, download to k_host
    auto eval_rhs = [&](DeviceArray<double>& d_N_in,
                         std::vector<double>& k_host) {
        d_rhs.zero();
        cudaError_t err = pbe_cuda::launch_aggregation_rhs(
            d_N_in.get(), d_x.get(), d_rhs.get(), p);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        PBE_CUDA_CHECK(cudaDeviceSynchronize());
        d_rhs.download(k_host);
    };

    for (int step = 0; step <= cfg::n_steps; ++step) {

        if (step % cfg::n_print == 0) {
            d_N.download(N_host);
            double M0_num = compute_M0(N_host);
            double M1_num = compute_M1(N_host, x_host);
            double M0_ana = ana.M0(t);
            double err_M0 = std::abs(M0_num - M0_ana) / M0_ana;
            double err_M1 = std::abs(M1_num - M1_ref) / M1_ref;
            print_table_row(t, M0_num, M0_ana, err_M0, M1_num, err_M1);
        }

        if (step == cfg::n_steps) break;

        // Download current N
        d_N.download(N_host);

        // k1 = R(N^n)
        eval_rhs(d_N, k1);

        // k2 = R(N^n + dt/2 * k1)
        for (int i = 0; i < cfg::n; ++i)
            N_stage[i] = N_host[i] + 0.5 * dt * k1[i];
        d_N_stage.upload(N_stage);
        eval_rhs(d_N_stage, k2);

        // k3 = R(N^n + dt/2 * k2)
        for (int i = 0; i < cfg::n; ++i)
            N_stage[i] = N_host[i] + 0.5 * dt * k2[i];
        d_N_stage.upload(N_stage);
        eval_rhs(d_N_stage, k3);

        // k4 = R(N^n + dt * k3)
        for (int i = 0; i < cfg::n; ++i)
            N_stage[i] = N_host[i] + dt * k3[i];
        d_N_stage.upload(N_stage);
        eval_rhs(d_N_stage, k4);

        // N^(n+1) = N^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for (int i = 0; i < cfg::n; ++i)
            N_host[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        d_N.upload(N_host);
        t += dt;
    }

    // Final check
    d_N.download(N_host);
    double M0_num = compute_M0(N_host);
    double M1_num = compute_M1(N_host, x_host);
    double err_M0 = std::abs(M0_num - ana.M0(cfg::t_end)) / ana.M0(cfg::t_end);
    double err_M1 = std::abs(M1_num - M1_ref) / M1_ref;

    print_separator();
    std::printf("Final t = %.2f\n", cfg::t_end);
    std::printf("  M0 error : %.4e  %s\n", err_M0, err_M0 < 0.01 ? "PASS" : "WARN");
    std::printf("  M1 error : %.4e  %s\n", err_M1, err_M1 < 1.0e-3 ? "PASS" : "WARN");
    print_separator();

    return EXIT_SUCCESS;
}