// =============================================================================
// examples/aggregation_product_kernel/main.cpp
//
// Validation: pure aggregation with product kernel β(u,v) = β₀ u v
// against the analytical Smoluchowski solution (exponential IC).
//
// Analytical moments (T = β₀ N₀ vc² t):
//   M₀(t) = N₀ exp(-T)
//   M₁(t) = N₀ vc             (conserved)
//   M₂(t) = N₀ vc² / (1-2T)  (diverges at T=0.5: gelation)
//
// The simulation is run to t_end < t_gel = 1/(2 β₀ N₀ vc²).
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
    // t_gel = 1/(2*beta0*N0*vc^2) = 0.5 — run to 60% of gelation time
    constexpr double t_end    = 0.3;
    constexpr int    n_steps  = 3000;
    constexpr int    n_print  = 300;
}

int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Aggregation example: product kernel\n");
    std::printf("=============================================================\n\n");

    cudaDeviceProp prop{};
    PBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    const auto x_host  = make_geometric_grid_range(cfg::n, cfg::v_min, cfg::v_max);
    const pbe_cuda::SectionalGrid grid(x_host);
    const double r      = x_host[1] / x_host[0];

    auto N_host = make_exponential_ic(x_host, cfg::N0, cfg::vc);

    ProductKernelAnalytical ana{cfg::N0, cfg::vc, cfg::beta0};

    DeviceArray<double> d_N(cfg::n), d_rhs(cfg::n);
    d_N.upload(N_host);

    pbe_cuda::PBEModelConfig model_config;
    model_config.grid = grid;
    model_config.aggregation_model =
        pbe_cuda::AggregationModel::product(cfg::beta0);
    pbe_cuda::CudaWorkspace workspace(pbe_cuda::CudaStream::external(0));
    const pbe_cuda::CudaPBEModel model(model_config, 256, workspace.stream());

    const double dt     = cfg::t_end / cfg::n_steps;
    const double M1_ref = ana.M1(0.0);
    const double t_gel  = 0.5 / (cfg::beta0 * cfg::N0 * cfg::vc * cfg::vc);

    std::printf("Grid    : n=%d, v=[%.1e, %.1e], r=%.4f\n",
                cfg::n, cfg::v_min, cfg::v_max, r);
    std::printf("Physics : β₀=%.2f, N₀=%.2f, vc=%.2f\n",
                cfg::beta0, cfg::N0, cfg::vc);
    std::printf("Time    : t_end=%.2f (%.0f%% of t_gel=%.2f), dt=%.2e (RK4)\n\n",
                cfg::t_end, 100.0*cfg::t_end/t_gel, t_gel, dt);

    std::printf("Validating M₀ and M₂ (product kernel) and M₁ conservation:\n\n");
    print_table_header("t", "M0 num", "M0 exact", "Err M0", "M2 num", "Err M2");

    // ---- Define RHS function --------------------------------------------
    auto rhs_func = [&](const DeviceArray<double>& N_in,
                        DeviceArray<double>&       rhs_out) 
    {
        cudaError_t err = model.compute_rhs(
            pbe_cuda::ConstDeviceRealView(N_in.get(), N_in.size()),
            pbe_cuda::DeviceRealView(rhs_out.get(), rhs_out.size()),
            workspace);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "RHS error: %s\n", cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        PBE_CUDA_CHECK(cudaDeviceSynchronize());
    };

    // ---- Time loop -------------------------------------------------------
    double t = 0.0;
    for (int step = 0; step <= cfg::n_steps; ++step) 
    {
        if (step % cfg::n_print == 0) 
        {
            d_N.download(N_host);
            double M0_num = compute_M0(N_host);
            double M2_num = compute_M2(N_host, x_host);
            double M0_ana = ana.M0(t);
            double M2_ana = ana.M2(t);
            double err_M0 = std::abs(M0_num - M0_ana) / M0_ana;
            double err_M2 = (M2_ana < 1.0e200)
                        ? std::abs(M2_num - M2_ana) / M2_ana : 0.0;
            print_table_row(t, M0_num, M0_ana, err_M0, M2_num, err_M2);
        }

        if (step == cfg::n_steps) break;
        rk4_step(d_N, d_rhs, rhs_func, dt, cfg::n);
        t += dt;
    }

    // ---- Final summary ---------------------------------------------------
    {
        d_N.download(N_host);
        double M0_num = compute_M0(N_host);
        double M1_num = compute_M1(N_host, x_host);
        double M2_num = compute_M2(N_host, x_host);
        double M0_ana = ana.M0(cfg::t_end);
        double M2_ana = ana.M2(cfg::t_end);
        double err_M0 = std::abs(M0_num - M0_ana) / M0_ana;
        double err_M1 = std::abs(M1_num - M1_ref) / M1_ref;
        double err_M2 = (M2_ana < 1.0e200)
                      ? std::abs(M2_num - M2_ana) / M2_ana : 0.0;

        print_separator();
        std::printf("Final t = %.2f  (tau = %.3f, t_gel = %.2f)\n",
                    cfg::t_end, ana.tau(cfg::t_end), ana.t_gel());
        std::printf("  M0 error : %.4e  %s\n", err_M0,
                    err_M0 < 0.01 ? "PASS" : "WARN");
        std::printf("  M2 error : %.4e  %s\n", err_M2,
                    err_M2 < 0.05 ? "PASS" : "WARN");
        std::printf("  M1 error : %.4e  %s\n", err_M1,
                    err_M1 < 1.0e-3 ? "PASS" : "WARN");
        print_separator();
    }

    return EXIT_SUCCESS;
}
