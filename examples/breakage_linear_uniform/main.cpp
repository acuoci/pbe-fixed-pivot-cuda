// =============================================================================
// examples/breakage_ziff_mcgrady/main.cpp
//
// Validation: pure breakage against the Ziff-McGrady analytical solution.
//   Selection : S(v) = S₀ v / vc   (linear)
//   Daughter  : uniform b(v|v') = 2/v'  (Gauss-Legendre quadrature)
//   IC        : exponential n(v,0) = (N₀/vc) exp(-v/vc)
//
// Analytical moments (τ = S₀ t):
//   M₀(t) = N₀ (1 + τ)
//   M₁(t) = N₀ vc             (conserved)
//   M₂(t) = 2 N₀ vc² / (1+τ)
//
// This is exactly the benchmark used in the associated paper (Section 4.2).
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>
#include "pbe_examples_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cfg {
    constexpr int    n        = 200;
    constexpr double v_min    = 1.0e-4;
    constexpr double v_max    = 1.0e1;
    constexpr double N0       = 1.0;
    constexpr double vc       = 1.0;
    constexpr double S0       = 1.0;

    // Uniform daughter distribution — 8-point Gauss-Legendre on (0,1)
    constexpr int    n_quad   = 8;
    constexpr double t_end    = 5.0;    // tau = S0*t_end = 5 
    constexpr int    n_steps  = 5000;
    constexpr int    n_print  = 500;
}

// 8-point Gauss-Legendre abscissae and weights on (0,1)
// Source: Abramowitz & Stegun, Table 25.4, transformed to (0,1)
static const double GL8_T[] = {
    0.019855071751232, 0.101666761293187, 0.237233795041836,
    0.408282678752175, 0.591717321247825, 0.762766204958164,
    0.898333238706813, 0.980144928248768
};
static const double GL8_W[] = {
    0.050614268145188, 0.111190517226687, 0.156853322938943,
    0.181341891689181, 0.181341891689181, 0.156853322938943,
    0.111190517226687, 0.050614268145188
};

int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Breakage example: Ziff-McGrady benchmark\n");
    std::printf(" Linear selection + uniform daughter (8-pt Gauss-Legendre)\n");
    std::printf("=============================================================\n\n");

    cudaDeviceProp prop{};
    PBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Grid
    const auto x_host   = make_geometric_grid_range(cfg::n, cfg::v_min, cfg::v_max);
    const double r       = x_host[1] / x_host[0];

    // Exponential IC
    auto N_host = make_exponential_ic(x_host, cfg::N0, cfg::vc);

    // Quadrature weights for uniform daughter b(v|v') = 2/v'
    // The birth kernel evaluates: v_frag = x[j] * t_q[q]
    // contrib = S(x[j]) * N[j] * bw_q[q]
    // where bw_q[q] = b(v_frag|x[j]) * x[j] * GL_weight
    //               = (2/x[j]) * x[j] * GL_weight = 2 * GL_weight
    std::vector<double> t_q_host(cfg::n_quad), bw_q_host(cfg::n_quad);
    for (int q = 0; q < cfg::n_quad; ++q) {
        t_q_host[q]  = GL8_T[q];
        bw_q_host[q] = 2.0 * GL8_W[q];   // 2 * GL_weight (uniform daughter)
    }

    // Analytical solution
    ZiffMcGradyAnalytical ana{cfg::N0, cfg::vc, cfg::S0};

    // Device arrays
    DeviceArray<double> d_x(cfg::n),    d_N(cfg::n),   d_rhs(cfg::n);
    DeviceArray<double> d_t_q(cfg::n_quad), d_bw_q(cfg::n_quad);
    d_x.upload(x_host);
    d_N.upload(N_host);
    d_t_q.upload(t_q_host);
    d_bw_q.upload(bw_q_host);

    pbe_cuda::BreakageParams p;
    p.n          = cfg::n;
    p.n_quad     = cfg::n_quad;
    p.selection  = pbe_cuda::BreakageSelection::Linear;
    p.S0         = cfg::S0;
    p.v_ref      = cfg::vc;
    p.block_size = 256;

    const double dt     = cfg::t_end / cfg::n_steps;
    const double M1_ref = ana.M1(0.0);

    std::printf("Grid    : n=%d, v=[%.1e, %.1e], r=%.4f\n",
                cfg::n, cfg::v_min, cfg::v_max, r);
    std::printf("Physics : S₀=%.2f, vc=%.2f, N₀=%.2f\n",
                cfg::S0, cfg::vc, cfg::N0);
    std::printf("Time    : t_end=%.1f (τ=%.1f), dt=%.2e\n\n",
                cfg::t_end, cfg::S0 * cfg::t_end, dt);

    std::printf("Validating M₀ (Ziff-McGrady) and M₁ conservation:\n\n");
    print_table_header("tau", "M0 num", "M0 exact", "Err M0",
                       "M1 num", "Err M1");

   // ---- Define RHS function --------------------------------------------
    auto rhs_func = [&](const DeviceArray<double>& N_in,
                        DeviceArray<double>&       rhs_out) 
    {
        rhs_out.zero();
        cudaError_t err = pbe_cuda::launch_breakage_rhs(
            N_in.get(), d_x.get(), d_t_q.get(), d_bw_q.get(),
            rhs_out.get(), p);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        PBE_CUDA_CHECK(cudaDeviceSynchronize());
    };

    double t = 0.0;
    for (int step = 0; step <= cfg::n_steps; ++step) 
    {
        if (step % cfg::n_print == 0) 
        {
            d_N.download(N_host);
            double M0_num = compute_M0(N_host);
            double M1_num = compute_M1(N_host, x_host);
            double M0_ana = ana.M0(t);
            double err_M0 = std::abs(M0_num - M0_ana) / M0_ana;
            double err_M1 = std::abs(M1_num - M1_ref) / M1_ref;
            // Print tau instead of t
            print_table_row(cfg::S0 * t, M0_num, M0_ana, err_M0, M1_num, err_M1);
        }

        if (step == cfg::n_steps) break;

        euler_step(d_N, d_rhs, rhs_func, dt, cfg::n);
        t += dt;
    }

    d_N.download(N_host);
    double M0_num = compute_M0(N_host);
    double M1_num = compute_M1(N_host, x_host);
    double err_M0 = std::abs(M0_num - ana.M0(cfg::t_end)) / ana.M0(cfg::t_end);
    double err_M1 = std::abs(M1_num - M1_ref) / M1_ref;

    print_separator();
    std::printf("Final τ = %.1f\n", cfg::S0 * cfg::t_end);
    std::printf("  M0 error : %.4e  %s\n", err_M0, err_M0 < 0.01 ? "PASS" : "WARN");
    std::printf("  M1 error : %.4e  %s\n", err_M1, err_M1 < 1.0e-4 ? "PASS" : "WARN");
    print_separator();

    return EXIT_SUCCESS;
}