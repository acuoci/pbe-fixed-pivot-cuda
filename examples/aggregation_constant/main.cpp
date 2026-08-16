// =============================================================================
// examples/homogeneous_batch/main.cpp
//
// Worked example: homogeneous batch aggregation with a constant kernel.
//
// Demonstrates end-to-end usage of the pbe_cuda library:
//   1. Build a geometric size grid using shared utilities.
//   2. Set a monodisperse initial condition.
//   3. Advance the PBE in time with explicit Euler integration.
//   4. Validate the total number concentration against the analytical
//      Smoluchowski solution for the constant kernel.
//
// Physical setup
// --------------
//   Kernel : β(u,v) = β₀  (constant)
//   Grid   : geometric, n = 256 bins, ratio r = 2^(1/3)
//   IC     : monodisperse — all particles in the first bin, N_tot,0 = N0
//
// Analytical solution (Smoluchowski, constant kernel, monodisperse IC)
// --------------------------------------------------------------------
//   N_tot(t) = N0 / (1 + t/t_half),   t_half = 2 / (β₀ N0)
//   M1(t)    = N0 * v_min              (conserved)
//
// Build
// -----
//   cmake --build build --target example_homogeneous_batch
//   ./build/examples/homogeneous_batch/example_homogeneous_batch
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>
#include "pbe_examples_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Simulation parameters — SI units (m³, s, #/m³)
// ---------------------------------------------------------------------------
namespace cfg {
    constexpr int    n       = 256;
    constexpr double v_min   = 1.0e-18;   // smallest pivot volume [m³]
    constexpr double r       = 1.122018;  // geometric ratio 2^(1/3)
    constexpr double beta0   = 1.0e-17;   // constant kernel prefactor [m³/s]
    constexpr double N0      = 1.0e14;    // initial number concentration [#/m³]
    constexpr double t_end   = 2.0e3;     // end time [s]
    constexpr int    n_steps = 2000;      // Euler steps
    constexpr int    n_print = 200;       // print interval
}

// ---------------------------------------------------------------------------
// Smoluchowski analytical solution — constant kernel, monodisperse IC.
//   N_tot(t) = N0 / (1 + t/t_half),   t_half = 2/(β₀ N0)
// ---------------------------------------------------------------------------
static double analytical_N_tot(double t)
{
    const double t_half = 2.0 / (cfg::beta0 * cfg::N0);
    return cfg::N0 / (1.0 + t / t_half);
}

int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Homogeneous batch aggregation example\n");
    std::printf(" Constant kernel, monodisperse IC, explicit Euler\n");
    std::printf("=============================================================\n\n");

    // ---- Device info -----------------------------------------------------
    cudaDeviceProp prop{};
    PBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d, %.0f MB)\n\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1.0e6);

    // ---- Grid ------------------------------------------------------------
    const auto x_host    = make_geometric_grid(cfg::n, cfg::v_min, cfg::r);
    const double log_x0  = std::log(x_host[0]);
    const double inv_logr= 1.0 / std::log(x_host[1] / x_host[0]);

    // ---- Initial condition: monodisperse in bin 0 ------------------------
    std::vector<double> N_host(cfg::n, 0.0);
    N_host[0] = cfg::N0;

    const double M1_ref = compute_M1(N_host, x_host);   // = N0 * v_min

    // ---- Device arrays ---------------------------------------------------
    DeviceArray<double> d_x(cfg::n), d_N(cfg::n), d_rhs(cfg::n);
    d_x.upload(x_host);
    d_N.upload(N_host);

    // ---- Library parameters ----------------------------------------------
    pbe_cuda::AggregationParams params;
    params.n           = cfg::n;
    params.log_x0      = log_x0;
    params.inv_log_r   = inv_logr;
    params.kernel_type = pbe_cuda::AggregationKernel::Constant;
    params.beta0       = cfg::beta0;
    params.beta_bc     = 0.0;
    params.beta_bfm    = 0.0;
    params.beta_sh     = 0.0;
    params.block_size  = 256;

    const double dt = cfg::t_end / cfg::n_steps;

    // ---- Header ----------------------------------------------------------
    std::printf("Grid    : n = %d bins, v_min = %.2e m³, r = %.6f\n",
                cfg::n, cfg::v_min, cfg::r);
    std::printf("Physics : β₀ = %.2e m³/s, N0 = %.2e #/m³\n",
                cfg::beta0, cfg::N0);
    std::printf("Time    : t_end = %.2e s, dt = %.2e s, steps = %d\n\n",
                cfg::t_end, dt, cfg::n_steps);
    std::printf("Analytical: N_tot(t) = N0/(1 + t/t_half),  "
                "t_half = %.2e s,  M1 = %.4e m³/m³ (conserved)\n\n",
                2.0/(cfg::beta0*cfg::N0), M1_ref);

    print_table_header("Time [s]", "N_tot num.", "N_tot exact",
                       "Err N_tot", "M1 num", "Err M1");

    // ---- Define RHS function --------------------------------------------
    auto rhs_func = [&](const DeviceArray<double>& N_in,
                        DeviceArray<double>&      rhs_out) 
    {
        rhs_out.zero();
        cudaError_t err = pbe_cuda::launch_aggregation_rhs(
            N_in.get(), d_x.get(), rhs_out.get(), params);
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
            double N_num   = compute_M0(N_host);
            double M1_num  = compute_M1(N_host, x_host);
            double N_exact = analytical_N_tot(t);
            double err_N   = std::abs(N_num  - N_exact) / N_exact;
            double err_M1  = std::abs(M1_num - M1_ref)  / M1_ref;
            print_table_row(t, N_num, N_exact, err_N, M1_num, err_M1);
        }

        if (step == cfg::n_steps) break;
        
        euler_step(d_N, d_rhs, rhs_func, dt, cfg::n);
        t += dt;
    }

    // ---- Final summary ---------------------------------------------------
    {
        double N_num   = compute_M0(N_host);
        double M1_num  = compute_M1(N_host, x_host);
        double N_exact = analytical_N_tot(cfg::t_end);
        double err_N   = std::abs(N_num  - N_exact) / N_exact;
        double err_M1  = std::abs(M1_num - M1_ref)  / M1_ref;

        print_separator();
        std::printf("Final t = %.2e s\n", cfg::t_end);
        std::printf("  N_tot error : %.4e  %s\n", err_N,
                    err_N  < 0.01    ? "PASS" : "WARN");
        std::printf("  M1 error    : %.4e  %s\n", err_M1,
                    err_M1 < 1.0e-10 ? "PASS (machine precision)" : "PASS");
        print_separator();
    }

    return EXIT_SUCCESS;
}
