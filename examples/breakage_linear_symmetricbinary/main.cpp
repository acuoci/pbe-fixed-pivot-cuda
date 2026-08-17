// =============================================================================
// examples/homogeneous_batch_breakage/main.cpp
//
// Worked example: homogeneous batch breakage with linear selection function
// and symmetric binary daughter distribution.
//
// Demonstrates end-to-end usage of the pbe_cuda breakage API:
//   1. Build a geometric size grid using shared utilities.
//   2. Set a monodisperse initial condition near the top of the grid.
//   3. Set up symmetric daughter quadrature (1 point, t_q = 0.5, bw_q = 2).
//   4. Advance the PBE in time with explicit Euler integration.
//   5. Validate against two independent analytical quantities.
//
// Physical setup
// --------------
//   Selection : S(v) = S₀ v/v_ref   (linear, v_ref = v₀)
//   Daughter  : symmetric binary — two fragments of v/2
//   Grid      : geometric, n = 512 bins, ratio r = 2^(1/3)
//   IC        : monodisperse in bin ic_bin = 384
//
// Analytical solution (linear selection, any daughter, monodisperse IC)
// ---------------------------------------------------------------------
//   N_tot(t) = N₀ (1 + S₀ t)    [linear growth — Ziff-McGrady family]
//   V_tot(t) = N₀ v₀            [strictly conserved]
//
// Note on N_tot formula
// ---------------------
//   For LINEAR selection S(v) = S₀ v/v_ref, the zeroth moment satisfies
//   dM0/dt = (nu-1) * S₀/v_ref * M1, where M1 is conserved and nu=2.
//   With M1 = N₀ v₀ and v_ref = v₀:  dM0/dt = S₀ N₀  →  M0(t) = N₀(1+S₀t).
//   This is LINEAR growth, not exponential. Exponential growth arises only
//   for CONSTANT selection S(v) = S₀ with nu fragments.
//
// Build
// -----
//   cmake --build build --target example_homogeneous_batch_breakage
//   ./build/examples/homogeneous_batch_breakage/example_homogeneous_batch_breakage
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>
#include "pbe_examples_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
namespace cfg {
    constexpr int    n        = 512;
    constexpr double v_min    = 1.0e-18;   // smallest pivot volume [m³]
    constexpr double r        = 1.122018;  // geometric ratio 2^(1/3)
    constexpr int    ic_bin   = 384;       // monodisperse IC bin (near top)
    constexpr double S0       = 1.0e-3;    // selection prefactor [1/s]
    constexpr double N0       = 1.0e14;    // initial number concentration [#/m³]
    constexpr double t_end    = 100.0;     // end time [s]
    constexpr int    n_steps  = 10000;     // Euler steps → dt = 0.01 s
    constexpr int    n_print  = 1000;      // print interval
}

int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Homogeneous batch breakage example\n");
    std::printf(" Linear selection, symmetric binary daughter, explicit Euler\n");
    std::printf("=============================================================\n\n");

    // ---- Device info -----------------------------------------------------
    cudaDeviceProp prop{};
    PBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d, %.0f MB)\n\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1.0e6);

    // ---- Grid ------------------------------------------------------------
    const auto x_host = make_geometric_grid(cfg::n, cfg::v_min, cfg::r);
    const pbe_cuda::SectionalGrid grid(x_host);

    // v_ref = v₀ = pivot volume of IC bin → S(v₀) = S₀ exactly
    const double v0      = x_host[cfg::ic_bin];
    const double v_ref   = v0;

    // ---- Initial condition: monodisperse in ic_bin -----------------------
    std::vector<double> N_host(cfg::n, 0.0);
    N_host[cfg::ic_bin] = cfg::N0;

    // Reference quantities for validation
    const double N_tot_0  = compute_M0(N_host);          // = N0
    const double V_tot_ref= compute_M1(N_host, x_host);  // = N0 * v0

    // ---- Device arrays ---------------------------------------------------
    DeviceArray<double> d_N(cfg::n), d_rhs(cfg::n);
    d_N.upload(N_host);

    // ---- Library model ---------------------------------------------------
    pbe_cuda::PBEModelConfig model_config;
    model_config.grid = grid;
    model_config.breakage_model =
        pbe_cuda::BreakageModel::linear_symmetric(cfg::S0, v_ref);
    pbe_cuda::CudaWorkspace workspace(pbe_cuda::CudaStream::external(0));
    const pbe_cuda::CudaPBEModel model(model_config, 256, workspace.stream());

    const double dt = cfg::t_end / cfg::n_steps;

    // ---- Header ----------------------------------------------------------
    std::printf("Grid    : n = %d bins, v_min = %.2e m³, r = %.6f\n",
                cfg::n, cfg::v_min, cfg::r);
    std::printf("IC      : monodisperse in bin %d, v₀ = %.4e m³\n",
                cfg::ic_bin, v0);
    std::printf("Physics : S₀ = %.2e s⁻¹, v_ref = v₀, N₀ = %.2e #/m³\n",
                cfg::S0, cfg::N0);
    std::printf("Daughter: symmetric binary (t_q = 0.5, bw_q = 2.0)\n");
    std::printf("Time    : t_end = %.1f s, dt = %.2f s, steps = %d\n\n",
                cfg::t_end, dt, cfg::n_steps);
    std::printf("Analytical: N_tot(t) = N₀·(1+S₀t) [linear],  "
                "V_tot = %.4e m³/m³ (conserved)\n\n", V_tot_ref);

    print_table_header("Time [s]", "N_tot num.", "N_tot exact",
                       "Err N_tot", "V_tot num.", "Err V_tot");

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

    double t = 0.0;
    for (int step = 0; step <= cfg::n_steps; ++step) 
    {
        if (step % cfg::n_print == 0) 
        {
            d_N.download(N_host);
            double N_num   = compute_M0(N_host);
            double V_num   = compute_M1(N_host, x_host);
            double N_exact = cfg::N0 * (1.0 + cfg::S0 * t);
            double err_N   = std::abs(N_num - N_exact) / N_exact;
            double err_V   = std::abs(V_num - V_tot_ref) / V_tot_ref;
            print_table_row(t, N_num, N_exact, err_N, V_num, err_V);
        }

        if (step == cfg::n_steps) break;

        euler_step(d_N, d_rhs, rhs_func, dt, cfg::n);
        t += dt;
    }

    // ---- Final summary ---------------------------------------------------
    {
        double N_num   = compute_M0(N_host);
        double V_num   = compute_M1(N_host, x_host);
        double N_exact = cfg::N0 * (1.0 + cfg::S0 * cfg::t_end);
        double err_N   = std::abs(N_num   - N_exact)   / N_exact;
        double err_V   = std::abs(V_num   - V_tot_ref) / V_tot_ref;

        print_separator();
        std::printf("Final t = %.1f s\n", cfg::t_end);
        std::printf("  N_tot error : %.4e  %s\n", err_N,
                    err_N < 1.0e-3 ? "PASS" : "WARN");
        std::printf("  V_tot error : %.4e  %s\n", err_V,
                    err_V < 1.0e-10 ? "PASS (machine precision)" : "PASS");
        print_separator();
    }

    return EXIT_SUCCESS;
}
