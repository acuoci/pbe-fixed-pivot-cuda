// =============================================================================
// examples/homogeneous_batch/main.cpp
//
// Worked example: homogeneous batch aggregation with a constant kernel.
//
// Demonstrates end-to-end usage of the pbe_cuda library:
//   1. Build a geometric size grid.
//   2. Set a monodisperse initial condition.
//   3. Advance the PBE in time with explicit Euler integration.
//   4. Validate the total number concentration against the analytical
//      Smoluchowski solution for the constant kernel.
//
// Physical setup
// --------------
//   Kernel   : β(u, v) = β₀  (constant)
//   Grid     : geometric, n = 256 bins, ratio r = 2^(1/3)
//   IC       : monodisperse — all particles in the first bin, N_tot,0 = N0
//   Domain   : volumes v ∈ [v_min, v_max]
//
// Analytical solution (Smoluchowski, constant kernel, monodisperse IC)
// --------------------------------------------------------------------
//   N_tot(t) = N0 / (1 + t / t_half)
//   t_half   = 2 / (β₀ * N0)
//
// Expected output
// ---------------
//   A table of time, N_tot (numerical), N_tot (analytical), relative error.
//   Final relative error should be < 1 % for the chosen dt and n.
//
// Build
// -----
//   cmake --build build --target example_homogeneous_batch
//   ./build/examples/homogeneous_batch/example_homogeneous_batch
//
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Minimal CUDA error-checking helper.
// Throws std::runtime_error with file/line context on failure.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            throw std::runtime_error(                                       \
                std::string("CUDA error at " __FILE__ ":")               \
                + std::to_string(__LINE__) + " — "                         \
                + cudaGetErrorString(_e));                                  \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Simulation parameters — all in consistent SI units (m³, s, #/m³).
// Modify these to explore different regimes.
// ---------------------------------------------------------------------------
namespace cfg {
    // Grid
    constexpr int    n       = 256;          // number of size bins
    constexpr double v_min   = 1.0e-18;     // smallest pivot volume [m³]
    constexpr double r       = 1.122018;    // geometric ratio  2^(1/3) ≈ 1.122018
                                             // each bin volume doubles every 3 bins

    // Physics
    constexpr double beta0   = 1.0e-17;     // constant aggregation kernel [m³/s]
    constexpr double N0      = 1.0e14;      // initial total number concentration [#/m³]

    // Time integration
    constexpr double t_end   = 2.0e3;       // end time [s]
    constexpr int    n_steps = 2000;        // number of Euler steps
    constexpr int    n_print = 200;         // print every n_print steps
}

// ---------------------------------------------------------------------------
// Build a geometric pivot grid x[i] = v_min * r^i
// ---------------------------------------------------------------------------
static std::vector<double> make_geometric_grid(int n, double v_min, double r)
{
    std::vector<double> x(n);
    x[0] = v_min;
    for (int i = 1; i < n; ++i)
        x[i] = x[i-1] * r;
    return x;
}

// ---------------------------------------------------------------------------
// Smoluchowski analytical solution — constant kernel, monodisperse IC.
//   N_tot(t) = N0 / (1 + t / t_half),   t_half = 2 / (β₀ * N0)
// ---------------------------------------------------------------------------
static double analytical_N_tot(double t)
{
    const double t_half = 2.0 / (cfg::beta0 * cfg::N0);
    return cfg::N0 / (1.0 + t / t_half);
}

// ---------------------------------------------------------------------------
// Sum the number distribution on the host to get N_tot.
// ---------------------------------------------------------------------------
static double sum_N(const std::vector<double>& N)
{
    return std::accumulate(N.begin(), N.end(), 0.0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Homogeneous batch aggregation example\n");
    std::printf(" Constant kernel, monodisperse IC, explicit Euler\n");
    std::printf("=============================================================\n\n");

    // ---- Print device info -----------------------------------------------
    int device_id = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::printf("GPU : %s  (sm_%d%d, %.0f MB)\n\n",
                prop.name,
                prop.major, prop.minor,
                prop.totalGlobalMem / 1.0e6);

    // ---- Build grid on host ----------------------------------------------
    const std::vector<double> x_host = make_geometric_grid(cfg::n, cfg::v_min, cfg::r);

    // Pre-compute grid parameters for the library
    const double log_x0    = std::log(x_host[0]);
    const double inv_log_r = 1.0 / std::log(x_host[1] / x_host[0]);

    // ---- Initial condition: monodisperse in bin 0 ------------------------
    // All number concentration in the first (smallest) bin.
    std::vector<double> N_host(cfg::n, 0.0);
    N_host[0] = cfg::N0;

    // ---- Allocate device memory ------------------------------------------
    double* d_N   = nullptr;
    double* d_x   = nullptr;
    double* d_rhs = nullptr;

    CUDA_CHECK(cudaMalloc(&d_N,   cfg::n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,   cfg::n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, cfg::n * sizeof(double)));

    // Copy grid to device (constant throughout the simulation)
    CUDA_CHECK(cudaMemcpy(d_x, x_host.data(),
                          cfg::n * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Set up library parameters ---------------------------------------
    pbe_cuda::AggregationParams params;
    params.n           = cfg::n;
    params.log_x0      = log_x0;
    params.inv_log_r   = inv_log_r;
    params.kernel_type = pbe_cuda::AggregationKernel::Constant;
    params.beta0       = cfg::beta0;
    params.beta_br     = 0.0;
    params.beta_sh     = 0.0;
    params.block_size  = 256;

    const double dt = cfg::t_end / cfg::n_steps;

    // ---- Print header ----------------------------------------------------
    std::printf("Grid    : n = %d bins, v_min = %.2e m³, r = %.6f\n",
                cfg::n, cfg::v_min, cfg::r);
    std::printf("Physics : β₀ = %.2e m³/s, N0 = %.2e #/m³\n",
                cfg::beta0, cfg::N0);
    std::printf("Time    : t_end = %.2e s, dt = %.2e s, steps = %d\n\n",
                cfg::t_end, dt, cfg::n_steps);

    std::printf("%-12s  %-14s  %-14s  %-12s\n",
                "Time [s]", "N_tot num.", "N_tot exact", "Rel. error");
    std::printf("%-12s  %-14s  %-14s  %-12s\n",
                "------------", "--------------",
                "--------------", "------------");

    // ---- Time loop -------------------------------------------------------
    // Copy initial condition to device
    CUDA_CHECK(cudaMemcpy(d_N, N_host.data(),
                          cfg::n * sizeof(double), cudaMemcpyHostToDevice));

    double t = 0.0;

    for (int step = 0; step <= cfg::n_steps; ++step) {

        // -- Print / validate at output intervals --------------------------
        if (step % cfg::n_print == 0) {
            CUDA_CHECK(cudaMemcpy(N_host.data(), d_N,
                                  cfg::n * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            double N_num   = sum_N(N_host);
            double N_exact = analytical_N_tot(t);
            double rel_err = std::abs(N_num - N_exact) / N_exact;

            std::printf("%-12.4e  %-14.6e  %-14.6e  %-12.4e\n",
                        t, N_num, N_exact, rel_err);
        }

        if (step == cfg::n_steps) break;   // last print done, exit loop

        // -- Compute RHS ---------------------------------------------------
        // Zero rhs before each kernel call (required by the library).
        CUDA_CHECK(cudaMemset(d_rhs, 0, cfg::n * sizeof(double)));

        cudaError_t err = pbe_cuda::launch_aggregation_rhs(
            d_N, d_x, d_rhs, params);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "launch_aggregation_rhs failed: %s\n",
                         cudaGetErrorString(err));
            return EXIT_FAILURE;
        }

        // Synchronise to catch any asynchronous kernel errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // -- Euler update: N[i] += dt * rhs[i] ----------------------------
        // Performed on the host for clarity; in a real solver this would
        // be a device kernel to avoid the round-trip.
        CUDA_CHECK(cudaMemcpy(N_host.data(), d_N,
                              cfg::n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        std::vector<double> rhs_host(cfg::n);
        CUDA_CHECK(cudaMemcpy(rhs_host.data(), d_rhs,
                              cfg::n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < cfg::n; ++i)
            N_host[i] += dt * rhs_host[i];

        CUDA_CHECK(cudaMemcpy(d_N, N_host.data(),
                              cfg::n * sizeof(double),
                              cudaMemcpyHostToDevice));

        t += dt;
    }

    // ---- Final error summary --------------------------------------------
    {
        double N_num   = sum_N(N_host);
        double N_exact = analytical_N_tot(cfg::t_end);
        double rel_err = std::abs(N_num - N_exact) / N_exact;

        std::printf("\n-------------------------------------------------------------\n");
        std::printf("Final relative error in N_tot at t = %.2e s : %.4e\n",
                    cfg::t_end, rel_err);
        if (rel_err < 0.01)
            std::printf("PASS — error < 1 %%\n");
        else
            std::printf("WARN — error >= 1 %% (consider reducing dt or increasing n)\n");
        std::printf("-------------------------------------------------------------\n");
    }

    // ---- Free device memory ---------------------------------------------
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_rhs));

    return EXIT_SUCCESS;
}