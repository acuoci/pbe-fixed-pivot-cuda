// =============================================================================
// examples/homogeneous_batch_breakage/main.cpp
//
// Worked example: homogeneous batch breakage with linear selection function
// and symmetric binary daughter distribution.
//
// Demonstrates end-to-end usage of the pbe_cuda breakage API:
//   1. Build a geometric size grid.
//   2. Set a monodisperse initial condition near the top of the grid.
//   3. Set up symmetric daughter quadrature (1 point, t_q = 0.5, bw_q = 2).
//   4. Advance the PBE in time with explicit Euler integration.
//   5. Validate against two independent analytical quantities.
//
// Physical setup
// --------------
//   Selection function : S(v) = S₀ * v / v_ref   (linear)
//   Daughter dist.     : symmetric binary — two fragments of v/2
//   Grid               : geometric, n = 128 bins, ratio r = 2^(1/3)
//   IC                 : monodisperse — all particles in bin ic_bin = 100
//
// Analytical solution (linear selection + symmetric daughter, monodisperse IC)
// ---------------------------------------------------------------------------
//   Total number : N_tot(t) = N₀ * exp(S₀ * t)       [grows exponentially]
//   Total volume : V_tot(t) = N₀ * v₀                 [strictly conserved]
//
//   The exponential growth of N_tot is the primary validation target.
//   Volume conservation is a secondary check on mass conservation.
//
// Quadrature for symmetric binary breakage
// -----------------------------------------
//   Symmetric breakage produces exactly two fragments of size v/2 each.
//   This is represented with a single quadrature point:
//     t_q  = [0.5]    (fragment volume fraction)
//     bw_q = [2.0]    (number of fragments × weight = 2 × 1)
//
//   The birth kernel then evaluates:
//     v_frag = x[j] * t_q[0] = x[j] / 2
//     contrib = S(x[j]) * N[j] * bw_q[0] = S(x[j]) * N[j] * 2
//   which correctly accounts for both fragments produced per breakage event.
//
// Stability
// ---------
//   For explicit Euler with linear selection S(v) = S₀ v/v_ref, the
//   stability limit is approximately dt < v_ref / (S₀ * v_max).
//   With the chosen parameters dt = 0.1 s is safely stable.
//
// Build
// -----
//   cmake --build build --target example_homogeneous_batch_breakage
//   ./build/examples/homogeneous_batch_breakage/example_homogeneous_batch_breakage
//
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// CUDA error-checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("CUDA error at " __FILE__ ":")                  \
                + std::to_string(__LINE__) + " — "                            \
                + cudaGetErrorString(_e));                                     \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
namespace cfg {
    // Grid
    constexpr int    n       = 512;          // number of size bins
    constexpr double v_min   = 1.0e-18;     // smallest pivot volume [m³]
    constexpr double r       = 1.122018;    // geometric ratio 2^(1/3)
    constexpr int    ic_bin  = 384;         // monodisperse IC bin index
                                             // (near top → room for fragments)

    // Physics — linear selection S(v) = S0 * v / v_ref
    constexpr double S0      = 1.0e-3;      // selection prefactor [1/s]
    constexpr double N0      = 1.0e14;      // initial number concentration [#/m³]

    // Quadrature — symmetric binary (1 point)
    constexpr int    n_quad  = 1;
    constexpr double t_q_val = 0.5;         // fragment fraction: v_frag = v/2
    constexpr double bw_q_val= 2.0;         // 2 fragments per event

    // Time integration
    constexpr double t_end   = 100.0;       // end time [s]
    constexpr int    n_steps = 10000;       // Euler steps  →  dt = 0.01 s
    constexpr int    n_print = 1000;        // print every n_print steps
}

// ---------------------------------------------------------------------------
// Build geometric grid
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
// Analytical solutions
//   N_tot(t) = N₀ * exp(S₀ * t)
//   V_tot    = N₀ * v₀              (conserved)
// ---------------------------------------------------------------------------
static double analytical_N_tot(double t)
{
    return cfg::N0 * std::exp(cfg::S0 * t);
}

// ---------------------------------------------------------------------------
// Compute N_tot and V_tot from host arrays
// ---------------------------------------------------------------------------
static double compute_N_tot(const std::vector<double>& N)
{
    return std::accumulate(N.begin(), N.end(), 0.0);
}

static double compute_V_tot(const std::vector<double>& N,
                             const std::vector<double>& x)
{
    double V = 0.0;
    for (int i = 0; i < static_cast<int>(N.size()); ++i)
        V += N[i] * x[i];
    return V;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Homogeneous batch breakage example\n");
    std::printf(" Linear selection, symmetric binary daughter, explicit Euler\n");
    std::printf("=============================================================\n\n");

    // ---- Device info -----------------------------------------------------
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d, %.0f MB)\n\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1.0e6);

    // ---- Build grid ------------------------------------------------------
    const std::vector<double> x_host = make_geometric_grid(cfg::n, cfg::v_min, cfg::r);

    // v_ref = v₀ = pivot volume of the IC bin → S(v₀) = S₀ exactly
    const double v0    = x_host[cfg::ic_bin];
    const double v_ref = v0;

    // ---- Initial condition: monodisperse in ic_bin -----------------------
    std::vector<double> N_host(cfg::n, 0.0);
    N_host[cfg::ic_bin] = cfg::N0;

    // Reference volume for conservation check
    const double V_tot_ref = cfg::N0 * v0;

    // ---- Quadrature arrays (host) ----------------------------------------
    // Symmetric binary: t_q = [0.5], bw_q = [2.0]
    const std::vector<double> t_q_host  = { cfg::t_q_val  };
    const std::vector<double> bw_q_host = { cfg::bw_q_val };

    // ---- Allocate device memory ------------------------------------------
    double* d_N    = nullptr;
    double* d_x    = nullptr;
    double* d_rhs  = nullptr;
    double* d_t_q  = nullptr;
    double* d_bw_q = nullptr;

    CUDA_CHECK(cudaMalloc(&d_N,    cfg::n       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,    cfg::n       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs,  cfg::n       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t_q,  cfg::n_quad  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bw_q, cfg::n_quad  * sizeof(double)));

    // Copy constants to device
    CUDA_CHECK(cudaMemcpy(d_x,    x_host.data(),
                          cfg::n      * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_q,  t_q_host.data(),
                          cfg::n_quad * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bw_q, bw_q_host.data(),
                          cfg::n_quad * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Library parameters ----------------------------------------------
    pbe_cuda::BreakageParams params;
    params.n          = cfg::n;
    params.n_quad     = cfg::n_quad;
    params.selection  = pbe_cuda::BreakageSelection::Linear;
    params.S0         = cfg::S0;
    params.v_ref      = v_ref;
    params.alpha      = 1.0;     // unused for Linear, set for completeness
    params.v_min      = 0.0;     // unused for Linear
    params.block_size = 256;

    const double dt = cfg::t_end / cfg::n_steps;

    // ---- Print setup summary ---------------------------------------------
    std::printf("Grid    : n = %d bins, v_min = %.2e m³, r = %.6f\n",
                cfg::n, cfg::v_min, cfg::r);
    std::printf("IC      : monodisperse in bin %d, v₀ = %.4e m³\n",
                cfg::ic_bin, v0);
    std::printf("Physics : S₀ = %.2e s⁻¹, v_ref = v₀, N₀ = %.2e #/m³\n",
                cfg::S0, cfg::N0);
    std::printf("Daughter: symmetric binary (t_q = 0.5, bw_q = 2.0)\n");
    std::printf("Time    : t_end = %.1f s, dt = %.2f s, steps = %d\n\n",
                cfg::t_end, dt, cfg::n_steps);

    std::printf("Analytical: N_tot(t) = N₀·exp(S₀·t),  V_tot = N₀·v₀ = %.4e m³/m³\n\n",
                V_tot_ref);

    std::printf("%-10s  %-14s  %-14s  %-12s  %-12s\n",
                "Time [s]", "N_tot num.", "N_tot exact", "Err N_tot",
                "Err V_tot");
    std::printf("%-10s  %-14s  %-14s  %-12s  %-12s\n",
                "----------", "--------------", "--------------",
                "------------", "------------");

    // ---- Copy IC to device -----------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_N, N_host.data(),
                          cfg::n * sizeof(double), cudaMemcpyHostToDevice));

    double t = 0.0;
    std::vector<double> rhs_host(cfg::n);

    // ---- Time loop -------------------------------------------------------
    for (int step = 0; step <= cfg::n_steps; ++step) {

        // -- Print / validate ----------------------------------------------
        if (step % cfg::n_print == 0) {
            CUDA_CHECK(cudaMemcpy(N_host.data(), d_N,
                                  cfg::n * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            double N_num   = compute_N_tot(N_host);
            double V_num   = compute_V_tot(N_host, x_host);
            double N_exact = analytical_N_tot(t);
            double err_N   = std::abs(N_num - N_exact) / N_exact;
            double err_V   = std::abs(V_num - V_tot_ref) / V_tot_ref;

            std::printf("%-10.2f  %-14.6e  %-14.6e  %-12.4e  %-12.4e\n",
                        t, N_num, N_exact, err_N, err_V);
        }

        if (step == cfg::n_steps) break;

        // -- Compute RHS ---------------------------------------------------
        CUDA_CHECK(cudaMemset(d_rhs, 0, cfg::n * sizeof(double)));

        cudaError_t err = pbe_cuda::launch_breakage_rhs(
            d_N, d_x, d_t_q, d_bw_q, d_rhs, params);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "launch_breakage_rhs failed: %s\n",
                         cudaGetErrorString(err));
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // -- Euler update --------------------------------------------------
        CUDA_CHECK(cudaMemcpy(N_host.data(),   d_N,
                              cfg::n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rhs_host.data(), d_rhs,
                              cfg::n * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < cfg::n; ++i)
            N_host[i] += dt * rhs_host[i];

        CUDA_CHECK(cudaMemcpy(d_N, N_host.data(),
                              cfg::n * sizeof(double), cudaMemcpyHostToDevice));
        t += dt;
    }

    // ---- Final summary ---------------------------------------------------
    {
        double N_num   = compute_N_tot(N_host);
        double V_num   = compute_V_tot(N_host, x_host);
        double N_exact = analytical_N_tot(cfg::t_end);
        double err_N   = std::abs(N_num   - N_exact)   / N_exact;
        double err_V   = std::abs(V_num   - V_tot_ref) / V_tot_ref;

        std::printf("\n-------------------------------------------------------------\n");
        std::printf("Final t = %.1f s\n", cfg::t_end);
        std::printf("  N_tot error : %.4e  ", err_N);
        std::printf("%s\n", err_N < 0.01 ? "PASS" : "WARN");
        std::printf("  V_tot error : %.4e  ", err_V);
        std::printf("%s\n", err_V < 1.0e-10 ? "PASS (machine precision)"
                                             : (err_V < 1.0e-6 ? "PASS" : "WARN"));
        std::printf("-------------------------------------------------------------\n");
    }

    // ---- Cleanup ---------------------------------------------------------
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_t_q));
    CUDA_CHECK(cudaFree(d_bw_q));

    return EXIT_SUCCESS;
}