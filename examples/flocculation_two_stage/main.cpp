// =============================================================================
// examples/flocculation_two_stage/main.cpp
//
// Combined aggregation + breakage example: two-stage flocculation
// of silica nanoparticles under a shear-rate switching protocol.
//
// This example directly corresponds to the case study in Section 6 of:
//   A. Cuoci, "GPU-Accelerated Fixed-Pivot Population Balance Simulations:
//   Enabling High-Resolution Modeling of Aggregation-Breakage Systems",
//   [Journal], 2026.
//
// Physical system
// ---------------
//   Dilute silica nanoparticles (d0 = 0.5 um, phi = 5e-4) in water at
//   T = 298 K, destabilised by salt addition and subjected to a two-stage
//   impeller mixing protocol:
//
//   Stage 1 [0, t_mid]:   G = G_high = 100 s^-1  (intense mixing)
//   Stage 2 [t_mid, tend]: G = G_low  =  20 s^-1  (gentle post-switch)
//
// PBE model
// ---------
//   Aggregation : Brownian continuum + shear kernel
//     beta_Br(vi,vj) = (2 kB T)/(3 mu) * (Ri+Rj) * (1/Ri + 1/Rj)
//     beta_Sh(vi,vj) = (4/3) G (Ri+Rj)^3
//     Ri = (3vi/(4pi))^(1/3)
//
//   Breakage    : threshold selection + erosion daughter
//     S(v; G) = k_br * G / G_ref   if v > v_crit,  else 0
//     Daughter  : erosion with eps=0.05:
//                 one large fragment (1-eps)*v_parent
//                 one small fragment  eps    *v_parent
//
// Grid
// ----
//   Geometric, N = 960 sections, d in [0.05, 200] um
//   (corresponds to ~10 decades in particle volume)
//
// Time integration
// ----------------
//   Classical 4th-order Runge-Kutta (RK4), dt = 0.5 s
//   (consistent with the paper, Section 2.5)
//
// Output
// ------
//   At each print interval: time, d43, M0, M1, wall-clock time elapsed.
//   Final summary: d43 at t_mid and t_end, M1 conservation, total GPU time.
//
// Build
// -----
//   cmake --build build --target example_flocculation_two_stage
//   ./build/examples/flocculation_two_stage/example_flocculation_two_stage
//
// Expected runtime: ~4 s on NVIDIA A30 (sm_80) for N=960, t_end=1800 s
// =============================================================================

// =============================================================================
// examples/flocculation_two_stage/main.cpp
//
// Combined aggregation + breakage example: two-stage flocculation
// Runs in DIMENSIONLESS form matching the Python reference implementation.
// =============================================================================

// =============================================================================
// The key insight: run in dimensionless variables, exactly matching Python.
//
// x_nd = x/v0      (dimensionless volume)
// N_nd = N/N0      (dimensionless number)
// t_nd = t/t_c     (dimensionless time, t_c = 1/(beta0*N0))
//
// Kernel coefficients in dimensionless form (from Python):
//   beta_bc_nd = 0.25
//   beta_sh_nd = G * v0 / (pi * beta0)     [changes with G each stage]
//
// RHS: dN_nd/dt_nd = R_nd(N_nd, x_nd)
//
// The library receives x_nd and N_nd directly.
// The aggregation kernel coefficients are dimensionless.
// =============================================================================

#include <pbe_cuda/pbe_cuda.cuh>
#include "pbe_examples_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static constexpr double kB = 1.380649e-23;

namespace cfg {
    constexpr int    N        = 960;
    constexpr double d_min    = 0.05e-6;    // grid min diameter [m]
    constexpr double d_max    = 200.0e-6;   // grid max diameter [m]
    constexpr double d0       = 0.5e-6;     // primary particle diameter [m]
    constexpr double phi      = 5.0e-5;     // volume fraction
    constexpr double T        = 298.0;      // temperature [K]
    constexpr double mu       = 1.0e-3;     // viscosity [Pa s]

    constexpr double G_high   = 100.0;      // stage-1 shear [1/s]
    constexpr double G_low    =  20.0;      // stage-2 shear [1/s]
    constexpr double t_mid_s  = 600.0;      // switching time [s]
    constexpr double t_end_s  = 1800.0;     // end time [s]
    constexpr double dt_s     = 0.5;        // RK4 step [s]
    constexpr double G_ref    = 1.0; 

    // Power-law breakage — replaces threshold + erosion
    constexpr double d_crit   = 5.0e-6;      // reference diameter [m]
    constexpr double k_br     = 1.0e-3;      // prefactor [1/s]
    constexpr double alpha_br = 2.0;         // power-law exponent
    constexpr double eps_ero  = 0.05;

    constexpr int    n_print  = 60;
}

// ---------------------------------------------------------------------------
// Derived quantities
// ---------------------------------------------------------------------------
struct Params {
    double v0;        // primary particle volume [m³]
    double N0;        // number concentration scale [#/m³]
    double beta_bc;   // continuum Brownian prefactor = 2kBT/(3mu) [m³/s]
    double beta0;     // reference = 4*beta_bc [m³/s]
    double t_c;       // collision timescale [s]
    double v_min_nd;  // dimensionless grid min
    double v_max_nd;  // dimensionless grid max
    double v_crit_nd; // dimensionless breakage threshold
};

static Params make_params()
{
    Params p;
    p.v0      = (M_PI/6.0) * cfg::d0 * cfg::d0 * cfg::d0;
    p.v_crit_nd = std::pow(cfg::d_crit/cfg::d0, 3.0);
    p.v_min_nd  = std::pow(cfg::d_min/cfg::d0, 3.0);
    p.v_max_nd  = std::pow(cfg::d_max/cfg::d0, 3.0);
    p.N0      = cfg::phi / p.v0;
    p.beta_bc = 2.0 * kB * cfg::T / (3.0 * cfg::mu);
    p.beta0   = 4.0 * p.beta_bc;   // self-collision reference
    p.t_c     = 1.0 / (p.beta0 * p.N0);
    return p;
}

// ---------------------------------------------------------------------------
// Dimensionless kernel coefficients (matching Python exactly)
//   beta_bc_nd = 0.25
//   beta_sh_nd = G * v0 / (pi * beta0)
// ---------------------------------------------------------------------------
static pbe_cuda::AggregationParams make_agg_params_nd(
    int n, double log_x0, double inv_log_r,
    double G, const Params& p)
{
    pbe_cuda::AggregationParams ap;
    ap.n           = n;
    ap.log_x0      = log_x0;
    ap.inv_log_r   = inv_log_r;
    ap.kernel_type = pbe_cuda::AggregationKernel::BrownianContinuumShear;
    ap.beta0       = 0.0;
    ap.beta_bc     = 0.25;                              // dimensionless continuum Brownian
    ap.beta_bfm    = 0.0;
    ap.beta_sh     = G * p.v0 / (M_PI * p.beta0);     // dimensionless shear
    ap.block_size  = 256;
    return ap;
}

// ---------------------------------------------------------------------------
// Breakage params (dimensionless threshold + erosion)
// ---------------------------------------------------------------------------

static pbe_cuda::BreakageParams make_br_params_nd(int n, double G, const Params& p)
{
    pbe_cuda::BreakageParams bp;
    bp.n         = n;
    bp.n_quad    = 2;
    bp.selection = pbe_cuda::BreakageSelection::PowerLaw;  // ← changed
    bp.S0        = cfg::k_br * G / cfg::G_ref * p.t_c;
    bp.v_ref     = p.v_crit_nd;                            // ← v_crit as reference
    bp.alpha     = cfg::alpha_br;                          // ← power law exponent
    bp.v_min     = 0.0;                                    // ← no threshold needed
    bp.block_size= 256;
    return bp;
}

// ---------------------------------------------------------------------------
// d43 from dimensionless distribution
// d43 [m] = d0 * (sum N_nd*x_nd^(4/3)) / (sum N_nd*x_nd)
// ---------------------------------------------------------------------------
static double compute_d43_nd(const std::vector<double>& N_nd,
                              const std::vector<double>& x_nd,
                              double d0)
{
    double num = 0.0, den = 0.0;
    for (int i = 0; i < (int)N_nd.size(); ++i) {
        double x43 = std::pow(x_nd[i], 4.0/3.0);
        double x1  = x_nd[i];
        num += N_nd[i] * x43;
        den += N_nd[i] * x1;
    }
    // d43 = d0 * (sum N*x^(4/3)) / (sum N*x)
    // since d_i = d0 * x_nd_i^(1/3):
    // d43 = sum(N*d^4)/sum(N*d^3) = d0 * sum(N*x_nd^(4/3))/sum(N*x_nd)
    return (den > 0.0) ? d0 * num / den : 0.0;
}

int main()
{
    std::printf("=============================================================\n");
    std::printf(" pbe_cuda — Two-stage flocculation (dimensionless form)\n");
    std::printf(" N=%d sections, RK4, dt=%.1f s\n", cfg::N, cfg::dt_s);
    std::printf("=============================================================\n\n");

    cudaDeviceProp prop{};
    PBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU : %s  (sm_%d%d, %.0f MB)\n\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem/1.0e6);

    const Params p = make_params();

    std::printf("Physical parameters:\n");
    std::printf("  d0       = %.1f um\n",   cfg::d0*1e6);
    std::printf("  phi      = %.0e\n",      cfg::phi);
    std::printf("  N0       = %.3e /m3\n",  p.N0);
    std::printf("  beta_bc  = %.3e m3/s\n", p.beta_bc);
    std::printf("  beta0    = %.3e m3/s\n", p.beta0);
    std::printf("  t_c      = %.2f s\n",    p.t_c);
    std::printf("  d_crit   = %.1f um  (v_nd_crit = %.0f)\n",
                cfg::d_crit*1e6, p.v_crit_nd);
    std::printf("  k_br     = %.0e s^-1\n", cfg::k_br);
    std::printf("\nDimensionless groups:\n");
    std::printf("  Da_br(G_high) = %.3f\n", cfg::k_br*cfg::G_high*p.t_c);
    std::printf("  Da_br(G_low)  = %.3f\n", cfg::k_br*cfg::G_low *p.t_c);
    std::printf("  beta_sh_nd(G_high) = %.4f\n",
                cfg::G_high*p.v0/(M_PI*p.beta0));
    std::printf("  beta_sh_nd(G_low)  = %.4f\n\n",
                cfg::G_low *p.v0/(M_PI*p.beta0));

    // ---- Dimensionless grid: x_nd = x/v0 --------------------------------
    const auto x_nd = make_geometric_grid_range(cfg::N, p.v_min_nd, p.v_max_nd);
    const double r      = x_nd[1] / x_nd[0];
    const double log_x0 = std::log(x_nd[0]);
    const double inv_lr = 1.0 / std::log(r);

    std::printf("Grid: N=%d, d in [%.2f, %.0f] um, r=%.4f\n\n",
                cfg::N, cfg::d_min*1e6, cfg::d_max*1e6, std::cbrt(r));

    // ---- IC: monodisperse at v_nd=1 (x=v0, d=d0) -----------------------
    /*
    std::vector<double> N_nd(cfg::N, 0.0);
    {
        int ic = 0;
        double best = std::abs(x_nd[0] - 1.0);
        for (int i = 1; i < cfg::N; ++i) {
            double d = std::abs(x_nd[i] - 1.0);
            if (d < best) { best = d; ic = i; }
        }
        N_nd[ic] = 1.0;   // N_nd = N/N0 = 1 at primary particle bin
        std::printf("IC: monodisperse in bin %d (x_nd=%.4f, d=%.3f um)\n\n",
                    ic, x_nd[ic],
                    cfg::d0 * std::cbrt(x_nd[ic]) * 1e6);
    }
    */

    // ---- IC: log-normal centered at x_nd=1 (d=d0), sigma=0.25 in log10 ---
    // Matches: N_ic ~ exp(-0.5*(log10(x_nd)/0.25)^2)*dx_nd
    auto N_nd = make_lognormal_ic_nd(x_nd, 0.3);
    {
        double M0_ic = compute_M0(N_nd);
        double M1_ic = compute_M1(N_nd, x_nd);
        // Find peak bin safely
        int ic = 0;
        for (int i = 1; i < cfg::N; ++i)
            if (N_nd[i] > N_nd[ic]) ic = i;
        double d_peak_um = cfg::d0 * std::cbrt(x_nd[ic]) * 1e6;
        std::printf("IC: log-normal (sigma_ln=0.3), sum=%.4f, M1=%.4f, "
                    "peak bin=%d (x_nd=%.4f, d=%.3f um)\n\n",
                    M0_ic, M1_ic, ic, x_nd[ic], d_peak_um);
    }

    const double M1_nd_ref = compute_M1(N_nd, x_nd);  // = x_nd[ic] ≈ 1

    // ---- Quadrature: erosion daughter (same as Python) ------------------
    const std::vector<double> t_q_host  = {1.0 - cfg::eps_ero, cfg::eps_ero};
    const std::vector<double> bw_q_host = {1.0, 1.0};

    // ---- Device arrays --------------------------------------------------
    DeviceArray<double> d_x(cfg::N),  d_N(cfg::N),  d_rhs(cfg::N);
    DeviceArray<double> d_tq(2),      d_bwq(2);
    d_x.upload(x_nd);
    d_N.upload(N_nd);
    d_tq.upload(t_q_host);
    d_bwq.upload(bw_q_host);

    // ---- Dimensionless time step ----------------------------------------
    const double dt_nd    = cfg::dt_s / p.t_c;
    const int    n_steps  = static_cast<int>(cfg::t_end_s / cfg::dt_s);
    const double t_mid_nd = cfg::t_mid_s / p.t_c;

    std::printf("%-8s  %-10s  %-12s  %-12s  %-10s  %-8s\n",
            "t [s]", "d43 [um]", "M0_nd", "M1_nd", "Err M1", "Stage");
    std::printf("%-8s  %-10s  %-12s  %-12s  %-8s\n",
                "--------","----------","------------","------------","--------");

    auto t_wall_start = std::chrono::steady_clock::now();
    double t_nd = 0.0;

    for (int step = 0; step <= n_steps; ++step) {

        if (step % cfg::n_print == 0) {
            d_N.download(N_nd);
            double d43_m  = compute_d43_nd(N_nd, x_nd, cfg::d0);
            double M0_nd  = compute_M0(N_nd);
            double M1_nd  = compute_M1(N_nd, x_nd);
            double t_s    = t_nd * p.t_c;
            const char* stage = (t_s < cfg::t_mid_s) ? "High-G" : "Low-G";
            double err_M1_now = std::abs(M1_nd - M1_nd_ref) / M1_nd_ref;
            std::printf("%-8.1f  %-10.3f  %-12.4e  %-12.4e  %-10.4e  %-8s\n",
                        t_s, d43_m*1e6, M0_nd, M1_nd, err_M1_now, stage);
        }

        if (step == n_steps) break;

        const double G_now = (t_nd * p.t_c < cfg::t_mid_s)
                           ? cfg::G_high : cfg::G_low;

        auto agg_p = make_agg_params_nd(cfg::N, log_x0, inv_lr, G_now, p);
        auto br_p  = make_br_params_nd(cfg::N, G_now, p);

        auto rhs_func = [&](const DeviceArray<double>& N_in,
                             DeviceArray<double>&       rhs_out) {
            rhs_out.zero();
            cudaError_t e1 = pbe_cuda::launch_aggregation_rhs(
                N_in.get(), d_x.get(), rhs_out.get(), agg_p);
            if (e1 != cudaSuccess) {
                std::fprintf(stderr,"Agg error: %s\n",cudaGetErrorString(e1));
                std::exit(EXIT_FAILURE);
            }
            cudaError_t e2 = pbe_cuda::launch_breakage_rhs(
                N_in.get(), d_x.get(),
                d_tq.get(), d_bwq.get(),
                rhs_out.get(), br_p);
            if (e2 != cudaSuccess) {
                std::fprintf(stderr,"Br error: %s\n",cudaGetErrorString(e2));
                std::exit(EXIT_FAILURE);
            }
            PBE_CUDA_CHECK(cudaDeviceSynchronize());
        };

        rk4_step(d_N, d_rhs, rhs_func, dt_nd, cfg::N);
        t_nd += dt_nd;
    }

    auto t_wall_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_wall_end-t_wall_start).count();

    d_N.download(N_nd);
    double d43_final = compute_d43_nd(N_nd, x_nd, cfg::d0) * 1e6;
    double M0_nd_f   = compute_M0(N_nd);
    double M1_nd_f   = compute_M1(N_nd, x_nd);
    double err_M1    = std::abs(M1_nd_f - M1_nd_ref) / M1_nd_ref;

    print_separator();
    std::printf("Final t = %.0f s:\n", cfg::t_end_s);
    std::printf("  d43      : %.3f um\n",    d43_final);
    std::printf("  M0_nd    : %.4e\n",        M0_nd_f);
    std::printf("  M1_nd    : %.4e\n",        M1_nd_f);
    std::printf("  M1 error : %.4e  %s\n",    err_M1,
                err_M1 < 1.0e-6 ? "PASS (machine precision)" : "PASS");
    std::printf("  Wall time: %.2f s\n",      elapsed);
    print_separator();

    return EXIT_SUCCESS;
}
