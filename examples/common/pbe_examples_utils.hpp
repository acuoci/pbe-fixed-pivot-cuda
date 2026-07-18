// =============================================================================
// examples/common/pbe_examples_utils.hpp
//
// Shared utilities for pbe_cuda worked examples.
//
// Contents:
//   Grid construction    : make_geometric_grid()
//   Moment computation   : compute_moment()
//   Analytical solutions : Smoluchowski (constant), Golovin (sum),
//                          product kernel, Ziff-McGrady (breakage)
//   Device helpers       : DeviceArray<T> RAII wrapper
//   Output helpers       : print_table_header(), print_table_row()
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define PBE_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("CUDA error at " __FILE__ ":")                     \
                + std::to_string(__LINE__) + " - "                             \
                + cudaGetErrorString(_e));                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// RAII device array — allocates on construction, frees on destruction.
// Eliminates boilerplate cudaMalloc/cudaFree pairs in examples.
// ---------------------------------------------------------------------------
template<typename T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t n) : n_(n), ptr_(nullptr) {
        PBE_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }

    ~DeviceArray() { if (ptr_) cudaFree(ptr_); }

    // Upload from host vector
    void upload(const std::vector<T>& h) {
        PBE_CUDA_CHECK(cudaMemcpy(ptr_, h.data(),
                                  n_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Download to host vector
    void download(std::vector<T>& h) const {
        h.resize(n_);
        PBE_CUDA_CHECK(cudaMemcpy(h.data(), ptr_,
                                  n_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() { PBE_CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T))); }

    T*          get()  const { return ptr_; }
    std::size_t size() const { return n_;   }

    // Non-copyable
    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

private:
    std::size_t n_;
    T*          ptr_;
};

// ---------------------------------------------------------------------------
// Grid construction
// ---------------------------------------------------------------------------

// Build a geometric grid: x[i] = v_min * r^i
inline std::vector<double> make_geometric_grid(int n, double v_min, double r)
{
    std::vector<double> x(n);
    x[0] = v_min;
    for (int i = 1; i < n; ++i)
        x[i] = x[i-1] * r;
    return x;
}

// Build a geometric grid spanning [v_min, v_max] with n points
inline std::vector<double> make_geometric_grid_range(int n,
                                                      double v_min,
                                                      double v_max)
{
    double r = std::pow(v_max / v_min, 1.0 / (n - 1));
    return make_geometric_grid(n, v_min, r);
}

// Exponential initial condition: N[i] = (N0/vc) * exp(-x[i]/vc) * dx[i]
// where dx[i] is approximated as (x[i+1]-x[i-1])/2 (trapezoidal width).
// For a geometric grid dx[i] = x[i]*(r-1/r)/2 = x[i]*(r^2-1)/(2r).
inline std::vector<double> make_exponential_ic_trap(const std::vector<double>& x,
                                                    double N0, double vc)
{
    int n = static_cast<int>(x.size());
    double r = x[1] / x[0];
    // cell width for geometric grid: x[i] * (r - 1/r) / 2
    double width_factor = (r - 1.0/r) / 2.0;
    std::vector<double> N(n);
    for (int i = 0; i < n; ++i)
        N[i] = (N0 / vc) * std::exp(-x[i] / vc) * x[i] * width_factor;
    return N;
}

// Accurate — integrates n(v) = (N0/vc)*exp(-v/vc) over each cell [v_lo, v_hi]
// using the exact integral: integral = N0*(exp(-v_lo/vc) - exp(-v_hi/vc))
// This gives machine-precision IC for exponential distributions.
inline std::vector<double> make_exponential_ic(const std::vector<double>& x,
                                                double N0, double vc)
{
    int n = static_cast<int>(x.size());
    double r = x[1] / x[0];
    double sqrt_r = std::sqrt(r);   // cell boundary = x[i] * sqrt(r)

    std::vector<double> N(n);
    for (int i = 0; i < n; ++i) {
        // Cell boundaries: v_lo = x[i]/sqrt(r), v_hi = x[i]*sqrt(r)
        double v_lo = x[i] / sqrt_r;
        double v_hi = x[i] * sqrt_r;
        // Exact integral of (N0/vc)*exp(-v/vc) over [v_lo, v_hi]
        N[i] = N0 * (std::exp(-v_lo / vc) - std::exp(-v_hi / vc));
    }
    return N;
}

// Log-normal IC in natural-log(x_nd) space, centered at x_nd=1 (d=d0).
// Matches Python make_initial_condition(x_nd, dx_nd, sigma=0.3):
//   N_ic ~ exp(-0.5*(ln(x_nd)/sigma)^2) * dx_nd,  normalised to sum(N)=1
// sigma=0.3 in ln space gives M1=1.1445 for the flocculation grid.
inline std::vector<double> make_lognormal_ic_nd(const std::vector<double>& x_nd,
                                                  double sigma_ln = 0.3)
{
    int n = static_cast<int>(x_nd.size());
    double log_r = std::log(x_nd[1] / x_nd[0]);   // = log_ratio
    std::vector<double> N(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double ln_x   = std::log(x_nd[i]);          // natural log, not log10
        double dx_nd_i = x_nd[i] * log_r;
        N[i] = std::exp(-0.5 * (ln_x / sigma_ln) * (ln_x / sigma_ln))
               * dx_nd_i;
        sum += N[i];
    }
    for (int i = 0; i < n; ++i) N[i] /= sum;
    return N;
}

// ---------------------------------------------------------------------------
// Moment computation
// ---------------------------------------------------------------------------

// k-th moment: Mk = sum_i x[i]^k * N[i]
inline double compute_moment(const std::vector<double>& N,
                              const std::vector<double>& x,
                              int k)
{
    double M = 0.0;
    int n = static_cast<int>(N.size());
    for (int i = 0; i < n; ++i) {
        double xk = 1.0;
        for (int j = 0; j < k; ++j) xk *= x[i];
        M += N[i] * xk;
    }
    return M;
}

// Convenience aliases
inline double compute_M0(const std::vector<double>& N) {
    double s = 0.0;
    for (double v : N) s += v;
    return s;
}

inline double compute_M1(const std::vector<double>& N,
                          const std::vector<double>& x) {
    return compute_moment(N, x, 1);
}

inline double compute_M2(const std::vector<double>& N,
                          const std::vector<double>& x) {
    return compute_moment(N, x, 2);
}

// ---------------------------------------------------------------------------
// Analytical solutions for aggregation
// ---------------------------------------------------------------------------

// ---- Constant kernel (Scott / Smoluchowski) --------------------------------
// IC: exponential n(v,0) = (N0/vc) exp(-v/vc)
// T   = β₀ N₀ t / 2
// tau = T / (1 + T)
// M0(t) = N0 (1 - tau)
// M1(t) = N0 vc                     (conserved)
// M2(t) = 2 N0 vc² + β₀ M1² t      (linear growth)
struct ConstantKernelAnalytical {
    double N0, vc, beta0;

    double tau(double t) const {
        double T = 0.5 * beta0 * N0 * t;
        return T / (1.0 + T);
    }
    double M0(double t) const { return N0 * (1.0 - tau(t)); }
    double M1(double)   const { return N0 * vc; }
    double M2(double t) const {
        double M2_0 = 2.0 * N0 * vc * vc;
        double M1_0 = N0 * vc;
        return M2_0 + beta0 * M1_0 * M1_0 * t;
    }
};

// ---- Sum kernel (Golovin) — exponential IC --------------------------------
// s   = β₀ N₀ vc t
// M0(t) = N0 exp(-s)
// M1(t) = N0 vc                     (conserved)
// M2: no closed form for exponential IC — validate M0 and M1 only
struct SumKernelAnalytical {
    double N0, vc, beta0;

    double s(double t)  const { return beta0 * N0 * vc * t; }
    double M0(double t) const { return N0 * std::exp(-s(t)); }
    double M1(double)   const { return N0 * vc; }
};

// ---- Product kernel — exponential IC -------------------------------------
// tau  = β₀ N₀ vc² t
// M0(t) = N0 (1 - tau/2)            (linear decay)
// M1(t) = N0 vc                     (conserved)
// M2(t) = 2 N0 vc² / (1 - β₀ M2(0) t)   (diverges at gelation)
struct ProductKernelAnalytical {
    double N0, vc, beta0;

    double tau(double t)  const { return beta0 * N0 * vc * vc * t; }
    double M0(double t)   const { return N0 * (1.0 - 0.5 * tau(t)); }
    double M1(double)     const { return N0 * vc; }
    double M2(double t)   const {
        double M2_0  = 2.0 * N0 * vc * vc;
        double denom = 1.0 - beta0 * M2_0 * t;
        if (denom <= 0.0) return 1.0e300;   // gelation
        return M2_0 / denom;
    }
    double t_gel() const { return 1.0 / (beta0 * 2.0 * N0 * vc * vc); }
};

// ---------------------------------------------------------------------------
// Analytical solutions for breakage
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Analytical solutions for breakage — verified against Ziff & McGrady (1985)
// ---------------------------------------------------------------------------

// ---- Ziff-McGrady: linear selection + uniform daughter -------------------
// IC : exponential n(v,0) = (N0/vc) exp(-v/vc)
// tau = (S0/v_ref) * vc * t;  with v_ref = vc → tau = S0 * t
//
// Exact solution: n(v,t) = (N0/vc) * (1+tau)² * exp(-(1+tau)*v/vc)
//
// Moments:
//   M0(t) = N0 * (1 + tau)         [linear growth]
//   M1(t) = N0 * vc                [conserved]
//   M2(t) = 2 N0 vc² / (1 + tau)  [decaying]
struct ZiffMcGradyAnalytical {
    double N0, vc, S0;

    double tau(double t)  const { return S0 * t; }
    double M0(double t)   const { return N0 * (1.0 + tau(t)); }
    double M1(double)     const { return N0 * vc; }
    double M2(double t)   const { return 2.0 * N0 * vc * vc / (1.0 + tau(t)); }

    // Continuous number density at volume v and time t
    double n(double v, double t) const {
        double T = 1.0 + tau(t);
        return (N0 / vc) * T * T * std::exp(-T * v / vc);
    }

    // Exact bin-integrated solution over [v_lo, v_hi]
    // Uses closed-form integral of the exponential density.
    double N_bin(double v_lo, double v_hi, double t) const {
        double lam = (1.0 + tau(t)) / vc;
        double exp_lo = std::exp(-lam * v_lo);
        double exp_hi = (std::isinf(v_hi)) ? 0.0 : std::exp(-lam * v_hi);
        return N0 * (1.0 + tau(t)) * (exp_lo - exp_hi);
    }
};

// ---- Constant selection + symmetric binary daughter ----------------------
// Valid for ANY initial condition (M0 formula is IC-independent).
//
//   S(v) = S0,  nu = 2 fragments per event
//
//   M0(t) = N0 * exp((nu-1) * S0 * t) = N0 * exp(S0 * t)
//   M1(t) = N0 * v0   [conserved]
//
// NOTE: This IS exponential growth — correct for constant selection with
// nu=2, regardless of IC.
//
// ---- Linear selection + any daughter (general result) --------------------
//   S(v) = S0 * v / v_ref,  exponential IC
//
//   M0(t) = N0 * (1 + S0 * t)    [linear growth — from Ziff-McGrady]
//   M1(t) = N0 * vc              [conserved]
//
// The homogeneous_batch_breakage example uses linear selection +
// symmetric binary + monodisperse IC. The monodisperse IC M0 formula
// is: M0(t) = N0 * (1 + S0 * t) for linear selection.
struct ConstantSymmetricBreakageAnalytical {
    double N0, v0, S0;
    // nu = 2 fragments, constant selection → exponential M0 growth
    double M0(double t) const { return N0 * std::exp(S0 * t); }
    double M1(double)   const { return N0 * v0; }
};

struct LinearSelectionAnalytical {
    double N0, vc, S0;
    // Linear selection, any daughter, exponential IC → linear M0 growth
    double M0(double t) const { return N0 * (1.0 + S0 * t); }
    double M1(double)   const { return N0 * vc; }
};

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

inline void print_separator(int width = 78) {
    for (int i = 0; i < width; ++i) std::putchar('-');
    std::putchar('\n');
}

inline void print_table_header(const char* c1, const char* c2,
                                const char* c3, const char* c4,
                                const char* c5, const char* c6) {
    std::printf("%-10s  %-12s  %-12s  %-12s  %-12s  %-10s\n",
                c1, c2, c3, c4, c5, c6);
    print_separator();
}

inline void print_table_row(double t,
                             double M_num, double M_ana, double err,
                             double M1_num, double err_M1) {
    std::printf("%-10.4f  %-12.4e  %-12.4e  %-12.4e  %-12.4e  %-10.4e\n",
                t, M_num, M_ana, err, M1_num, err_M1);
}

// ---------------------------------------------------------------------------
// ODE integrators
//
// Both integrators accept a callable rhs_func with signature:
//   void rhs_func(const DeviceArray<double>& N_in,
//                 DeviceArray<double>&       rhs_out)
//
// The callable must:
//   1. Zero rhs_out before accumulating (use rhs_out.zero())
//   2. Call launch_aggregation_rhs and/or launch_breakage_rhs
//   3. Call cudaDeviceSynchronize() before returning
//
// Arithmetic (N update) is performed on the host for clarity and
// portability. In a production solver, replace with a device kernel.
//
// Usage example:
//
//   auto rhs = [&](const DeviceArray<double>& N_in,
//                  DeviceArray<double>&       rhs_out) {
//       rhs_out.zero();
//       pbe_cuda::launch_aggregation_rhs(
//           N_in.get(), d_x.get(), rhs_out.get(), agg_params);
//       PBE_CUDA_CHECK(cudaDeviceSynchronize());
//   };
//
//   for (int step = 0; step < n_steps; ++step)
//       euler_step(d_N, d_rhs, rhs, dt, n);
//
//   for (int step = 0; step < n_steps; ++step)
//       rk4_step(d_N, d_rhs, rhs, dt, n);
// ---------------------------------------------------------------------------

// ---- Explicit Euler -------------------------------------------------------
// N^(n+1) = N^n + dt * R(N^n)
// Cost: 1 RHS evaluation per step.
template<typename RhsFunc>
void euler_step(DeviceArray<double>& d_N,
                DeviceArray<double>& d_rhs,
                RhsFunc&&            rhs_func,
                double               dt,
                int                  n)
{
    rhs_func(d_N, d_rhs);

    // Update on host (clear and simple; replace with device kernel if needed)
    std::vector<double> N_h(n), rhs_h(n);
    d_N.download(N_h);
    d_rhs.download(rhs_h);
    for (int i = 0; i < n; ++i)
        N_h[i] += dt * rhs_h[i];
    d_N.upload(N_h);
}

// ---- Classical 4th-order Runge-Kutta -------------------------------------
// k1 = R(N^n)
// k2 = R(N^n + dt/2 * k1)
// k3 = R(N^n + dt/2 * k2)
// k4 = R(N^n + dt   * k3)
// N^(n+1) = N^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
// Cost: 4 RHS evaluations per step.
template<typename RhsFunc>
void rk4_step(DeviceArray<double>& d_N,
              DeviceArray<double>& d_rhs,
              RhsFunc&&            rhs_func,
              double               dt,
              int                  n)
{
    // Reusable host buffers — allocated once, reused across stages
    std::vector<double> N_h(n), k1(n), k2(n), k3(n), k4(n), stage(n);

    d_N.download(N_h);

    // Temporary device arrays for stage evaluations
    DeviceArray<double> d_stage(n), d_k(n);

    // k1 = R(N^n)
    rhs_func(d_N, d_k);
    d_k.download(k1);

    // k2 = R(N^n + dt/2 * k1)
    for (int i = 0; i < n; ++i) stage[i] = N_h[i] + 0.5 * dt * k1[i];
    d_stage.upload(stage);
    rhs_func(d_stage, d_k);
    d_k.download(k2);

    // k3 = R(N^n + dt/2 * k2)
    for (int i = 0; i < n; ++i) stage[i] = N_h[i] + 0.5 * dt * k2[i];
    d_stage.upload(stage);
    rhs_func(d_stage, d_k);
    d_k.download(k3);

    // k4 = R(N^n + dt * k3)
    for (int i = 0; i < n; ++i) stage[i] = N_h[i] + dt * k3[i];
    d_stage.upload(stage);
    rhs_func(d_stage, d_k);
    d_k.download(k4);

    // Update
    for (int i = 0; i < n; ++i)
        N_h[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    d_N.upload(N_h);
}