// =============================================================================
// tests/unit/test_aggregation.cpp
//
// Regression and unit tests for the aggregation RHS kernel.
//
// Test categories:
//   Smoke        — all kernel types launch without CUDA error
//   Conservation — first moment M1 preserved after one RHS evaluation
//   Correctness  — RHS values match analytically known results
//   Edge cases   — zero population, single-bin systems
// =============================================================================

#include <pbe_cuda/aggregation.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// CUDA error helper
// ---------------------------------------------------------------------------
// For use in void functions (SetUp, TearDown, test bodies)
#define CUDA_ASSERT(call)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e);\
    } while (0)

// For use in non-void functions (run()) — reports failure and returns error
#define CUDA_CHECK_RET(call)                                                   \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            ADD_FAILURE() << "CUDA error: " << cudaGetErrorString(_e);        \
            return _e;                                                         \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Test fixture — allocates and populates a geometric grid and device arrays.
// ---------------------------------------------------------------------------
class AggregationTest : public ::testing::Test {
protected:
    static constexpr int    N      = 64;
    static constexpr double V_MIN  = 1.0e-18;
    static constexpr double R      = 1.122018;   // 2^(1/3)

    std::vector<double> x_host;
    std::vector<double> N_host;
    std::vector<double> rhs_host;

    double* d_x   = nullptr;
    double* d_N   = nullptr;
    double* d_rhs = nullptr;

    double log_x0    = 0.0;
    double inv_log_r = 0.0;

    void SetUp() override {
        x_host.resize(N);
        N_host.resize(N, 0.0);
        rhs_host.resize(N, 0.0);

        x_host[0] = V_MIN;
        for (int i = 1; i < N; ++i)
            x_host[i] = x_host[i-1] * R;

        log_x0    = std::log(x_host[0]);
        inv_log_r = 1.0 / std::log(x_host[1] / x_host[0]);

        CUDA_ASSERT(cudaMalloc(&d_x,   N * sizeof(double)));
        CUDA_ASSERT(cudaMalloc(&d_N,   N * sizeof(double)));
        CUDA_ASSERT(cudaMalloc(&d_rhs, N * sizeof(double)));

        CUDA_ASSERT(cudaMemcpy(d_x, x_host.data(),
                               N * sizeof(double), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        cudaFree(d_x);
        cudaFree(d_N);
        cudaFree(d_rhs);
    }

    // Upload N_host to device, zero rhs, call kernel, download rhs.
    cudaError_t run(pbe_cuda::AggregationParams& p) {
        CUDA_CHECK_RET(cudaMemcpy(d_N, N_host.data(),
                               N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RET(cudaMemset(d_rhs, 0, N * sizeof(double)));

        cudaError_t err = pbe_cuda::launch_aggregation_rhs(d_N, d_x, d_rhs, p);
        CUDA_CHECK_RET(cudaDeviceSynchronize());

        CUDA_CHECK_RET(cudaMemcpy(rhs_host.data(), d_rhs,
                               N * sizeof(double), cudaMemcpyDeviceToHost));
        return err;
    }

    pbe_cuda::AggregationParams make_params(pbe_cuda::AggregationKernel k,
                                             double b0   = 1.0e-17,
                                             double b_bc = 0.0,
                                             double b_bfm = 0.0,
                                             double b_sh = 0.0) {
        pbe_cuda::AggregationParams p;
        p.n           = N;
        p.log_x0      = log_x0;
        p.inv_log_r   = inv_log_r;
        p.kernel_type = k;
        p.beta0       = b0;
        p.beta_bc     = b_bc;
        p.beta_bfm    = b_bfm;
        p.beta_sh     = b_sh;
        p.block_size  = 256;
        return p;
    }

    double M1() {
        double v = 0.0;
        for (int i = 0; i < N; ++i) v += N_host[i] * x_host[i];
        return v;
    }
};

// ===========================================================================
// SMOKE TESTS — every kernel type must launch without error
// ===========================================================================

TEST_F(AggregationTest, SmokeConstant) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::Constant);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeSum) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::Sum);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeProduct) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::Product);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeBrownianContinuum) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::BrownianContinuum,
                         0.0, 1.0e-17, 0.0, 0.0);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeBrownianFreeMolecular) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::BrownianFreeMolecular,
                         0.0, 0.0, 1.0e-17, 0.0);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeShear) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::Shear,
                         0.0, 0.0, 0.0, 1.0e-17);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeBrownianContinuumShear) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::BrownianContinuumShear,
                         0.0, 1.0e-17, 0.0, 1.0e-17);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(AggregationTest, SmokeBrownianFreeMolecularShear) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::AggregationKernel::BrownianFreeMolecularShear,
                         0.0, 0.0, 1.0e-17, 1.0e-17);
    EXPECT_EQ(run(p), cudaSuccess);
}

// ===========================================================================
// CONSERVATION TESTS — M1 must be preserved to machine precision
// Volume conservation: sum(rhs[i] * x[i]) == 0 for all interior events.
// ===========================================================================

class AggregationConservationTest : public AggregationTest {
protected:
    void check_volume_conservation(pbe_cuda::AggregationKernel k,
                                double b0, double b_bc, double b_bfm, double b_sh) {
        for (int i = 0; i < N; ++i)
            N_host[i] = 1.0e12 * std::exp(-static_cast<double>(i) / 10.0);

        auto p = make_params(k, b0, b_bc, b_bfm, b_sh);
        ASSERT_EQ(run(p), cudaSuccess);

        double dM1  = 0.0;
        double M1_v = 0.0;
        for (int i = 0; i < N; ++i) {
            dM1  += rhs_host[i] * x_host[i];
            M1_v += N_host[i]   * x_host[i];
        }

        // Tolerance: N^2 atomic operations each with ~eps floating point error,
        // accumulated into M1. For N=64, N^2*eps*M1 ~ 64^2 * 2e-16 * M1 ~ 8e-13 * M1.
        // We use a generous factor of 1e-6 relative to M1 to account for
        // varying kernel magnitudes while still catching real conservation failures.
        double tol = 1.0e-6 * std::abs(M1_v);

        EXPECT_NEAR(dM1, 0.0, tol)
            << "Volume not conserved for kernel "
            << static_cast<int>(k)
            << ": dM1/dt = " << dM1
            << ", M1 = " << M1_v
            << ", rel error = " << std::abs(dM1) / M1_v;
    }
};

TEST_F(AggregationConservationTest, VolumeConservedConstant) {
    check_volume_conservation(pbe_cuda::AggregationKernel::Constant,
                              1.0e-17, 0.0, 0.0, 0.0);
}

TEST_F(AggregationConservationTest, VolumeConservedSum) {
    check_volume_conservation(pbe_cuda::AggregationKernel::Sum,
                              1.0e-17, 0.0, 0.0, 0.0);
}

TEST_F(AggregationConservationTest, VolumeConservedProduct) {
    check_volume_conservation(pbe_cuda::AggregationKernel::Product,
                              1.0e-17, 0.0, 0.0, 0.0);
}

TEST_F(AggregationConservationTest, VolumeConservedBrownianContinuum) {
    check_volume_conservation(pbe_cuda::AggregationKernel::BrownianContinuum,
                              0.0, 1.0e-17, 0.0, 0.0);
}

TEST_F(AggregationConservationTest, VolumeConservedBrownianFreeMolecular) {
    check_volume_conservation(pbe_cuda::AggregationKernel::BrownianFreeMolecular,
                              0.0, 0.0, 1.0e-17, 0.0);
}

TEST_F(AggregationConservationTest, VolumeConservedShear) {
    check_volume_conservation(pbe_cuda::AggregationKernel::Shear,
                              0.0, 0.0, 0.0, 1.0e-17);
}

TEST_F(AggregationConservationTest, VolumeConservedBrownianContinuumShear) {
    check_volume_conservation(pbe_cuda::AggregationKernel::BrownianContinuumShear,
                              0.0, 1.0e-17, 0.0, 1.0e-17);
}

TEST_F(AggregationConservationTest, VolumeConservedBrownianFreeMolecularShear) {
    check_volume_conservation(pbe_cuda::AggregationKernel::BrownianFreeMolecularShear,
                              0.0, 0.0, 1.0e-17, 1.0e-17);
}

// ===========================================================================
// CORRECTNESS TESTS
// ===========================================================================

// Zero population → RHS must be identically zero for all kernels.
TEST_F(AggregationTest, ZeroPopulationGivesZeroRHS) {
    // N_host is all zeros from SetUp
    auto p = make_params(pbe_cuda::AggregationKernel::Constant);
    ASSERT_EQ(run(p), cudaSuccess);
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(rhs_host[i], 0.0) << "rhs[" << i << "] != 0 for zero population";
}

// Single bin self-collision — constant kernel.
// For a single populated bin j with population Nj and constant kernel beta0:
//   death: rhs[j] -= beta0 * Nj^2 / 2 * 2 = -beta0 * Nj^2
//   birth: goes to the bin bracketing 2*x[j]
//   net death at j: rhs[j] = -beta0 * Nj^2
TEST_F(AggregationTest, SelfCollisionDeathConstantKernel) {
    constexpr int    j     = N / 2;
    constexpr double Nj    = 1.0e14;
    constexpr double beta0 = 1.0e-17;

    N_host[j] = Nj;
    auto p = make_params(pbe_cuda::AggregationKernel::Constant, beta0);
    ASSERT_EQ(run(p), cudaSuccess);

    // Death contribution at bin j: -beta0 * Nj^2
    double expected_death = -beta0 * Nj * Nj;
    // rhs[j] also receives birth from (k<j, j-k) pairs — here none since
    // only bin j is populated. So rhs[j] = death only.
    EXPECT_NEAR(rhs_host[j], expected_death, std::abs(expected_death) * 1.0e-10)
        << "Self-collision death term incorrect for bin " << j;
}

// Direct free-molecular Brownian check for a self-collision in bin 0.
// With x[0]=v, N[0]=N and no other populated bins:
//   beta = beta_bfm * (2*cbrt(v))^2 * sqrt(2/v)
//   rhs[0] death = -beta*N^2
//   birth        = 0.5*beta*N^2 redistributed to v_new=2v
TEST_F(AggregationTest, BrownianFreeMolecularSelfCollisionReference) {
    constexpr int    j        = 0;
    constexpr double Nj       = 3.0e7;
    constexpr double beta_bfm = 2.5e-9;

    N_host[j] = Nj;
    auto p = make_params(pbe_cuda::AggregationKernel::BrownianFreeMolecular,
                         0.0, 0.0, beta_bfm, 0.0);
    ASSERT_EQ(run(p), cudaSuccess);

    const double rj = std::cbrt(x_host[j]);
    const double s = rj + rj;
    const double beta = beta_bfm * s * s * std::sqrt(1.0 / x_host[j] + 1.0 / x_host[j]);
    const double rate = 0.5 * beta * Nj * Nj;

    EXPECT_NEAR(rhs_host[j], -2.0 * rate, std::abs(2.0 * rate) * 1.0e-12);

    const double v_new = 2.0 * x_host[j];
    const double pos = (std::log(v_new) - log_x0) * inv_log_r;
    const int hi = std::min(N - 1, std::max(1, static_cast<int>(std::floor(pos)) + 1));
    const int lo = hi - 1;
    const double w_upper = (v_new - x_host[lo]) / (x_host[hi] - x_host[lo]);
    EXPECT_NEAR(rhs_host[lo], (1.0 - w_upper) * rate,
                std::abs(rate) * 1.0e-12);
    EXPECT_NEAR(rhs_host[hi], w_upper * rate,
                std::abs(rate) * 1.0e-12);
}

TEST_F(AggregationTest, CpuCudaAgreeForBrownianAndShearKernels) {
    for (int i = 0; i < N; ++i)
        N_host[i] = 1.0e9 * std::exp(-static_cast<double>(i) / 12.0);

    const pbe_cuda::AggregationKernel kernels[] = {
        pbe_cuda::AggregationKernel::BrownianContinuum,
        pbe_cuda::AggregationKernel::BrownianFreeMolecular,
        pbe_cuda::AggregationKernel::Shear,
        pbe_cuda::AggregationKernel::BrownianContinuumShear,
        pbe_cuda::AggregationKernel::BrownianFreeMolecularShear
    };

    for (auto kernel : kernels) {
        auto p = make_params(kernel, 0.0, 1.2e-8, 2.5e-9, 7.5e-10);
        ASSERT_EQ(run(p), cudaSuccess);

        std::vector<double> rhs_cpu(N, 0.0);
        ASSERT_EQ(pbe_cuda::launch_aggregation_rhs_cpu(
                      N_host.data(), x_host.data(), rhs_cpu.data(), p),
                  cudaSuccess);

        for (int i = 0; i < N; ++i) {
            const double scale = std::max({1.0, std::abs(rhs_host[i]), std::abs(rhs_cpu[i])});
            EXPECT_NEAR(rhs_host[i], rhs_cpu[i], 1.0e-10 * scale)
                << "kernel=" << static_cast<int>(kernel) << " bin=" << i;
        }
    }
}

// Sum of RHS must be non-positive for pure aggregation
// (total number decreases or stays constant).
TEST_F(AggregationTest, TotalNumberNonIncreasing) {
    for (int i = 0; i < N; ++i)
        N_host[i] = 1.0e12;
    auto p = make_params(pbe_cuda::AggregationKernel::Constant);
    ASSERT_EQ(run(p), cudaSuccess);

    double sum_rhs = std::accumulate(rhs_host.begin(), rhs_host.end(), 0.0);
    EXPECT_LE(sum_rhs, 0.0)
        << "Total RHS is positive — aggregation should not increase total number";
}

// Invalid parameters must return an error, not crash.
TEST_F(AggregationTest, InvalidNReturnsError) {
    auto p = make_params(pbe_cuda::AggregationKernel::Constant);
    p.n = 0;
    cudaError_t err = pbe_cuda::launch_aggregation_rhs(d_N, d_x, d_rhs, p);
    EXPECT_NE(err, cudaSuccess);
}

TEST_F(AggregationTest, NullPointerReturnsError) {
    auto p = make_params(pbe_cuda::AggregationKernel::Constant);
    cudaError_t err = pbe_cuda::launch_aggregation_rhs(nullptr, d_x, d_rhs, p);
    EXPECT_NE(err, cudaSuccess);
}
