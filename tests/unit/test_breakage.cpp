// =============================================================================
// tests/unit/test_breakage.cpp
//
// Regression and unit tests for the breakage RHS kernels.
//
// Test categories:
//   Smoke        — all selection function types launch without CUDA error
//   Conservation — first moment M1 preserved after one RHS evaluation
//   Correctness  — death term, birth rate, zero population
//   Edge cases   — invalid parameters
// =============================================================================

#include <pbe_cuda/breakage.cuh>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <numeric>

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
// Test fixture
// ---------------------------------------------------------------------------
class BreakageTest : public ::testing::Test {
protected:
    static constexpr int    N     = 64;
    static constexpr double V_MIN = 1.0e-18;
    static constexpr double R     = 1.122018;

    // Symmetric binary quadrature: 1 point
    static constexpr int    N_QUAD  = 1;
    static constexpr double T_Q_VAL = 0.5;
    static constexpr double BW_Q_VAL= 2.0;

    std::vector<double> x_host;
    std::vector<double> N_host;
    std::vector<double> rhs_host;
    std::vector<double> t_q_host  = { T_Q_VAL  };
    std::vector<double> bw_q_host = { BW_Q_VAL };

    double* d_x    = nullptr;
    double* d_N    = nullptr;
    double* d_rhs  = nullptr;
    double* d_t_q  = nullptr;
    double* d_bw_q = nullptr;

    void SetUp() override {
        x_host.resize(N);
        N_host.resize(N, 0.0);
        rhs_host.resize(N, 0.0);

        x_host[0] = V_MIN;
        for (int i = 1; i < N; ++i)
            x_host[i] = x_host[i-1] * R;

        CUDA_ASSERT(cudaMalloc(&d_x,    N       * sizeof(double)));
        CUDA_ASSERT(cudaMalloc(&d_N,    N       * sizeof(double)));
        CUDA_ASSERT(cudaMalloc(&d_rhs,  N       * sizeof(double)));
        CUDA_ASSERT(cudaMalloc(&d_t_q,  N_QUAD  * sizeof(double)));
        CUDA_ASSERT(cudaMalloc(&d_bw_q, N_QUAD  * sizeof(double)));

        CUDA_ASSERT(cudaMemcpy(d_x,    x_host.data(),
                               N      * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_t_q,  t_q_host.data(),
                               N_QUAD * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_bw_q, bw_q_host.data(),
                               N_QUAD * sizeof(double), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        cudaFree(d_x);   cudaFree(d_N);
        cudaFree(d_rhs); cudaFree(d_t_q); cudaFree(d_bw_q);
    }

    cudaError_t run(pbe_cuda::BreakageParams& p) {
        CUDA_CHECK_RET(cudaMemcpy(d_N, N_host.data(),
                               N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RET(cudaMemset(d_rhs, 0, N * sizeof(double)));

        cudaError_t err = pbe_cuda::launch_breakage_rhs(
            d_N, d_x, d_t_q, d_bw_q, d_rhs, p);
        CUDA_CHECK_RET(cudaDeviceSynchronize());

        CUDA_CHECK_RET(cudaMemcpy(rhs_host.data(), d_rhs,
                               N * sizeof(double), cudaMemcpyDeviceToHost));
        return err;
    }

    pbe_cuda::BreakageParams make_params(pbe_cuda::BreakageSelection sel,
                                         double S0    = 1.0e-3,
                                         double v_ref = 1.0e-18,
                                         double alpha = 1.0,
                                         double v_min = 0.0) {
        pbe_cuda::BreakageParams p;
        p.n         = N;
        p.n_quad    = N_QUAD;
        p.selection = sel;
        p.S0        = S0;
        p.v_ref     = v_ref;
        p.alpha     = alpha;
        p.v_min     = v_min;
        p.block_size= 256;
        return p;
    }

    double M1() {
        double v = 0.0;
        for (int i = 0; i < N; ++i) v += N_host[i] * x_host[i];
        return v;
    }
};

// ===========================================================================
// SMOKE TESTS — all selection functions must launch without error
// ===========================================================================

TEST_F(BreakageTest, SmokeConstant) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::BreakageSelection::Constant);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(BreakageTest, SmokeLinear) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::BreakageSelection::Linear,
                         1.0e-3, x_host[N/2]);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(BreakageTest, SmokePowerLaw) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::BreakageSelection::PowerLaw,
                         1.0e-3, x_host[N/2], 2.0);
    EXPECT_EQ(run(p), cudaSuccess);
}

TEST_F(BreakageTest, SmokeThreshold) {
    N_host[N/2] = 1.0e14;
    auto p = make_params(pbe_cuda::BreakageSelection::Threshold,
                         1.0e-3, 1.0, 1.0, x_host[N/4]);
    EXPECT_EQ(run(p), cudaSuccess);
}

// ===========================================================================
// CONSERVATION TESTS — M1 preserved (dM1/dt = 0) for all selection functions
// ===========================================================================

class BreakageConservationTest : public BreakageTest {
protected:
    void check_volume_conservation(pbe_cuda::BreakageParams p) {
        for (int i = 0; i < N; ++i)
            N_host[i] = 1.0e12 * std::exp(-static_cast<double>(i) / 10.0);

        ASSERT_EQ(run(p), cudaSuccess);

        double dM1  = 0.0;
        double M1_v = 0.0;
        for (int i = 0; i < N; ++i) {
            dM1  += rhs_host[i] * x_host[i];
            M1_v += N_host[i]   * x_host[i];
        }

        // Symmetric binary daughter: fragments land exactly on grid points for
        // geometric grids with r=2^(1/3), giving near-exact conservation.
        // For continuous daughters (uniform, power-law), quadrature introduces
        // O(h^2) discretisation error — not a floating point error.
        // We use 1e-4 * M1 as tolerance, which is tight enough to catch real
        // conservation failures but loose enough for quadrature-limited cases.
        double tol = 1.0e-4 * std::abs(M1_v);

        EXPECT_NEAR(dM1, 0.0, tol)
            << "Volume not conserved for BreakageSelection "
            << static_cast<int>(p.selection)
            << ": dM1/dt = " << dM1
            << ", M1 = " << M1_v
            << ", rel error = " << std::abs(dM1) / M1_v;
    }
};

TEST_F(BreakageConservationTest, VolumeConservedConstant) {
    check_volume_conservation(
        make_params(pbe_cuda::BreakageSelection::Constant, 1.0e-3));
}

TEST_F(BreakageConservationTest, VolumeConservedLinear) {
    check_volume_conservation(
        make_params(pbe_cuda::BreakageSelection::Linear, 1.0e-3, x_host[N/2]));
}

TEST_F(BreakageConservationTest, VolumeConservedPowerLaw) {
    check_volume_conservation(
        make_params(pbe_cuda::BreakageSelection::PowerLaw,
                    1.0e-3, x_host[N/2], 2.0));
}

TEST_F(BreakageConservationTest, VolumeConservedThreshold) {
    // Threshold with v_min = x[N/4]: upper half of bins break, lower half do not
    check_volume_conservation(
        make_params(pbe_cuda::BreakageSelection::Threshold,
                    1.0e-3, 1.0, 1.0, x_host[N/4]));
}

// ===========================================================================
// CORRECTNESS TESTS
// ===========================================================================

// Zero population → RHS identically zero.
TEST_F(BreakageTest, ZeroPopulationGivesZeroRHS) {
    auto p = make_params(pbe_cuda::BreakageSelection::Constant);
    ASSERT_EQ(run(p), cudaSuccess);
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(rhs_host[i], 0.0) << "rhs[" << i << "] != 0 for zero population";
}

// Death term — constant selection, single populated bin j.
// Expected: rhs[j] receives death = -S0 * N[j] PLUS birth from all parents
// whose fragment lands on j. With only bin j populated, the death at j is
// -S0 * N[j], and the birth contribution from j itself lands on bin j-3
// (since x[j]/2 = x[j-3] for geometric grid with r = 2^(1/3)).
// Therefore rhs[j] == -S0 * N[j] exactly (no birth at j from itself).
TEST_F(BreakageTest, DeathTermConstantSelection) {
    constexpr int    j  = N / 2;
    constexpr double Nj = 1.0e14;
    constexpr double S0 = 1.0e-3;

    N_host[j] = Nj;
    // Use v_ref = x[j] so S(x[j]) = S0 for Linear too; here Constant.
    auto p = make_params(pbe_cuda::BreakageSelection::Constant, S0);
    ASSERT_EQ(run(p), cudaSuccess);

    double expected = -S0 * Nj;
    EXPECT_NEAR(rhs_host[j], expected, std::abs(expected) * 1.0e-10)
        << "Death term incorrect at bin " << j;
}

// Total number must increase for pure breakage (each event creates 2 fragments
// from 1 parent with symmetric binary daughter).
TEST_F(BreakageTest, TotalNumberNonDecreasing) {
    for (int i = 0; i < N; ++i)
        N_host[i] = 1.0e12;
    auto p = make_params(pbe_cuda::BreakageSelection::Constant);
    ASSERT_EQ(run(p), cudaSuccess);

    double sum_rhs = std::accumulate(rhs_host.begin(), rhs_host.end(), 0.0);
    EXPECT_GE(sum_rhs, 0.0)
        << "Total RHS is negative — breakage should not decrease total number";
}

// Threshold selection — bins below v_min must have rhs == 0 (death suppressed).
TEST_F(BreakageTest, ThresholdSuppressesBelowVmin) {
    constexpr int threshold_bin = N / 2;
    for (int i = 0; i < N; ++i)
        N_host[i] = 1.0e12;

    double v_min = x_host[threshold_bin];
    auto p = make_params(pbe_cuda::BreakageSelection::Threshold,
                         1.0e-3, 1.0, 1.0, v_min);
    ASSERT_EQ(run(p), cudaSuccess);

    // Bins strictly below the threshold bin receive no death contribution.
    // (They may receive birth from fragments of larger bins.)
    // At minimum, death term for bins below threshold must be zero.
    // We verify this by checking that rhs is non-negative below threshold_bin
    // (only birth, no death).
    for (int i = 0; i < threshold_bin; ++i)
        EXPECT_GE(rhs_host[i], 0.0)
            << "Bin " << i << " below threshold has negative rhs (unexpected death)";
}

// Invalid parameters must return error.
TEST_F(BreakageTest, InvalidNReturnsError) {
    auto p = make_params(pbe_cuda::BreakageSelection::Constant);
    p.n = 0;
    cudaError_t err = pbe_cuda::launch_breakage_rhs(
        d_N, d_x, d_t_q, d_bw_q, d_rhs, p);
    EXPECT_NE(err, cudaSuccess);
}

TEST_F(BreakageTest, NullPointerReturnsError) {
    auto p = make_params(pbe_cuda::BreakageSelection::Constant);
    cudaError_t err = pbe_cuda::launch_breakage_rhs(
        nullptr, d_x, d_t_q, d_bw_q, d_rhs, p);
    EXPECT_NE(err, cudaSuccess);
}