# =============================================================================
# tests/CMakeLists.txt
#
# Regression and unit tests for pbe_cuda.
# Uses GoogleTest, either from a system installation or downloaded automatically.
#
# Priority (in order):
#   1. If GTEST_ROOT is set:    use the provided installation (no download)
#   2. If GTest is found in standard paths: use it (no download)
#   3. Otherwise:               download v1.12.1 via FetchContent
#
# Usage examples:
#   # Let CMake find or download automatically:
#   cmake .. -DPBE_BUILD_TESTS=ON
#
#   # Point to a specific installation (e.g. on a cluster without internet):
#   cmake .. -DPBE_BUILD_TESTS=ON -DGTEST_ROOT=/path/to/gtest
#
# Run tests:
#   ctest --test-dir build --output-on-failure -V
# =============================================================================

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ── GoogleTest: system install or FetchContent fallback ───────────────────────
if(DEFINED GTEST_ROOT OR DEFINED ENV{GTEST_ROOT})
    # --- Option 1: explicit path provided by the user ------------------------
    message(STATUS "pbe_cuda tests: using GoogleTest from GTEST_ROOT")
    find_package(GTest REQUIRED)

else()
    # --- Try to find a system installation first -----------------------------
    find_package(GTest QUIET)

    if(GTest_FOUND)
        # --- Option 2: found in standard system paths ------------------------
        message(STATUS "pbe_cuda tests: using system GoogleTest ${GTest_VERSION}")

    else()
        # --- Option 3: download via FetchContent -----------------------------
        message(STATUS "pbe_cuda tests: GoogleTest not found — downloading v1.12.1")

        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        58d77fa8070e8cec2dc1ed015d66b454c8d78850  # v1.12.1
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)

    endif()
endif()

# ── Helper macro ──────────────────────────────────────────────────────────────
macro(pbe_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target}
        PRIVATE
            pbe_cuda::pbe_cuda
            GTest::gtest_main
    )
    if(ENABLE_CUDA)
        target_link_libraries(${target} PRIVATE CUDA::cudart)
    endif()
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD          17
        CXX_STANDARD_REQUIRED ON
    )
    include(GoogleTest)
    gtest_discover_tests(${target})
endmacro()

pbe_add_test(test_cpu_rhs unit/test_cpu_rhs.cpp)

if(ENABLE_CUDA)
    pbe_add_test(test_aggregation unit/test_aggregation.cpp)
    pbe_add_test(test_breakage    unit/test_breakage.cpp)
endif()
