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

# ── Helper function ───────────────────────────────────────────────────────────
function(pbe_add_test target source)
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
    if(ARGN)
        gtest_discover_tests(${target} PROPERTIES LABELS "${ARGN}")
    else()
        gtest_discover_tests(${target})
    endif()
    set(PBE_TEST_TARGETS ${PBE_TEST_TARGETS} ${target} PARENT_SCOPE)
endfunction()

pbe_add_test(test_array_view unit/test_array_view.cpp unit)
pbe_add_test(test_backend unit/test_backend.cpp unit)
pbe_add_test(test_aggregation_model unit/test_aggregation_model.cpp unit)
pbe_add_test(test_breakage_model unit/test_breakage_model.cpp unit)
pbe_add_test(test_cpu_pbe_model unit/test_cpu_pbe_model.cpp unit)
pbe_add_test(test_fixed_pivot unit/test_fixed_pivot.cpp unit)
pbe_add_test(test_model_config unit/test_model_config.cpp unit)
pbe_add_test(test_sectional_grid unit/test_sectional_grid.cpp unit)
pbe_add_test(test_source_model unit/test_source_model.cpp unit)

pbe_add_test(test_cpu_rhs verification/test_cpu_rhs.cpp verification)

if(ENABLE_CUDA)
    pbe_add_test(test_cuda_backend unit/test_cuda_backend.cpp unit cuda)
    pbe_add_test(test_cuda_pbe_model unit/test_cuda_pbe_model.cpp unit cuda)
    pbe_add_test(test_aggregation unit/test_aggregation.cpp verification cuda)
    pbe_add_test(test_breakage    unit/test_breakage.cpp verification cuda)
endif()

add_custom_target(check_unit
    COMMAND ${CMAKE_CTEST_COMMAND} --label-regex unit --output-on-failure
    DEPENDS ${PBE_TEST_TARGETS}
    USES_TERMINAL
)

add_custom_target(check_verification
    COMMAND ${CMAKE_CTEST_COMMAND} --label-regex verification --output-on-failure
    DEPENDS ${PBE_TEST_TARGETS}
    USES_TERMINAL
)

if(ENABLE_CUDA)
    add_custom_target(check_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --label-regex cuda --output-on-failure
        DEPENDS ${PBE_TEST_TARGETS}
        USES_TERMINAL
    )
endif()
