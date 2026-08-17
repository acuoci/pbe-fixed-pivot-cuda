cmake_minimum_required(VERSION 3.18)

# Build with CUDA by default.  Set -DENABLE_CUDA=OFF to build a pure C++
# serial CPU library with no CUDA toolkit, nvcc, CUDA sources, or cudart link.
option(ENABLE_CUDA "Build with CUDA support" ON)

# ── CUDA architecture — MUST be set before project() ─────────────────────────
# CMake initialises CMAKE_CUDA_ARCHITECTURES during project(), so setting it
# afterwards has no effect on the default.
# Override at configure time with: cmake -DCMAKE_CUDA_ARCHITECTURES="80;86" ..
if(ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # sm_80 = A100, A30
    # sm_86 = RTX 3090, A40, RTX 30xx
    # Both are Ampere but different compute capabilities —
    # sm_86 code does NOT run on sm_80 hardware.
    set(CMAKE_CUDA_ARCHITECTURES 80 86 CACHE STRING "CUDA architectures to compile for" FORCE)
endif()

# ── Project definition ────────────────────────────────────────────────────────
if(ENABLE_CUDA)
    project(pbe_cuda
        VERSION      1.0.0
        DESCRIPTION  "Fast CUDA implementation of aggregation and breakage terms in PBE using the fixed pivot sectional method"
        LANGUAGES    CXX CUDA
    )
else()
    project(pbe_cuda
        VERSION      1.0.0
        DESCRIPTION  "Serial CPU implementation of aggregation and breakage terms in PBE using the fixed pivot sectional method"
        LANGUAGES    CXX
    )
endif()

# ── CMake modules & policies ──────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # Generates compile_commands.json for clangd/VS Code

# ── C++ / CUDA standards ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD         17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified — defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ── User-facing options ───────────────────────────────────────────────────────
option(PBE_BUILD_SHARED    "Build pbe_cuda as a shared library (.so)" OFF)
option(PBE_BUILD_STATIC    "Build pbe_cuda as a static library (.a)"  ON)
option(PBE_BUILD_EXAMPLES  "Build the worked examples"                 ON)
option(PBE_BUILD_TESTS     "Build the unit/regression tests"           ON)
option(PBE_BUILD_BENCHMARKS "Build optional performance benchmarks"    OFF)
option(PBE_ENABLE_WARNINGS "Enable extra compiler warnings"            ON)

# ── Library target(s) ─────────────────────────────────────────────────────────
set(PBE_SOURCES
    src/aggregation_cpu.cpp
    src/breakage_cpu.cpp
    src/sectional_grid.cpp
    src/source_cpu.cpp
)

set(PBE_PUBLIC_HEADERS
    include/pbe_cuda/pbe_cuda.cuh
    include/pbe_cuda/array_view.hpp
    include/pbe_cuda/backend.hpp
    include/pbe_cuda/cuda_compat.cuh
    include/pbe_cuda/cpu_pbe_model.hpp
    include/pbe_cuda/cuda_pbe_model.hpp
    include/pbe_cuda/source.cuh
    include/pbe_cuda/source_model.hpp
    include/pbe_cuda/aggregation.cuh
    include/pbe_cuda/aggregation_model.hpp
    include/pbe_cuda/breakage.cuh
    include/pbe_cuda/breakage_model.hpp
    include/pbe_cuda/model_config.hpp
    include/pbe_cuda/sectional_grid.hpp
)

if(ENABLE_CUDA)
    list(APPEND PBE_SOURCES
        src/aggregation.cu
        src/breakage.cu
        src/source.cu
        src/utils.cu
    )
endif()

# Helper function to configure common properties on a library target
function(pbe_configure_target target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    set_target_properties(${target_name} PROPERTIES
        POSITION_INDEPENDENT_CODE   ON
        VERSION                     ${PROJECT_VERSION}
        SOVERSION                   ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER               "${PBE_PUBLIC_HEADERS}"
    )
    target_compile_definitions(${target_name}
        PUBLIC $<$<BOOL:${ENABLE_CUDA}>:PBE_ENABLE_CUDA>
    )
    if(ENABLE_CUDA)
        set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
    if(PBE_ENABLE_WARNINGS)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:  -Wall -Wextra -Wpedantic>
        )
        if(ENABLE_CUDA)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall,-Wextra>
            )
        endif()
    endif()
endfunction()

if(PBE_BUILD_STATIC)
    add_library(pbe_cuda_static STATIC ${PBE_SOURCES})
    pbe_configure_target(pbe_cuda_static)
    set_target_properties(pbe_cuda_static PROPERTIES OUTPUT_NAME pbe_cuda)
    add_library(pbe_cuda::static ALIAS pbe_cuda_static)
    message(STATUS "pbe_cuda: building static library")
endif()

if(PBE_BUILD_SHARED)
    add_library(pbe_cuda_shared SHARED ${PBE_SOURCES})
    pbe_configure_target(pbe_cuda_shared)
    set_target_properties(pbe_cuda_shared PROPERTIES OUTPUT_NAME pbe_cuda)
    add_library(pbe_cuda::shared ALIAS pbe_cuda_shared)
    message(STATUS "pbe_cuda: building shared library")
endif()

# Convenience alias: pbe_cuda::pbe_cuda points to whichever is built
# (static takes priority if both are built)
if(PBE_BUILD_STATIC)
    add_library(pbe_cuda::pbe_cuda ALIAS pbe_cuda_static)
elseif(PBE_BUILD_SHARED)
    add_library(pbe_cuda::pbe_cuda ALIAS pbe_cuda_shared)
endif()

# Common utilities header — interface target so examples can include it
add_library(pbe_examples_common INTERFACE)
target_include_directories(pbe_examples_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
)

# ── Sub-directories ───────────────────────────────────────────────────────────
if(PBE_BUILD_EXAMPLES)
    if(ENABLE_CUDA)
        add_subdirectory(examples/aggregation_constant)
        add_subdirectory(examples/aggregation_sum)
        add_subdirectory(examples/aggregation_product)
        add_subdirectory(examples/breakage_linear_uniform)
        add_subdirectory(examples/breakage_linear_symmetricbinary)
        add_subdirectory(examples/flocculation_two_stage)
    else()
        message(STATUS "pbe_cuda: examples are skipped when ENABLE_CUDA=OFF")
    endif()
endif()

if(PBE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PBE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ── Installation rules ────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(PBE_BUILD_STATIC)
    install(TARGETS pbe_cuda_static
        EXPORT      pbe_cudaTargets
        ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pbe_cuda
    )
endif()
if(PBE_BUILD_SHARED)
    install(TARGETS pbe_cuda_shared
        EXPORT      pbe_cudaTargets
        LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pbe_cuda
    )
endif()

# Export targets so downstream projects can use find_package(pbe_cuda)
install(EXPORT pbe_cudaTargets
    FILE        pbe_cudaTargets.cmake
    NAMESPACE   pbe_cuda::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pbe_cuda
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pbe_cudaConfigVersion.cmake"
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pbe_cudaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pbe_cudaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pbe_cuda
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/pbe_cudaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/pbe_cudaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pbe_cuda
)
