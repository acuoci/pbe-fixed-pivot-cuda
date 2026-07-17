# Find the CUDA toolkit so we can link cudart and get CUDA include paths.
# This is needed because main.cpp includes cuda_runtime.h directly and is
# compiled by the host C++ compiler, not nvcc.
find_package(CUDAToolkit REQUIRED)

add_executable(example_aggregation_constant
    main.cpp
)

target_link_libraries(example_aggregation_constant
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart
        pbe_examples_common
)

set_target_properties(example_aggregation_constant PROPERTIES
    CXX_STANDARD          17
    CXX_STANDARD_REQUIRED ON
)