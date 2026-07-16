# Find the CUDA toolkit so we can link cudart and get CUDA include paths.
# This is needed because main.cpp includes cuda_runtime.h directly and is
# compiled by the host C++ compiler, not nvcc.
find_package(CUDAToolkit REQUIRED)

add_executable(example_homogeneous_batch
    main.cpp
)

target_link_libraries(example_homogeneous_batch
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart          # provides cuda_runtime.h include path + libcudart
)

set_target_properties(example_homogeneous_batch PROPERTIES
    CXX_STANDARD          17
    CXX_STANDARD_REQUIRED ON
)