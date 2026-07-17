find_package(CUDAToolkit REQUIRED)

add_executable(example_breakage_linear_uniform main.cpp)

target_link_libraries(example_breakage_linear_uniform
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart
        pbe_examples_common
)

set_target_properties(example_breakage_linear_uniform PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)