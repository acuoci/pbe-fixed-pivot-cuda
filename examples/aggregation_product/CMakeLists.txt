find_package(CUDAToolkit REQUIRED)

add_executable(example_aggregation_product main.cpp)

target_link_libraries(example_aggregation_product
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart
        pbe_examples_common
)

set_target_properties(example_aggregation_product PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)