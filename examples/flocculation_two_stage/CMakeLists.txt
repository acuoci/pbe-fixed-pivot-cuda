find_package(CUDAToolkit REQUIRED)

add_executable(example_flocculation_two_stage main.cpp)

target_link_libraries(example_flocculation_two_stage
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart
        pbe_examples_common
)

set_target_properties(example_flocculation_two_stage PROPERTIES
    CXX_STANDARD          17
    CXX_STANDARD_REQUIRED ON
)