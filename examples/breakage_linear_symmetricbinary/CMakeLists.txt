find_package(CUDAToolkit REQUIRED)

add_executable(example_breakage_linear_symmetricbinary
    main.cpp
)

target_link_libraries(example_breakage_linear_symmetricbinary
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart
        pbe_examples_common
)

set_target_properties(example_breakage_linear_symmetricbinary PROPERTIES
    CXX_STANDARD          17
    CXX_STANDARD_REQUIRED ON
)