find_package(CUDAToolkit REQUIRED)

add_executable(example_homogeneous_batch_breakage
    main.cpp
)

target_link_libraries(example_homogeneous_batch_breakage
    PRIVATE
        pbe_cuda::pbe_cuda
        CUDA::cudart
)

set_target_properties(example_homogeneous_batch_breakage PROPERTIES
    CXX_STANDARD          17
    CXX_STANDARD_REQUIRED ON
)