add_executable(example_homogeneous_batch
    main.cpp
)

target_link_libraries(example_homogeneous_batch
    PRIVATE pbe_cuda::pbe_cuda
)

set_target_properties(example_homogeneous_batch PROPERTIES
    CXX_STANDARD          20
    CXX_STANDARD_REQUIRED ON
)