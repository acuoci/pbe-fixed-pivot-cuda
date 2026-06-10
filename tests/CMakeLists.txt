# Lightweight testing without external dependencies.
# Replace with GoogleTest later if needed:
#   find_package(GTest REQUIRED)

add_executable(test_aggregation unit/test_aggregation.cpp)
target_link_libraries(test_aggregation PRIVATE pbe_cuda::pbe_cuda)
add_test(NAME aggregation COMMAND test_aggregation)

add_executable(test_breakage unit/test_breakage.cpp)
target_link_libraries(test_breakage PRIVATE pbe_cuda::pbe_cuda)
add_test(NAME breakage COMMAND test_breakage)