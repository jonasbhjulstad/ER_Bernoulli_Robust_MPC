
function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen Sycl_Graph ${ONEAPI_LIBRARIES} cppitertools Math)
endfunction()


add_custom_executable(array_constraints)
add_custom_executable(sycl_mutex)

if(${SYCL_GRAPH_USE_SYCL})
add_custom_executable(bitonic_sort)
add_custom_executable(sycl_device_info)
add_custom_executable(sycl_copy_profiling)
# add_custom_executable(erdos_renyi_gen_sycl)
add_custom_executable(device_copy)
add_custom_executable(edges_sort)
add_custom_executable(sycl_parallel_cond)
add_custom_executable(sycl_kernel_emit)
add_custom_executable(sycl_rng_construction_time)
endif()
# add_executable(csv_regression csv_regression.cpp)
# target_link_libraries(csv_regression PUBLIC Sycl_Graph Eigen3::Eigen ${ONEAPI_LIBRARIES} Regressors Graph_Math Features DataFrame Sycl_Graph_Eigen)