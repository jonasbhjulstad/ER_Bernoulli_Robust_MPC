add_library(Sycl_Graph INTERFACE Graph.hpp)
target_link_libraries(Sycl_Graph INTERFACE Common)
target_include_directories(Sycl_Graph INTERFACE ${SYCL_GRAPH_INCLUDE_DIRS})