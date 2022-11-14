add_library(FROLS_Sycl INTERFACE)
target_include_directories(FROLS_Sycl INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(FROLS_Sycl INTERFACE ${ONEAPI_LIBS})
