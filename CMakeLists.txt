cmake_minimum_required(VERSION 3.28)

project(pmoo LANGUAGES CUDA CXX)

set(CUDA_PROPAGATE_HOST_FLAGS OFF) # Cant be set through CMakePresets.json

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/src/problems)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/metrics)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/algorithms)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/benchmark)
