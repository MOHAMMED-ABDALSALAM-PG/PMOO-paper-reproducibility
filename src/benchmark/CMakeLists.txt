set(BENCHMARK_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
)

add_executable(benchmark ${BENCHMARK_SRC})

target_link_libraries(benchmark PUBLIC problems metrics algorithms)