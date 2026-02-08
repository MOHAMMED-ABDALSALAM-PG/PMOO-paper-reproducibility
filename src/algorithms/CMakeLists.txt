set(ALGORITHMS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
)

add_library(algorithms STATIC ${ALGORITHMS_SRC})

target_include_directories(algorithms PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(algorithms PUBLIC problems)
