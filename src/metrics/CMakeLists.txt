file(GLOB METRICS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_library(metrics STATIC ${METRICS_SRC})

target_include_directories(metrics PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(metrics PUBLIC problems)