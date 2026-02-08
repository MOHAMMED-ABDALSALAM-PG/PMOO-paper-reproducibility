file(GLOB SUITES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/suites/*.cu)

set(PROBLEMS_SRC
    ${SUITES_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.cu
)

add_library(problems STATIC ${PROBLEMS_SRC})

target_include_directories(problems PUBLIC ${CMAKE_SOURCE_DIR}/include)