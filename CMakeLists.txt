cmake_minimum_required(VERSION 3.16.3)
project(edge_switching)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(OpenMP REQUIRED)

add_subdirectory(extlib/hopscotch-map)
add_subdirectory(extlib/networkit)
add_subdirectory(extlib/robin-map)
add_subdirectory(extlib/range-v3)

add_library(libedgeswitch INTERFACE)

if (APPLE)
    target_include_directories(libedgeswitch INTERFACE include /usr/local/Cellar/google-sparsehash/2.0.4/include/ extlib/shuffle/include)
else()
    target_include_directories(libedgeswitch INTERFACE include extlib/shuffle/include)
endif()

target_link_libraries(libedgeswitch
        INTERFACE
        tlx range-v3 networkit
        hopscotch_map robin_map OpenMP::OpenMP_CXX)

add_executable(edge_switching main.cpp)
target_link_libraries(edge_switching PRIVATE libedgeswitch)

add_executable(benchmark_shuffle benchmark_shuffle.cpp)
target_link_libraries(benchmark_shuffle PRIVATE libedgeswitch)

add_executable(benchmark_algorithms benchmark_algorithms.cpp)
target_link_libraries(benchmark_algorithms PRIVATE libedgeswitch)

enable_testing()
add_subdirectory(source/tests)