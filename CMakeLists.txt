cmake_minimum_required(VERSION 3.15)
project(edge_switching)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_PROFILE} -O2 -g -fno-omit-frame-pointer -DNDEBUG -fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls")

option(ENABLE_ITT "Enable Intels Instrumentation API" OFF)

find_package(OpenMP REQUIRED)

add_subdirectory(extlib/hopscotch-map)
add_subdirectory(extlib/networkit)
add_subdirectory(extlib/robin-map-adopted)
add_subdirectory(extlib/range-v3)

add_library(libedgeswitch INTERFACE)

add_library(libgengraph
        extlib/gengraph/gengraph/include/box_list.cpp
        extlib/gengraph/gengraph/include/degree_sequence.cpp
        extlib/gengraph/gengraph/include/graph_molloy_hash.cpp
        extlib/gengraph/gengraph/include/graph_molloy_optimized.cpp
        extlib/gengraph/gengraph/include/powerlaw.cpp
        extlib/gengraph/gengraph/include/random.cpp
)
target_include_directories(libgengraph PUBLIC extlib/gengraph/gengraph/include/)

if (APPLE)
    target_include_directories(libedgeswitch INTERFACE include /usr/local/Cellar/google-sparsehash/2.0.4/include/ extlib/shuffle/include)
else()
    target_include_directories(libedgeswitch INTERFACE include extlib/shuffle/include)
endif()

target_link_libraries(libedgeswitch
        INTERFACE
        tlx range-v3 networkit atomic
        hopscotch_map robin_map OpenMP::OpenMP_CXX)


if(ENABLE_ITT)
    target_include_directories(libedgeswitch INTERFACE /opt/intel/vtune_profiler/sdk/include/)
    target_link_directories(libedgeswitch INTERFACE /opt/intel/vtune_profiler/sdk/lib64)
    target_link_libraries(libedgeswitch INTERFACE ittnotify ${CMAKE_DL_LIBS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -D ENABLE_ITT")
endif()

add_executable(edge_switching main.cpp)
target_link_libraries(edge_switching PRIVATE libedgeswitch libgengraph)

add_executable(parallel_naive parallel_naive.cpp)
target_link_libraries(parallel_naive PRIVATE libedgeswitch)

add_executable(benchmark_parallel_naive benchmark_parallel_naive.cpp)
target_link_libraries(benchmark_parallel_naive PRIVATE libedgeswitch)

#add_executable(benchmark_shuffle benchmark_shuffle.cpp)
#target_link_libraries(benchmark_shuffle PRIVATE libedgeswitch)

add_executable(benchmark_algorithms benchmark_algorithms.cpp)
target_link_libraries(benchmark_algorithms PRIVATE libedgeswitch libgengraph)

add_executable(count_duplicates count_duplicates.cpp)
target_link_libraries(count_duplicates libedgeswitch)

add_executable(convert_large convert_large.cpp)
target_link_libraries(convert_large libedgeswitch)

enable_testing()
add_subdirectory(source/tests)