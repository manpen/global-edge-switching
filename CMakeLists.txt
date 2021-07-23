cmake_minimum_required(VERSION 3.18)
project(edge_switching)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_subdirectory(extlib/hopscotch-map)
add_subdirectory(extlib/networkit)
add_subdirectory(extlib/robin-map)
add_subdirectory(extlib/range-v3)

add_library(libedgeswitch INTERFACE)
target_include_directories(libedgeswitch
        INTERFACE include)
target_link_libraries(libedgeswitch
        INTERFACE
         tlx range-v3 networkit
         hopscotch_map robin_map )

add_executable(edge_switching main.cpp)
target_link_libraries(edge_switching PRIVATE libedgeswitch)

enable_testing()
add_subdirectory(source/tests)