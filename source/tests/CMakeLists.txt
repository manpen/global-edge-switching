add_executable(TestAlgorithms TestAlgorithms.cpp)
target_link_libraries(TestAlgorithms libedgeswitch)

add_executable(TestParallelEdgeSet TestParallelEdgeSet.cpp)
target_link_libraries(TestParallelEdgeSet libedgeswitch)

add_executable(TestAlgorithmParallelGlobal TestAlgorithmParallelGlobal.cpp)
target_link_libraries(TestAlgorithmParallelGlobal libedgeswitch)

add_executable(TestLinearCongruentialMap TestLinearCongruentialMap.cpp)
target_link_libraries(TestLinearCongruentialMap libedgeswitch)

add_executable(TestAutocorrelationAnalysis TestAutocorrelationAnalysis.cpp)
target_link_libraries(TestAutocorrelationAnalysis libedgeswitch)
