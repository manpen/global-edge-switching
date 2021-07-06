#pragma once

#include <span>
#include <vector>
#include <random>

#include <es/Graph.hpp>


namespace es {

struct AlgorithmBase {
public:
    // interface to be implemented
    AlgorithmBase() = delete;
    AlgorithmBase(const Graph& graph)
        : input_graph_(graph)
    {}

    virtual size_t do_switches(std::mt19937_64 &gen, size_t num_switches) = 0;

    virtual Graph get_graph() = 0;

protected:
    const Graph& input_graph_;

};

}