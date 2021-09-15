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
    AlgorithmBase(const NetworKit::Graph& graph)
        : input_graph_(graph)
    {}

    virtual size_t do_switches(std::mt19937_64 &gen, size_t num_switches, bool autocorrelation) = 0;

    virtual NetworKit::Graph get_graph() = 0;

    virtual void enable_logging(unsigned l = 1) {
        log_level_ = l;
    };

protected:
    const NetworKit::Graph& input_graph_;
    unsigned log_level_ = 0;

};

}