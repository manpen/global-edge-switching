#pragma once

#include <es/algorithms/AlgorithmBase.hpp>
#include <networkit/randomization/EdgeSwitching.hpp>

namespace es {

struct AlgorithmNetworKit : public AlgorithmBase {
public:
    AlgorithmNetworKit(const NetworKit::Graph &graph) //
        : AlgorithmBase(graph), algo_(graph, 10, false), num_edges_(graph.numberOfEdges()) {

    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        algo_.setNumberOfSwitchesPerEdge(static_cast<double>(num_switches) / num_edges_);
        algo_.run();
        return algo_.getNumberOfAffectedEdges() / 2;
    }

    NetworKit::Graph get_graph() override {
        return algo_.getGraph();
    }

private:
    NetworKit::EdgeSwitching algo_;
    size_t num_edges_;
};

}
