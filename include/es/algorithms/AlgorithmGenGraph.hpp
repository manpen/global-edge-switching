#pragma once

#include <memory>

#include <es/algorithms/AlgorithmBase.hpp>
#include <networkit/graph/Graph.hpp>

#include <header.h>
#include <graph_molloy_hash.h>

namespace es {

struct AlgorithmGenGraph : public AlgorithmBase {
public:
    AlgorithmGenGraph(const NetworKit::Graph &graph) //
        : AlgorithmBase(graph), num_edges_(graph.numberOfEdges()) {

        auto hard_copy = to_hard_copy(graph);
        algo_ = std::make_unique<graph_molloy_hash>(hard_copy.get());
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        return algo_->shuffle_without_connectivity_test(num_switches);
    }

    NetworKit::Graph get_graph() override {
        abort();
    }

private:
    std::unique_ptr<graph_molloy_hash> algo_;
    size_t num_edges_;

    std::unique_ptr<int[]> to_hard_copy(const NetworKit::Graph &graph) const {
        assert(graph.numberOfNodes() < std::numeric_limits<int>::max());
        assert(graph.numberOfEdges() < std::numeric_limits<int>::max() / 2);

        const auto n = static_cast<int>(graph.numberOfNodes());
        const auto m = static_cast<int>(graph.numberOfEdges());
        auto hard_copy = std::make_unique<int[]>(2+n+m);

        hard_copy[0] = n;
        hard_copy[1] = 2 * m;

        for(int u=0; u < n; ++u)
            hard_copy[u + 2] = graph.degree(u);

        auto it = hard_copy.get() + n + 2;
        for(int u=0; u < n; ++u) {
            for(int v : graph.neighborRange(u)) {
                if (u < v)
                    *(it++) = v;
            }
        }

        assert(it == hard_copy.get() + 2 + n + m);

        return hard_copy;
    }

};

}
