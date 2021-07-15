#pragma once

#include <es/RandomBits.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <es/AdjacencyVector.hpp>

namespace es {

struct AlgorithmAdjacencyVector : public AlgorithmBase {
public:
    AlgorithmAdjacencyVector(const Graph& input_graph)
        : AlgorithmBase(input_graph),
        graph_(input_graph.degrees())
    {
        edges_.reserve(input_graph.number_of_edges());

        for(auto edge : input_graph.edges()) {
            auto [u, v] = to_nodes(edge);
            graph_.add_edge(u, v);
            edges_.push_back(edge);
        }
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        shuffle::RandomBits fair_coin;

        std::uniform_int_distribution<size_t> distr(0, edges_.size() - 1);

        size_t successful_switches = 0;
        while (num_switches--) {
            auto i1 = distr(gen);
            auto i2 = distr(gen);

            auto [u, v] = to_nodes(edges_[i1]);
            auto [x, y] = to_nodes(edges_[i2]);

            // rewire to u-x, v-y
            if (u == x || u == y || v == x || v == y)
                continue; // prevent self-loops or noops

            if (fair_coin(gen))
                std::swap(x, y);

            if (graph_.has_edge(u,x) || graph_.has_edge(v, y))
                continue;

            graph_.remove_edge(u, v);
            graph_.remove_edge(x, y);
            graph_.add_edge(u, x);
            graph_.add_edge(y, v);

            edges_[i1] = to_edge(u, x);
            edges_[i2] = to_edge(v, y);

            ++successful_switches;
        }

        return successful_switches;
    }

    Graph get_graph() override {
        Graph result(input_graph_.number_of_nodes(), input_graph_.number_of_edges());

        graph_.for_each([&] (auto u, auto v) {result.add_edge(u, v);});

        return result;
    }

private:
    AdjacencyVector graph_;
    std::vector<edge_t> edges_;
};



}
