#pragma once

#include <google/dense_hash_set>

#include <es/RandomBits.hpp>
#include <es/algorithms/AlgorithmBase.hpp>

namespace es {

template<typename Set = std::unordered_set <edge_t>>
struct AlgorithmSet : public AlgorithmBase {
public:
    AlgorithmSet(const NetworKit::Graph& graph)
        : AlgorithmBase(graph)
    {
        edge_set_.max_load_factor(0.7);
        prepare_hashset(edge_set_, graph.numberOfEdges());

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            auto res = edge_set_.insert(edge);
            assert(res.second);
        });
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        shuffle::RandomBits fair_coin;

        size_t successful_switches = 0;
        while (num_switches--) {
            auto o1 = *(edge_set_.sample(gen));
            auto o2 = *(edge_set_.sample(gen));

            auto[u, v] = to_nodes(o1);
            auto[x, y] = to_nodes(o2);

            // rewire to u-x, v-y

            if (u == x || u == y || v == x || v == y)
                continue; // prevent self-loops or noops

            if (fair_coin(gen))
                std::swap(x, y);

            const auto e1 = to_edge(u, x);
            const auto e2 = to_edge(v, y);

            auto ins1 = edge_set_.insert(e1);
            if (!ins1.second) {
                continue; // prevent parallel edges
            }

            if (!edge_set_.insert(e2).second) {
                edge_set_.erase(ins1.first);
                continue; // prevent parallel edges
            }

            edge_set_.erase(o1);
            edge_set_.erase(o2);

            ++successful_switches;
        }

        return successful_switches;
    }

    NetworKit::Graph get_graph() override {
        NetworKit::Graph result(input_graph_.numberOfNodes());
        for(auto e : edge_set_) {
            auto[u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

private:
    Set edge_set_;

    template<typename HS>
    void prepare_hashset(HS &hs, size_t num_edges) {
        hs.reserve(num_edges);
    }

    template<class Value, class HashFcn, class EqualKey, class Alloc>
    void prepare_hashset(google::dense_hash_set <Value, HashFcn, EqualKey, Alloc> &hs, size_t) {
        hs.set_empty_key(std::numeric_limits<edge_t>::max());
        hs.set_deleted_key(std::numeric_limits<edge_t>::max() - 1);
    }
};



}
