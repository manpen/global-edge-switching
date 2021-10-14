#pragma once

#include <random>
#include <vector>

#include <google/dense_hash_set>
#include <es/RandomBits.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <tlx/container/btree_set.hpp>

namespace es {

template<typename Set = std::unordered_set<edge_t>>
struct AlgorithmVectorSet : public AlgorithmBase {
public:
    AlgorithmVectorSet(const NetworKit::Graph &graph) : AlgorithmBase(graph) {
        prepare_hashset(edge_set_, graph.numberOfEdges());
        edge_list_.reserve(graph.numberOfEdges());

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
            auto res = edge_set_.insert(edge);
            assert(res.second);
        });
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        assert(!edge_list_.empty());
        std::uniform_int_distribution<size_t> distr{0, edge_list_.size() - 1};
        shuffle::RandomBits fair_coin;

        size_t successful_switches = 0;
        while (num_switches--) {
            const auto index1 = distr(gen);
            const auto index2 = distr(gen);

            auto[u, v] = to_nodes(edge_list_[index1]);
            auto[x, y] = to_nodes(edge_list_[index2]);

            if (fair_coin(gen))
                std::swap(x, y);

            // rewire to u-x, v-y
            if (u == x || v == y) continue; // prevent self-loops

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

            edge_set_.erase(edge_list_[index1]);
            edge_set_.erase(edge_list_[index2]);

            edge_list_[index1] = e1;
            edge_list_[index2] = e2;
            ++successful_switches;
        }

        return successful_switches;
    }

    NetworKit::Graph get_graph() override {
        NetworKit::Graph result(input_graph_.numberOfNodes());
        for(auto e : edge_list_) {
            auto[u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

    const std::vector<edge_t>& get_edgelist() const {
        return edge_list_;
    }

private:
    std::vector<edge_t> edge_list_;
    Set edge_set_;

    template<typename HS>
    void prepare_hashset(HS &hs, size_t num_edges) {
        hs.reserve(num_edges);
    }

    template<class Value, class HashFcn, class EqualKey, class Alloc>
    void prepare_hashset(google::dense_hash_set<Value, HashFcn, EqualKey, Alloc> &hs, size_t) {
        hs.set_empty_key(std::numeric_limits<edge_t>::max());
        hs.set_deleted_key(std::numeric_limits<edge_t>::max() - 1);
    }

    void prepare_hashset(tlx::btree_set<long unsigned int> &hs, size_t num_edges) {
    }
};

}
