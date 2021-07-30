#pragma once

#include <random>
#include <thread>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>

#include <shuffle/algorithms/FisherYates.hpp>
#include <shuffle/algorithms/InplaceScatterShuffle.hpp>

namespace es {

template<size_t NumThreads, typename Set>
struct AlgorithmParallelGlobalES : public AlgorithmBase {
public:
    AlgorithmParallelGlobalES(const NetworKit::Graph &graph)
    : AlgorithmBase(graph), edge_set_(graph.numberOfEdges()) {
        edge_list_.reserve(graph.numberOfEdges());

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
            auto res = edge_set_.insert(edge);
            assert(res);
        });
    }

    size_t do_switches(std::mt19937_64& gen, size_t num_switches) {
        assert(!edge_list_.empty());

        size_t num_rounds = 2 * (num_switches / edge_list_.size());
        std::vector<size_t> successful_local(NumThreads, 0);
        std::vector<std::mt19937_64> gen_local(NumThreads, std::mt19937_64(gen()));

        omp_set_num_threads(NumThreads);

        while (num_rounds--) {
            shuffle::GeneratorProvider gen_prov(gen);
            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);

            #pragma omp parallel num_threads(NumThreads)
            {
                const auto thread_id = omp_get_thread_num();

                size_t edges_per_thread = edge_list_.size() / NumThreads;
                size_t beg = thread_id * edges_per_thread;
                size_t end = thread_id + 1 == NumThreads ? edge_list_.size() : beg + edges_per_thread;

                std::mt19937_64 &gen = gen_local[thread_id];
                shuffle::RandomBits fair_coin;

                for (size_t i = beg; i + 1 < end; i += 2) {
                    const auto index1 = i;
                    const auto index2 = i + 1;

                    auto [u, v] = to_nodes(edge_list_[index1]);
                    auto [x, y] = to_nodes(edge_list_[index2]);

                    if (fair_coin(gen))
                        std::swap(x, y);

                    // rewire to u-x, v-y
                    if (u == x || v == y) continue; // prevent self-loops

                    const edge_t e1 = to_edge(u, x);
                    const edge_t e2 = to_edge(v, y);

                    if (!edge_set_.insert(e1)) {
                        continue; // prevent self-loops
                    }

                    if (!edge_set_.insert(e2)) {
                        edge_set_.erase(e1);
                        continue; // prevent self-loops
                    }

                    edge_set_.erase(edge_list_[index1]);
                    edge_set_.erase(edge_list_[index2]);

                    edge_list_[index1] = e1;
                    edge_list_[index2] = e2;

                    successful_local[thread_id]++;
                }
            }
        }

        return std::accumulate(successful_local.begin(), successful_local.end(), 0);
    }

    NetworKit::Graph get_graph() override {
        NetworKit::Graph result(input_graph_.numberOfNodes());
        for (auto e : edge_list_) {
            auto [u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

private:
    std::vector<edge_t> edge_list_;
    Set edge_set_;
};

}