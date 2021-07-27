#pragma once

#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>

namespace es {

template<size_t NumThreads, size_t HashTableSize, typename HashFcn = edge_hash_crc32>
struct AlgorithmParallelGlobalES : public AlgorithmBase {
public:
    AlgorithmParallelGlobalES(const NetworKit::Graph &graph)
    : AlgorithmBase(graph), inserted_(HashTableSize * graph.numberOfEdges()) {
        edge_list_.reserve(graph.numberOfEdges());

        for (auto& e : inserted_) e.store(0);

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
            inserted_[hash_(edge) % inserted_.size()].fetch_add(1);
        });
    }

    size_t do_switches(std::mt19937_64& gen, size_t num_switches) {
        assert(!edge_list_.empty());

        size_t num_rounds = num_switches / (edge_list_.size() / 2.);
        size_t successful_switches = 0;
        std::vector<std::mt19937_64> gen_local(NumThreads, std::mt19937_64(gen()));

        while (num_rounds--) {
            std::shuffle(edge_list_.begin(), edge_list_.end(), gen);

            std::vector<size_t> successful_local(NumThreads, 0);
            auto perform_switches = [&](size_t thread_id, size_t beg, size_t end) {

                std::mt19937_64 &gen = gen_local[thread_id];
                shuffle::RandomBits fair_coin;

                for (size_t i = beg; i + 1 < end; i += 2) {
                    const auto index1 = i;
                    const auto index2 = i + 1;

                    const edge_t e1 = edge_list_[index1];
                    const edge_t e2 = edge_list_[index2];

                    auto[u, v] = to_nodes(e1);
                    auto[x, y] = to_nodes(e2);

                    if (fair_coin(gen))
                        std::swap(x, y);

                    // rewire to u-x, v-y
                    if (u == x || v == y) continue; // prevent self-loops

                    const edge_t e3 = to_edge(u, x);
                    const edge_t e4 = to_edge(v, y);

                    // attempt to claim creation of e1 and e2
                    const size_t e3_insert_id = hash_(e3) % inserted_.size();
                    const size_t e4_insert_id = hash_(e4) % inserted_.size();
                    size_t expected3 = 0;
                    if (!inserted_[e3_insert_id].compare_exchange_strong(expected3, 1,
                                                                         std::memory_order_release,
                                                                         std::memory_order_relaxed)) continue;
                    size_t expected4 = 0;
                    if (!inserted_[e4_insert_id].compare_exchange_strong(expected4, 1,
                                                                         std::memory_order_release,
                                                                         std::memory_order_relaxed)) {
                        inserted_[e3_insert_id].store(0, std::memory_order_release);
                        continue;
                    }

                    edge_list_[index1] = e3;
                    edge_list_[index2] = e4;

                    // free the edges
                    inserted_[hash_(e1) % inserted_.size()].fetch_sub(1);
                    inserted_[hash_(e2) % inserted_.size()].fetch_sub(1);

                    ++successful_local[thread_id];
                }
            };

            std::vector<std::thread> threads;
            size_t edges_per_thread = edge_list_.size() / NumThreads;
            for (size_t t = 0; t < NumThreads; ++t) {
                size_t beg = t * edges_per_thread;
                size_t end = t + 1 == NumThreads ? edge_list_.size() : beg + edges_per_thread;
                threads.emplace_back(perform_switches, t, beg, end);
            }
            for (size_t t = 0; t < NumThreads; ++t) {
                threads[t].join();
                successful_switches += successful_local[t];
            }
        }

        return successful_switches;
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
    std::vector<std::atomic<size_t>> inserted_;
    std::vector<edge_t> edge_list_;
    HashFcn hash_;
};

}