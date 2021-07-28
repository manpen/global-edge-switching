#pragma once

#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>

namespace es {

template<size_t NumThreads, size_t HashTableSize, typename HashFcn = edge_hash_crc32>
struct AlgorithmParallelVector : public AlgorithmBase {
public:
    AlgorithmParallelVector(const NetworKit::Graph &graph)
    : AlgorithmBase(graph), available_(graph.numberOfEdges()), inserted_(HashTableSize * graph.numberOfEdges()) {
        edge_list_.reserve(graph.numberOfEdges());

        for (auto& e : available_) e.store(true);
        for (auto& e : inserted_) e.store(0);

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
            inserted_[hash_(edge) % inserted_.size()].fetch_add(1);
        });
    }

    size_t do_switches(std::mt19937_64& gen, size_t num_switches) {
        assert(!edge_list_.empty());

        size_t switches_per_thread = num_switches / NumThreads;
        size_t successful_switches = 0;
        std::vector<size_t> successful_local(NumThreads, 0);
        std::vector<std::mt19937_64> gen_local(NumThreads, std::mt19937_64(gen()));
        std::uniform_int_distribution<size_t> edge_distr(0, edge_list_.size() - 1);

        auto perform_switches = [&](size_t thread_id, size_t num_local) {

            std::mt19937_64& gen = gen_local[thread_id];
            shuffle::RandomBits fair_coin;

            while (num_local--) {
                const auto index1 = edge_distr(gen);
                const auto index2 = edge_distr(gen);

                if (index1 == index2) continue;

                // attempt to claim these edges
                bool available;
                do {
                    available = true;
                    available_[index1].compare_exchange_weak(available, false,
                                                             std::memory_order_release,
                                                             std::memory_order_relaxed);
                    if (available) {
                        available_[index2].compare_exchange_weak(available, false,
                                                                 std::memory_order_release,
                                                                 std::memory_order_relaxed);
                        if (available) break;
                        available_[index1].store(true, std::memory_order_release);
                    }
                } while (!available);

                const edge_t e1 = edge_list_[index1];
                const edge_t e2 = edge_list_[index2];

                auto [u, v] = to_nodes(e1);
                auto [x, y] = to_nodes(e2);

                if (fair_coin(gen))
                    std::swap(x, y);

                // rewire to u-x, v-y
                if (u == x || v == y) {
                    available_[index1].store(true, std::memory_order_release);
                    available_[index2].store(true, std::memory_order_release);
                    continue; // prevent self-loops
                }

                const edge_t e3 = to_edge(u, x);
                const edge_t e4 = to_edge(v, y);

                // attempt to claim creation of e1 and e2
                const size_t e3_insert_id = hash_(e3) % inserted_.size();
                const size_t e4_insert_id = hash_(e4) % inserted_.size();
                size_t expected3 = 0;
                if (!inserted_[e3_insert_id].compare_exchange_strong(expected3, 1,
                                                                     std::memory_order_release,
                                                                     std::memory_order_relaxed)) {
                    available_[index1].store(true, std::memory_order_release);
                    available_[index2].store(true, std::memory_order_release);
                    continue;
                }
                size_t expected4 = 0;
                if (!inserted_[e4_insert_id].compare_exchange_strong(expected4, 1,
                                                                     std::memory_order_release,
                                                                     std::memory_order_relaxed)) {
                    available_[index1].store(true, std::memory_order_release);
                    available_[index2].store(true, std::memory_order_release);
                    inserted_[e3_insert_id].store(0, std::memory_order_release);
                    continue;
                }

                edge_list_[index1] = e3;
                edge_list_[index2] = e4;

                // free the edges
                available_[index1].store(true, std::memory_order_release);
                available_[index2].store(true, std::memory_order_release);
                inserted_[hash_(e1) % inserted_.size()].fetch_sub(1);
                inserted_[hash_(e2) % inserted_.size()].fetch_sub(1);

                ++successful_local[thread_id];
            }
        };

        std::vector<std::thread> threads;
        for (size_t t = 0; t < NumThreads; ++t) {
            threads.emplace_back(perform_switches, t, switches_per_thread);
        }
        for (size_t t = 0; t < NumThreads; ++t) {
            threads[t].join();
            successful_switches += successful_local[t];
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
    std::vector<std::atomic<bool>> available_;
    std::vector<std::atomic<size_t>> inserted_;
    std::vector<edge_t> edge_list_;
    HashFcn hash_;
};

}