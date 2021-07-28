#pragma once

#include <random>
#include <thread>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>

namespace es {

template<size_t NumThreads, typename Set>
struct AlgorithmParallelVectorSet : public AlgorithmBase {
public:
    AlgorithmParallelVectorSet(const NetworKit::Graph &graph)
            : AlgorithmBase(graph), available_(graph.numberOfEdges()), edge_set_(graph.numberOfEdges()) {
        edge_list_.reserve(graph.numberOfEdges());

        for (auto& e : available_) e.store(true);

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
            auto res = edge_set_.insert(edge);
            assert(res);
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

                auto [u, v] = to_nodes(edge_list_[index1]);
                auto [x, y] = to_nodes(edge_list_[index2]);

                if (fair_coin(gen))
                    std::swap(x, y);

                // rewire to u-x, v-y
                if (u == x || v == y) {
                    available_[index1].store(true, std::memory_order_release);
                    available_[index2].store(true, std::memory_order_release);
                    continue; // prevent self-loops
                }

                const edge_t e1 = to_edge(u, x);
                const edge_t e2 = to_edge(v, y);

                if (!edge_set_.insert(e1)) {
                    available_[index1].store(true, std::memory_order_release);
                    available_[index2].store(true, std::memory_order_release);
                    continue; // prevent self-loops
                }

                if (!edge_set_.insert(e2)) {
                    available_[index1].store(true, std::memory_order_release);
                    available_[index2].store(true, std::memory_order_release);
                    edge_set_.erase(e1);
                    continue; // prevent self-loops
                }

                edge_set_.erase(edge_list_[index1]);
                edge_set_.erase(edge_list_[index2]);

                edge_list_[index1] = e1;
                edge_list_[index2] = e2;

                // free the edges
                available_[index1].store(true, std::memory_order_release);
                available_[index2].store(true, std::memory_order_release);

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
    std::vector<edge_t> edge_list_;
    Set edge_set_;
};

}