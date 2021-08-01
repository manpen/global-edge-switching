#pragma once

#include <random>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>
#include <es/ParallelEdgeSet.hpp>
#include <es/RandomBits.hpp>

namespace es {

struct AlgorithmParallelNaive : public AlgorithmBase {
    using edge_set_type = ParallelEdgeSet<>;

public:
    AlgorithmParallelNaive(const NetworKit::Graph &graph, double load_factor = 2.0, double chunk_factor = 1.0)
        : AlgorithmBase(graph), edge_set_(graph.numberOfEdges(), load_factor), chunk_factor_(chunk_factor) {
        edge_list_.reserve(graph.numberOfEdges());

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
        });

        auto m = edge_list_.size();

        #pragma omp parallel for
        for(size_t i = 0; i < m; ++i) {
            auto [u, v] = to_nodes(edge_list_[i]);
            auto unlocked_ticket = edge_set_.insert(u, v);
            assert(unlocked_ticket != nullptr);
        }
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        using ticket_t = edge_set_type::iterator_type;
        const auto num_switches_requested = num_switches;
        assert(!edge_list_.empty());

        std::vector<std::mt19937_64> gens;
        {
            auto n = omp_get_max_threads();
            gens.reserve(n);
            while(n--) {
                gens.emplace_back(gen());
            }
        }

        size_t successful_switches = 0;
        size_t sync_rejects = 0;

        while (true) {
            incpwl::ScopedTimer timer;
            auto chunk_size = std::min<size_t>(num_switches, edge_list_.capacity() * chunk_factor_);

            #pragma omp parallel reduction(+:successful_switches,sync_rejects)
            {
                auto tid = static_cast<unsigned>(omp_get_thread_num());
                shuffle::RandomBits fair_coin;
                auto gen = gens[tid];

                std::uniform_int_distribution<size_t> uniform_distr{0, edge_list_.size() - 1};
                auto sample_edge = [&] {
                    while(true) {
                        auto idx = uniform_distr(gen);
                        auto edge = __atomic_load_n(&edge_list_[idx], __ATOMIC_CONSUME);
                        auto [u, v] = to_nodes(edge);
                        auto acq = edge_set_.acquire(u, v, tid);

                        if (acq.second != nullptr)
                            return std::make_tuple(u, v, idx, acq.second);

                        ++sync_rejects;
                    }
                };

                #pragma omp for schedule(dynamic,1<<8)
                for(size_t i = 0; i < chunk_size; ++i) {
                    auto [u, v, edge_id1, ticket1] = sample_edge();
                    auto [x, y, edge_id2, ticket2] = sample_edge();

                    assert(ticket1);
                    assert(ticket2);

                    if (fair_coin(gen))
                        std::swap(x, y);


                    edge_set_type::iterator_type ticket3{nullptr};
                    edge_set_type::iterator_type ticket4{nullptr};

                    if (u != x && v != y) { // otherwise we have a self-loop and can stop early
                        ticket3 = edge_set_.insert(u, x, tid);
                        ticket4 = ticket3 ? edge_set_.insert(v, y, tid) : nullptr;
                    }

                    if (ticket4) {
                        // successful: commit new edges
                        __atomic_store_n(&edge_list_[edge_id1], to_edge(u, x), __ATOMIC_RELEASE);
                        __atomic_store_n(&edge_list_[edge_id2], to_edge(v, y), __ATOMIC_RELEASE);
                        edge_set_.release(ticket3);
                        edge_set_.release(ticket4);

                        // and erase the old ones
                        edge_set_.erase_and_release(ticket1);
                        edge_set_.erase_and_release(ticket2);

                        ++successful_switches;
                    } else {
                        // reject: erase tickets and release unmodified edges
                        if (ticket3)
                            edge_set_.erase_and_release(ticket3);

                        edge_set_.release(ticket1);
                        edge_set_.release(ticket2);
                    }
                }

                gens[tid] = gen;
            }

            num_switches -= chunk_size;

            if (logging_)
                timer.report("chunk");

            if (!num_switches)
                break;

            edge_set_.rebuild();
        }

        if (logging_) {
            std::cout << "PERF num_switches=" << num_switches_requested << ",num_successful_switches=" << successful_switches
                      << ",num_sync_rejects=" << sync_rejects << "\n";
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

    void enable_logging(bool val = true) {
        logging_ = val;
    }

private:
    std::vector<edge_t> edge_list_;
    edge_set_type edge_set_;
    double chunk_factor_;
    bool logging_{false};

};

}
