#pragma once

#include <random>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>
#include <es/ParallelEdgeSet.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/LinearCongruentialMap.hpp>

namespace es {

    struct AlgorithmParallelNaiveGlobalGaps : public AlgorithmBase {
        using edge_set_type = ParallelEdgeSet<>;
        using lin_con_map_t = linear_congruential_map::LinearCongruentialMap;


    public:
        AlgorithmParallelNaiveGlobalGaps(const NetworKit::Graph &graph, double load_factor = 2.0)
                : AlgorithmBase(graph), edge_set_(graph.numberOfEdges(), load_factor) {
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

            const auto m = edge_list_.size();
            size_t num_rounds = 2 * (num_switches / edge_list_.size());

            size_t successful_switches = 0;
            size_t sync_rejects = 0;

            const auto m_prime = linear_congruential_map::next_prime(edge_list_.size());
            for(size_t round=0; round < num_rounds; ++round) {
                if (round)
                    edge_set_.rebuild();

                incpwl::ScopedTimer timer;

                const lin_con_map_t map = linear_congruential_map::LinearCongruentialMap::get_map(m_prime, gen);

#pragma omp parallel reduction(+:successful_switches,sync_rejects)
                {
                    auto tid = static_cast<unsigned>(omp_get_thread_num());
                    auto end = edge_list_.size() - 1;

#pragma omp for schedule(dynamic,1024)
                    for(size_t i=0; i<end; i+=2) {
                        const auto id1 = map(i);
                        const auto id2 = map(i+1);

                        if (id1 >= m || id2 >= m) continue;

                        auto e1 = edge_list_[id1];
                        auto e2 = edge_list_[id2];

                        auto [u, v] = to_nodes(e1);
                        auto [x, y] = to_nodes(e2);

                        swap_if(e1 < e2, x, y);

                        const edge_t e3 = to_edge(u, x);
                        const edge_t e4 = to_edge(v, y);

                        bool trivial_reject = (u == x || v == y || e3 == e4 || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4);
                        if (trivial_reject)
                            continue;

                        auto ticket3 = edge_set_.insert(u, x, tid);
                        if (!ticket3)
                            continue; // prevent multi-edges

                        auto ticket4 = edge_set_.insert(v, y, tid);
                        if (!ticket4) {
                            edge_set_.erase_and_release(ticket3);
                            continue; // prevent multi-edges
                        }

                        // successful: commit new edges
                        edge_list_[id1] = to_edge(u, x);
                        edge_list_[id2] = to_edge(v, y);

                        edge_set_.release(ticket3);
                        edge_set_.release(ticket4);

                        // and erase the old ones
                        edge_set_.erase(u, v);
                        edge_set_.erase(x, y);

                        ++successful_switches;
                    }
                }

                if (logging_)
                    timer.report("round");
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
        bool logging_{false};

    };

}
