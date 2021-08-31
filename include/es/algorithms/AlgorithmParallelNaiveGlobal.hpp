#pragma once

#include <random>
#include <vector>

#include <shuffle/algorithms/InplaceScatterShuffle.hpp>

#include <es/algorithms/AlgorithmBase.hpp>
#include <es/ParallelEdgeSet.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>

namespace es {

struct AlgorithmParallelNaiveGlobal : public AlgorithmBase {
    using edge_set_type = ParallelEdgeSet<>;

public:
    AlgorithmParallelNaiveGlobal(const NetworKit::Graph &graph, double load_factor = 2.0)
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

        size_t num_rounds = 2 * (num_switches / edge_list_.size());

        size_t successful_switches = 0;
        size_t sync_rejects = 0;

        while (true) {
            incpwl::ScopedTimer timer;

            shuffle::GeneratorProvider gen_prov(gen);
            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);

            #pragma omp parallel reduction(+:successful_switches,sync_rejects)
            {
                auto tid = static_cast<unsigned>(omp_get_thread_num());
                shuffle::RandomBits fair_coin;
                auto gen = gens[tid];

                size_t edges_per_thread = edge_list_.size() / omp_get_num_threads();
                size_t beg = tid * edges_per_thread;
                size_t end = tid + 1 == omp_get_num_threads() ? edge_list_.size() : beg + edges_per_thread;

                for (size_t i = beg; i + 1 < end; i += 2) {
                    auto [u, v] = to_nodes(edge_list_[i]);
                    auto [x, y] = to_nodes(edge_list_[i + 1]);

                    if (fair_coin(gen))
                        std::swap(x, y);

                    if (u == x || v == y)
                        continue; // prevent self-loops

                    auto ticket3 = edge_set_.insert(u, x, tid);
                    if (!ticket3)
                        continue; // prevent multi-edges

                    auto ticket4 = edge_set_.insert(v, y, tid);
                    if (!ticket4) {
                        edge_set_.erase_and_release(ticket3);
                        continue; // prevent multi-edges
                    }

                    // successful: commit new edges
                    edge_list_[i] = to_edge(u, x);
                    edge_list_[i + 1] = to_edge(v, y);

                    edge_set_.release(ticket3);
                    edge_set_.release(ticket4);

                    // and erase the old ones
                    edge_set_.erase(u, v);
                    edge_set_.erase(x, y);

                    ++successful_switches;
                }

                gens[tid] = gen;
            }

            num_rounds--;

            if (logging_)
                timer.report("round");

            if (!num_rounds)
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
    bool logging_{false};

};

}
