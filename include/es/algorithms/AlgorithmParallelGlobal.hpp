#pragma once

#include <random>
#include <vector>

#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/EdgeDependencies.hpp>

namespace es {

struct AlgorithmParallelGlobal : public AlgorithmBase {
    using EdgeDependenciesStore = EdgeDependencies<edge_hash_crc32>;

public:
    AlgorithmParallelGlobal(const NetworKit::Graph& graph, double load_factor = 4.0)
        : AlgorithmBase(graph),
        edge_dependencies(graph.numberOfEdges(), load_factor) {
        edge_list_.reserve(graph.numberOfEdges());

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
        });
    }

    size_t do_switches(std::mt19937_64& gen, size_t num_switches) {
        const auto num_switches_requested = num_switches;
        assert(!edge_list_.empty());

        size_t num_rounds = 2 * (num_switches / edge_list_.size());
        size_t successful_switches = 0;

        shuffle::GeneratorProvider gen_prov(gen);
        for(size_t round=0; round < num_rounds; ++round) {
            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);
            successful_switches += do_round();
            edge_dependencies.next_round();
        }

        return successful_switches;
    }

    size_t do_round(bool logging = false) {
        size_t successful_switches = 0;

        if (edge_list_.size() % 2) {
            // in case we've an uneven number of edges, the last one wont be switched;
            // we still need to announce that it exists
            edge_dependencies.announce_erase(edge_list_.back(), EdgeDependenciesStore::kNone_);
        }

        #pragma omp parallel reduction(+:successful_switches)
        {
            const auto t = static_cast<size_t>(omp_get_num_threads());
            const auto tid = static_cast<size_t>(omp_get_thread_num());

            const size_t num_switches = edge_list_.size() / 2;
            const size_t beg = (num_switches * tid) / t;
            const size_t end = (num_switches * (tid + 1)) / t;

            auto edgeid_to_switchid = [&] (size_t edge_id) {
                return (edge_id - beg) * t + tid;
            };

            for (size_t edge_id = beg; edge_id < end; ++edge_id) {
                const auto switch_id = edgeid_to_switchid(edge_id);
                const auto e1 = edge_list_[2*edge_id];
                const auto e2 = edge_list_[2*edge_id+1];

                auto [u, v] = to_nodes(e1);
                auto [x, y] = to_nodes(e2);

                swap_if(e1 < e2, x, y);

                const auto e3 = to_edge(u, x);
                const auto e4 = to_edge(v, y);

                if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) { // prevent self-loops
                    edge_dependencies.announce_erase(e1, EdgeDependenciesStore::kNone_);
                    edge_dependencies.announce_erase(e2, EdgeDependenciesStore::kNone_);
                    continue;
                }

                edge_dependencies.announce_erase(e1, switch_id);
                edge_dependencies.announce_erase(e2, switch_id);
                edge_dependencies.announce_insert(e3, switch_id);
                edge_dependencies.announce_insert(e4, switch_id);
            }

            #pragma omp barrier

            for (size_t edge_id = beg; edge_id < end; ++edge_id) {
                const auto switch_id = edgeid_to_switchid(edge_id);
                const auto e1 = edge_list_[2*edge_id];
                const auto e2 = edge_list_[2*edge_id+1];

                if (logging) {
                    #pragma omp critical
                    std::cout << "A" << switch_id << " ";
                }

                auto [u, v] = to_nodes(e1);
                auto [x, y] = to_nodes(e2);

                swap_if(e1 < e2, x, y);

                const auto e3 = to_edge(u, x);
                const auto e4 = to_edge(v, y);

                if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) // prevent self-loops
                    continue;

                auto wait_for_dependency = [&] (auto eid) {
                    bool erase_collision = true;
                    bool insert_collision = true;
                    do {
                        auto [erasing_switch, erase_resolved, inserting_switch, insert_resolved] = edge_dependencies
                            .lookup_dependencies(eid);
                        if (erasing_switch > switch_id) break;
                        if (erase_resolved) {
                            erase_collision = false;
                            if (inserting_switch >= switch_id) {
                                insert_collision = false;
                            }
                            if (insert_resolved) break;
                        }
                    } while(erase_collision || insert_collision);

                    return erase_collision || insert_collision;
                };

                auto e3_collision = wait_for_dependency(e3);
                auto collision = e3_collision || wait_for_dependency(e4);

                if (collision) {
                    edge_dependencies.announce_erase_failed(e1, switch_id);
                    edge_dependencies.announce_erase_failed(e2, switch_id);
                    edge_dependencies.announce_insert_failed(e3, switch_id);
                    edge_dependencies.announce_insert_failed(e4, switch_id);
                    continue;
                }

                edge_list_[2*edge_id] = e3;
                edge_list_[2*edge_id+1] = e4;

                edge_dependencies.announce_erase_succeeded(e1, switch_id);
                edge_dependencies.announce_erase_succeeded(e2, switch_id);
                edge_dependencies.announce_insert_succeeded(e3, switch_id);
                edge_dependencies.announce_insert_succeeded(e4, switch_id);

                if (logging) {
                    #pragma omp critical
                    std::cout << "S" << switch_id << " ";
                }

                ++successful_switches;
            }
        }

        if (logging) std::cout << std::endl;

        return successful_switches;
    }

    void do_switches(const std::vector<size_t>& rho, size_t num_threads) {
        assert(!edge_list_.empty());
        assert(!rho.empty());

        std::vector<edge_t> edge_list_permuted(rho.size());
        size_t edges_per_thread = rho.size() / num_threads;
        size_t num_edges = edges_per_thread * num_threads;
        for (size_t i = 0; i + 1 < num_edges; i += 2) {
            size_t switch_id = i / 2;
            size_t thread_id = switch_id % num_threads;
            size_t local_switch_id = switch_id / num_threads;
            size_t offset = edges_per_thread * thread_id + 2 * local_switch_id;
            edge_list_permuted[offset] = edge_list_[rho[i]];
            edge_list_permuted[offset + 1] = edge_list_[rho[i + 1]];
        }
        edge_list_ = edge_list_permuted;

        do_round(true);
    }

    NetworKit::Graph get_graph() override {
        #ifndef NDEBUG
        {
            auto sorted = edge_list_;
            std::sort(sorted.begin(), sorted.end());
            auto copy = sorted;
            assert(std::unique(copy.begin(), copy.end()) == copy.end());
        }
        #endif

        NetworKit::Graph result(input_graph_.numberOfNodes());
        for (auto e : edge_list_) {
            auto [u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

private:
    EdgeDependenciesStore edge_dependencies;
    std::vector<edge_t> edge_list_;
};

}
