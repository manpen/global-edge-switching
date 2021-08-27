#pragma once

#include <random>
#include <vector>

#include <es/algorithms/AlgorithmBase.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/EdgeDependencies.hpp>

namespace es {

struct AlgorithmParallel : public AlgorithmBase {
public:
    AlgorithmParallel(const NetworKit::Graph& graph, double load_factor = 4.0)
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

        size_t num_rounds = 20;
        size_t switches_per_round = num_switches / num_rounds;
        size_t successful_switches = 0;

        std::vector<std::mt19937_64> gens;
        auto n = omp_get_max_threads();
        gens.reserve(n);
        while (n--) gens.emplace_back(gen());

        while (true) {
            const size_t kNoSwitch = EdgeDependencies<es::edge_hash_crc32>::kNone_;
            const edge_t kNone = std::numeric_limits<edge_t>::max();

            std::vector<edge_t> edges_for_switchings(2 * switches_per_round, kNone);
            std::uniform_int_distribution<size_t> edge_index_distr(0, edge_list_.size() - 1);

            #pragma omp parallel reduction(+:successful_switches)
            {
                size_t t = omp_get_num_threads();
                size_t tid = omp_get_thread_num();

                auto& gen = gens[tid];

                size_t m = edge_list_.size();
                size_t edges_per_thread = m / t;
                size_t beg = tid * edges_per_thread;
                size_t end = beg + edges_per_thread;

                for (size_t i = beg; i + 1 < end; i += 2) {
                    edge_dependencies.announce_erase(edge_list_[i], kNoSwitch);
                    edge_dependencies.announce_erase(edge_list_[i + 1], kNoSwitch);
                }

                if (edges_per_thread % 2 == 1) {
                    edge_dependencies.announce_erase(edge_list_[end - 1], kNoSwitch);
                }

                if (tid + 1 == t) {
                    size_t r = m % t;
                    for (size_t i = 0; i < r; ++i) {
                        edge_dependencies.announce_erase(edge_list_[end + i], kNoSwitch);
                    }
                }

                #pragma omp barrier

                size_t edges_for_switches_per_thread = (2 * switches_per_round) / t;
                size_t switches_beg = tid * edges_for_switches_per_thread;
                size_t switches_end = tid + 1 == t ? 2 * switches_per_round : switches_beg + edges_for_switches_per_thread;

                for (size_t i = switches_beg; i + 1 < switches_end; i += 2) {
                    size_t switch_id = ((i % edges_per_thread) / 2) * t + tid;

                    size_t index1 = edge_index_distr(gen);
                    size_t index2 = edge_index_distr(gen);

                    if (index1 == index2) continue;

                    edge_t e1 = edge_list_[index1];
                    edge_t e2 = edge_list_[index2];

                    auto [u, v] = to_nodes(e1);
                    auto [x, y] = to_nodes(e2);

                    if (e1 < e2) std::swap(x, y);

                    edge_t e3 = to_edge(u, x);
                    edge_t e4 = to_edge(v, y);

                    if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) { // prevent self-loops
                        continue;
                    }

                    edge_dependencies.announce_erase(e1, switch_id);
                    edge_dependencies.announce_erase(e2, switch_id);
                    edge_dependencies.announce_insert(e3, switch_id);
                    edge_dependencies.announce_insert(e4, switch_id);

                    edges_for_switchings[i] = e1;
                    edges_for_switchings[i + 1] = e2;
                }

                #pragma omp barrier

                for (size_t i = beg; i + 1 < end; i += 2) {
                    size_t switch_id = ((i % edges_per_thread) / 2) * t + tid;

                    size_t e1 = edges_for_switchings[i];
                    size_t e2 = edges_for_switchings[i + 1];

                    auto [u, v] = to_nodes(e1);
                    auto [x, y] = to_nodes(e2);

                    if (e1 < e2) std::swap(x, y);

                    size_t e3 = to_edge(u, x);
                    size_t e4 = to_edge(v, y);

                    if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) // prevent self-loops
                        continue;

                    bool e1_erase_collision = true;
                    while (e1_erase_collision) {
                        auto [erasing_switch, erase_resolved, inserting_switch, insert_resolved] = edge_dependencies
                                .lookup_dependencies(e1);
                        if (erasing_switch >= switch_id) {
                            e1_erase_collision = false;
                        }
                        if (erase_resolved) break;
                    }
                    if (e1_erase_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }

                    bool e2_erase_collision = true;
                    while (e2_erase_collision) {
                        auto [erasing_switch, erase_resolved, inserting_switch, insert_resolved] = edge_dependencies
                                .lookup_dependencies(e2);
                        if (erasing_switch >= switch_id) {
                            e2_erase_collision = false;
                        }
                        if (erase_resolved) break;
                    }
                    if (e2_erase_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }

                    bool e3_erase_collision = true;
                    bool e3_insert_collision = true;
                    while (e3_erase_collision || e3_insert_collision) {
                        auto [erasing_switch, erase_resolved, inserting_switch, insert_resolved] = edge_dependencies
                                .lookup_dependencies(e3);
                        if (erasing_switch > switch_id) break;
                        if (erase_resolved) {
                            e3_erase_collision = false;
                            if (inserting_switch >= switch_id) {
                                e3_insert_collision = false;
                            }
                            if (insert_resolved) break;
                        }
                    }
                    if (e3_erase_collision || e3_insert_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }

                    bool e4_erase_collision = true;
                    bool e4_insert_collision = true;
                    while (e4_erase_collision || e4_insert_collision) {
                        auto [erasing_switch, erase_resolved, inserting_switch, insert_resolved] = edge_dependencies
                                .lookup_dependencies(e4);
                        if (erasing_switch > switch_id) break;
                        if (erase_resolved) {
                            e4_erase_collision = false;
                            if (inserting_switch >= switch_id) {
                                e4_insert_collision = false;
                            }
                            if (insert_resolved) break;
                        }
                    }
                    if (e4_erase_collision || e4_insert_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }

                    edge_list_[i] = e3;
                    edge_list_[i + 1] = e4;

                    edge_dependencies.announce_erase_succeeded(e1, switch_id);
                    edge_dependencies.announce_erase_succeeded(e2, switch_id);
                    edge_dependencies.announce_insert_succeeded(e3, switch_id);
                    edge_dependencies.announce_insert_succeeded(e4, switch_id);

                    ++successful_switches;
                }
            }

            num_rounds--;

            if (!num_rounds)
                break;

            edge_dependencies.next_round();
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
    EdgeDependencies<edge_hash_crc32> edge_dependencies;
    std::vector<edge_t> edge_list_;
};

}
