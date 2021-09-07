#pragma once

#include <random>
#include <vector>

#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/EdgeDependenciesNoWait.hpp>

namespace es {

struct AlgorithmParallelGlobalNoWaitV2 : public AlgorithmBase {
    using EdgeDependenciesStore = EdgeDependenciesNoWait<edge_hash_crc32>;

public:
    AlgorithmParallelGlobalNoWaitV2(const NetworKit::Graph& graph, double load_factor = 4.0)
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
        for (size_t r = 0; r < num_rounds; ++r) {
            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);
            successful_switches += do_round();
            edge_dependencies.next_round();
        }

        return successful_switches;
    }

    size_t do_round(bool logging = false) {
        const size_t kNoSwitch = std::numeric_limits<size_t>::max();
        const size_t num_switches = edge_list_.size() / 2;

#ifdef NDEBUG
        constexpr size_t kBatchSize = 1024;
#else
        constexpr size_t kBatchSize = 2;
#endif

        size_t successful_switches = 0;

        if (edge_list_.size() % 2) {
            // in case we've an uneven number of edges, the last one wont be switched;
            // we still need to announce that it exists
            edge_dependencies.announce_erase(edge_list_.back(), kNoSwitch);
        }

        std::vector<int> switch_done(num_switches, 0);
        std::atomic<bool> all_done;

        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, kBatchSize)
            for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                const edge_t e1 = edge_list_[2 * switch_id];
                const edge_t e2 = edge_list_[2 * switch_id + 1];

                auto [u, v] = to_nodes(e1);
                auto [x, y] = to_nodes(e2);

                swap_if(e1 < e2, x, y);

                const edge_t e3 = to_edge(u, x);
                const edge_t e4 = to_edge(v, y);

                if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) { // prevent self-loops
                    edge_dependencies.announce_erase(e1, kNoSwitch);
                    edge_dependencies.announce_erase(e2, kNoSwitch);
                    switch_done[switch_id] = 1;
                    continue;
                }

                edge_dependencies.announce_erase(e1, switch_id);
                edge_dependencies.announce_erase(e2, switch_id);
            }
        }

        do {
            all_done = true;

            #pragma omp parallel reduction(+:successful_switches)
            {
                #pragma omp for schedule(dynamic, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    if (switch_done[switch_id] == 1) continue;

                    const edge_t e1 = edge_list_[2 * switch_id];
                    const edge_t e2 = edge_list_[2 * switch_id + 1];

                    auto [u, v] = to_nodes(e1);
                    auto [x, y] = to_nodes(e2);

                    swap_if(e1 < e2, x, y);

                    const edge_t e3 = to_edge(u, x);
                    const edge_t e4 = to_edge(v, y);

                    if (check_erase_dependency(e3, switch_id) != COLLISION) {
                        if (check_erase_dependency(e4, switch_id) != COLLISION) {
                            if (edge_dependencies.announce_insert_if_minimum(e3, switch_id)) {
                                edge_dependencies.announce_insert_if_minimum(e4, switch_id);
                            }
                        }
                    }
                }

                #pragma omp for schedule(dynamic, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    if (switch_done[switch_id] == 1) continue;

                    const edge_t e1 = edge_list_[2 * switch_id];
                    const edge_t e2 = edge_list_[2 * switch_id + 1];

                    if (logging) {
                        #pragma omp critical
                        std::cout << "A" << switch_id << " ";
                    }

                    auto [u, v] = to_nodes(e1);
                    auto [x, y] = to_nodes(e2);

                    swap_if(e1 < e2, x, y);

                    const edge_t e3 = to_edge(u, x);
                    const edge_t e4 = to_edge(v, y);

                    auto [e3_erase_result, e3_insert_result] = check_dependencies(e3, switch_id);
                    auto [e4_erase_result, e4_insert_result] = check_dependencies(e4, switch_id);
                    bool collision = e3_erase_result == COLLISION || e3_insert_result == COLLISION ||
                                     e4_erase_result == COLLISION || e4_insert_result == COLLISION;
                    bool skip = !collision && (e3_erase_result == SKIP || e3_insert_result == SKIP ||
                                               e4_erase_result == SKIP || e4_insert_result == SKIP);
                    if (skip) {
                        all_done.store(false, std::memory_order_release);
                        continue;
                    }
                    if (collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        if (e3_insert_result == NO_COLLISION) edge_dependencies.announce_insert_failed(e3, switch_id);
                        if (e4_insert_result == NO_COLLISION) edge_dependencies.announce_insert_failed(e4, switch_id);
                        switch_done[switch_id] = 1;
                        continue;
                    }

                    edge_list_[2 * switch_id] = e3;
                    edge_list_[2 * switch_id + 1] = e4;

                    edge_dependencies.announce_erase_succeeded(e1, switch_id);
                    edge_dependencies.announce_erase_succeeded(e2, switch_id);
                    edge_dependencies.announce_insert_succeeded(e3, switch_id);
                    edge_dependencies.announce_insert_succeeded(e4, switch_id);

                    switch_done[switch_id] = 1;

                    if (logging) {
                        #pragma omp critical
                        std::cout << "S" << switch_id << " ";
                    }

                    ++successful_switches;
                }
            }
        } while (!all_done);

        if (logging) std::cout << std::endl;

        return successful_switches;
    }

    void do_switches (const std::vector<size_t>& rho, size_t num_threads) {
        assert(!edge_list_.empty());
        assert(!rho.empty());

        std::vector<edge_t> edge_list_permuted;
        edge_list_permuted.reserve(rho.size());
        for (const size_t& r : rho) {
            edge_list_permuted.push_back(edge_list_[r]);
        }
        edge_list_ = std::move(edge_list_permuted);

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

    enum DependencyResult {
        SKIP = 0,
        COLLISION = 1,
        NO_COLLISION = 2
    };

    DependencyResult check_erase_dependency(edge_t eid, size_t switch_id) {
        auto [erasing_switch, erase_resolved] = edge_dependencies.lookup_erase(eid);
        if (erasing_switch > switch_id) return COLLISION;
        if (erase_resolved) return NO_COLLISION;
        return SKIP;
    }

    DependencyResult check_insert_dependency(edge_t eid, size_t switch_id) {
        auto [inserting_switch, insert_resolved] = edge_dependencies.lookup_insert(eid);
        if (insert_resolved) return COLLISION;
        if (inserting_switch == switch_id) return NO_COLLISION;
        return SKIP;
    }

    std::pair<DependencyResult, DependencyResult> check_dependencies(edge_t eid, size_t switch_id) {
        auto [erasing_switch, erase_resolved, inserting_switch, insert_resolved] =
                edge_dependencies.lookup_dependencies(eid);
        DependencyResult erase_result, insert_result;
        if (erasing_switch > switch_id) erase_result = COLLISION;
        else if (erase_resolved) erase_result = NO_COLLISION;
        else erase_result = SKIP;
        if (insert_resolved) insert_result = COLLISION;
        else if (inserting_switch == switch_id) insert_result = NO_COLLISION;
        else insert_result = SKIP;
        return {erase_result, insert_result};
    }
};

}
