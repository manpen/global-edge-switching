#pragma once

#include <random>
#include <vector>

#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/EdgeDependenciesNoWaitV3.hpp>

namespace es {

struct AlgorithmParallelGlobalNoWaitV3 : public AlgorithmBase {
    using EdgeDependenciesStore = EdgeDependenciesNoWaitV3<edge_hash_crc32>;
    static constexpr size_t kPrefetch = 1;

    AlgorithmParallelGlobalNoWaitV3(const NetworKit::Graph &graph, double load_factor = 2.0) : AlgorithmBase(graph),
                                                                                               edge_dependencies(graph.numberOfEdges(),
                                                                                                                 load_factor) {
        edge_list_.reserve(graph.numberOfEdges());
        switch_cache_.resize(graph.numberOfEdges() / 2);

        graph.forEdges([&](NetworKit::node u, NetworKit::node v) {
            auto edge = to_edge(u, v);
            assert(edge >> 63 == 0);

            edge_list_.emplace_back(edge);
        });
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        const auto num_switches_requested = num_switches;
        assert(!edge_list_.empty());

        size_t num_rounds = 2 * (num_switches / edge_list_.size());
        size_t successful_switches = 0;

        shuffle::GeneratorProvider gen_prov(gen);
        for (size_t r = 0; r < num_rounds; ++r) {
            if (r) edge_dependencies.next_round(); // clear hash table
            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);
            successful_switches += do_round();

            if (false) {
                auto sorted = edge_list_;
                std::sort(sorted.begin(), sorted.end());
                auto copy = sorted;
                assert(std::unique(copy.begin(), copy.end()) == copy.end());
                std::cout << " Unique!\n";
            }
        }

        return successful_switches;
    }

    size_t do_round(bool logging = false) {
        const size_t kNoSwitch = EdgeDependenciesStore::kLastSwitch;
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

        std::atomic<bool> all_done = true;

        const auto invalid_cache_pointer = reinterpret_cast<EdgeDependenciesStore::iterator_t>(this);

        #pragma omp parallel reduction(+:successful_switches)
        {
            auto fetch_switch_infos = [&](auto switch_id) -> std::tuple<edge_t, edge_t, edge_t, edge_t, bool> {
                const edge_t e1 = edge_list_[2 * switch_id];
                const edge_t e2 = edge_list_[2 * switch_id + 1];

                auto[u, v] = to_nodes(e1);
                auto[x, y] = to_nodes(e2);

                swap_if(e1 < e2, x, y);

                const edge_t e3 = to_edge(u, x);
                const edge_t e4 = to_edge(v, y);

                bool trivial_reject = (u == x || v == y || e3 == e4 || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4);

                return {e1, e2, e3, e4, trivial_reject};
            };

            auto try_process = [&](auto switch_id) {
                // returns false if switch was delayed
                auto &cache = switch_cache_[switch_id];
                auto[e1, e2, e3, e4, trivial_reject] = fetch_switch_infos(switch_id);

                if (logging) {
                    #pragma omp critical
                    std::cout << "A" << switch_id << " ";
                }

                if (trivial_reject)
                    return true;

                bool skip = false; // only valid if collision == false
                bool collision = false;

                auto check_erase_dependency = [&](auto iter) {
                    if (!iter) return;

                    auto[erasing_switch, erase_resolved] = iter->get_erase_switch_id_and_resolved();
                    const bool col = erasing_switch > switch_id;

                    collision |= col;
                    skip |= !erase_resolved;
                };

                auto check_insert_dependency = [&](auto iter) {
                    auto[inserting_switch, insert_resolved] = iter->get_insert_switch_id_and_resolved();

                    collision |= insert_resolved;
                    skip |= (inserting_switch != switch_id);

                    return inserting_switch == switch_id;
                };

                auto e3_insert_owner = check_insert_dependency(cache.e3_insert);
                auto e4_insert_owner = check_insert_dependency(cache.e4_insert);

                if (!collision) {
                    if (cache.e3_erase == invalid_cache_pointer) {
                        cache.e3_erase = edge_dependencies.find(e3, EdgeDependenciesStore::DependencyType::Erase);
                        cache.e4_erase = edge_dependencies.find(e4, EdgeDependenciesStore::DependencyType::Erase);
                    }
                    check_erase_dependency(cache.e3_erase);
                    check_erase_dependency(cache.e4_erase);
                }

                if (!collision && skip)
                    return false;

                if (collision) {
                    cache.e1_erase->announce_erase_failed(switch_id);
                    cache.e2_erase->announce_erase_failed(switch_id);

                    if (e3_insert_owner) cache.e3_insert->announce_insert_failed(switch_id);
                    if (e4_insert_owner) cache.e4_insert->announce_insert_failed(switch_id);

                } else {

                    edge_list_[2 * switch_id] = e3;
                    edge_list_[2 * switch_id + 1] = e4;

                    cache.e1_erase->announce_erase_succeeded(switch_id);
                    cache.e2_erase->announce_erase_succeeded(switch_id);
                    cache.e3_insert->announce_insert_succeeded(switch_id);
                    cache.e4_insert->announce_insert_succeeded(switch_id);

                    ++successful_switches;
                }

                if (logging) {
                    #pragma omp critical
                    std::cout << "S" << switch_id << " ";
                }

                return true;
            };

            auto prefetch = [&](auto switch_id) {
                const auto &cache = switch_cache_[switch_id];
                __builtin_prefetch(cache.e1_erase, 1, 1);
                __builtin_prefetch(cache.e2_erase, 1, 1);
                __builtin_prefetch(cache.e3_erase, 0, 0);
                __builtin_prefetch(cache.e4_erase, 0, 0);
                __builtin_prefetch(cache.e3_insert, 1, 1);
                __builtin_prefetch(cache.e4_insert, 1, 1);
            };

            std::vector<size_t> switch_ids, delayed_switch_ids;
            delayed_switch_ids.reserve(num_switches / omp_get_num_threads() / 8);

            // the first round is special since we iterate over all switch ids and write out
            // those that had to be delayed; in the next rounds we will only iterate over those!
            {
                #pragma omp for schedule(static, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    // returns false if switch can be cancelled
                    auto &cache = switch_cache_[switch_id];
                    auto[e1, e2, e3, e4, trivial_reject] = fetch_switch_infos(switch_id);

                    auto to_announce = trivial_reject ? kNoSwitch : switch_id;

                    cache.e1_erase = edge_dependencies.announce_erase(e1, to_announce);
                    cache.e2_erase = edge_dependencies.announce_erase(e2, to_announce);

                    if (trivial_reject) continue;

                    cache.e3_insert = edge_dependencies.announce_insert_if_minimum(e3, switch_id);
                    cache.e4_insert = edge_dependencies.announce_insert_if_minimum(e4, switch_id);
                    cache.e3_erase = invalid_cache_pointer;
                }

                #pragma omp for schedule(static, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    if (TLX_LIKELY(kPrefetch && switch_id < num_switches - kPrefetch))
                        prefetch(switch_id + kPrefetch);
                    auto done = try_process(switch_id);
                    if (!done)
                        delayed_switch_ids.push_back(switch_id);
                }
            }

            while (true) {
                switch_ids.swap(delayed_switch_ids);
                delayed_switch_ids.clear();
                delayed_switch_ids.reserve(switch_ids.size() / 8);

                // check whether we can quit
                {
                    if (!switch_ids.empty())
                        all_done = false;

                    #pragma omp barrier

                    if (all_done)
                        break;

                    #pragma omp barrier

                    all_done = true;
                }

                auto num_remaining_switches = switch_ids.size();

                // retry to acquire edge
                for (auto switch_id: switch_ids) {
                    auto &cache = switch_cache_[switch_id];
                    cache.e3_insert->announce_insert_if_minimum(switch_id);
                    cache.e4_insert->announce_insert_if_minimum(switch_id);
                }

                #pragma omp barrier

                for (size_t i = 0; i < num_remaining_switches; ++i) {
                    if (TLX_LIKELY(kPrefetch && i < num_remaining_switches - kPrefetch))
                        prefetch(switch_ids[i + kPrefetch]);

                    auto done = try_process(switch_ids[i]);
                    if (!done)
                        delayed_switch_ids.push_back(switch_ids[i]);
                }
            }
        }

        return successful_switches;
    }

    void do_switches(const std::vector<size_t> &rho, size_t num_threads) {
        assert(!edge_list_.empty());
        assert(!rho.empty());

        std::vector<edge_t> edge_list_permuted;
        edge_list_permuted.reserve(rho.size());
        for (const size_t &r: rho) {
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
        for (auto e: edge_list_) {
            auto[u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

private:
    EdgeDependenciesStore edge_dependencies;
    std::vector<edge_t> edge_list_;

    struct Cache {
        EdgeDependenciesStore::iterator_t e1_erase;
        EdgeDependenciesStore::iterator_t e2_erase;
        EdgeDependenciesStore::iterator_t e3_erase;
        EdgeDependenciesStore::iterator_t e4_erase;
        EdgeDependenciesStore::iterator_t e3_insert;
        EdgeDependenciesStore::iterator_t e4_insert;
    };
    std::vector<Cache> switch_cache_;

};

}
