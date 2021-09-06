#pragma once

#include <random>
#include <vector>

#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/EdgeDependenciesNoWaitV4.hpp>

namespace es {

struct AlgorithmParallelGlobalNoWaitV4 : public AlgorithmBase {
    using EdgeDependenciesStore = EdgeDependenciesNoWaitV4<edge_hash_crc32>;
    static constexpr size_t kPrefetch = 1;

    AlgorithmParallelGlobalNoWaitV4(const NetworKit::Graph &graph, double load_factor = 2.0) : AlgorithmBase(graph),
                                                                                               edge_dependencies(graph.numberOfEdges(),
                                                                                                                 load_factor) {
        edge_list_.reserve(graph.numberOfEdges());
        new_edge_list_.resize(graph.numberOfEdges());

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

            new_edge_list_.swap(edge_list_);

#ifndef NDEBUG
            {
                auto sorted = edge_list_;
                std::sort(sorted.begin(), sorted.end());
                auto copy = sorted;
                assert(std::unique(copy.begin(), copy.end()) == copy.end());
                std::cout << " Unique!\n";
            }
#endif
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
            new_edge_list_.back() = edge_list_.back();
        }

        std::atomic<bool> all_done = true;

        constexpr auto kInvalidEdge = std::numeric_limits<edge_t>::max();

        #pragma omp parallel reduction(+:successful_switches)
        {
            auto fetch_iterators = [&] (size_t switch_id) -> std::array<EdgeDependenciesStore::iterator_t, 4> {
                auto e1 = decode_iterator(edge_list_[2*switch_id]);
                auto e2 = decode_iterator(edge_list_[2*switch_id+1]);
                auto e3 = decode_iterator(new_edge_list_[2*switch_id]);
                auto e4 = decode_iterator(new_edge_list_[2*switch_id+1]);
                return {e1, e2, e3, e4};
            };

            auto try_process = [&](auto switch_id) {
                if (logging) {
                    #pragma omp critical
                    std::cout << "A" << switch_id << " ";
                }

                if (edge_list_[2 * switch_id] == kInvalidEdge) // encodes trivial reject
                    return true;

                bool skip = false; // only valid if collision == false
                bool collision = false;

                auto check_erase_dependency = [&](auto iter) {
                    auto [uninit, erasing_switch, erase_resolved] = iter->get_erase_switch_id_and_resolved();

                    const bool col = !uninit && erasing_switch > switch_id;

                    collision |= col;
                    skip |= !uninit && !erase_resolved;
                };

                auto check_insert_dependency = [&](auto iter) {
                    auto[inserting_switch, insert_resolved] = iter->get_insert_switch_id_and_resolved();

                    collision |= insert_resolved;
                    skip |= (inserting_switch != switch_id);

                    return inserting_switch == switch_id;
                };

                auto[e1_it, e2_it, e3_it, e4_it] = fetch_iterators(switch_id);

                auto e3_insert_owner = check_insert_dependency(e3_it);
                auto e4_insert_owner = check_insert_dependency(e4_it);
                check_erase_dependency(e3_it);
                check_erase_dependency(e4_it);

                if (collision) {
                    new_edge_list_[2 * switch_id] = e1_it->edge();
                    new_edge_list_[2 * switch_id + 1] = e2_it->edge();

                    e1_it->announce_erase_failed(switch_id);
                    e2_it->announce_erase_failed(switch_id);

                    if (e3_insert_owner) e3_it->announce_insert_failed(switch_id);
                    if (e4_insert_owner) e4_it->announce_insert_failed(switch_id);

                } else if (skip) {
                    return false;

                } else {
                    new_edge_list_[2 * switch_id] = e3_it->edge();
                    new_edge_list_[2 * switch_id + 1] = e4_it->edge();

                    e1_it->announce_erase_succeeded(switch_id);
                    e2_it->announce_erase_succeeded(switch_id);
                    e3_it->announce_insert_succeeded(switch_id);
                    e4_it->announce_insert_succeeded(switch_id);

                    ++successful_switches;
                }

                assert(new_edge_list_[2*switch_id] != 18446744073709551615ull);
                assert(new_edge_list_[2*switch_id+1] != 18446744073709551615ull);

                if (logging) {
                    #pragma omp critical
                    std::cout << "S" << switch_id << " ";
                }

                return true;
            };

            auto prefetch = [&](auto switch_id) {
                auto [e1_it, e2_it, e3_it, e4_it] = fetch_iterators(switch_id);
                __builtin_prefetch(e1_it, 1, 1);
                __builtin_prefetch(e2_it, 1, 1);
                __builtin_prefetch(e3_it, 1, 1);
                __builtin_prefetch(e4_it, 1, 1);
            };

            std::vector<size_t> switch_ids, delayed_switch_ids;
            delayed_switch_ids.reserve(num_switches / omp_get_num_threads() / 8);

            // the first round is special since we iterate over all switch ids and write out
            // those that had to be delayed; in the next rounds we will only iterate over those!
            {
                #pragma omp for schedule(static, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    edge_t e1 = edge_list_[2 * switch_id];
                    edge_t e2 = edge_list_[2 * switch_id + 1];

                    auto[u, v] = to_nodes(e1);
                    auto[x, y] = to_nodes(e2);

                    swap_if(e1 < e2, x, y);

                    const edge_t e3 = to_edge(u, x);
                    const edge_t e4 = to_edge(v, y);

                    bool trivial_reject = (u == x || v == y || e3 == e4 || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4);

                    auto to_announce = trivial_reject ? kNoSwitch : switch_id;

                    EdgeDependenciesStore::hint_t e1_hint, e2_hint, e3_hint, e4_hint;

                    e1_hint = edge_dependencies.prefetch(e1);
                    e2_hint = edge_dependencies.prefetch(e2);

                    if (!trivial_reject) {
                        e3_hint = edge_dependencies.prefetch(e3);
                        e4_hint = edge_dependencies.prefetch(e4);
                    }

                    edge_list_[2 * switch_id] = encode_iterator( edge_dependencies.announce_erase(e1_hint, to_announce) );
                    edge_list_[2 * switch_id + 1] = encode_iterator( edge_dependencies.announce_erase(e2_hint, to_announce) );

                    if (trivial_reject) {
                        edge_list_[2 * switch_id] = kInvalidEdge;
                        new_edge_list_[2 * switch_id] = e1;
                        new_edge_list_[2 * switch_id + 1] = e2;
                    } else {
                        new_edge_list_[2 * switch_id] = encode_iterator(
                            edge_dependencies.announce_insert_if_minimum(e3_hint, switch_id));
                        new_edge_list_[2 * switch_id + 1] = encode_iterator(
                            edge_dependencies.announce_insert_if_minimum(e4_hint, switch_id));

                        {
#ifndef NDEBUG
                            auto[e1_it, e2_it, e3_it, e4_it] = fetch_iterators(switch_id);
                            assert(e1_it->edge() == e1);
                            assert(e2_it->edge() == e2);
                            assert(e3_it->edge() == e3);
                            assert(e4_it->edge() == e4);
#endif
                        }
                    }
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
                    auto [e1_it, e2_it, e3_it, e4_it] = fetch_iterators(switch_id);
                    e3_it->announce_insert_if_minimum(switch_id);
                    e4_it->announce_insert_if_minimum(switch_id);
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
    std::vector<edge_t> new_edge_list_;

    static edge_t encode_iterator(EdgeDependenciesStore::iterator_t ptr) {
        static_assert(sizeof(ptr) <= sizeof(edge_t));
        return reinterpret_cast<edge_t>(ptr);
    }

    static EdgeDependenciesStore::iterator_t decode_iterator(edge_t edge) {
        return reinterpret_cast<EdgeDependenciesStore::iterator_t>(edge);
    }

};

}
