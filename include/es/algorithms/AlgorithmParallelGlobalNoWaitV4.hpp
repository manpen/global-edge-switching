#pragma once

#include <random>
#include <vector>
#include <numeric>

#include <es/EdgeDependenciesNoWaitV4.hpp>
#include <es/RandomBits.hpp>
#include <es/ScopedTimer.hpp>
#include <es/algorithms/AlgorithmBase.hpp>
#include <shuffle/algorithms/InplaceScatterShuffle.hpp>

namespace es {

template <bool UsePrefetching = true>
struct AlgorithmParallelGlobalNoWaitV4 : public AlgorithmBase {
    using EdgeDependenciesStore = EdgeDependenciesNoWaitV4<edge_hash_crc32>;
    static constexpr size_t kPrefetch = UsePrefetching ? 1 : 0;

    AlgorithmParallelGlobalNoWaitV4(const NetworKit::Graph &graph, double lazyness = 0.01, double load_factor = 2.0) //
        : AlgorithmBase(graph), edge_dependencies(graph.numberOfEdges(), load_factor), lazyness_(lazyness) {
        assert(0. <= lazyness && lazyness < 1.0);
        new_edge_list_.resize(graph.numberOfEdges());

        if (graph.numberOfEdges() < 1e6) {
            edge_list_.reserve(graph.numberOfEdges());
            graph.forEdges([&](NetworKit::node u, NetworKit::node v) {
                auto edge = to_edge(u, v);
                assert(edge >> 63 == 0);

                edge_list_.emplace_back(edge);
            });
        } else {
            std::vector<size_t> local_counts(omp_get_max_threads());
            std::vector<size_t> partial_sums(omp_get_max_threads());
            edge_list_.resize(graph.numberOfEdges());

            #pragma omp parallel
            {
                const auto tid = static_cast<size_t>(omp_get_thread_num());
                const auto num_threads = static_cast<size_t>(omp_get_num_threads());
                const auto n = graph.numberOfNodes();

                size_t local_count = 0;
                for(auto u = tid; u < n; u += num_threads) {
                    for (auto v : graph.neighborRange(u)) {
                        if (v > u) continue;
                        ++local_count;
                    }
                }
                #pragma omp critical
                local_counts[tid] = local_count;

                #pragma omp barrier

                #pragma omp single // implies barrier at end
                std::partial_sum(begin(local_counts), end(local_counts), begin(partial_sums));
                assert(partial_sums.back() == graph.numberOfEdges());

                auto it = begin(edge_list_) + (partial_sums[tid] - local_count);
                for(auto u = tid; u < n; u += num_threads) {
                    for (auto v : graph.neighborRange(u)) {
                        if (v > u) continue;

                        auto edge = to_edge(u, v);
                        assert(edge >> 63 == 0);

                        *(it++) = edge;
                    }
                }
            }
        }
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches, bool autocorrelation = false) {
        const auto num_switches_requested = num_switches;
        assert(!edge_list_.empty());

        size_t num_rounds = 2 * (num_switches / edge_list_.size());
        size_t successful_switches = 0;

        shuffle::GeneratorProvider gen_prov(gen);

        for (size_t r = 0; r < num_rounds; ++r) {
            if (r)
                edge_dependencies.next_round(); // clear hash table

            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);
            size_t num_switches_in_round = edge_list_.size() / 2;
            if (lazyness_ > 0.0) {
                auto num_lazy = std::binomial_distribution<size_t>{num_switches_in_round, lazyness_}(gen);
                num_switches_in_round -= num_lazy;
            }

            successful_switches += do_round(num_switches_in_round, autocorrelation);

            new_edge_list_.swap(edge_list_);

#ifndef NDEBUG
            if (false) {
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

    size_t do_round(size_t num_switches, bool autocorrelation, bool logging = false) {
        const size_t kNoSwitch = EdgeDependenciesStore::kLastSwitch;

#ifdef NDEBUG
        constexpr size_t kBatchSize = 1024;
#else
        constexpr size_t kBatchSize = 2;
#endif

        size_t successful_switches = 0;

        // make sure the hash table contains the edges contained in lazy switches
        for (auto i = 2 * num_switches; i < edge_list_.size(); ++i) {
            edge_dependencies.announce_erase(edge_list_[i], kNoSwitch);
            new_edge_list_[i] = edge_list_[i];
        }

        std::atomic<size_t> total_remaining = 0;
        std::atomic<size_t> total_successful_switches = 0;
        constexpr auto kInvalidEdge = std::numeric_limits<edge_t>::max();

        if (log_level_) {
            std::cout << "#start " << num_switches << '\n';
        }

        if (autocorrelation) omp_set_num_threads(1);

#pragma omp parallel reduction(+ : successful_switches)
        {
            size_t successful_switches = 0;
            incpwl::ScopedTimer timer;

            auto fetch_iterators = [&](size_t switch_id) -> std::array<EdgeDependenciesStore::iterator_t, 4> {
                auto e1 = decode_iterator(edge_list_[2 * switch_id]);
                auto e2 = decode_iterator(edge_list_[2 * switch_id + 1]);
                auto e3 = decode_iterator(new_edge_list_[2 * switch_id]);
                auto e4 = decode_iterator(new_edge_list_[2 * switch_id + 1]);
                return {e1, e2, e3, e4};
            };

            auto try_process = [&](auto switch_id) {
                if (edge_list_[2 * switch_id] == kInvalidEdge) // encodes trivial reject
                    return true;

                bool skip = false; // only valid if collision == false
                bool collision = false;

                auto check_erase_dependency = [&](auto iter) {
                    if (collision) return;
                    auto[uninit, erasing_switch, erase_resolved] =
                    iter->get_erase_switch_id_and_resolved();

                    const bool col = !uninit && erasing_switch > switch_id;

                    collision |= col;
                    skip |= !uninit && !erase_resolved;
                };

                auto check_insert_dependency = [&](auto iter) {
                    auto[inserting_switch, insert_resolved] =
                    iter->get_insert_switch_id_and_resolved();

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
                    new_edge_list_[2 * switch_id] = e1_it->edge(std::memory_order_relaxed);
                    new_edge_list_[2 * switch_id + 1] = e2_it->edge(std::memory_order_relaxed);

                    e1_it->announce_erase_failed(switch_id);
                    e2_it->announce_erase_failed(switch_id);

                    if (e3_insert_owner)
                        e3_it->announce_insert_failed(switch_id);
                    if (e4_insert_owner)
                        e4_it->announce_insert_failed(switch_id);

                } else if (skip) {
                    return false;

                } else {
                    new_edge_list_[2 * switch_id] = e3_it->edge(std::memory_order_relaxed);
                    new_edge_list_[2 * switch_id + 1] = e4_it->edge(std::memory_order_relaxed);

                    e1_it->announce_erase_succeeded(switch_id);
                    e2_it->announce_erase_succeeded(switch_id);
                    e3_it->announce_insert_succeeded(switch_id);
                    e4_it->announce_insert_succeeded(switch_id);

                    ++successful_switches;
                }

                if (logging) {
#pragma omp critical
                    std::cout << "S" << switch_id << " ";
                }

                return true;
            };

            auto prefetch = [&](auto switch_id) {
                auto[e1_it, e2_it, e3_it, e4_it] = fetch_iterators(switch_id);
                __builtin_prefetch(e1_it, 1, 1);
                __builtin_prefetch(e2_it, 1, 1);
                __builtin_prefetch(e3_it, 1, 1);
                __builtin_prefetch(e4_it, 1, 1);
            };

            std::vector<size_t> switch_ids, delayed_switch_ids;
            delayed_switch_ids.reserve(num_switches / omp_get_num_threads() / 8);

            // the first round is special since we iterate over all switch ids and
            // write out those that had to be delayed; in the next rounds we will only
            // iterate over those!
            {
                using hint_pair = std::pair<EdgeDependenciesStore::hint_t, EdgeDependenciesStore::hint_t>;
                size_t prefetched_switch = std::numeric_limits<size_t>::max();
                hint_pair prefetched_hints;

                #pragma omp for schedule(static, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    auto hints_valid = (prefetched_switch == switch_id);
                    auto hints = prefetched_hints;

                    if (TLX_LIKELY(kPrefetch && switch_id + kPrefetch < num_switches)) {
                        prefetched_switch = switch_id + kPrefetch;
                        prefetched_hints = { //
                            edge_dependencies.prefetch(edge_list_[2 * prefetched_switch]),
                            edge_dependencies.prefetch(edge_list_[2 * prefetched_switch + 1])};
                    }

                    edge_t e1 = edge_list_[2 * switch_id];
                    edge_t e2 = edge_list_[2 * switch_id + 1];

                    if (TLX_UNLIKELY(!hints_valid)) {
                        hints = {edge_dependencies.prefetch(e1), edge_dependencies.prefetch(e2)};
                    }

                    auto[u, v] = to_nodes(e1);
                    auto[x, y] = to_nodes(e2);

                    swap_if(e1 < e2, x, y);

                    const edge_t e3 = to_edge(u, x);
                    const edge_t e4 = to_edge(v, y);

                    bool trivial_reject = (u == x || v == y || e3 == e4 || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4);

                    auto to_announce = trivial_reject ? kNoSwitch : switch_id;

                    edge_list_[2 * switch_id] = encode_iterator(edge_dependencies.announce_erase(hints.first, to_announce));
                    edge_list_[2 * switch_id + 1] = encode_iterator(edge_dependencies.announce_erase(hints.second, to_announce));

                    if (TLX_UNLIKELY(trivial_reject)) {
                        edge_list_[2 * switch_id] = kInvalidEdge;
                        new_edge_list_[2 * switch_id] = e1;
                        new_edge_list_[2 * switch_id + 1] = e2;
                    } else {
                        new_edge_list_[2 * switch_id] = e3;
                        new_edge_list_[2 * switch_id + 1] = e4;
                    }
                }

                prefetched_switch = std::numeric_limits<size_t>::max();
#pragma omp for schedule(static, kBatchSize)
                for (size_t switch_id = 0; switch_id < num_switches; ++switch_id) {
                    auto hints_valid = (prefetched_switch == switch_id);
                    auto hints = prefetched_hints;

                    if (TLX_LIKELY(kPrefetch && switch_id + kPrefetch < num_switches)) {
                        prefetched_switch = switch_id + kPrefetch;
                        prefetched_hints = { //
                            edge_dependencies.prefetch(new_edge_list_[2 * prefetched_switch]),
                            edge_dependencies.prefetch(new_edge_list_[2 * prefetched_switch + 1])};
                    }

                    if (edge_list_[2 * switch_id] == kInvalidEdge)
                        continue; // trivial reject

                    if (TLX_UNLIKELY(!hints_valid)) {
                        hints = { //
                            edge_dependencies.prefetch(new_edge_list_[2 * switch_id]),
                            edge_dependencies.prefetch(new_edge_list_[2 * switch_id + 1])};
                    }

                    bool reject = false;

                    auto find_or_insert = [&](auto hint) {
                        const auto it = edge_dependencies.find_or_insert(hint).first;

                        // if our target edge will be deleted after this this switch,
                        // we have a collision can do not need to announce the target
                        // insertion
                        auto[uninit, erasing_switch, erase_resolved] =
                        it->get_erase_switch_id_and_resolved(std::memory_order_relaxed);
                        reject |= erase_resolved || (!uninit && (erasing_switch > switch_id));

                        return it;
                    };

                    auto e3_it = find_or_insert(hints.first);
                    if (TLX_LIKELY(!reject)) {
                        auto e4_it = find_or_insert(hints.second);

                        if (TLX_LIKELY(!reject)) {
                            new_edge_list_[2 * switch_id] = encode_iterator(e3_it);
                            new_edge_list_[2 * switch_id + 1] = encode_iterator(e4_it);

                            e3_it->announce_insert_if_minimum(switch_id);
                            e4_it->announce_insert_if_minimum(switch_id);
                            continue;
                        }
                    }

                    assert(reject);
                    auto e1_it = decode_iterator(edge_list_[2 * switch_id]);
                    auto e2_it = decode_iterator(edge_list_[2 * switch_id + 1]);

                    e1_it->announce_erase_failed(switch_id);
                    e2_it->announce_erase_failed(switch_id);

                    new_edge_list_[2 * switch_id] = e1_it->edge(std::memory_order_relaxed);
                    new_edge_list_[2 * switch_id + 1] = e2_it->edge(std::memory_order_relaxed);
                    edge_list_[2 * switch_id] = kInvalidEdge;
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

            size_t previous_total_remaining = 0;
            size_t previous_sucessful_switches = 0;
            double previous_timer = 0.0;
            while (true) {
                switch_ids.swap(delayed_switch_ids);
                delayed_switch_ids.clear();
                delayed_switch_ids.reserve(switch_ids.size() / 8);

                {
                    // synchronize progress made so far
                    if (!switch_ids.empty())
                        total_remaining += switch_ids.size();

                    if (successful_switches)
                        total_successful_switches += successful_switches;

#pragma omp barrier
                    auto remaining = total_remaining.load(std::memory_order_relaxed);

                    #pragma omp master
                    if (log_level_) {
                        auto time = timer.elapsedSeconds();
                        std::cout << "#remaining " << (remaining - previous_total_remaining) << "," << (total_successful_switches.load(std::memory_order_relaxed) - previous_sucessful_switches) << ',' << (time - previous_timer) << '\n';
                        previous_sucessful_switches = total_successful_switches.load(std::memory_order_relaxed);
                        previous_timer = time;
                    }

                    if (previous_total_remaining == remaining)
                        break;

                    previous_total_remaining = remaining;
                }

                successful_switches = 0;
                auto num_remaining_switches = switch_ids.size();

                // retry to acquire edge
                for (auto switch_id: switch_ids) {
                    auto[e1_it, e2_it, e3_it, e4_it] = fetch_iterators(switch_id);
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

        return total_successful_switches.load(std::memory_order_relaxed);
    }

    void do_switches(const std::vector<size_t> &rho, size_t num_threads, bool autocorrelation = false) {
        assert(!edge_list_.empty());
        assert(!rho.empty());

        std::vector<edge_t> edge_list_permuted;
        edge_list_permuted.reserve(rho.size());
        for (const size_t &r: rho) {
            edge_list_permuted.push_back(edge_list_[r]);
        }
        edge_list_ = std::move(edge_list_permuted);

        do_round(0, autocorrelation);
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

    const std::vector<edge_t>& get_edgelist() const {
        return edge_list_;
    }

private:
    EdgeDependenciesStore edge_dependencies;

    std::vector<edge_t> edge_list_;
    std::vector<edge_t> new_edge_list_;

    double lazyness_;

    static edge_t encode_iterator(EdgeDependenciesStore::iterator_t ptr) {
        static_assert(sizeof(ptr) <= sizeof(edge_t));
        return reinterpret_cast<edge_t>(ptr);
    }

    static EdgeDependenciesStore::iterator_t decode_iterator(edge_t edge) {
        return reinterpret_cast<EdgeDependenciesStore::iterator_t>(edge);
    }
};

} // namespace es
