#pragma once

#include <random>
#include <vector>
#include <thread>

#include <es/algorithms/AlgorithmBase.hpp>
#include <es/ParallelEdgeSet.hpp>
#include <es/RandomBits.hpp>

#include <shuffle/random/UniformRange.hpp>
#include <time.h>

namespace es {

template<unsigned PrefetchDepth = 4>
struct AlgorithmParallelNaiveGlobalV2Impl : public AlgorithmBase {
    using edge_set_type = ParallelEdgeSet<>;

public:
    AlgorithmParallelNaiveGlobalV2Impl(const NetworKit::Graph &graph, double lazyness = 0.01, double load_factor = 2.0) //
        : AlgorithmBase(graph), edge_set_(graph.numberOfEdges(), load_factor), num_edges_(graph.numberOfEdges()), laziness_(lazyness) //
    {
        edge_list_.reserve(num_edges_);

        graph.forEdges([&](NetworKit::node u, NetworKit::node v) {
            edge_list_.emplace_back(to_edge(u, v));
        });

        #pragma omp parallel for
        for (size_t i = 0; i < num_edges_; ++i) {
            auto[u, v] = to_nodes(edge_list_[i]);
            auto unlocked_ticket = edge_set_.insert(u, v);
            assert(unlocked_ticket != nullptr);
        }
    }


    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        using ticket_t = edge_set_type::iterator_type;
        const auto num_switches_requested = num_switches;
        assert(!edge_list_.empty());

        size_t num_rounds = 2 * (num_switches / num_edges_);

        size_t successful_switches = 0;
        size_t sync_retries = 0;
        size_t sync_collisions = 0;

        shuffle::GeneratorProvider gen_prov(gen);

        constexpr size_t kPrefetch = PrefetchDepth;

        for (size_t round = 0; round < num_rounds; ++round) {
            if (round)
                edge_set_.rebuild();

            incpwl::ScopedTimer timer;

            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);

            auto index_end = 2 * std::binomial_distribution<size_t>{num_edges_ / 2, 1 - laziness_}(gen);

            if (index_end < kPrefetch + 128) {
                for (size_t i = 0; i < index_end; i += 2) {
                    Switch<false> sw(edge_list_.data(), &edge_set_, 0, i);
                    successful_switches += sw.commit();
                }
            } else {
                #pragma omp parallel reduction(+:successful_switches, sync_retries, sync_collisions)
                {
                    auto tid = static_cast<unsigned>(omp_get_thread_num());

                    if constexpr (kPrefetch == 0) {
                        #pragma omp for schedule(dynamic, 1024)
                        for (size_t i = 0; i < index_end; i += 2) {
                            Switch<false> sw(edge_list_.data(), &edge_set_, tid, i);
                            successful_switches += sw.commit();
                        }
                    } else {
                        std::array<Switch<>, kPrefetch> pipeline;
                        size_t p_idx = 0;

                        auto commit_switch = [&](auto &sw) {
                            successful_switches += sw.commit();
                            sync_retries += sw.retries;
                            sync_collisions += (sw.retries > 0);
                        };

                        #pragma omp for schedule(dynamic, 1024)
                        for (size_t i = 0; i < index_end; i += 2) {
                            auto &cur = pipeline[p_idx % kPrefetch];

                            if (TLX_LIKELY(p_idx >= kPrefetch))
                                commit_switch(cur);

                            cur = Switch<>(edge_list_.data(), &edge_set_, tid, i);
                            ++p_idx;
                        }

                        for (size_t i = 0; i < kPrefetch; ++i, ++p_idx) {
                            commit_switch(pipeline[p_idx % kPrefetch]);
                        }
                    }
                }
            }

            if (log_level_ > 1)
                timer.report("round");
        }

        if (log_level_) {
            std::cout << "PERF num_switches=" << num_switches_requested << ",num_successful_switches=" << successful_switches
                      << ",num_sync_retries=" << sync_retries << ",num_sync_collisions=" << sync_collisions << std::endl;
        }

        return successful_switches;
    }

    NetworKit::Graph get_graph() override {
        NetworKit::Graph result(input_graph_.numberOfNodes());
        for (auto e: edge_list_) {
            auto[u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

private:
    std::vector<edge_t> edge_list_;
    edge_set_type edge_set_;
    size_t num_edges_;
    double laziness_;

    template<bool DoPrefetch = true>
    struct Switch {
        // provided
        edge_t *edge_list;
        edge_set_type *edge_set;
        unsigned tid;
        size_t index;

        // computed
        edge_t e1, e2, e3, e4;
        edge_set_type::hint_t e1_hint, e2_hint, e3_hint, e4_hint;
        bool failed;
        long retries;

        void prefetch(void *ptr) {
            __builtin_prefetch(ptr, 1, 1);
        }

        Switch() : failed(true) {}

        Switch(edge_t *edge_list, edge_set_type *edge_set, unsigned tid, size_t index) //
            : edge_list(edge_list), edge_set(edge_set), tid(tid), index(index) //
        {
            e1 = edge_list[index];
            e2 = edge_list[index + 1];

            auto[u, v] = to_nodes(e1);
            auto[x, y] = to_nodes(e2);

            swap_if(e1 > e2, x, y);

            e3 = to_edge(u, x);
            e4 = to_edge(v, y);

            // rewire to u-x, v-y
            failed = u == x || v == y || e3 == e4 || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4;
            if (failed) {
                return;
            }

            e1_hint = edge_set->prefetch<DoPrefetch>(u, v);
            {
                auto[x1, y1] = to_nodes(e2);
                e2_hint = edge_set->prefetch<DoPrefetch>(x1, y1);
            }

            e3_hint = edge_set->prefetch<DoPrefetch>(u, x);
            e4_hint = edge_set->prefetch<DoPrefetch>(v, y);
        }

        bool commit() {
            if (failed) {
                return false;
            }

            auto[u, v] = to_nodes(e1);
            auto[x, y] = to_nodes(e2);
            auto[U, V] = to_nodes(e3);
            auto[X, Y] = to_nodes(e4);

            if (failed) {
                retries = 0;
                return false;
            }

            retries = -1;
            while (true) {
                if (++retries) {
                    // make spin-lock a little bit less spinny
                    std::this_thread::yield();
                }

                auto ticket1 = edge_set->acquire(e1_hint, u, v, tid).second;
                if (!ticket1) continue;

                auto ticket2 = edge_set->acquire(e2_hint, x, y, tid).second;
                if (!ticket2) {
                    edge_set->release(ticket1);
                    continue;
                }

                auto ticket3 = edge_set->insert(e3_hint, U, V, tid);
                if (!ticket3) {
                    // edge e3 exists: reject
                    edge_set->release(ticket1);
                    edge_set->release(ticket2);
                    return false;
                }

                auto ticket4 = edge_set->insert(e4_hint, X, Y, tid);
                if (!ticket4) {
                    // edge e4 exists: reject
                    edge_set->erase_and_release(ticket3);
                    edge_set->release(ticket1);
                    edge_set->release(ticket2);
                    return false;
                }

                // successful
                edge_list[index] = e3;
                edge_list[index + 1] = e4;
                edge_set->erase_and_release(ticket1);
                edge_set->erase_and_release(ticket2);
                edge_set->release(ticket3);
                edge_set->release(ticket4);
                return true;
            }
        }
    };
};


using AlgorithmParallelNaiveGlobalV2 = AlgorithmParallelNaiveGlobalV2Impl<>;
using AlgorithmParallelNaiveGlobalV2NoPrefetch = AlgorithmParallelNaiveGlobalV2Impl<0>;

}
