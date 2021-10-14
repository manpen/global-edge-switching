#pragma once

#include <random>
#include <vector>

#include <shuffle/random/UniformRange.hpp>
#include <shuffle/algorithms/InplaceScatterShuffle.hpp>

#include <tsl/robin_set.h>
#include <tsl/robin_map.h>

#include <es/algorithms/AlgorithmBase.hpp>
#include <tlx/container/btree_set.hpp>

namespace es {

template<bool Global, bool UsePrefetching = true, bool CaptureStats = false>
struct AlgorithmVectorRobin : public AlgorithmBase {
public:
    AlgorithmVectorRobin(const NetworKit::Graph &graph, double load_factor = 0.25) : AlgorithmBase(graph) {
        edge_list_.reserve(graph.numberOfEdges());
        edge_set_.reserve(graph.numberOfEdges());
        edge_set_.max_load_factor(load_factor);

        graph.forEdges([&](NetworKit::node u, NetworKit::node v) {
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
            auto res = edge_set_.insert(edge);
            assert(res.second);
        });
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        assert(!edge_list_.empty());
        const auto m = edge_list_.size();

        size_t successful_switches = 0;
        if (Global) {
            shuffle::RandomBits fair_coin;

            while (true) {
                constexpr size_t kPrefetchSwitches = UsePrefetching ? 8 : 0;
                shuffle::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen);
                size_t num_switch_in_global_step = std::min(edge_list_.size() / 2, num_switches);
                auto num_lazy = std::binomial_distribution<size_t>{num_switch_in_global_step, lazyness_}(gen);

                size_t i = -2;
                successful_switches += do_n_switches<kPrefetchSwitches, true>(num_switch_in_global_step - num_lazy, [&] {
                    i += 2;
                    return std::make_tuple(i, i + 1, fair_coin(gen));
                });

                num_switches -= num_switch_in_global_step;
                if (!num_switches) {
                    report_stats();
                    return successful_switches;
                }
            }

        } else {
            constexpr size_t kPrefetchSwitches = UsePrefetching ? 4 : 0;
            constexpr size_t kPrefetchIndices = 2 * kPrefetchSwitches;
            std::array<size_t, kPrefetchIndices> prefetch_cache;
            size_t prefetch_i = 0;
            auto sample_and_prefetch = [&] {
                auto i = shuffle::nearlydivisionless(m, gen);
                if (kPrefetchIndices == 0)
                    return i;

                assert(i < m);
                __builtin_prefetch(edge_list_.data() + i, 1, 1);

                auto ret = prefetch_cache[prefetch_i];
                prefetch_cache[prefetch_i] = i;
                prefetch_i = (prefetch_i + 1) % kPrefetchIndices;

                return ret;
            };
            for (size_t i = 0; i < kPrefetchIndices; ++i)
                sample_and_prefetch();

            while (true) {
                const size_t num_chopped_switches = std::min(edge_list_.size()/2 + 1, num_switches);
                successful_switches += do_n_switches<kPrefetchSwitches, false>(num_chopped_switches, [&] {
                    auto index1 = sample_and_prefetch();
                    auto index2 = sample_and_prefetch();
                    return std::make_tuple(index1, index2, index1 < index2);
                });

                num_switches -= num_chopped_switches;
                if (!num_switches) {
                    report_stats();
                    return successful_switches;
                }
            }
        }
    }

    NetworKit::Graph get_graph() override {
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
    using Set = tsl::robin_set<edge_t, edge_hash_crc32>;
    std::vector<edge_t> edge_list_;
    Set edge_set_;
    std::vector<std::pair<size_t, size_t>> outcome_at_depth_;

    double lazyness_{0.01};

    void report_stats() {
        if constexpr(CaptureStats) {
            std::cout << "#robin-stat-depth";
            for (auto &oc: outcome_at_depth_)
                std::cout << ':' << oc.first << ',' << oc.second;
            std::cout << std::endl;
        }
    }

    // carries out num_switches where the edge index and direction is queried using the
    // nextDescriptor callback. Returns the number of successful switches
    template<size_t kPrefetchSwitches, bool kGlobal, typename NextDescriptor>
    size_t do_n_switches(size_t num_switches, NextDescriptor nextDescriptor) {
        // we split the switch into two steps: first we compute the hashes and prefetch
        // the hash set's buckets in the constructor. The actual switch is done in the
        // method commit. the idea is to some independent work in between to allow the
        // cpu to fetch the data.
        struct Switch {
            edge_t *e1, *e2;
            edge_t e1_copy, e2_copy;
            bool direction;
            Set *edge_set;


            edge_t e3, e4;
            size_t hash_e1, hash_e2, hash_e3, hash_e4;
            bool failed;

            Switch() : failed{true} {}

            Switch(edge_t *e1, edge_t *e2, bool direction, Set *edge_set) : e1(e1), e2(e2), direction(direction), edge_set(edge_set) {
#ifdef NDEBUG
                if (!kGlobal)
#endif
                {
                    e1_copy = *e1;
                    e2_copy = *e2;
                }


                prepare();
            }

            void prepare() {
                auto[u, v] = to_nodes(*e1);
                auto[x, y] = to_nodes(*e2);

                swap_if(direction, x, y);

                e3 = to_edge(u, x);
                e4 = to_edge(v, y);

                // rewire to u-x, v-y
                failed = u == x || v == y;
                if (failed) {
                    return;
                }

                hash_e3 = edge_set->prefetch(e3);
                hash_e4 = edge_set->prefetch(e4);

                hash_e1 = edge_set->prefetch(*e1);
                hash_e2 = edge_set->prefetch(*e2);
            }

            bool commit() {
                if (TLX_UNLIKELY(!kGlobal && (e1_copy != *e1 || e2_copy != *e2))) {
                    prepare();
                } else {
                    assert(e1_copy == *e1);
                    assert(e2_copy == *e2);
                }

                if (failed) return false;

                auto ins1 = edge_set->insert_hash(hash_e3, e3);
                if (!ins1.second) {
                    return false; // prevent parallel edges
                }

                if (!edge_set->insert_hash(hash_e4, e4).second) {
                    edge_set->erase(e3, hash_e3);
                    return false; // prevent parallel edges
                }

                edge_set->erase(*e1, hash_e1);
                edge_set->erase(*e2, hash_e2);

                *e1 = e3;
                *e2 = e4;

                return true;
            }
        };

        // invoke the switching class
        size_t successful_switches = 0;

        std::conditional_t<CaptureStats,  tsl::robin_map<edge_t, unsigned, edge_hash_crc32>, int> depth;
        if constexpr(CaptureStats) depth.reserve(2*num_switches);
        auto commit_switch = [&] (Switch &sw) {
            auto success = sw.commit();
            successful_switches += success;

            if constexpr(CaptureStats) {
                auto d = std::max(depth[sw.e3]++, depth[sw.e4]++);
                while(outcome_at_depth_.size() <= d)
                    outcome_at_depth_.emplace_back(0, 0);

                (success ? outcome_at_depth_[d].second : outcome_at_depth_[d].first) += 1;
            }
        };

        if (!kPrefetchSwitches || num_switches < kPrefetchSwitches) {
            // base case without prefetch
            for (size_t i = 0; i < num_switches; ++i) {
                auto[index1, index2, direction] = nextDescriptor();

                Switch sw(edge_list_.data() + index1, edge_list_.data() + index2, direction, &edge_set_);
                commit_switch(sw);
            }

        } else {
            std::array<Switch, kPrefetchSwitches> prefetch_pipeline;
            size_t prefetch_cursor = 0;
            auto prefetch_switch = [&] { ;
                auto[index1, index2, direction] = nextDescriptor();

                prefetch_pipeline[prefetch_cursor] = Switch(edge_list_.data() + index1, edge_list_.data() + index2, direction,
                                                            &edge_set_);
                prefetch_cursor = (prefetch_cursor + 1) % kPrefetchSwitches;
            };


            for (size_t i = 0; i < kPrefetchSwitches; ++i)
                prefetch_switch();
            assert(prefetch_cursor == 0);

            for (size_t i = kPrefetchSwitches; i < num_switches; ++i) {
                commit_switch(prefetch_pipeline[prefetch_cursor]);
                prefetch_switch();
            }

            for (size_t i = 0; i < kPrefetchSwitches; ++i) {
                commit_switch(prefetch_pipeline[prefetch_cursor]);
                prefetch_cursor = (prefetch_cursor + 1) % kPrefetchSwitches;
            }
        }
        return successful_switches;
    }

};

}
