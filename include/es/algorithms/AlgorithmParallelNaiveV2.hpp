#pragma once

#include <random>
#include <vector>
#include <thread>

#include <es/algorithms/AlgorithmBase.hpp>
#include <es/ParallelEdgeSet.hpp>
#include <es/RandomBits.hpp>

#include <shuffle/random/UniformRange.hpp>

namespace es {

struct AlgorithmParallelNaiveV2 : public AlgorithmBase {
    using edge_set_type = ParallelEdgeSet<>;

public:
    AlgorithmParallelNaiveV2(const NetworKit::Graph &graph, double load_factor = 2.0, double chunk_factor = 1.0) //
        : AlgorithmBase(graph), edge_set_(graph.numberOfEdges(), load_factor), chunk_factor_(chunk_factor),
          num_edges_(graph.numberOfEdges()) //
          {
        edge_list_ = std::make_unique<std::atomic<edge_t>[]>(num_edges_);
        auto it = edge_list_.get();

        graph.forEdges([&](NetworKit::node u, NetworKit::node v) {
            auto edge = to_edge(u, v);
            (it++)->store(edge, std::memory_order_relaxed);
        });
        assert(it == edge_list_.get() + num_edges_);

        #pragma omp parallel for
        for (size_t i = 0; i < num_edges_; ++i) {
            auto[u, v] = to_nodes(edge_list_[i].load(std::memory_order_relaxed));
            auto unlocked_ticket = edge_set_.insert(u, v);
            assert(unlocked_ticket != nullptr);
        }
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        using ticket_t = edge_set_type::iterator_type;
        const auto num_switches_requested = num_switches;

        std::vector<std::mt19937_64> gens;
        {
            auto n = omp_get_max_threads();
            gens.reserve(n);
            while (n--) {
                gens.emplace_back(gen());
            }
        }

        size_t successful_switches = 0;
        size_t sync_retries = 0;
        size_t sync_collisions = 0;

        while (true) {
            incpwl::ScopedTimer timer;
            auto chunk_size = std::min<size_t>(num_switches, num_edges_ * chunk_factor_);

            #pragma omp parallel reduction(+:successful_switches, sync_retries, sync_collisions)
            {
                auto tid = static_cast<unsigned>(omp_get_thread_num());
                auto num_threads = static_cast<unsigned>(omp_get_num_threads());
                auto gen = gens[tid];

                auto local_switches = ((tid + 1) * chunk_size / num_threads) - (tid * chunk_size / num_threads);

                constexpr size_t kPrefetch = 4;

                auto new_random_switch = [&] {
                    return Switch(edge_list_.get(), &edge_set_, tid, shuffle::nearlydivisionless(num_edges_, gen),
                                  shuffle::nearlydivisionless(num_edges_, gen));
                };

                if (!kPrefetch || local_switches < kPrefetch) {
                    for (size_t i = 0; i < local_switches; ++i) {
                        auto s = new_random_switch();
                        s.stage1();
                        successful_switches += s.stage2();
                    }
                } else {
                    std::array<Switch, kPrefetch> pipeline;

                    // we do a constant amount of addition work and exploit that
                    // a switches constructor and phase1 do not affect our data structures.
                    // so it's safe to create a few more switches and call stage1 too often
                    // to reduce the complexity of the boundary cases.
                    for (size_t i = 0; i < kPrefetch; ++i) {
                        pipeline[i] = new_random_switch();
                        if (i < kPrefetch / 2)
                            pipeline[i].stage1();
                    }

                    size_t pipeline_i = 0;
                    for (size_t i = 0; i < local_switches; ++i) {
                        successful_switches += pipeline[pipeline_i].stage2();
                        sync_retries += pipeline[pipeline_i].retries;
                        sync_collisions += !!pipeline[pipeline_i].retries;

                        pipeline[pipeline_i] = new_random_switch();
                        pipeline[(pipeline_i + kPrefetch / 2) % kPrefetch].stage1();
                        pipeline_i = (pipeline_i + 1) % kPrefetch;
                    }
                }

                gens[tid] = gen;
            }

            num_switches -= chunk_size;

            if (logging_ > 1)
                timer.report("chunk");

            if (!num_switches)
                break;

            edge_set_.rebuild();
        }

        if (logging_) {
            std::cout << "PERF num_switches=" << num_switches_requested << ",num_successful_switches=" << successful_switches
                      << ",num_sync_retries=" << sync_retries << ",num_sync_collisions=" << sync_collisions << std::endl;
        }

        return successful_switches;
    }

    NetworKit::Graph get_graph() override {
        NetworKit::Graph result(input_graph_.numberOfNodes());
        for (size_t i = 0; i < num_edges_; ++i) {
            auto[u, v] = to_nodes(edge_list_[i].load(std::memory_order_relaxed));
            result.addEdge(u, v);
        }
        return result;
    }

    void enable_logging(unsigned val = 1) {
        logging_ = val;
    }

private:
    std::unique_ptr<std::atomic<edge_t>[]> edge_list_;
    edge_set_type edge_set_;
    size_t num_edges_;
    double chunk_factor_;
    unsigned logging_{0};

    struct Switch {
        // provided
        std::atomic<edge_t> *edge_list;
        edge_set_type *edge_set;
        unsigned tid;
        size_t index1, index2;

        // computed
        edge_t e1, e2, e3, e4;
        edge_set_type::hint_t e1_hint, e2_hint, e3_hint, e4_hint;
        bool failed;
        long retries;

        void prefetch(void *ptr) {
            __builtin_prefetch(ptr, 1, 1);
        }

        Switch() : failed(true) {}

        Switch(std::atomic<edge_t> *edge_list, edge_set_type *edge_set, unsigned tid, size_t index1, size_t index2) : edge_list(
            edge_list), edge_set(edge_set), tid(tid), index1(index1), index2(index2) {

            prefetch(edge_list + index1);
            prefetch(edge_list + index2);
        }

        bool stage1() {
            e1 = edge_list[index1].load(std::memory_order_consume);
            e2 = edge_list[index2].load(std::memory_order_consume);

            auto[u, v] = to_nodes(e1);
            auto[x, y] = to_nodes(e2);

            swap_if(index1 > index2, x, y);

            // rewire to u-x, v-y
            failed = u == x || v == y;
            if (failed) {
                return false;
            }

            e1_hint = edge_set->prefetch(u, v);
            {
                auto[x1, y1] = to_nodes(e2);
                e2_hint = edge_set->prefetch(x1, y1);
            }

            e3 = to_edge(u, x);
            e4 = to_edge(v, y);

            e3_hint = edge_set->prefetch(u, x);
            e4_hint = edge_set->prefetch(v, y);

            return true;
        }

        bool stage2() {
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

                if (e1 != edge_list[index1].load(std::memory_order_consume) || e2 != edge_list[index2].load(std::memory_order_consume)) {
                    if (!stage1()) // race condition on the sample data structure
                        return false;
                }

                auto[u, v] = to_nodes(e1);
                auto[x, y] = to_nodes(e2);

                auto ticket1 = edge_set->acquire(e1_hint, u, v, tid).second;
                if (!ticket1) continue;

                auto ticket2 = edge_set->acquire(e2_hint, x, y, tid).second;
                if (!ticket2) {
                    edge_set->release(ticket1);
                    continue;
                }

                auto[U, V] = to_nodes(e3);

                auto ticket3 = edge_set->insert(e3_hint, U, V, tid);
                if (!ticket3) {
                    // edge exists: reject
                    edge_set->release(ticket1);
                    edge_set->release(ticket2);
                    return false;
                }

                auto[X, Y] = to_nodes(e4);
                auto ticket4 = edge_set->insert(e4_hint, X, Y, tid);
                if (!ticket4) {
                    edge_set->erase_and_release(ticket3);
                    edge_set->release(ticket1);
                    edge_set->release(ticket2);
                    return false;
                }

                // successful
                edge_list[index1].store(e3, std::memory_order_release);
                edge_list[index2].store(e4, std::memory_order_release);
                edge_set->erase_and_release(ticket1);
                edge_set->erase_and_release(ticket2);
                edge_set->release(ticket3);
                edge_set->release(ticket4);
                return true;
            }
        }
    };

};

}
