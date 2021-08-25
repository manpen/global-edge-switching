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
public:
    AlgorithmParallelGlobal(const NetworKit::Graph &graph, double load_factor = 2.0)
        : AlgorithmBase(graph),
        edge_dependencies(graph.numberOfEdges(), load_factor) {
        edge_list_.reserve(graph.numberOfEdges());

        graph.forEdges([&](NetworKit::node u, NetworKit::node v){
            auto edge = to_edge(u, v);
            edge_list_.emplace_back(edge);
        });
    }

    size_t do_switches(std::mt19937_64 &gen, size_t num_switches) {
        const auto num_switches_requested = num_switches;
        assert(!edge_list_.empty());

        size_t num_rounds = 2 * (num_switches / edge_list_.size());
        size_t successful_switches = 0;

        while (true) {
            shuffle::GeneratorProvider gen_prov(gen);
            shuffle::parallel::iss_shuffle(edge_list_.begin(), edge_list_.end(), gen_prov);

            #pragma omp parallel
            {
                size_t t = omp_get_num_threads();
                size_t tid = omp_get_thread_num();

                size_t m = edge_list_.size();
                size_t edges_per_thread = m / t;
                size_t beg = tid * edges_per_thread;
                size_t end = beg + edges_per_thread;

                for (size_t i = beg; i + 1 < end; i += 2) {
                    size_t switch_id = ((i % edges_per_thread) / 2) * t + tid;

                    size_t e1 = edge_list_[i];
                    size_t e2 = edge_list_[i + 1];

                    auto[u, v] = to_nodes(e1);
                    auto[x, y] = to_nodes(e2);

                    if (e1 < e2) std::swap(x, y);

                    size_t e3 = to_edge(u, x);
                    size_t e4 = to_edge(v, y);

                    if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) { // switch is definitely illegal, announce that edges wont be erased
                        edge_dependencies.announce_erase(e1, edge_dependencies.kNone_);
                        edge_dependencies.announce_erase(e2, edge_dependencies.kNone_);
                        continue;
                    }

                    edge_dependencies.announce_erase(e1, switch_id);
                    edge_dependencies.announce_erase(e2, switch_id);
                    edge_dependencies.announce_insert(e3, switch_id);
                    edge_dependencies.announce_insert(e4, switch_id);
                }

                if (edges_per_thread % 2 == 1) {
                    edge_t e = edge_list_[end - 1];
                    edge_dependencies.announce_erase(e, edge_dependencies.kNone_);
                }

                if (tid + 1 == t) {
                    size_t r = m % t;
                    for (size_t i = 0; i < r; ++i) {
                        edge_t e = edge_list_[end + i];
                        edge_dependencies.announce_erase(e, edge_dependencies.kNone_);
                    }
                }
            }

            #pragma omp parallel reduction(+:successful_switches)
            {
                size_t t = omp_get_num_threads();
                size_t tid = omp_get_thread_num();

                size_t m = edge_list_.size();
                size_t edges_per_thread = m / t;
                size_t beg = tid * edges_per_thread;
                size_t end = beg + edges_per_thread;

                for (size_t i = beg; i + 1 < end; i += 2) {
                    size_t switch_id = ((i % edges_per_thread) / 2) * t + tid;

                    size_t e1 = edge_list_[i];
                    size_t e2 = edge_list_[i + 1];

                    auto [u, v] = to_nodes(e1);
                    auto [x, y] = to_nodes(e2);

                    if (e1 < e2) std::swap(x, y);

                    size_t e3 = to_edge(u, x);
                    size_t e4 = to_edge(v, y);

                    if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) // prevent self-loops
                        continue;

                    bool e3_insert_collision = false;
                    while (!e3_insert_collision) {
                        auto [inserting_switch, resolved] = edge_dependencies.lookup_insert(e3);
                        if (inserting_switch >= switch_id) break;
                        if (resolved) e3_insert_collision = true;
                    }
                    if (e3_insert_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }
                    bool e3_erase_collision = false;
                    while (!e3_erase_collision) {
                        auto [erasing_switch, resolved] = edge_dependencies.lookup_erase(e3);
                        if (erasing_switch > switch_id) e3_erase_collision = true;
                        if (resolved) break;
                    }
                    if (e3_erase_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }

                    bool e4_insert_collision = false;
                    while (!e4_insert_collision) {
                        auto [inserting_switch, resolved] = edge_dependencies.lookup_insert(e4);
                        if (inserting_switch >= switch_id) break;
                        if (resolved) e4_insert_collision = true;
                    }
                    if (e4_insert_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }
                    bool e4_erase_collision = false;
                    while (!e4_erase_collision) {
                        auto [erasing_switch, resolved] = edge_dependencies.lookup_erase(e4);
                        if (erasing_switch > switch_id) e4_erase_collision = true;
                        if (resolved) break;
                    }
                    if (e4_erase_collision) {
                        edge_dependencies.announce_erase_failed(e1, switch_id);
                        edge_dependencies.announce_erase_failed(e2, switch_id);
                        edge_dependencies.announce_insert_failed(e3, switch_id);
                        edge_dependencies.announce_insert_failed(e4, switch_id);
                        continue;
                    }

                    // successful: commit new edges
                    edge_list_[i] = e3;
                    edge_list_[i + 1] = e4;
                    edge_dependencies.announce_insert_succeeded(e3, switch_id);
                    edge_dependencies.announce_insert_succeeded(e4, switch_id);

                    // and erase the old ones
                    edge_dependencies.announce_erase_succeeded(e1, switch_id);
                    edge_dependencies.announce_erase_succeeded(e2, switch_id);

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

        std::mutex output_mutex;
        std::cout << "Successful: ";
        #pragma omp parallel num_threads(num_threads)
        {
            size_t t = omp_get_num_threads();
            size_t tid = omp_get_thread_num();

            size_t m = edge_list_.size();
            size_t edges_per_thread = m / t;
            size_t beg = tid * edges_per_thread;
            size_t end = beg + edges_per_thread;

            for (size_t i = beg; i + 1 < end; i += 2) {
                size_t switch_id = ((i % edges_per_thread) / 2) * t + tid;

                size_t e1 = edge_list_[i];
                size_t e2 = edge_list_[i + 1];

                auto[u, v] = to_nodes(e1);
                auto[x, y] = to_nodes(e2);

                if (e1 < e2) std::swap(x, y);

                size_t e3 = to_edge(u, x);
                size_t e4 = to_edge(v, y);

                if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4) { // switch is definitely illegal, announce that edges wont be erased
                    edge_dependencies.announce_erase(e1, edge_dependencies.kNone_);
                    edge_dependencies.announce_erase(e2, edge_dependencies.kNone_);
                    continue;
                }

                edge_dependencies.announce_erase(e1, switch_id);
                edge_dependencies.announce_erase(e2, switch_id);
                edge_dependencies.announce_insert(e3, switch_id);
                edge_dependencies.announce_insert(e4, switch_id);
            }

            if (edges_per_thread % 2 == 1) {
                edge_t e = edge_list_[end - 1];
                edge_dependencies.announce_erase(e, edge_dependencies.kNone_);
            }

            if (tid + 1 == t) {
                size_t r = m % t;
                for (size_t i = 0; i < r; ++i) {
                    edge_t e = edge_list_[end + i];
                    edge_dependencies.announce_erase(e, edge_dependencies.kNone_);
                }
            }
        }

        #pragma omp parallel num_threads(num_threads)
        {
            size_t t = omp_get_num_threads();
            size_t tid = omp_get_thread_num();

            size_t m = edge_list_.size();
            size_t edges_per_thread = m / t;
            size_t beg = tid * edges_per_thread;
            size_t end = beg + edges_per_thread;

            for (size_t i = beg; i + 1 < end; i += 2) {
                size_t switch_id = ((i % edges_per_thread) / 2) * t + tid;

                {
                    std::lock_guard lock(output_mutex);
                    std::cout << " A" << switch_id;
                }

                size_t e1 = edge_list_[i];
                size_t e2 = edge_list_[i + 1];

                auto [u, v] = to_nodes(e1);
                auto [x, y] = to_nodes(e2);

                if (e1 < e2) std::swap(x, y);

                size_t e3 = to_edge(u, x);
                size_t e4 = to_edge(v, y);

                if (u == x || v == y || e1 == e3 || e1 == e4 || e2 == e3 || e2 == e4)
                    continue;

                bool e3_insert_collision = false;
                while (!e3_insert_collision) {
                    auto [inserting_switch, resolved] = edge_dependencies.lookup_insert(to_edge(u, x));
                    if (inserting_switch >= switch_id) break;
                    if (resolved) e3_insert_collision = true;
                }
                if (e3_insert_collision) {
                    {
                        std::lock_guard lock(output_mutex);
                        std::cout << " IC" << switch_id;
                    }
                    edge_dependencies.announce_erase_failed(e1, switch_id);
                    edge_dependencies.announce_erase_failed(e2, switch_id);
                    edge_dependencies.announce_insert_failed(e3, switch_id);
                    edge_dependencies.announce_insert_failed(e4, switch_id);
                    continue;
                }
                bool e3_erase_collision = false;
                while (!e3_erase_collision) {
                    auto [erasing_switch, resolved] = edge_dependencies.lookup_erase(to_edge(u, x));
                    if (erasing_switch > switch_id) e3_erase_collision = true;
                    if (resolved) break;
                }
                if (e3_erase_collision) {
                    {
                        std::lock_guard lock(output_mutex);
                        std::cout << " EC" << switch_id;
                    }
                    edge_dependencies.announce_erase_failed(e1, switch_id);
                    edge_dependencies.announce_erase_failed(e2, switch_id);
                    edge_dependencies.announce_insert_failed(e3, switch_id);
                    edge_dependencies.announce_insert_failed(e4, switch_id);
                    continue;
                }

                bool e4_insert_collision = false;
                while (!e4_insert_collision) {
                    auto [inserting_switch, resolved] = edge_dependencies.lookup_insert(to_edge(v, y));
                    if (inserting_switch >= switch_id) break;
                    if (resolved) e4_insert_collision = true;
                }
                if (e4_insert_collision) {
                    {
                        std::lock_guard lock(output_mutex);
                        std::cout << " IC" << switch_id;
                    }
                    edge_dependencies.announce_erase_failed(e1, switch_id);
                    edge_dependencies.announce_erase_failed(e2, switch_id);
                    edge_dependencies.announce_insert_failed(e3, switch_id);
                    edge_dependencies.announce_insert_failed(e4, switch_id);
                    continue;
                }
                bool e4_erase_collision = false;
                while (!e4_erase_collision) {
                    auto [erasing_switch, resolved] = edge_dependencies.lookup_erase(to_edge(v, y));
                    if (erasing_switch > switch_id) e4_erase_collision = true;
                    if (resolved) break;
                }
                if (e4_erase_collision) {
                    {
                        std::lock_guard lock(output_mutex);
                        std::cout << " EC" << switch_id;
                    }
                    edge_dependencies.announce_erase_failed(e1, switch_id);
                    edge_dependencies.announce_erase_failed(e2, switch_id);
                    edge_dependencies.announce_insert_failed(e3, switch_id);
                    edge_dependencies.announce_insert_failed(e4, switch_id);
                    continue;
                }

                // successful: commit new edges
                edge_list_[i] = to_edge(u, x);
                edge_list_[i + 1] = to_edge(v, y);
                edge_dependencies.announce_insert_succeeded(e3, switch_id);
                edge_dependencies.announce_insert_succeeded(e4, switch_id);

                // and erase the old ones
                edge_dependencies.announce_erase_succeeded(e1, switch_id);
                edge_dependencies.announce_erase_succeeded(e2, switch_id);

                std::lock_guard lock(output_mutex);
                std::cout << " S" << switch_id;
            }
        }
        std::cout << std::endl;
    }

    NetworKit::Graph get_graph() override {
        NetworKit::Graph result(input_graph_.numberOfNodes());
        for(auto e : edge_list_) {
            auto[u, v] = to_nodes(e);
            result.addEdge(u, v);
        }
        return result;
    }

private:
    EdgeDependencies<edge_hash_crc32> edge_dependencies;
    std::vector<edge_t> edge_list_;
};

}
