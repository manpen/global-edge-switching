#include <iostream>

#include <random>
#include <unordered_set>


#include <tsl/hopscotch_set.h>
#include <tsl/robin_set.h>

#include <es/Graph.hpp>
#include <es/Generators.hpp>

#include <es/algorithms/AlgorithmAdjecencyVector.hpp>
#include <es/algorithms/AlgorithmSet.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>

using namespace es;

template <typename Algo>
void run_test(std::string_view label, node_t n, edge_t target_m, std::mt19937_64 &gen) {
    double p = std::min(1.0, (2.0 * target_m) / n / (n - 1));
    auto input_graph = generate_gnp(n, p, gen);

    Algo es(input_graph);

    {
        const auto requested_swichtes = 2 * target_m;
        const auto sucessful_switches = es.do_switches(gen, requested_swichtes);
    }

    auto output_graph = es.get_graph();

// check degrees are maintained
    edge_t num_edges = 0;
    for(node_t i = 0; i < n; ++i) {
        num_edges += input_graph.degree_of(i);
        if (input_graph.degree_of(i) != output_graph.degree_of(i))
            abort();
    }

    if (num_edges / 2 != input_graph.number_of_edges())
        abort();

// check that there are not self-loops or multi-edges
    std::unordered_set<edge_t> edges;
    for(auto e : output_graph.edges()) {
        auto r = edges.insert(e);
        if (!r.second) abort();

        auto [u, v] = to_nodes(e);
        if (u == v) abort();
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 100000;
    edge_t target_m = n * 9;

    for(auto n : {10, 100, 1000}) {
        for(auto d : {1, 5, 10}) {
            if (d >= n) continue;

            edge_t target_m = n * d;

            for(int repeat = 0; repeat < 5; ++repeat) {
                /*run_test<AlgorithmSet<tsl::robin_set<
                    edge_t, edge_hash_crc32, std::equal_to<edge_t>, std::allocator<edge_t>, false, tsl::rh::prime_growth_policy
                >>>("robin-s", n, target_m, gen); */

                run_test<AlgorithmAdjacencyVector>("aj", n, target_m, gen);
                run_test<AlgorithmVectorSet<google::dense_hash_set<edge_t, edge_hash_crc32>>>("dense", n, target_m, gen);
                run_test<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", n, target_m, gen);

                std::cout << "\n";
            }
        }
    }

    return 0;
}
