#include <iostream>

#include <random>
#include <unordered_set>


#include <tsl/hopscotch_set.h>
#include <tsl/robin_set.h>

#include <es/Graph.hpp>

#include <es/algorithms/AlgorithmAdjecencyVector.hpp>
#include <es/algorithms/AlgorithmSet.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>

using namespace es;

template <typename Algo>
void run_test(std::string_view label, node_t n, edge_t target_m, std::mt19937_64 &gen) {
    double p = std::min(1.0, (2.0 * target_m) / n / (n - 1));
    auto input_graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();

    Algo es(input_graph);

    {
        const auto requested_swichtes = 2 * target_m;
        const auto sucessful_switches = es.do_switches(gen, requested_swichtes);
    }

    auto output_graph = es.get_graph();

// check degrees are maintained
    edge_t num_edges = 0;
    for(node_t i = 0; i < n; ++i) {
        num_edges += input_graph.degree(i);
        if (input_graph.degree(i) != output_graph.degree(i))
            abort();
    }

    if (num_edges / 2 != input_graph.numberOfEdges())
        abort();

// check that there are not self-loops or multi-edges
    std::unordered_set<edge_t> edges;
    output_graph.forEdges([&](NetworKit::node u, NetworKit::node v){
        if (u == v) abort();

        auto e = to_edge(u, v);
        auto r = edges.insert(e);
        if (!r.second) abort();
    });
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
                run_test<AlgorithmAdjacencyVector>("aj", n, target_m, gen);
                run_test<AlgorithmVectorSet<google::dense_hash_set<edge_t, edge_hash_crc32>>>("dense", n, target_m, gen);
                run_test<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", n, target_m, gen);

                std::cout << "\n";
            }
        }
    }

    return 0;
}
