#include <iostream>
#include <random>
#include <unordered_set>

#include <tsl/hopscotch_set.h>
#include <tsl/robin_set.h>

#include <es/Graph.hpp>
#include <es/algorithms/AlgorithmAdjecencyVector.hpp>
#include <es/algorithms/AlgorithmParallelNaive.hpp>
#include <es/algorithms/AlgorithmParallelNaiveGlobal.hpp>
#include <es/algorithms/AlgorithmParallelNaiveGlobalV2.hpp>
#include <es/algorithms/AlgorithmSet.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <es/algorithms/AlgorithmParallelNaive.hpp>
#include <es/algorithms/AlgorithmParallelNaiveV2.hpp>
#include <es/algorithms/AlgorithmGlobal.hpp>
#include <es/algorithms/AlgorithmVectorRobin.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV4.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>

#include <tlx/die.hpp>

using namespace es;

template <typename Algo>
void run_test(std::string_view label, node_t n, edge_t target_m, std::mt19937_64 &gen) {
    double p = std::min(1.0, (2.0 * target_m) / n / (n - 1));
    auto input_graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();

    Algo es(input_graph);

    {
        const auto requested_switches = 2 * target_m;
        es.do_switches(gen, requested_switches);
    }

    auto output_graph = es.get_graph();

    // check degrees are maintained
    edge_t num_edges = 0;
    for (node_t i = 0; i < n; ++i) {
        num_edges += input_graph.degree(i);
        tlx_die_unequal(input_graph.degree(i), output_graph.degree(i));
    }

    tlx_die_unequal(num_edges / 2, input_graph.numberOfEdges());

    // check that there are no self-loops or multi-edges
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

            std::cout << "n=" << std::setw(4) << n << ",d=" << std::setw(4) << d << "  | ";
            for (int repeat = 0; repeat < 100; ++repeat) {
                // production algorithms
                {
                    run_test<AlgorithmVectorRobin<false>>("robin-v2", n, target_m, gen);
                    run_test<AlgorithmVectorRobin<true>>("global-robin", n, target_m, gen);
                    run_test<AlgorithmParallelGlobalNoWaitV4<true>>("parallel-global-nowait-v4", n, target_m, gen);
                    run_test<AlgorithmParallelNaiveV2>("par-v2", n, target_m, gen);
                    run_test<AlgorithmParallelNaiveV2NoPrefetch>("par-v2-np", n, target_m, gen);
                    run_test<AlgorithmParallelNaiveGlobalV2>("parallel-naive-global-v2", n, target_m, gen);
                    run_test<AlgorithmParallelNaiveGlobalV2NoPrefetch>("parallel-naive-global-v2", n, target_m, gen);
                }

                // discontinued algorithms
                if (false) {
                    run_test<AlgorithmAdjacencyVector>("aj", n, target_m, gen);
                    run_test<AlgorithmParallelNaive>("par", n, target_m, gen);
                    //run_test<AlgorithmVectorSet<google::dense_hash_set<edge_t, edge_hash_crc32>>>("dense", n, target_m, gen);
                    run_test<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", n, target_m, gen);
                    run_test<AlgorithmGlobal<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", n, target_m, gen);
                    run_test<AlgorithmParallelNaive>("parallel-naive", n, target_m, gen);
                    run_test<AlgorithmParallelNaiveGlobal>("parallel-naive-global", n, target_m, gen);
                }

                if (repeat % 10 == 0)
                    std::cout << ' ';
                std::cout << '.' << std::flush;
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
