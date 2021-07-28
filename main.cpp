#include <iostream>

#include <cassert>
#include <random>
#include <vector>
#include <unordered_set>
#include <string_view>
#include <functional>

#include <tsl/hopscotch_set.h>
#include <tsl/robin_set.h>

#include <es/Graph.hpp>
#include <es/ScopedTimer.hpp>
#include <es/RandomBits.hpp>

#include <es/algorithms/AlgorithmSet.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <es/algorithms/AlgorithmAdjecencyVector.hpp>
#include <es/algorithms/AlgorithmParallelNaive.hpp>

#include <es/AdjacencyVector.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>

using namespace es;

template <typename Algo>
void run_benchmark(std::string_view label, node_t n, edge_t target_m, std::mt19937_64 &gen) {
    double p = (2.0 * target_m) / n / (n - 1);
    auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();

    incpwl::ScopedTimer init_timer;
    Algo es(graph);
    std::cout << label << ": n=" << n << ",m=" << graph.numberOfEdges() << "\n";
    std::cout << label << ": Init " << init_timer.elapsedSeconds() << "s\n";

    {
        incpwl::ScopedTimer timer;
        const auto switches_per_edge = 10;
        const auto requested_switches = switches_per_edge * target_m;
        const auto sucessful_switches = es.do_switches(gen, requested_switches);
        std::cout << label << ": Switches requested: " << requested_switches << "\n";
        std::cout << label << ": Switches successful: " << (100. * sucessful_switches / requested_switches) << "\n";
        std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
        std::cout << label << ": Switches per second: " << switches_per_edge / timer.elapsedSeconds() << "M \n";
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 1<<20;
    edge_t target_m = n * 5;

    for (int repeat = 0; repeat < 5; ++repeat) {
        run_benchmark<AlgorithmAdjacencyVector>("aj", n, target_m, gen);

        auto max_threads = omp_get_max_threads();
        for(unsigned threads=1; threads <= max_threads; ++threads) {
            omp_set_num_threads(threads);
            run_benchmark<AlgorithmParallelNaive>("par-naive-" + std::to_string(threads), n, target_m, gen);
        }

        run_benchmark<AlgorithmVectorSet<google::dense_hash_set<edge_t, edge_hash_crc32>>>("dense", n, target_m, gen);
        run_benchmark<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", n, target_m, gen);

        std::cout << "\n";
    }

    //run_benchmark<EdgeSwitch_VectorSet<tsl::hopscotch_set<edge_t, edge_hash>>>("hps", n, target_m, gen);

    //run_benchmark<EdgeSwitch_VectorSet<std::unordered_set<edge_t, edge_hash>>>("std::uset", n, target_m, gen);

    return 0;

}
