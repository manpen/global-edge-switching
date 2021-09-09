#include <iostream>

#include <cassert>
#include <random>
#include <vector>
#include <unordered_set>
#include <string_view>
#include <functional>

#include <tsl/hopscotch_set.h>
#include <tsl/robin_set.h>

#include <es/AdjacencyVector.hpp>
#include <es/Graph.hpp>
#include <es/ScopedTimer.hpp>
#include <es/RandomBits.hpp>

#include <es/algorithms/AlgorithmSet.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <es/algorithms/AlgorithmAdjecencyVector.hpp>
#include <es/algorithms/AlgorithmParallelNaive.hpp>
#include <es/algorithms/AlgorithmParallelNaiveGlobal.hpp>
#include <es/algorithms/AlgorithmParallelGlobal.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWait.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV2.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/HavelHakimiGenerator.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>

using namespace es;

template <typename Algo>
void run_benchmark(std::string_view label, node_t n, edge_t target_m, std::mt19937_64 &gen, bool detailed = false) {
    double p = (2.0 * target_m) / n / (n - 1);
    auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();
    edge_t m = graph.numberOfEdges();

    Algo es(graph);

    {
        incpwl::ScopedTimer timer;
        const auto switches_per_edge = 100;
        const auto requested_switches = switches_per_edge * m;
        const auto sucessful_switches = es.do_switches(gen, requested_switches);
        if (detailed) {
            std::cout << label << ": Switches successful: " << (100. * sucessful_switches / requested_switches) << "% \n";
            std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
            std::cout << label << ": Switches per second: " << requested_switches / timer.elapsedSeconds() * 1e-6 << "M" << std::endl;
        }
        std::cout << "Estimated randomization time: " << timer.elapsedSeconds() * (1. * requested_switches / sucessful_switches) << "s \n";
    }
}

template <typename Algo>
void run_benchmark(std::string_view label, NetworKit::Graph graph, std::mt19937_64 &gen, bool detailed = true) {
    edge_t m = graph.numberOfEdges();

    Algo es(graph);

    {
        incpwl::ScopedTimer timer;
        const auto switches_per_edge = 10;
        const auto requested_switches = switches_per_edge * m;
        const auto sucessful_switches = es.do_switches(gen, requested_switches);
        if (detailed) {
            std::cout << label << ": Switches successful: " << (100. * sucessful_switches / requested_switches) << "% \n";
            std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
            std::cout << label << ": Switches per second: " << requested_switches / timer.elapsedSeconds() * 1e-6 << "M" << std::endl;
        }
        std::cout << label << ": Successful switches per second: " << (1. * sucessful_switches / m) / timer.elapsedSeconds() << "m \n";
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 1<<20;
    edge_t target_m = n * 1.44;

    for (int repeat = 0; repeat < 5; ++repeat) {
        NetworKit::PowerlawDegreeSequence ds_gen(1, n - 1, -2.1);
        std::vector<NetworKit::count> ds;
        bool realizable;
        do {
            ds_gen.run();
            ds = ds_gen.getDegreeSequence(n);
            realizable = NetworKit::HavelHakimiGenerator(ds).isRealizable();
        } while (!realizable);
        auto graph = NetworKit::HavelHakimiGenerator(ds).generate();

        //run_benchmark<AlgorithmAdjacencyVector>("aj", n, target_m, gen);
        //run_benchmark<AlgorithmAdjacencyVector>("aj-sorted", n, target_m, gen, true);
        /*run_benchmark<AlgorithmSet<tsl::robin_set<
            edge_t, edge_hash_crc32, std::equal_to<edge_t>, std::allocator<edge_t>, false, tsl::rh::prime_growth_policy
        >>>("robin-s", n, target_m, gen);*/
        //run_benchmark<AlgorithmVectorSet<google::dense_hash_set<edge_t, edge_hash_crc32>>>("dense", n, target_m, gen);

        omp_set_num_threads(4);

        run_benchmark<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", graph, gen);
        run_benchmark<AlgorithmParallelNaive>("parallel-naive", graph, gen);
        run_benchmark<AlgorithmParallelNaiveGlobal>("parallel-global-naive", graph, gen);
        run_benchmark<AlgorithmParallelGlobal>("parallel-global", graph, gen);
        run_benchmark<AlgorithmParallelGlobalNoWait>("parallel-global-no-wait", graph, gen);
        run_benchmark<AlgorithmParallelGlobalNoWaitV2>("parallel-global-no-wait-v2", graph, gen);

        std::cout << "\n";
    }

    //run_benchmark<EdgeSwitch_VectorSet<tsl::hopscotch_set<edge_t, edge_hash>>>("hps", n, target_m, gen);
    //run_benchmark<EdgeSwitch_VectorSet<std::unordered_set<edge_t, edge_hash>>>("std::uset", n, target_m, gen);

    return 0;

}
