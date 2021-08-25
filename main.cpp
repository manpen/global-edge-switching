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
#include <es/ThreadsafeSetLockedList.hpp>

#include <es/algorithms/AlgorithmSet.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <es/algorithms/AlgorithmAdjecencyVector.hpp>
#include <es/algorithms/AlgorithmParallelVector.hpp>
#include <es/algorithms/AlgorithmParallelGlobalES.hpp>
#include <es/algorithms/AlgorithmParallelVectorSet.hpp>
#include <es/algorithms/AlgorithmParallelNaive.hpp>
#include <es/algorithms/AlgorithmParallelNaiveGlobal.hpp>
#include <es/algorithms/AlgorithmParallelGlobal.hpp>

#include <es/AdjacencyVector.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/HavelHakimiGenerator.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>

using namespace es;

template <typename Algo>
void run_benchmark(std::string_view label, node_t n, edge_t target_m, std::mt19937_64 &gen, bool sorted = false) {
    double p = (2.0 * target_m) / n / (n - 1);
    auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();
    edge_t m = graph.numberOfEdges();

    Algo es(graph);

    {
        incpwl::ScopedTimer timer;
        const auto switches_per_edge = 100;
        const auto requested_swichtes = switches_per_edge * m;
        const auto sucessful_switches = es.do_switches(gen, requested_swichtes);
        std::cout << label << ": Switches successful: " << (1. * sucessful_switches / m) << "M \n";
        std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
        std::cout << label << ": Switches per second: " << (1. * sucessful_switches / m) / timer.elapsedSeconds() << "M \n";
    }
}

template <typename Algo>
void run_benchmark(std::string_view label, NetworKit::Graph graph, std::mt19937_64 &gen) {
    edge_t m = graph.numberOfEdges();

    Algo es(graph);

    {
        incpwl::ScopedTimer timer;
        const auto switches_per_edge = 10;
        const auto requested_swichtes = switches_per_edge * m;
        const auto sucessful_switches = es.do_switches(gen, requested_swichtes);
        std::cout << label << ": Switches successful: " << (1. * sucessful_switches / m) << "M \n";
        std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
        std::cout << label << ": Switches per second: " << (1. * sucessful_switches / m) / timer.elapsedSeconds() << "M \n";
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 1<<20;
    edge_t target_m = n * 1.44;

    for (int repeat = 0; repeat < 1; ++repeat) {
        NetworKit::PowerlawDegreeSequence ds_gen(1, n - 1, -2.5);
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

        run_benchmark<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", graph, gen);
        omp_set_num_threads(4);
        run_benchmark<AlgorithmParallelNaive>("parallel-naive", graph, gen);
        omp_set_num_threads(4);
        run_benchmark<AlgorithmParallelNaiveGlobal>("parallel-global-naive", graph, gen);
        omp_set_num_threads(4);
        run_benchmark<AlgorithmParallelGlobal>("parallel-global", graph, gen);
        //run_benchmark<AlgorithmParallelVectorSet<4, ThreadsafeSetLockedList<edge_t, edge_hash_crc32>>>("parallel-ll", graph, gen);
        //run_benchmark<AlgorithmParallelGlobalES<4, ThreadsafeSetLockedList<edge_t, edge_hash_crc32>>>("parallel-global-ll", graph, gen);

        std::cout << "\n";
    }

    //run_benchmark<EdgeSwitch_VectorSet<tsl::hopscotch_set<edge_t, edge_hash>>>("hps", n, target_m, gen);
    //run_benchmark<EdgeSwitch_VectorSet<std::unordered_set<edge_t, edge_hash>>>("std::uset", n, target_m, gen);

    return 0;

}
