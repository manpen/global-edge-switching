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
#include <es/algorithms/AlgorithmParallelNaiveGlobalGaps.hpp>
#include <es/algorithms/AlgorithmParallelGlobal.hpp>
#include <es/algorithms/AlgorithmParallelGlobalGaps.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV2.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV3.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV4.hpp>
#include <es/algorithms/AlgorithmNetworKit.hpp>
#include <es/algorithms/AlgorithmGenGraph.hpp>
#include <es/algorithms/AlgorithmGlobal.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/HavelHakimiGenerator.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>
#include <networkit/auxiliary/Random.hpp>

#ifdef ENABLE_ITT
#define ITT_ENABLED(X) X
#include <ittnotify.h>
#else
#define ITT_ENABLED(X)
#endif

using namespace es;

template <typename Algo>
void run_benchmark(std::string_view label, NetworKit::Graph& graph, std::mt19937_64 &gen, bool detailed = true) {
    Algo es(graph);

    {
        incpwl::ScopedTimer timer;
        edge_t m = graph.numberOfEdges();
        const auto switches_per_edge = 10;
        const auto requested_switches = switches_per_edge * m;

        es.enable_logging();

        ITT_ENABLED(__itt_pause());
        const auto sucessful_switches = es.do_switches(gen, requested_switches);
        ITT_ENABLED(__itt_resume());

        if (detailed) {
            std::cout << label << ": Switches successful: " << (100. * sucessful_switches / requested_switches) << "% \n";
            std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
            std::cout << label << ": Switches per second: " << requested_switches / timer.elapsedSeconds() * 1e-6 << "M" << std::endl;
        }
        std::cout << label << ": Estimated randomization time: " << timer.elapsedSeconds() * (1. * requested_switches / sucessful_switches) << "s \n\n";
    }
}

int main() {
    ITT_ENABLED(__itt_pause());
    std::mt19937_64 gen{0};
    Aux::Random::setSeed(1337, true);

    node_t n = 1<<20;

    NetworKit::Graph graph;
    {
        incpwl::ScopedTimer timer("Seed graph");
        NetworKit::PowerlawDegreeSequence ds_gen(1, n - 1, -2.01);
        std::vector<NetworKit::count> ds;
        ds_gen.run();
        ds = ds_gen.getDegreeSequence(n);
        graph = NetworKit::HavelHakimiGenerator(ds, false).generate();
    }
    std::cout << "Nodes: " << graph.numberOfNodes() << " Edges: " << graph.numberOfEdges() << std::endl;

    for (int repeat = 0; repeat < 5; ++repeat) {
        run_benchmark<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>("robin", graph, gen);
        run_benchmark<AlgorithmGenGraph>("gengraph", graph, gen);
        run_benchmark<AlgorithmNetworKit>("nk", graph, gen);
        run_benchmark<AlgorithmGlobal<tsl::robin_set<edge_t, edge_hash_crc32>>>("global", graph, gen);
        run_benchmark<AlgorithmParallelNaive>("parallel-naive", graph, gen);
        run_benchmark<AlgorithmParallelGlobal>("parallel-global", graph, gen);
        run_benchmark<AlgorithmParallelGlobalGaps>("parallel-global-gaps", graph, gen);
        run_benchmark<AlgorithmParallelNaiveGlobal>("parallel-global-naive", graph, gen);
        run_benchmark<AlgorithmParallelNaiveGlobalGaps>("parallel-global-gaps-naive", graph, gen);
        run_benchmark<AlgorithmParallelGlobalNoWaitV2>("parallel-global-no-wait-v2", graph, gen);
        run_benchmark<AlgorithmParallelGlobalNoWaitV3>("parallel-global-no-wait-v3", graph, gen);
        run_benchmark<AlgorithmParallelGlobalNoWaitV4<true>>("parallel-global-no-wait-v4", graph, gen);

        std::cout << "\n";
    }

    return 0;
}
