#include <iostream>

#include <random>
#include <vector>
#include <string_view>

#include <es/Graph.hpp>
#include <es/ScopedTimer.hpp>

#include <es/algorithms/AlgorithmParallelGlobalNoWaitV4.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/HavelHakimiGenerator.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>
#include <networkit/auxiliary/Random.hpp>

using namespace es;

template <typename Algo>
void run_benchmark(std::string_view label, NetworKit::Graph graph, std::mt19937_64 &gen, bool detailed = true) {
    Algo es(graph);

    {
        incpwl::ScopedTimer timer;
        edge_t m = graph.numberOfEdges();
        const auto switches_per_edge = 10;
        const auto requested_switches = switches_per_edge * m;

        es.enable_logging();

        const auto sucessful_switches = es.do_switches(gen, requested_switches);

        if (detailed) {
            std::cout << label << ": Switches successful: " << (100. * sucessful_switches / requested_switches) << "% \n";
            std::cout << label << ": Runtime " << timer.elapsedSeconds() << "s\n";
            std::cout << label << ": Switches per second: " << requested_switches / timer.elapsedSeconds() * 1e-6 << "M" << std::endl;
        }
        std::cout << label << ": Estimated randomization time: " << timer.elapsedSeconds() * (1. * requested_switches / sucessful_switches) << "s \n\n";
    }
}

int main() {
    std::mt19937_64 gen{0};
    Aux::Random::setSeed(1337, true);


    for(double gamma : {3.0, 2.5, 2.1, 2.01, 2.001}) {
        node_t n = 1024;
        while(true) {
            NetworKit::PowerlawDegreeSequence ds_gen(1, std::pow(n, 1.0 / (gamma - 1.0)), -gamma);
            std::vector<NetworKit::count> ds;
            NetworKit::Graph graph;
            while (true) {
                ds_gen.run();
                ds = ds_gen.getDegreeSequence(n);
                auto gen = NetworKit::HavelHakimiGenerator(ds, false);
                if (gen.isRealizable()) {
                    graph = gen.generate();
                    break;
                }
            }

            std::cout << "#new " << gamma << ',' << graph.numberOfNodes() << ',' << graph.numberOfEdges() << std::endl;
            run_benchmark<AlgorithmParallelGlobalNoWaitV4<true>>("parallel-global-no-wait-v4", graph, gen);

            if (graph.numberOfEdges() > 1e8)
                break;
            n *= 2;
        }
    }

    return 0;
}
