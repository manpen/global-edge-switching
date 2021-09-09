#include <iostream>

#include <cassert>
#include <random>
#include <vector>
#include <unordered_set>
#include <string_view>
#include <functional>

#include <tsl/hopscotch_set.h>
#include <tsl/robin_set.h>


#include <es/algorithms/AlgorithmParallelNaive.hpp>

#include <es/AdjacencyVector.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/auxiliary/Random.hpp>


using namespace es;

int main() {
    std::mt19937_64 gen{0};

    for (int repeat = 0; repeat < 5; ++repeat) {
        for (unsigned n_exp : {16, 20, 22, 24}) {
            for (unsigned avg_deg : {10}) {
                for (double load_factor : {2.0}) {
                    for (double chunk_factor : {0.1, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0}) {
                        Aux::Random::setSeed(0, true);

                        node_t n = 1llu << n_exp;
                        edge_t target_m = n * avg_deg;

                        double p = (2.0 * target_m) / n / (n - 1);
                        auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();

                        std::cout << "NEWRUN: n_exp=" << n_exp << ",avg_deg="<<avg_deg<<",load_factor="<<load_factor<<",chunk_factor="<<chunk_factor << "\n";

                        incpwl::ScopedTimer init_timer;
                        AlgorithmParallelNaive es(graph, load_factor, chunk_factor);
                        std::cout << "Init " << init_timer.elapsedSeconds() << "s\n";

                        {
                            incpwl::ScopedTimer timer;
                            const auto switches_per_edge = 10;
                            const auto requested_swichtes = switches_per_edge * target_m;
                            const auto sucessful_switches = es.do_switches(gen, requested_swichtes);
                            std::cout << "Switches successful: " << (100. * sucessful_switches / requested_swichtes) << "\n";
                            std::cout << "Runtime " << timer.elapsedSeconds() << "s\n";
                            std::cout << "Switches per second: " << (1e-6 * requested_swichtes / timer.elapsedSeconds()) << "M\n";
                        }
                    }
                }
            }
        }
    }

    return 0;

}
