#include <iostream>
#include <cassert>
#include <random>
#include <string_view>
#include <vector>

#include <es/Graph.hpp>
#include <es/ScopedTimer.hpp>
#include <shuffle/algorithms/FisherYates.hpp>
#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>

using namespace es;

void benchmark_shuffle(node_t n, edge_t target_m, size_t num_threads, std::mt19937_64 &gen) {
    double p = (2.0 * target_m) / n / (n - 1);
    auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();

    std::vector<edge_t> edge_list;
    edge_list.reserve(graph.numberOfEdges());
    graph.forEdges([&](NetworKit::node u, NetworKit::node v){
        auto edge = to_edge(u, v);
        edge_list.emplace_back(edge);
    });

    omp_set_num_threads(num_threads);
    {
        incpwl::ScopedTimer timer;
        shuffle::GeneratorProvider gen_prov(gen);
        shuffle::parallel::iss_shuffle(edge_list.begin(), edge_list.end(), gen_prov);
        std::cout << "PUs: " << num_threads << ": Runtime " << timer.elapsedSeconds() << "s\n";
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 1<<20;
    edge_t target_m = n * 1.44;

    for (int repeat = 0; repeat < 5; ++repeat) {
        for (size_t num_threads = 2; num_threads <= 32; num_threads += 2) {
            benchmark_shuffle(n, target_m, num_threads, gen);
        }
        std::cout << "\n";
    }
    return 0;
}
