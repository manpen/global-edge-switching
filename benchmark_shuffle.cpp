#include <iostream>
#include <cassert>
#include <execution>
#include <random>
#include <string_view>
#include <vector>

#include <es/Graph.hpp>
#include <es/ScopedTimer.hpp>
#include <shuffle/algorithms/FisherYates.hpp>
#include <shuffle/algorithms/InplaceScatterShuffle.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>

using namespace es;

void benchmark_shuffle(NetworKit::Graph& graph, size_t num_threads, std::mt19937_64 &gen) {
    std::vector<edge_t> edge_list;
    edge_list.reserve(graph.numberOfEdges());
    graph.forEdges([&](NetworKit::node u, NetworKit::node v){
        auto edge = to_edge(u, v);
        edge_list.emplace_back(edge);
    });

    omp_set_num_threads(num_threads);
    {
        incpwl::ScopedTimer timer;
        for (int r = 0; r < 20; ++r) {
            shuffle::GeneratorProvider gen_prov(gen, num_threads);
            shuffle::parallel::iss_shuffle(edge_list.begin(), edge_list.end(), gen_prov);
        }
        std::cout << "PUs: " << num_threads << ": Runtime " << timer.elapsedSeconds() << "s\n";
    }
}

void benchmark_shuffle2(NetworKit::Graph& graph, size_t num_threads, std::mt19937_64 &gen) {
    std::vector<edge_t> edge_list;
    edge_list.reserve(graph.numberOfEdges());
    graph.forEdges([&](NetworKit::node u, NetworKit::node v){
        auto edge = to_edge(u, v);
        edge_list.emplace_back(edge);
    });

    size_t threads_per_shuffle = 2;
    size_t num_lists = num_threads / threads_per_shuffle;
    {
        incpwl::ScopedTimer timer;
        omp_set_num_threads(num_lists);
        #pragma omp parallel
        {
            size_t num_rounds = 20 / omp_get_num_threads();
            size_t tid = omp_get_thread_num();
            std::vector<edge_t> edge_list_local;
            edge_list_local.resize(edge_list.size());
            std::iota(edge_list_local.begin(), edge_list_local.end(), 0);
            omp_set_num_threads(threads_per_shuffle);
            for (int r = 0; r < num_rounds; ++r) {
                shuffle::GeneratorProvider gen_prov(gen, threads_per_shuffle);
                shuffle::parallel::iss_shuffle(edge_list_local.begin(), edge_list_local.end(), gen_prov);
            }
        };
        std::cout << "PUs: " << num_threads << ": Runtime2 " << timer.elapsedSeconds() << "s\n";
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 1<<20;
    edge_t target_m = n * 1.44;

    for (int repeat = 0; repeat < 5; ++repeat) {
        double p = (2.0 * target_m) / n / (n - 1);
        auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();
        for (size_t num_threads = 2; num_threads <= 8; num_threads += 2) {
            benchmark_shuffle(graph, num_threads, gen);
        }
        for (size_t num_threads = 2; num_threads <= 8; num_threads += 2) {
            benchmark_shuffle2(graph, num_threads, gen);
        }
        std::cout << "\n";
    }
    return 0;
}
