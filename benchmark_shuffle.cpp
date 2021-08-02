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
    std::vector<edge_t> edge_list1 = edge_list;
    {
        incpwl::ScopedTimer timer;
        shuffle::GeneratorProvider gen_prov(gen, num_threads);
        shuffle::parallel::iss_shuffle(edge_list1.begin(), edge_list1.end(), gen_prov);
        std::cout << "PUs: " << num_threads << ": Runtime " << timer.elapsedSeconds() << "s\n";
    }
    std::vector<std::mt19937_64> gen_local(num_threads, std::mt19937_64(gen()));
    std::vector<edge_t> edge_list2 = edge_list;
    {
        incpwl::ScopedTimer timer;
        size_t r = num_threads;
        std::shuffle(edge_list2.begin(), edge_list2.begin() + r, gen);
        while (r < edge_list2.size() / 2) {
            #pragma omp parallel num_threads(num_threads)
            {
                const size_t thread_id = omp_get_thread_num();

                std::mt19937_64 &gen = gen_local[thread_id];
                shuffle::RandomBits fair_coin;

                const size_t block_size = r / num_threads;
                const size_t beg = block_size * thread_id;
                const size_t end = beg + block_size;
                for (size_t i = beg; i < end; ++i) {
                    if (fair_coin(gen)) std::swap(edge_list2[i], edge_list2[r + i]);
                }
            }
            r *= 2;
        }
        std::cout << "PUs: " << num_threads << ": Runtime2 " << timer.elapsedSeconds() << "s\n";
    }
}

int main() {
    std::mt19937_64 gen{0};

    node_t n = 1<<20;
    edge_t target_m = n * 50.44;

    for (int repeat = 0; repeat < 1; ++repeat) {
        for (size_t num_threads = 2; num_threads <= 8; num_threads += 2) {
            benchmark_shuffle(n, target_m, num_threads, gen);
        }
        std::cout << "\n";
    }
    return 0;
}
