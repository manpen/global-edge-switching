#include <iostream>

#include <random>
#include <unordered_set>

#include <es/Graph.hpp>
#include <es/algorithms/AlgorithmParallelGlobal.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWait.hpp>

#include <networkit/generators/ErdosRenyiGenerator.hpp>

using namespace es;

template <typename Algo>
void run_permutation_test(std::string_view label, node_t n, edge_t target_m, size_t num_threads, std::mt19937_64 &gen) {
    if (target_m % (2 * num_threads) != 0) {
        assert(target_m > 2 * num_threads);
        target_m -= target_m % (2 * num_threads);
    }
    double p = std::min(1.0, (2.0 * target_m) / n / (n - 1));
    NetworKit::Graph input_graph;
    do {
        input_graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();
    } while (input_graph.numberOfEdges() == 0 || input_graph.numberOfEdges() % (2 * num_threads) != 0);

    std::vector<size_t> rho(input_graph.numberOfEdges());
    std::iota(rho.begin(), rho.end(), 0);
    std::shuffle(rho.begin(), rho.end(), gen);

    omp_set_num_threads(num_threads);

    Algo es_par(input_graph);
    es_par.do_switches(rho, num_threads);
    NetworKit::Graph output_graph_par = es_par.get_graph();

    omp_set_num_threads(1);

    Algo es_seq(input_graph);
    es_seq.do_switches(rho, 1);
    NetworKit::Graph output_graph_seq = es_seq.get_graph();

    if (output_graph_par.numberOfEdges() != output_graph_seq.numberOfEdges())
        abort();

    output_graph_par.forEdges([&](NetworKit::node a, NetworKit::node b){
        if (!output_graph_seq.hasEdge(a, b))
            abort();
    });
}

template <typename Algo>
void run_basic_test(std::string_view label, node_t n, edge_t target_m, size_t num_threads, std::mt19937_64 &gen) {
    double p = std::min(1.0, (2.0 * target_m) / n / (n - 1));
    auto input_graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();
    size_t m = input_graph.numberOfEdges();

    omp_set_num_threads(num_threads);

    Algo es(input_graph);

    {
        const auto requested_swichtes = 2 * m;
        const auto sucessful_switches = es.do_switches(gen, requested_swichtes);
    }

    auto output_graph = es.get_graph();

    // check degrees are maintained
    edge_t num_edges = 0;
    for (node_t i = 0; i < n; ++i) {
        num_edges += input_graph.degree(i);
        if (input_graph.degree(i) != output_graph.degree(i))
            abort();
    }

    if (num_edges / 2 != input_graph.numberOfEdges())
        abort();

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

    size_t num_threads = 2;

    for (auto n : {10, 100, 1000}) {
        for (auto d : {1, 5, 10}) {
            for (auto t : {2, 4, 6, 8}) {
                if (d >= n) continue;

                edge_t target_m = (n * d) / 2;
                if (target_m < 4 * num_threads) continue;

                for (int repeat = 0; repeat < 100; ++repeat) {
                    run_basic_test<AlgorithmParallelGlobal>("parallel-global", n, target_m, num_threads, gen);
                    run_permutation_test<AlgorithmParallelGlobal>("parallel-global", n, target_m, num_threads, gen);
                    run_permutation_test<AlgorithmParallelGlobalNoWait>("parallel-global-no-wait", n, target_m, num_threads, gen);

                    std::cout << "n=" << n << " d=" << d << " t=" << t << "\n";
                }
            }
        }
    }

    return 0;
}
