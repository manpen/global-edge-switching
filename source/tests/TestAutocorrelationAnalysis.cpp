#include <es/autocorrelation/AutocorrelationAnalysis.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>
#include <networkit/generators/HavelHakimiGenerator.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV4.hpp>
#include <tlx/die.hpp>
#include <vector>

int main() {
    // transitions
    const transition_counter_t x{4, 8, 16, 2};
    const transition_counter_t hat_x = x.compute_independent_model_prediction();
    die_unless(hat_x.x00 == 12. * 20 / 30);
    die_unless(hat_x.x01 == 12. * 10 / 30);
    die_unless(hat_x.x10 == 18. * 20 / 30);
    die_unless(hat_x.x11 == 18. * 10 / 30);

    // ggt, kgv
    die_unless(ggt(5, 7) == 1);
    die_unless(ggt(7, 14) == 7);
    die_unless(ggt(14, 28) == 14);
    die_unless(kgv(2, 3) == 6);
    die_unless(kgv(6, 60) == 60);
    die_unless(kgv(1440, 175) == 50400);
    const std::vector<size_t> two_to_6 = {2, 3, 4, 5, 6};
    const std::vector<size_t> one_to_7 = {1, 2, 3, 4, 5, 6, 7};
    const std::vector<size_t> one_to_10_m7 = {1, 2, 3, 4, 5, 6, 8, 9, 10};
    const std::vector<size_t> power_of_twos_p5 = {2, 4, 5, 8, 16, 32};
    die_unless(kgv(two_to_6) == 60);
    die_unless(kgv(one_to_7) == 420);
    die_unless(kgv(one_to_10_m7) == 360);
    die_unless(kgv(power_of_twos_p5) == 160);
    const std::vector<size_t> test_1 = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 30};

    // time series
    const es::node_t n = 2000;
    const size_t graphseed = 1337;
    Aux::Random::setSeed(1337, true);
    NetworKit::Graph graph;
    {
        NetworKit::PowerlawDegreeSequence ds_gen(1, n - 1, -2.01);
        std::vector<NetworKit::count> ds;
        ds_gen.run();
        ds = ds_gen.getDegreeSequence(n);
        graph = NetworKit::HavelHakimiGenerator(ds, false).generate();
    }
    std::random_device rd;
    auto seed = rd();
    std::mt19937_64 gen(seed);
    const size_t min_snapshots = 400;
    const size_t max_snapshots = min_snapshots;
    const std::vector<size_t> thinnings = {1, 2, 3, 4, 5};
    AutocorrelationAnalysis<es::AlgorithmVectorSet<tsl::robin_set<es::edge_t, es::edge_hash_crc32>>> ts(graph, gen, test_1, min_snapshots, "Robin", "PLD-2.01", graphseed, seed, 1, max_snapshots, omp_get_max_threads());
    AutocorrelationAnalysis<es::AlgorithmParallelGlobalNoWaitV4> ts2(graph, gen, test_1, min_snapshots, "NoWaitV4", "PLD-2.01", graphseed, seed, 1, max_snapshots, omp_get_max_threads());

    return 0;
}