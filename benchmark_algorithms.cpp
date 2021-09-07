#include <iostream>
#include <fstream>

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

void run_benchmark(std::string_view algo, NetworKit::Graph graph, std::mt19937_64& gen, bool detailed = true) {
    std::unique_ptr<AlgorithmBase> es;
    if (algo == "robin") {
        es = std::make_unique<AlgorithmVectorSet<tsl::robin_set<edge_t, edge_hash_crc32>>>(graph);
    } else if (algo == "naive") {
        es = std::make_unique<AlgorithmParallelNaive>(graph);
    } else if (algo == "global-naive") {
        es = std::make_unique<AlgorithmParallelNaiveGlobal>(graph);
    } else if (algo == "global") {
        es = std::make_unique<AlgorithmParallelGlobal>(graph);
    } else if (algo == "global-no-wait") {
        es = std::make_unique<AlgorithmParallelGlobalNoWaitV2>(graph);
    }

    {
        incpwl::ScopedTimer timer;
        const edge_t m = graph.numberOfEdges();
        const auto switches_per_edge = 10;
        const auto requested_switches = switches_per_edge * m;
        const auto sucessful_switches = es->do_switches(gen, requested_switches);
        if (detailed) {
            std::cout << "Switches successful: " << (100. * sucessful_switches / requested_switches) << "% \n";
            std::cout << "Runtime " << timer.elapsedSeconds() << "s\n";
            std::cout << "Switches per second: " << requested_switches / timer.elapsedSeconds() * 1e-6 << "M" << std::endl;
        }
        std::cout << "Successful switches per second: " << (1. * sucessful_switches / m) / timer.elapsedSeconds() << "m \n";
    }
}

void benchmark_file(int argc, const char** argv) {
    if (argc < 3)
        throw std::runtime_error("Missing filename");
    std::string filename{argv[2]};
    double gamma = -1;
    edge_t m = 0;
    std::vector<edge_t> degree_sequence;
    {
        std::ifstream ifile{filename};
        std::string line;
        getline(ifile, line);
        gamma = std::stod(line);
        while (getline(ifile, line)) {
            edge_t degree = std::stoul(line);
            degree_sequence.push_back(degree);
            m += degree;
        }
        if (!std::is_sorted(degree_sequence.begin(), degree_sequence.end(), std::greater<edge_t>()))
            throw std::runtime_error("Degree sequence is not sorted");
    }
    if (!NetworKit::HavelHakimiGenerator(degree_sequence).isRealizable())
        throw std::runtime_error("Degree sequence not realizable");
    size_t threads = argc > 3 ? std::stoi(argv[3]) : 4;
    size_t repeats = argc > 4 ? std::stoi(argv[4]) : 1;
    std::string algo = argc > 5 ? argv[5] : "global-no-wait";
    bool detailed = argc > 6 ? std::string(argv[6]) == "verbose" : false;

    std::cout << "Starting file experiment with parameters" << std::endl
              << "algo=" << algo << std::endl
              << "file=" << filename << std::endl
              << "n=" << degree_sequence.size() << std::endl
              << "m=" << m << std::endl
              << "gamma_estimate=" << gamma << std::endl
              << "p=" << threads << std::endl
              << "repeats=" << repeats << std::endl << std::endl;

    omp_set_num_threads(threads);

    std::mt19937_64 gen{0};
    while (repeats--) {
        auto graph = NetworKit::HavelHakimiGenerator(degree_sequence).generate();

        run_benchmark(algo, graph, gen, detailed);

        std::cout << "\n";
    }
}

void benchmark_random(int argc, const char** argv) {
    node_t n = argc > 2 ? (1<<std::stoi(argv[2])) : 1<<20;
    edge_t target_m = argc > 3 ? n * std::stod(argv[3]) : n * 1.44;
    size_t threads = argc > 4 ? std::stoi(argv[4]) : 4;
    size_t repeats = argc > 5 ? std::stoi(argv[5]) : 1;
    std::string algo = argc > 6 ? argv[6] : "global-no-wait";
    bool detailed = argc > 7 ? std::string(argv[7]) == "verbose" : false;

    std::cout << "Starting random-graph experiment with parameters" << std::endl
              << "algo=" << algo << std::endl
              << "n=" << n << std::endl
              << "m=" << target_m << std::endl
              << "p=" << threads << std::endl
              << "repeats=" << repeats << std::endl << std::endl;

    omp_set_num_threads(threads);

    std::mt19937_64 gen{0};
    while (repeats--) {
        double p = (2.0 * target_m) / n / (n - 1);
        auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();

        run_benchmark(algo, graph, gen, detailed);

        std::cout << "\n";
    }
}

void benchmark_powerlaw(int argc, const char** argv) {
    node_t n = argc > 2 ? (1<<std::stoi(argv[2])) : 1<<20;
    edge_t target_m = argc > 3 ? n * std::stod(argv[3]) : n * 1.44;
    double gamma = argc > 4 ? std::stod(argv[4]) : 2.5;
    size_t threads = argc > 5 ? std::stoi(argv[5]) : 4;
    size_t repeats = argc > 6 ? std::stoi(argv[6]) : 1;
    std::string algo = argc > 7 ? argv[7] : "global-no-wait";
    bool detailed = argc > 8 ? std::string(argv[8]) == "verbose" : false;

    std::cout << "Starting powerlaw-graph experiment with parameters" << std::endl
              << "algo=" << algo << std::endl
              << "n=" << n << std::endl
              << "m=" << target_m << std::endl
              << "gamma=" << std::setprecision(3) << gamma << std::endl
              << "p=" << threads << std::endl
              << "repeats=" << repeats << std::endl << std::endl;

    omp_set_num_threads(threads);

    std::mt19937_64 gen{0};
    while (repeats--) {
        NetworKit::PowerlawDegreeSequence ds_gen(1, n - 1, -gamma);
        std::vector<NetworKit::count> ds;
        bool realizable;
        do {
            ds_gen.run();
            ds = ds_gen.getDegreeSequence(n);
            realizable = NetworKit::HavelHakimiGenerator(ds).isRealizable();
        } while (!realizable);
        auto graph = NetworKit::HavelHakimiGenerator(ds).generate();

        run_benchmark(algo, graph, gen, detailed);

        std::cout << "\n";
    }
}

int main(int argc, const char** argv) {
    std::string mode = argc > 1 ? argv[1] : "random";
    if (mode == "random") benchmark_random(argc, argv);
    else if (mode == "powerlaw") benchmark_powerlaw(argc, argv);
    else if (mode == "file") benchmark_file(argc, argv);

    return 0;
}
