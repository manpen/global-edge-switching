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

NetworKit::Graph read_graph(std::string filename) {
    std::ifstream ifile{filename};
    std::string line;
    getline(ifile, line);
    node_t n = std::stoll(line.substr(2));
    NetworKit::Graph graph(n);
    while (getline(ifile, line)) {
        if (line.starts_with('%')) continue;
        size_t sep = line.find(',');
        node_t u = std::stoll(line.substr(0, sep));
        node_t v = std::stoll(line.substr(sep + 1));
        graph.addEdge(u, v);
    }
    return graph;
}

void write_graph(const NetworKit::Graph& graph, std::string filename) {
    std::ofstream ofile{filename};
    ofile << "% " << graph.numberOfNodes() << "\n";
    ofile << "% " << graph.numberOfEdges() << "\n";
    graph.forEdges([&](NetworKit::node u, NetworKit::node v){
        ofile << u << "," << v << "\n";
    });
    ofile << std::flush;
}

void generate_random_graph(int argc, const char** argv) {
    if (argc < 3)
        throw std::runtime_error("Missing output filename");
    std::string filename{argv[2]};
    node_t n = argc > 3 ? (1<<std::stoi(argv[3])) : 1<<20;
    edge_t target_m = argc > 4 ? n * std::stod(argv[4]) : n * 1.44;

    double p = (2.0 * target_m) / n / (n - 1);
    auto graph = NetworKit::ErdosRenyiGenerator(n, p, false, false).generate();
    write_graph(graph, filename);
}

void generate_powerlaw_graph(int argc, const char** argv) {
    if (argc < 3)
        throw std::runtime_error("Missing output filename");
    std::string filename{argv[2]};
    node_t n = argc > 3 ? (1<<std::stoi(argv[3])) : 1<<20;
    double gamma = argc > 4 ? std::stod(argv[4]) : 2.5;
    edge_t d_min = argc > 5 ? std::stod(argv[5]) : 1;

    NetworKit::PowerlawDegreeSequence ds_gen(d_min, n - 1, -gamma);
    std::vector<NetworKit::count> ds;
    bool realizable;
    do {
        ds_gen.run();
        ds = ds_gen.getDegreeSequence(n);
        realizable = NetworKit::HavelHakimiGenerator(ds).isRealizable();
    } while (!realizable);
    auto graph = NetworKit::HavelHakimiGenerator(ds).generate();
    write_graph(graph, filename);
}

void benchmark_on_file(int argc, const char** argv) {
    if (argc < 3)
        throw std::runtime_error("Missing filename");
    std::string filename{argv[2]};
    auto graph = read_graph(filename);
    size_t threads = argc > 3 ? std::stoi(argv[3]) : 4;
    size_t repeats = argc > 4 ? std::stoi(argv[4]) : 1;
    std::string algo = argc > 5 ? argv[5] : "global-no-wait";
    size_t switches = argc > 6 ? std::stoi(argv[6]) : 10;
    bool detailed = argc > 7 ? std::string(argv[7]) == "verbose" : false;

    std::cout << "Starting experiment with parameters" << std::endl
              << "algo=" << algo << std::endl
              << "switches=" << switches << "m" << std::endl
              << "file=" << filename << std::endl
              << "n=" << graph.numberOfNodes() << std::endl
              << "m=" << graph.numberOfEdges() << std::endl
              << "p=" << threads << std::endl
              << "repeats=" << repeats << std::endl << std::endl;

    omp_set_num_threads(threads);

    std::unique_ptr<AlgorithmBase> es;
    double init_time;
    {
        incpwl::ScopedTimer timer;
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
        init_time = timer.elapsedSeconds();
    }

    std::mt19937_64 gen{0};
    while (repeats--) {
        incpwl::ScopedTimer timer;
        const edge_t m = graph.numberOfEdges();
        const auto requested_switches = switches * m;
        const auto sucessful_switches = es->do_switches(gen, requested_switches);
        double run_time = timer.elapsedSeconds();
        if (detailed) {
            std::cout << "Switches successful: " << (100. * sucessful_switches / requested_switches) << "% \n";
            std::cout << "Runtime: " << run_time << "s\n";
            std::cout << "Initialization time: " << init_time << "s \n";
            std::cout << "Runtime + Initialization: " << run_time + init_time << "s\n";
            std::cout << "Switches per second: " << requested_switches / run_time * 1e-6 << "M \n";
            std::cout << "Successful switches per second: " << sucessful_switches / run_time * 1e-6 << "M \n";
            std::cout << "Runtime for 1m successful switches: " << run_time * (1. * m / sucessful_switches) << "s \n";
            std::cout << "Runtime for 1m successful switches + Initialization: " << init_time + run_time * (1. * m / sucessful_switches) << "s \n";
        }
        std::cout << "Runtime for 10m successful switches: " << run_time * (1. * switches * m / sucessful_switches) << "s \n";
        std::cout << "Runtime for 10m successful switches + Initialization: " << init_time + run_time * (1. * switches * m / sucessful_switches) << "s \n";
        std::cout << std::endl;
    }
}

int main(int argc, const char** argv) {
    std::string mode = argc > 1 ? argv[1] : "bench";
    if (mode == "bench") benchmark_on_file(argc, argv);
    else if (mode == "generate-random") generate_random_graph(argc, argv);
    else if (mode == "generate-powerlaw") generate_powerlaw_graph(argc, argv);

    return 0;
}