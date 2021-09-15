#include <iostream>
#include <tlx/cmdline_parser.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/EdgeListReader.hpp>
#include <es/autocorrelation/AutocorrelationAnalysis.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV4.hpp>

struct autocorrelation_config_t {
    const NetworKit::Graph& g;
    const std::vector<size_t>& thinnings;
    const size_t min_snapshots;
    const size_t max_snapshots;
    const std::string graphlabel;
    const size_t switches_per_edge;

    autocorrelation_config_t(const NetworKit::Graph& g_,
                             const std::vector<size_t>& thinnings_,
                             size_t min_snapshots_,
                             size_t max_snapshots_,
                             const std::string& graphlabel_,
                             size_t switches_per_edge_)
            : g(g_),
            thinnings(thinnings_),
            min_snapshots(min_snapshots_),
            max_snapshots(max_snapshots_),
            graphlabel(graphlabel_),
            switches_per_edge(switches_per_edge_) { }
};

template <typename Algo>
void run_realworld_autocorrelation_analysis(const autocorrelation_config_t& c, std::mt19937_64& gen, const std::string& algolabel, unsigned seed) {
    const size_t fake_graphseed = 0;
    AutocorrelationAnalysis<Algo> aa(c.g, gen, c.thinnings, c.min_snapshots, algolabel, c.graphlabel, fake_graphseed, seed, c.switches_per_edge, c.max_snapshots);
}

int main(int argc, char *argv[]) {
    tlx::CmdlineParser cp;
    cp.set_description("Autocorrelation Analysis");

    unsigned algo = 0;
    cp.add_param_unsigned("algo", algo, "Algorithm; 1=Robin, 2=Global");

    std::string input_fn;
    cp.add_param_string("input", input_fn, "Input Graph File");

    unsigned runs = 0;
    cp.add_param_unsigned("runs", runs, "Runs");

    unsigned seed = std::random_device{}();
    cp.add_unsigned("seed", seed, "Autocorrelation Seed");

    unsigned min_snapshots;
    cp.add_unsigned("minsnaps", min_snapshots, "Minimum Number of Snapshots / Thinning");

    unsigned max_snapshots;
    cp.add_unsigned("maxsnaps", max_snapshots, "Maximum Number of Snapshots / Thinning");

    unsigned switches_per_edge = 1;
    cp.add_unsigned("switchesperedge", switches_per_edge, "Switches / Edge");

    std::vector<std::string> thinnings_str;
    cp.add_param_stringlist("thinnings", thinnings_str, "Thinning Values e.g. --thinnings 1 2 3 4");

    // evaluate command line parser
    if (!cp.process(argc, argv)) {
        return -1;
    }
    
    std::cout << "# successfully processed command line arguments" << std::endl;

    std::vector<size_t> thinnings;
    thinnings.reserve(thinnings_str.size());
    for (auto& thinning_str : thinnings_str) {
        std::stringstream tss(thinning_str);
        size_t thinning;
        tss >> thinning;
        if (!tss) {
            return 0;
        }
        thinnings.push_back(thinning);
    }

    NetworKit::EdgeListReader edgelist_reader = NetworKit::EdgeListReader(' ', 0, "%", true);
    NetworKit::Graph g = edgelist_reader.read(input_fn);
    std::cout << "# successfully loaded edgelist file " << input_fn << std::endl;

    
    const autocorrelation_config_t config(g, thinnings, min_snapshots, max_snapshots, input_fn, switches_per_edge);

    // run autocorrelation analysis
    std::cout << "type,algo,"
              << "graphlabel,"
              << "n,"
              << "m,"
              << "chainlength,"
              << "min snapshots/thinning,"
              << "max snapshots/thinning,"
              << "switches/edge,"
              << "thinning,"
              << "snapshots/thinning,"
              << "successful switches,"
              << "independent edges,"
              << "non-independent edges,"
              << "independent none-edges,"
              << "non-independent none-edges,"
              << "graphseed,"
              << "seed" << std::endl;
    for (unsigned run = 0; run < runs; run++) {
        std::cout << "# run " << run << std::endl;
        std::mt19937_64 gen(seed);

        switch (algo) {
            case 1:
                run_realworld_autocorrelation_analysis<es::AlgorithmVectorSet<tsl::robin_set<es::edge_t, es::edge_hash_crc32>>>
                        (config, gen, "ES-Robin", seed);
                break;
            case 2:
                run_realworld_autocorrelation_analysis<es::AlgorithmParallelGlobalNoWaitV4>
                        (config, gen, "ES-Global-NoWait-V4", seed);
                break;
            default:
                break;
        }

        // generate new random seed for the next run
        seed = std::random_device{}();
    }

    return 0;
}
