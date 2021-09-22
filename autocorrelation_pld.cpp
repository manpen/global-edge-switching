#include <iostream>
#include <string.h>
#include <tlx/cmdline_parser.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/EdgeListReader.hpp>
#include <es/autocorrelation/AutocorrelationAnalysis.hpp>
#include <es/algorithms/AlgorithmParallelGlobal.hpp>
#include <es/algorithms/AlgorithmParallelGlobalNoWaitV4.hpp>
#include <es/algorithms/AlgorithmVectorSet.hpp>
#include <tsl/robin_set.h>
#include <networkit/generators/HavelHakimiGenerator.hpp>
#include <networkit/generators/PowerlawDegreeSequence.hpp>
#include <networkit/auxiliary/Random.hpp>

struct autocorrelation_config_t {
    const NetworKit::Graph& g;
    const std::vector<size_t>& thinnings;
    const size_t min_snapshots;
    const size_t max_snapshots;
    const std::string graphlabel;
    const size_t switches_per_edge;
    const bool skip_non_orig;
    const std::string output_fn_prefix;

    autocorrelation_config_t(const NetworKit::Graph& g_,
                             const std::vector<size_t>& thinnings_,
                             size_t min_snapshots_,
                             size_t max_snapshots_,
                             const std::string& graphlabel_,
                             size_t switches_per_edge_,
                             bool skip_non_orig_,
                             const std::string& output_fn_prefix_)
            : g(g_),
              thinnings(thinnings_),
              min_snapshots(min_snapshots_),
              max_snapshots(max_snapshots_),
              graphlabel(graphlabel_),
              switches_per_edge(switches_per_edge_),
              skip_non_orig(skip_non_orig_),
              output_fn_prefix(output_fn_prefix_)
    { }
};

template <typename Algo>
void run_pld_autocorrelation_analysis(const autocorrelation_config_t& c, std::mt19937_64& gen, const std::string& algolabel, unsigned graphseed, unsigned seed, int pu_id) {
    AutocorrelationAnalysis<Algo> aa(c.g,
                                     gen,
                                     c.thinnings,
                                     c.min_snapshots,
                                     algolabel,
                                     c.graphlabel,
                                     graphseed,
                                     seed,
                                     c.output_fn_prefix,
                                     pu_id,
                                     c.switches_per_edge,
                                     c.max_snapshots,
                                     c.skip_non_orig);
}

int main(int argc, char *argv[]) {
    tlx::CmdlineParser cp;
    cp.set_description("Autocorrelation Analysis");

    unsigned algo = 0;
    cp.add_param_unsigned("algo", algo, "Algorithm; 1=Robin, 2=Global");

    unsigned runs = 0;
    cp.add_param_unsigned("runs", runs, "Runs");

    unsigned n;
    cp.add_param_unsigned("n", n, "Number of Nodes");

    double gamma = -2.01;
    cp.add_param_double("gamma", gamma, "Degree Exponent Gamma");

    unsigned graphseed = std::random_device{}();
    cp.add_unsigned("graphseed", graphseed, "Graph Generation Seed");

    unsigned seed = std::random_device{}();
    cp.add_unsigned("seed", seed, "Autocorrelation Seed");

    unsigned min_snapshots;
    cp.add_unsigned("minsnaps", min_snapshots, "Minimum Number of Snapshots / Thinning");

    unsigned max_snapshots;
    cp.add_unsigned("maxsnaps", max_snapshots, "Maximum Number of Snapshots / Thinning");

    unsigned switches_per_edge = 1;
    cp.add_unsigned("switchesperedge", switches_per_edge, "Switches / Edge");

    bool skip_non_orig = false;
    cp.add_flag("skipnonorig", skip_non_orig, "Skip Non-Original Edges");

    int pus = 1;
    cp.add_int("pus", pus, "Number of PUs");

    std::string output_fn_prefix;
    cp.add_param_string("outputfnprefix", output_fn_prefix, "Output Filename Prefix");

    std::vector<std::string> thinnings_str;
    cp.add_param_stringlist("thinnings", thinnings_str, "Thinning Values e.g. --thinnings 1 2 3 4");


    // evaluate command line parser
    if (!cp.process(argc, argv)) {
        return -1;
    }

    std::cout << "# successfully processed command line arguments" << std::endl;

    if (gamma > 2)
        gamma = -gamma;

    Aux::Random::setSeed(graphseed, true);

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

    const size_t maxdeg = static_cast<size_t>(std::pow(n, 1./(-gamma - 1.)));
    std::cout << "# setting maximum degree to " << maxdeg << std::endl;
    const std::string graphname =  "pld" + std::to_string(-gamma);

    // run autocorrelation analysis
#pragma omp parallel for num_threads(pus)
    for (unsigned run = 0; run < runs; run++) {
        const int pu_id = omp_get_thread_num();
        std::cout << "# run " << run << std::endl;
        Aux::Random::setSeed(graphseed, true);
        std::mt19937_64 gen((seed ? pus == 1 : Aux::Random::integer()));

        NetworKit::Graph g;
        bool done = false;
        while (!done) {
            try {
                {
                    NetworKit::PowerlawDegreeSequence ds_gen(1, maxdeg, gamma);
                    std::vector<NetworKit::count> ds;
                    ds_gen.run();
                    ds = ds_gen.getDegreeSequence(n);
                    g = NetworKit::HavelHakimiGenerator(ds, false).generate();
                }
                std::cout << "# successfully generated graph instance" << std::endl;
                done = true;
            } catch (const std::runtime_error& error) {
                std::cout << "# failed to generate graph instance" << std::endl;
            }
        }

        const autocorrelation_config_t config(g, thinnings, min_snapshots, max_snapshots, graphname, switches_per_edge, skip_non_orig, output_fn_prefix);


        switch (algo) {
            case 1:
                run_pld_autocorrelation_analysis<es::AlgorithmVectorSet<tsl::robin_set<es::edge_t, es::edge_hash_crc32>>>
                        (config, gen, "ES-Robin", graphseed, seed, pu_id);
                break;
            case 2:
                run_pld_autocorrelation_analysis<es::AlgorithmParallelGlobalNoWaitV4>
                        (config, gen, "ES-Global-NoWait-V4", graphseed, seed, pu_id);
                break;
            default:
                break;
        }
    }

    return 0;
}
