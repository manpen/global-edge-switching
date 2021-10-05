#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <es/Graph.hpp>

#include <charconv>
#include <networkit/io/MemoryMappedFile.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <tlx/die.hpp>
#include <range/v3/view.hpp>

using namespace es;

int main(int argc, char* argv[]) {
    std::vector<es::edge_t> edges;
    edges.reserve(1llu << 33);
    auto min_node = std::numeric_limits<node_t>::max();
    auto max_node = std::numeric_limits<node_t>::min();

    {
        NetworKit::MemoryMappedFile in(argv[1]);
        auto it = in.cbegin();
        for (size_t line = 1;; ++line) {
            auto prune_whitespace = [&] {
                for (; it != in.cend() && (*it == ' ' || *it == ','); ++it);
            };

            auto read_num = [&] {
                node_t num;
                die_if(it == in.cend());

                auto res = std::from_chars(it, in.cend(), num);
                die_unless(res.ec == std::errc());

                it = res.ptr;

                min_node = std::min(num, min_node);
                max_node = std::max(num, max_node);

                return num;
            };

            prune_whitespace();
            if (it == in.cend())
                break;


            auto u = read_num();
            prune_whitespace();
            auto v = read_num();
            prune_whitespace();

            edges.push_back(to_edge(u, v));

            if (it == in.cend())
                break;

            die_unless(*it == '\n' || *it == '\r');
            while (it != in.cend() && (*it == '\n' || *it == '\r')) {
                ++it;
                prune_whitespace();
            }

            if (line % 100'000'000 == 0)
                std::cout << "Line " << line << std::endl;
        }
    }


    std::cout << "Sort " << edges.size() << " edges" << std::endl;
    Aux::Parallel::sort(edges.begin(), edges.end());

    auto end = std::unique(edges.begin(), edges.end());
    edges.erase(end, edges.end());
    std::cout << "Found " << edges.size() << "edges" << std::endl;

    std::vector<node_t> degrees(1 + max_node - min_node);
    {
        std::ofstream edges_out(std::string(argv[1]) + ".edges");
        edges_out << "%n=" << degrees.size() << ",m=" << edges.size() << "\n";

        for (auto e: edges) {
            auto[u, v] = to_nodes(e);
            u -= min_node;
            v -= min_node;
            degrees[u]++;
            degrees[v]++;

            edges_out << u << ',' << v << '\n';
        }
    }

    std::map<size_t, size_t> deg_distr;
    for(auto deg : degrees)
        if (deg)
            ++deg_distr[deg];

    std::ofstream deg_out(std::string(argv[1]) + ".degs");
    deg_out << "%n=" << degrees.size() << ",m=" << edges.size() << "\n";
    for (auto [deg, n] : deg_distr)
        deg_out << deg << "," << n << "\n";

    return 0;
}