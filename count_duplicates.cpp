#include <iostream>
#include <string>
#include <unordered_set>
#include <fstream>
#include <es/Graph.hpp>

using namespace es;

int main(int argc, char* argv[]) {
    std::string line;

    std::ifstream ifile{argv[1]};

    size_t multiedges = 0;
    std::unordered_set<edge_t> edges;
    edges.reserve(10'000'000);
    while (getline(ifile, line)) {
        if (line.starts_with('%'))
            continue;

        size_t sep = line.find(',');
        node_t u = std::stoll(line.substr(0, sep));
        node_t v = std::stoll(line.substr(sep + 1));

        auto edge = to_edge(u, v);

        multiedges += !edges.insert(edge).second;
    }

    std::cout << argv[1] << ":" << multiedges << std::endl;
    return !!multiedges;
}