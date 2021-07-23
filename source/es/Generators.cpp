#include <es/Generators.hpp>

namespace es {

NetworKit::Graph generate_gnp(node_t n, double p, std::mt19937_64 &gen) {
    NetworKit::Graph result(n, n * p * n);

    std::geometric_distribution<edge_t> distr{p};

    node_t u = 1;
    edge_t v = distr(gen);
    while (true) {
        while (v >= u) {
            v -= u;
            ++u;
            if (u == n) return result;
        }

        result.addEdge(u, v);

        v += 1 + distr(gen);
    }
}

}
