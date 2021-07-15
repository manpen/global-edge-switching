#include <tlx/die.hpp>

#include <es/Graph.hpp>
#include <es/Generators.hpp>

#include <range/v3/all.hpp>

void test_sort() {
    std::mt19937_64 gen(0);

    for(int repeat = 1; repeat < 10; ++repeat) {
        es::Graph g = es::generate_gnp(repeat * 1000, 0.05, gen);

        auto[sorted_g, inv_mapping] = g.get_sorted_by_degree();

        die_unequal(ranges::accumulate(g.degrees(), 0u), ranges::accumulate(sorted_g.degrees(), 0u));
        die_unless(ranges::is_sorted(sorted_g.degrees(), std::greater<es::node_t>{}));

        die_unequal(g.edges().size(), sorted_g.edges().size());
        for (auto[oe, me] : ranges::views::zip(g.edges(), sorted_g.edges())) {
            auto[u, v] = es::to_nodes(me);
            die_unequal(oe, es::to_edge(inv_mapping[u], inv_mapping[v]));
        }
    }
}

int main() {
    test_sort();
}