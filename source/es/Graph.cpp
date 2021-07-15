#include <es/Graph.hpp>

#include <vector>

#include <range/v3/all.hpp>

namespace es {

std::pair<Graph, std::vector<node_t>> Graph::get_sorted_by_degree(bool ascending) const {
    auto sorted = ranges::views::zip(degrees_, ranges::views::ints(0u, number_of_nodes())) //
                  | ranges::to<std::vector> //
                  | ranges::actions::sort;

    if (!ascending)
        ranges::reverse(sorted);

    auto mapping = sorted | ranges::views::values | ranges::to<std::vector>;

    auto result = *this;
    auto inv_mapping = result.apply_mapping(mapping);
    return {result, inv_mapping};
}

std::vector<node_t> Graph::apply_mapping(std::span<node_t> mapping) {
    std::vector<node_t> inv_mapping(number_of_nodes());
    for (auto[i, mapped] : ranges::views::enumerate(mapping)) {
        inv_mapping[mapped] = i;
    }

    degrees_ = inv_mapping //
               | ranges::views::transform([&](auto i) { return degrees_[i]; }) //
               | ranges::to<std::vector>;

    edges_ = edges_ //
             | ranges::views::transform([&](edge_t e) {
        auto[u, v] = to_nodes(e);
        return to_edge(mapping[u], mapping[v]);
    }) //
             | ranges::to<std::vector>;

    return inv_mapping;
}

}