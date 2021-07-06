#pragma once

#include <nmmintrin.h>

namespace es {

using node_t = uint32_t;
using edge_t = uint64_t;

inline edge_t to_edge(node_t a, node_t b) {
    bool swap = (a > b);
    const node_t tmp = (a ^ b) * swap;
    a ^= tmp;
    b ^= tmp;
    return (static_cast<edge_t>(a) << 32) | b;
}

inline std::pair <node_t, node_t> to_nodes(edge_t e) {
    return {static_cast<node_t>(e >> 32), static_cast<node_t>(e)};
}

struct edge_hash_crc32 {
    size_t operator()(edge_t e) const noexcept {
        auto l = _mm_crc32_u64(0, e);
        auto h = _mm_crc32_u64(l, e);

        return (l | (h << 32));
    }
};

struct Graph {
    Graph(node_t number_of_nodes, size_t number_of_edges = 0)
        : degrees_(number_of_nodes)
    {
        if (number_of_edges)
            edges_.reserve(number_of_edges);
    }


    [[nodiscard]] node_t number_of_nodes() const noexcept {return degrees_.size();}
    [[nodiscard]] node_t number_of_edges() const noexcept {return edges_.size();}

    void add_edge(edge_t e) {
        auto [u, v] = to_nodes(e);
        add_edge(u, v);
    }

    void add_edge(node_t u, node_t v) {
        ++degrees_[u];
        ++degrees_[v];
        edges_.push_back(to_edge(u, v));
    }

    [[nodiscard]] const auto& edges() const noexcept {return edges_;}

    [[nodiscard]] node_t degree_of(node_t u) const noexcept {
        return degrees_[u];
    }

private:
    std::vector<edge_t> edges_;
    std::vector<node_t> degrees_;
};

}