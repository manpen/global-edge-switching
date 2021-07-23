#pragma once

#include <networkit/graph/Graph.hpp>

#include <vector>
#include <cstdint>
#include <nmmintrin.h>

namespace es {

using node_t = std::uint32_t;
using edge_t = std::uint64_t;

inline edge_t to_edge(node_t a, node_t b) {
    bool swap = (a > b);
    const node_t tmp = (a ^ b) * swap;
    a ^= tmp;
    b ^= tmp;
    return (static_cast<edge_t>(a) << 32) | b;
}

inline std::pair<node_t, node_t> to_nodes(edge_t e) {
    return {static_cast<node_t>(e >> 32), static_cast<node_t>(e)};
}

inline std::vector<node_t> degree_sequence_of(const NetworKit::Graph& graph) {
    std::vector<node_t> degrees;
    degrees.reserve(2 * graph.numberOfEdges());
    graph.forNodes([&](NetworKit::node v){
        degrees.push_back(graph.degree(v));
    });
    return degrees;
}

struct edge_hash_crc32 {
    size_t operator()(edge_t e) const noexcept {
        auto l = _mm_crc32_u64(0, e);
        auto h = _mm_crc32_u64(l, e);

        return (l | (h << 32));
    }
};

}