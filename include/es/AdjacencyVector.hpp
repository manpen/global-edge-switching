#pragma once
#ifndef UNIFORM_PLD_SAMPLING_ADJACENCYVECTOR_H
#define UNIFORM_PLD_SAMPLING_ADJACENCYVECTOR_H

#include <iosfwd>
#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include <span>

#include <es/Graph.hpp>

#include <tlx/define.hpp>
#include <range/v3/view.hpp>
#include <range/v3/algorithm.hpp>

namespace es {
using AdjacencyList = std::vector<std::unordered_set<node_t>>;

class ConfigurationModel;

class AdjacencyVector {
    using boundary_t = std::pair<edge_t, edge_t>;
    friend ConfigurationModel;

public:
    AdjacencyVector() : boundaries_{0} {}

    AdjacencyVector(AdjacencyVector &&) = default;

    AdjacencyVector(const std::vector<node_t> &degree_sequence, node_t slack_added_to_each_node_t = 0) {
        boundaries_.reserve(degree_sequence.size() + 1);
        edge_t num_edges = 0;
        for (auto d : degree_sequence) {
            boundaries_.emplace_back(num_edges, num_edges);
            num_edges += d + std::max(d, slack_added_to_each_node_t);
        }
        boundaries_.emplace_back(num_edges, num_edges); // sentinel to avoid boundary checks
        adj_vec_.resize(num_edges);
    }

    AdjacencyVector &operator=(AdjacencyVector &&) = default;

private: // copy is expensive, so make using it explicit via copy()
    AdjacencyVector(const AdjacencyVector &) = default;

    AdjacencyVector &operator=(const AdjacencyVector &) = default;

public:
    //! return copy of the data structure
    AdjacencyVector copy() { return {*this}; }

    //! returns the number of nodes supported
    [[nodiscard]] edge_t num_nodes() const noexcept {
        return boundaries_.empty() ? 0 : boundaries_.size() - 1;
    }

    //! returns maximum number of neighbors that can be assigned to this node_t
    [[nodiscard]] edge_t capacity(node_t u) const noexcept {
        assert(u < num_nodes());
        return boundaries_[u + 1].first - boundaries_[u].first;
    }

    //! returns degree of node_t u, i.e. number of neighbors
    [[nodiscard]] edge_t degree(node_t u) const noexcept {
        assert(u < num_nodes());
        return boundaries_[u].second - boundaries_[u].first;
    }

    //! sequence from 0 to n-1
    [[nodiscard]] auto nodes() const noexcept {
        return ranges::views::ints(node_t(0), static_cast<node_t>(num_nodes()));
    }

    //! returns a sequence of all degrees
    [[nodiscard]] auto degrees() const noexcept {
        return nodes() | ranges::views::transform([&](node_t u) { return degree(u); });
    }

    //! view of all neighbors of node_t u
    [[nodiscard]] auto neighbors(node_t u) const noexcept {
        assert(u < num_nodes());
        return adj_vec_ | ranges::views::slice(boundaries_[u].first, boundaries_[u].second);
    }

    //! view of all neighbors of node_t u
    [[nodiscard]] auto neighbors(node_t u) noexcept {
        assert(u < num_nodes());
        return adj_vec_ | ranges::views::slice(boundaries_[u].first, boundaries_[u].second);
    }

    //! view of all neighbors of node_t u
    [[nodiscard]] auto unique_neighbors(node_t u) const noexcept {
        assert(u < num_nodes());
        return neighbors(u) | ranges::views::unique;
    }

    //! view of views, i.e. [neighbors(0), ...]
    [[nodiscard]] auto neighborhoods() const noexcept {
        return nodes() | ranges::views::transform([&](node_t u) { return neighbors(u); });
    }

    //! return view of std::pair<node_t, node_t> that contains all edges in an arbitrary order
    //! all edges (u, v) hold u <= v
    [[nodiscard]] auto edges() const noexcept {
        return nodes() | ranges::views::for_each([&](node_t u) { //
            return neighbors(u) | ranges::views::filter([=](auto v) { return u <= v; }) |
                   ranges::views::transform([=](node_t v) { return std::pair<node_t, node_t>{u, v}; });
        });
    }

    //! returns true if edge (u, v) exists
    [[nodiscard]] bool has_edge(node_t u, node_t v) const noexcept {
        if (u < v) std::swap(u, v);
        auto neigh = neighbors(u);
        auto it = ranges::lower_bound(neigh, v);
        return it != neigh.end() && *it == v;
    }

    void add_edge(node_t u, node_t v) noexcept {
        if (u < v) std::swap(u, v);
        assert(degree(u) < capacity(u));

        auto begin = adj_vec_.begin() + boundaries_[u].first;
        auto it = adj_vec_.begin() + boundaries_[u].second;
        boundaries_[u].second++;

        // insertion sort
        if (begin == it) {
            *begin = v;
            return;
        }

        *it = v;
        while (true) {
            auto prev = std::prev(it);
            if (*prev <= *it) break;
            std::swap(*prev, *it);
            if (prev == begin) break;
            it = prev;
        }

        assert(ranges::is_sorted(neighbors(u)));
    }

    //! remove edge (u, v) and (v, u); if all_occurrences == true all instances of an edge
    //! with possibly multiplicity > 1 are removed
    //! returns the number of instances (u, v) was removed
    size_t remove_edge(node_t u, node_t v, bool all_occurrences = false) noexcept {
        if (u < v) std::swap(u, v);
        auto neigh = neighbors(u);
        auto it = ranges::find(neigh, v);

        assert(it != neigh.end());
        auto it2 = it + 1;

        if (all_occurrences) {
            while (it2 != neigh.end() && *it2 == v)
                ++it2;
        }

        std::move(it2, neigh.end(), it);

        boundaries_[u].second -= (it2 - it);
        assert(boundaries_[u].first <= boundaries_[u].second);

        return static_cast<size_t>(it2 - it);
    }

    //! erase all neighbors while keeping capacities
    void clear() noexcept {
        for (auto &b : boundaries_)
            b.second = b.first;
    }


    //! iterate through all edges stored; for each edge {u, v} with multiplicity m
    //! the callback cb(u, v, m) is invoked; each edge is either called with u <= v but not for v >= u
    template<typename Callback>
    void for_each(Callback cb) const noexcept {
        for (node_t u : nodes()) {
            for_each<Callback>(u, cb);
        }
    }

    template<typename Callback>
    void for_each(node_t u, Callback cb) const noexcept {
        auto neigh = neighbors(u);
        for(auto v : neigh)
            cb(u, v);
    }


private:
    std::vector<boundary_t> boundaries_;
    std::vector<node_t> adj_vec_;

};

}

#endif // UNIFORM_PLD_SAMPLING_ADJACENCYVECTOR_H
