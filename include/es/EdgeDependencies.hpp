#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

namespace es {

template<typename HashFcn>
struct EdgeDependencies {
public:
    EdgeDependencies(size_t num_edges, double load_factor) : dependencies_(load_factor * 2 * num_edges) {}

    void cleanup_last_round(size_t num_threads = omp_get_max_threads()) {
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < dependencies_.size(); ++i) {
            auto& dep = dependencies_[i];
            dep.edge = kEmpty_;
            dep.resolved = true;
            dep.erasing_switch = 0;
            dep.inserting_switches.clear();
            dep.unlocked = true;
        }
    }

    void announce_erase(edge_t edge, size_t sid) {
        auto [iter, _] = insert_or_find(edge);
        iter->erasing_switch = sid;
        iter->resolved = false;
    }

    void announce_insert(edge_t edge, size_t sid) {
        auto [iter, _] = insert_or_find(edge);
        bool unlocked;
        do {
            unlocked = true;
            iter->unlocked.compare_exchange_weak(unlocked, false,
                                                 std::memory_order_release,
                                                 std::memory_order_consume);
        } while (!unlocked);
        iter->inserting_switches.emplace_back(sid, false);
        iter->unlocked.store(true, std::memory_order_release);
    }

    void announce_erase_succeeded(edge_t edge) {
        auto iter = find_existing(edge);
        iter->resolved = true;
    }

    void announce_erase_failed(edge_t edge) {
        auto iter = find_existing(edge);
        iter->erasing_switch = kNone_;
    }

    void announce_insert_succeeded(edge_t edge, size_t sid) {
        auto iter = find_existing(edge);
        for (auto& insert : iter->inserting_switches) {
            if (insert.first == sid) {
                insert.second = true;
            }
        }
    }

    void announce_insert_failed(edge_t edge, size_t sid) {
        auto iter = find_existing(edge);
        for (auto& insert : iter->inserting_switches) {
            if (insert.first == sid) {
                insert.first = kNone_;
            }
        }
    }

    std::pair<size_t, bool> lookup_erase(edge_t edge) const {
        const auto iter = find_existing(edge);
        return {iter->erasing_switch, iter->resolved};
    }

    std::pair<size_t, bool> lookup_insert(edge_t edge) const {
        const auto iter = find_existing(edge);
        size_t earliest = kNone_;
        bool resolved = false;
        for (auto& insert : iter->inserting_switches) {
            if (insert.first < earliest) {
                earliest = insert.first;
                resolved = insert.second;
            }
        }
        return {earliest, resolved};
    }

private:
    const static size_t kEmpty_ = std::numeric_limits<edge_t>::max();
    const static size_t kNone_ = std::numeric_limits<size_t>::max();

    struct EdgeDependency {
        std::atomic<edge_t> edge = kEmpty_;
        bool resolved = true;
        size_t erasing_switch = 0;
        std::vector<std::pair<size_t, bool>> inserting_switches;
        std::atomic<bool> unlocked = true;
    };

    std::vector<EdgeDependency> dependencies_;
    HashFcn hash_func_;

    using iterator_t = typename std::vector<EdgeDependency>::iterator;
    using const_iterator_t = typename std::vector<EdgeDependency>::const_iterator;

    std::pair<iterator_t, bool> insert_or_find(edge_t edge) {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            edge_t edge_at_iter = iter->edge.load(std::memory_order_acquire);
            if (edge_at_iter == kEmpty_) {
                iter->edge.compare_exchange_weak(edge_at_iter, edge,
                                                 std::memory_order_release,
                                                 std::memory_order_consume);
                if (edge_at_iter == kEmpty_) return {iter, true};
            }
            if (edge_at_iter == edge)  return {iter, false};
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
    }

    iterator_t find_existing(edge_t edge) {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            edge_t edge_at_iter = iter->edge.load(std::memory_order_acquire);
            if (edge_at_iter == edge)  return iter;
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
    }

    const_iterator_t find_existing(edge_t edge) const {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            edge_t edge_at_iter = iter->edge.load(std::memory_order_acquire);
            if (edge_at_iter == edge) return iter;
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
    }
};

}