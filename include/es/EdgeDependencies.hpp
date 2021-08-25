#pragma once

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

namespace es {

template<typename HashFcn>
struct EdgeDependencies {
public:
    const static size_t kNone_ = std::numeric_limits<size_t>::max();
    const static edge_t kEmpty_ = std::numeric_limits<edge_t>::max();
    const static int kLocked_ = std::numeric_limits<int>::max();

    EdgeDependencies(size_t num_edges, double load_factor) : dependencies_(load_factor * 4 * num_edges), round_(0) {}

    void next_round() {
        round_++;
    }

    void announce_erase(edge_t edge, size_t sid) {
        auto iter = insert(edge);
        iter->type = ERASE;
        iter->switch_id = sid;
        iter->resolved = false;
    }

    void announce_insert(edge_t edge, size_t sid) {
        auto iter = insert(edge);
        iter->type = INSERT;
        iter->switch_id = sid;
        iter->resolved = false;
    }

    void announce_erase_succeeded(edge_t edge, size_t sid) {
        auto iter = find_erase(edge, sid);
        iter->resolved = true;
    }

    void announce_erase_failed(edge_t edge, size_t sid) {
        auto iter = find_erase(edge, sid);
        iter->switch_id = kNone_;
    }

    void announce_insert_succeeded(edge_t edge, size_t sid) {
        auto iter = find_insert(edge, sid);
        iter->resolved = true;
    }

    void announce_insert_failed(edge_t edge, size_t sid) {
        auto iter = find_insert(edge, sid);
        iter->switch_id = kNone_;
    }

    std::pair<size_t, bool> lookup_erase(edge_t edge) const {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            if (iter->round < round_) return {0, true};
            if (iter->edge == edge && iter->type == ERASE) break;
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
        return {iter->switch_id, iter->resolved};
    }

    std::pair<size_t, bool> lookup_insert(edge_t edge) const {
        size_t earliest = kNone_;
        bool resolved = false;
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            if (iter->round < round_) break;
            if (iter->edge == edge && iter->type == INSERT && iter->switch_id < earliest) {
                earliest = iter->switch_id;
                resolved = iter->resolved;
            }
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
        return {earliest, resolved};
    }

private:
    enum DependencyType : char {
        NONE = 0,
        ERASE = 1,
        INSERT = 2
    };

    struct EdgeDependency {
        std::atomic<int> round = -1;
        edge_t edge = kEmpty_;
        size_t switch_id = 0;
        bool resolved = true;
        DependencyType type = NONE;
    };

    std::vector<EdgeDependency> dependencies_;
    HashFcn hash_func_;
    int round_;

    using iterator_t = typename std::vector<EdgeDependency>::iterator;
    using const_iterator_t = typename std::vector<EdgeDependency>::const_iterator;

    iterator_t insert(edge_t edge) {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            int round_at_iter = iter->round.load(std::memory_order_acquire);
            if (round_at_iter < round_) {
                iter->round.compare_exchange_strong(round_at_iter, round_,
                                                    std::memory_order_release,
                                                    std::memory_order_consume);
                if (round_at_iter < round_) {
                    iter->edge = edge;
                    return iter;
                }
            }
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
    }

    iterator_t find_erase(edge_t edge, size_t sid) {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            if (iter->edge == edge && iter->type == ERASE && iter->switch_id == sid) return iter;
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
    }

    iterator_t find_insert(edge_t edge, size_t sid) {
        size_t bucket = hash_func_(edge) % dependencies_.size();
        auto iter = dependencies_.begin() + bucket;
        while (true) {
            if (iter->edge == edge && iter->type == INSERT && iter->switch_id == sid) return iter;
            iter++;
            if (iter == dependencies_.end()) iter = dependencies_.begin();
        }
    }
};

}